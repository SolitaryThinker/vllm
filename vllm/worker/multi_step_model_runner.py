import dataclasses
from dataclasses import dataclass, field
from typing import (TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Type,
                    TypeVar, Union)
try:
    from vllm.attention.backends.flash_attn import FlashAttentionMetadata
except ModuleNotFoundError:
    # vllm_flash_attn is not installed, use the identical ROCm FA metadata
    from vllm.attention.backends.rocm_flash_attn import (
        ROCmFlashAttentionMetadata as FlashAttentionMetadata)

from ..model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.worker.model_runner_base import (
    ModelRunnerBase, ModelRunnerInputBase, ModelRunnerInputBuilderBase,
    _add_attn_metadata_broadcastable_dict,
    _add_sampling_metadata_broadcastable_dict,
    _init_attn_metadata_from_tensor_dict,
    _init_sampling_metadata_from_tensor_dict)
from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata
from vllm.worker.model_runner import GPUModelRunnerBase
from vllm.logger import init_logger
from vllm.distributed import get_pp_group, get_tp_group
from vllm.sequence import (IntermediateTensors, SamplerOutput,
                           SequenceGroupMetadata, SequenceOutput,
                           CompletionSequenceGroupOutput, Logprob)
from vllm import _custom_ops as ops

import torch

logger = init_logger(__name__)


@dataclass
class ModelOutput:
    """The output of a single model forward pass.

    The sampler_output_ready_event is set when the tensors in
    sampler_output are ready (the model+sampler forward pass has
    completed). We use the event to synchronize the GPU->CPU transfer,
    which we want to only run when the data has been written to the
    GPU tensors. Until the event is ready, the tensors in sampler_output
    will have garbage data.
    """
    sampler_output: SamplerOutput
    sampler_output_ready_event: torch.cuda.Event
    pythonized: bool = False

    def pythonize(self, input_metadata: ModelInputForGPUWithSamplingMetadata,
                  copy_stream: torch.cuda.Stream) -> None:
        """Pythonize the output. Blocking."""
        if not self.pythonized:
            self._pythonize_sampler_output_wait_on_event(
                input_metadata, copy_stream)
            self.pythonized = True

    def maybe_pythonize(self,
                        input_metadata: ModelInputForGPUWithSamplingMetadata,
                        copy_stream: torch.cuda.Stream) -> None:
        """Pythonize the output if ready, else return None. Non-blocking."""
        if not self.pythonized:
            self.pythonized = self._pythonize_sampler_output_if_event_ready(
                input_metadata, copy_stream)

    def _pythonize_sampler_output_wait_on_event(
            self, input_metadata: ModelInputForGPUWithSamplingMetadata,
            copy_stream: torch.cuda.Stream) -> None:
        self.sampler_output_ready_event.synchronize()
        with torch.cuda.stream(copy_stream):
            _pythonize_sampler_output(input_metadata, self.sampler_output)

    def _pythonize_sampler_output_if_event_ready(
            self, input_metadata: ModelInputForGPUWithSamplingMetadata,
            copy_stream: torch.cuda.Stream) -> bool:
        if self.sampler_output_ready_event.query():
            with torch.cuda.stream(copy_stream):
                _pythonize_sampler_output(input_metadata, self.sampler_output)
            return True
        return False


# @dataclass(frozen=True)
class ModelInputForGPUWithMultiStepMetadata(
        ModelInputForGPUWithSamplingMetadata):
    outputs: List[ModelOutput] = field(default_factory=list)
    is_multi_step: bool = False
    is_last_step: bool = False
    is_first_multi_step: bool = False
    step_cuda_events: List[torch.cuda.Event] = field(
        default_factory=lambda: [torch.cuda.Event(blocking=True)] * 2)

    def __init__(self, *args, **kwargs):
        self.current_step = kwargs.pop('current_step', 0)
        self.outputs = kwargs.pop('outputs', [])
        self.is_multi_step = kwargs.pop('is_multi_step', False)
        self.is_last_step = kwargs.pop('is_last_step', False)
        self.is_first_multi_step = kwargs.pop('is_first_multi_step', False)
        self.step_cuda_events = [torch.cuda.Event(blocking=True)] * 2
        super().__init__(*args, **kwargs)

    def record_step_event(self):
        self.step_cuda_events[self.current_step %
                              2] = torch.cuda.Event(blocking=True)
        self.step_cuda_events[self.current_step % 2].record()

    def wait_previous_step(self):
        self.step_cuda_events[(self.current_step + 1) % 2].wait()

    def add_sampler_output(self, sampler_output: SamplerOutput):
        self.outputs.append(
            ModelOutput(sampler_output=sampler_output,
                        sampler_output_ready_event=None,
                        pythonized=False))


class MultiStepModelRunnerBase(
        GPUModelRunnerBase[ModelInputForGPUWithMultiStepMetadata]):

    def __init__(self, base_model_runner: ModelRunnerBase, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # uses the base model runner to execute the model and wraps it with
        # multi-step logic
        self._base_model_runner: ModelRunnerBase = base_model_runner

        self.is_multi_step = self.scheduler_config.is_multi_step
        # used to copy tensors from GPU to CPU asynchronously
        self._copy_stream = torch.cuda.Stream()

    def make_model_input_from_broadcasted_tensor_dict(
            self,
            tensor_dict: Dict[str,
                              Any]) -> ModelInputForGPUWithSamplingMetadata:
        return self._base_model_runner.make_model_input_from_broadcasted_tensor_dict(
            tensor_dict)

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForGPUWithSamplingMetadata:
        return self._base_model_runner.prepare_model_input(
            seq_group_metadata_list, virtual_engine, finished_requests_ids)

    def load_model(self) -> None:
        return self._base_model_runner.load_model()

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        return self._base_model_runner.save_sharded_state(
            path, pattern, max_size)

    def save_tensorized_model(self,
                              tensorizer_config: TensorizerConfig) -> None:
        return self._base_model_runner.save_tensorized_model(tensorizer_config)

    def profile_run(self) -> None:
        return self._base_model_runner.profile_run()

    def remove_all_loras(self):
        return self._base_model_runner.remove_all_loras()

    def capture_model(self, kv_caches: List[List]) -> None:
        return self._base_model_runner.capture_model(kv_caches)

    @property
    def vocab_size(self) -> int:
        return self._base_model_runner.vocab_size


class MultiStepModelRunner(MultiStepModelRunnerBase):

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForGPUWithMultiStepMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:
        """ 
        Execute the model for a single step and update multi-step
        metadata
        """
        assert num_steps == 1, "MultiStepModelRunner only supports num_steps=1"

        # path for warm up runs
        if not model_input.is_multi_step:
            return self._base_model_runner.execute_model(
                model_input, kv_caches, intermediate_tensors, num_steps)

        debug_multi_step = False
        if debug_multi_step:
            print(
                f'=======step {model_input.current_step} for {model_input.virtual_engine}============='
            )
            print(f'is_multi_step: {model_input.is_multi_step}')
            print(f'is_last_step: {model_input.is_last_step}')
            print(f'current_step: {model_input.current_step}')
            print(f'is_first_multi_step: {model_input.is_first_multi_step}')

        # some pre-execute model logic for multi-step:
        #   - if it's the first step, we need to reset the sampling tensors
        #   - if it's not the first step, we need to advance the step using the
        #   appended sampler output from last iteration
        #   - also maybe pythonize if CPU is ahead of GPU

        # explicitly block on the previous step's forward to make sure we
        # don't clobber any GPU tensors still in use
        model_input.wait_previous_step()
        model_input = self._advance_step(
            model_input, model_input.outputs[-1].sampler_output)
        if model_input.sampling_metadata:
            model_input.sampling_metadata.reuse_sampling_tensors = True

        # make sure we skip the sampler on the lask rank and only pythonize
        # if CPU is ahead
        if self.is_driver_worker and get_pp_group().is_last_rank:
            self._base_model_runner.model.sampler.include_gpu_probs_tensor = True
            model_input.sampling_metadata.skip_sampler_cpu_output = True
            for output in model_input.outputs:
                output.maybe_pythonize(model_input, self._copy_stream)

        current_stream = torch.cuda.current_stream()

        # Execute the model
        output = self._base_model_runner.execute_model(model_input,
                                                       kv_caches,
                                                       intermediate_tensors,
                                                       num_steps=1)

        # record the event for the current step so that the next step can sync
        model_input.record_step_event()

        if get_pp_group().is_last_rank and self.is_driver_worker:
            assert len(
                output) == 1, "MultiStepModelRunner only supports single step"

            # event for the pythonization so that we only pythonize if the
            # tensors are ready. May be able to be combined with the step event
            output_ready_event = torch.cuda.Event()
            output_ready_event.record(current_stream)
            model_input.outputs.append(
                ModelOutput(output[0], output_ready_event, False))

        model_input.current_step += 1

        if not get_pp_group().is_last_rank:
            # Should be IntermediateTensors
            assert isinstance(output, IntermediateTensors)
            return output
        if not self.is_driver_worker:
            return []

        # Pythonize the output and block if needed since it is the last step
        if model_input.is_last_step:
            outputs = []
            for output in model_input.outputs:
                output.pythonize(model_input, self._copy_stream)
                outputs.append(output.sampler_output)
            return outputs

        # should be [SamplerOutput]
        return output

    def _update_flash_attn_metadata(self, attn_metadata, num_seqs,
                                    num_queries):
        assert isinstance(attn_metadata, FlashAttentionMetadata)

        if num_seqs != num_queries:
            assert num_seqs > num_queries
            assert attn_metadata.use_cuda_graph

        assert attn_metadata.num_prefills == 0
        assert attn_metadata.num_prefill_tokens == 0
        assert attn_metadata.num_decode_tokens == num_seqs
        assert attn_metadata.slot_mapping.shape == (num_seqs, )

        assert len(attn_metadata.seq_lens) == num_seqs
        assert attn_metadata.seq_lens_tensor.shape == (num_seqs, )
        assert attn_metadata.max_query_len == 1
        assert attn_metadata.max_prefill_seq_len == 0
        assert attn_metadata.max_decode_seq_len == max(attn_metadata.seq_lens)

        assert attn_metadata.query_start_loc.shape == (num_queries + 1, )
        assert attn_metadata.seq_start_loc.shape == (num_seqs + 1, )

        assert attn_metadata.context_lens_tensor.shape == (num_queries, )

        assert attn_metadata.block_tables.shape[0] == num_seqs

        # Update query lengths. Note that we update only queries and not seqs,
        # since tensors may be padded due to captured cuda graph batch size
        for i in range(num_queries):
            attn_metadata.seq_lens[i] += 1
        attn_metadata.max_decode_seq_len = max(attn_metadata.seq_lens)

    def _update_sampling_metadata(self, sampling_metadata, num_seqs,
                                  num_queries):

        assert sampling_metadata.num_prompts == 0
        assert len(sampling_metadata.seq_groups) == num_queries
        assert sampling_metadata.selected_token_indices.shape == (
            num_queries, )
        # assert sampling_metadata.categorized_sample_indices == TODO: Add if needed # noqa: E501

        # Verify that all sequences are decodes
        for i in range(num_queries):
            seq_group = sampling_metadata.seq_groups[i]

            assert seq_group.is_prompt is False  # No prompt
            assert seq_group.prompt_logprob_indices == []  # No prompt
            assert seq_group.sample_indices == [i]  # Simple
            assert seq_group.seq_len is None  # Decode
            assert seq_group.query_len is None  # Decode

    def _advance_step(
            self, model_input: ModelInputForGPUWithSamplingMetadata,
            out: SamplerOutput) -> ModelInputForGPUWithSamplingMetadata:
        # model_input.current_step += 1
        assert model_input.seq_lens is not None
        assert model_input.query_lens is not None
        assert model_input.attn_metadata is not None

        num_seqs = len(model_input.seq_lens)
        num_queries = len(model_input.query_lens)
        attn_metadata = model_input.attn_metadata
        assert isinstance(attn_metadata, FlashAttentionMetadata)
        self._update_flash_attn_metadata(attn_metadata, num_seqs, num_queries)

        # Update GPU tensors
        ops.advance_step(num_seqs=num_seqs,
                         num_queries=num_queries,
                         block_size=self.block_size,
                         input_tokens=model_input.input_tokens,
                         sampled_token_ids=out.sampled_token_ids,
                         input_positions=model_input.input_positions,
                         seq_lens=attn_metadata.seq_lens_tensor,
                         slot_mapping=attn_metadata.slot_mapping,
                         block_tables=attn_metadata.block_tables)

        # Update sampling_metadata
        # model_input.seq_lens = attn_metadata.seq_lens
        # sampling_metadata = model_input.sampling_metadata
        # self._update_sampling_metadata(sampling_metadata, num_seqs,
        #                                num_queries)
        for i in range(num_queries):
            model_input.seq_lens[i] = attn_metadata.seq_lens[i]

        return model_input


def _pythonize_sampler_output(
        model_input: ModelInputForGPUWithSamplingMetadata,
        output: SamplerOutput) -> SamplerOutput:
    # TODO(will): fix logprobs

    # samples generation should have been skipped
    assert not output.outputs

    # print shapes
    # print('output.sampled_token_ids', output.sampled_token_ids.shape)
    # print('output.logprobs', output.logprobs.shape)
    samples = torch.empty(*output.sampled_token_ids.shape,
                          dtype=output.sampled_token_ids.dtype,
                          device="cpu",
                          pin_memory=True)
    # prompt_logprobs = torch.empty(
    #     *output.prompt_logprobs.shape,
    #     dtype=output.prompt_logprobs.dtype,
    #     device="cpu",
    #     pin_memory=True)

    # CPU GPU sync
    # logprobs = logprobs.copy_(output.logprobs, non_blocking=False)
    samples = samples.copy_(output.sampled_token_ids, non_blocking=False)

    samples_list = samples.tolist()
    del samples
    # logprobs = logprobs.tolist()
    # print('samples', samples)
    # print('logprobs', logprobs)

    # from vllm.model_executor.layers.sampler import _get_logprobs

    # output.sampled_token_ids = output.sampled_token_ids.cpu()
    # token_ids = output.sampled_token_ids.tolist()
    sampling_metadata = model_input.sampling_metadata

    for (seq_group, sample_result) in zip(sampling_metadata.seq_groups,
                                          samples_list):
        seq_ids = seq_group.seq_ids
        # next_token_ids, parent_ids = sample_result
        next_token_ids = sample_result
        parent_ids = [0]
        seq_outputs: List[SequenceOutput] = []
        for parent_id, next_token_id in zip(parent_ids, next_token_ids):
            # print('SequenceOutput', seq_ids[parent_id], next_token_id)
            # XXX Hard coded logprob
            seq_outputs.append(
                SequenceOutput(seq_ids[parent_id], next_token_id,
                               {next_token_id: Logprob(logprob=42)}))
        # print('CompletionSequenceGroupOutput', seq_outputs)
        output.outputs.append(CompletionSequenceGroupOutput(seq_outputs, None))
    assert len(output.outputs) > 0
