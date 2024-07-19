from typing import List, Optional, Tuple
import dataclasses

import torch

from vllm.attention import AttentionMetadata
try:
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper
    from flashinfer.decode import CUDAGraphBatchDecodeWithPagedKVCacheWrapper
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
    FLASHINFER_WORKSPACE_BUFFER_SIZE = 256 * 1024 * 1024
except ImportError:
    BatchDecodeWithPagedKVCacheWrapper = None
    CUDAGraphBatchDecodeWithPagedKVCacheWrapper = None
    BatchPrefillWithPagedKVCacheWrapper = None
    FLASHINFER_WORKSPACE_BUFFER_SIZE = 0

from vllm import _custom_ops as ops
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, MultiModalConfig, ParallelConfig,
                         SchedulerConfig, PromptAdapterConfig)
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.sequence import (Logprob, CompletionSequenceGroupOutput, IntermediateTensors, 
                           SamplerOutput, SequenceGroupMetadata, SequenceOutput)
from vllm.worker.model_runner import (ModelInputForGPUWithSamplingMetadata,
                                      ModelRunner)

logger = init_logger(__name__)


# TODO(will) use this class for T1DraftModelRunner
class MultiStepModelRunner(ModelRunner):
    """Specialized model runner for multi-step decoding model.
    Since the draft model always execute k forward passes consecutively to
    generate k speculative tokens in a single speculative decoding step,
    we could get rid of most CPU-GPU synchronization and data transfer
    overheads by keeping model input and output tensors on GPU all the time.

    This runner is still under development so there's no performance gain
    at this moment. Currently we adopt a temporary solution that caches the
    seq_group_metadata_list for multi-step execution, so that we can
    leverage existing prepare_model_input to be compatible with the current
    execution flow, but we plan to remove this cache and avoid calling
    prepare_model_input in execute_model at all.
    
    The detail development plan includes:
    1. Use "update_model_input" to update existing model_input without
       creating a new one.
    2. Improve the performance of "update_model_input" with a GPU kernel.
    3. Support TP > 1 (this requires some designs because we do not expect
       any broadcasting inside execute_model).
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        kv_cache_dtype: Optional[str] = "auto",
        prompt_adapter_config: Optional[PromptAdapterConfig] = None,
        is_driver_worker: bool = False,
        multimodal_config: Optional[MultiModalConfig] = None,
        return_hidden_states: bool = False,
    ):
        if return_hidden_states:
            raise ValueError(
                "return_hidden_states is not supported for MultiStepModelRunner."
            )

        super().__init__(
            model_config=model_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            cache_config=cache_config,
            load_config=load_config,
            lora_config=lora_config,
            kv_cache_dtype=kv_cache_dtype,
            is_driver_worker=is_driver_worker,
            prompt_adapter_config=prompt_adapter_config,
            multimodal_config=multimodal_config,
            return_hidden_states=return_hidden_states,
        )

        # TODO: Remove this cache when we are able to update model_input
        # directly in advance_step.
        self.cached_seq_group_metadata_list: Optional[
            List[SequenceGroupMetadata]] = None

        # self.cached_block_tables: Optional[torch.Tensor] = None
        # self.cached_seq_lens: Optional[torch.Tensor] = None

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None,
    ) -> ModelInputForGPUWithSamplingMetadata:
        """A temporary solution that caches the seq_group_metadata_list
        for multi-step execution.
        TODO: In-place update model_input and remove this function.
        """
        self.cached_seq_group_metadata_list = seq_group_metadata_list
        # self.cached_block_tables, self.cached_seq_lens = self._cache_block_tables(seq_group_metadata_list)
        return super().prepare_model_input(
            seq_group_metadata_list,
            finished_requests_ids=finished_requests_ids)

    def update_model_input(
            self, model_input: ModelInputForGPUWithSamplingMetadata,
            last_output: SamplerOutput
    ) -> ModelInputForGPUWithSamplingMetadata:
        """Prepare the model inputs for the next step.
        TODO: In-place update model_input instead of calling
        prepare_model_input.
        """

        # Append the output token to the sequence data.
        assert self.cached_seq_group_metadata_list is not None
        for seq_group_metadata, sequence_group_outputs in zip(
                self.cached_seq_group_metadata_list, last_output.outputs):
            seq_group_metadata.is_prompt = False

            for seq_output in sequence_group_outputs.samples:
                seq = seq_group_metadata.seq_data[seq_output.parent_seq_id]

                token_id = seq_output.output_token
                token_logprob = seq_output.logprobs[token_id]

                seq.append_token_id(token_id, token_logprob.logprob)
                seq.update_num_computed_tokens(1)

        return self.prepare_model_input(self.cached_seq_group_metadata_list)

    def update_model_input_torch(
            self, model_input: ModelInputForGPUWithSamplingMetadata,
            last_output: SamplerOutput
    ) -> ModelInputForGPUWithSamplingMetadata:
        """Prepare the model inputs for the next step in-place.
        """
        for seq_group_metadata in self.cached_seq_group_metadata_list:
            model_input.input_tokens
            # seq_group_metadata.prepare_model_input(model_input)
        # model_input.input_tokens.
        return model_input

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForGPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        # Since we do not broadcast data inside execute_model anymore,
        # we need to figure out the best way to support TP > 1 in this
        # case, because we will at least need to broadcast the sampled
        # tokens to all workers.
        if not self.is_driver_worker:
            raise ValueError("TP1DraftModelRunner only supports TP=1.")

        if self.lora_config:
            raise ValueError("LoRA is not supported for MultiStepModelRunner.")
        
        if get_pp_group().is_last_rank:
            self.model.sampler.include_gpu_probs_tensor = True


        virtual_engine = model_input.virtual_engine
        outputs: List[SamplerOutput] = []
        # print(f'------doing {num_steps} steps------')
        for step in range(num_steps):
            # print(f'\t\t\t------ON step {step} -----')
            if step != num_steps - 1:
                # model_input.sampling_metadata.skip_sampler_cpu_output = False
                model_input.sampling_metadata.skip_sampler_cpu_output = True
            else:
                if get_pp_group().is_last_rank:
                    model_input.sampling_metadata.skip_sampler_cpu_output = False
                else:
                    model_input.sampling_metadata.skip_sampler_cpu_output = True

            # print('skip_sampler_cpu_output', model_input.sampling_metadata.skip_sampler_cpu_output)
            # print('=====self.attn_backend.get_name()', self.attn_backend.get_name())
            if self.attn_backend.get_name() == "flashinfer":
                assert model_input.attn_metadata is not None
                assert model_input.input_tokens is not None
                if self.flashinfer_decode_workspace_buffer is None:
                    self.flashinfer_decode_workspace_buffer = torch.empty(
                        FLASHINFER_WORKSPACE_BUFFER_SIZE,
                        dtype=torch.uint8,
                        device=self.device)
                    self.flashinfer_decode_wrapper = \
                        BatchDecodeWithPagedKVCacheWrapper(
                        self.flashinfer_decode_workspace_buffer, "NHD")
                    self.flashinfer_prefill_workspace_buffer = torch.empty(
                        FLASHINFER_WORKSPACE_BUFFER_SIZE,
                        dtype=torch.uint8,
                        device=self.device)
                    self.flashinfer_prefill_wrapper = \
                        BatchPrefillWithPagedKVCacheWrapper(
                        self.flashinfer_prefill_workspace_buffer, "NHD")

                model_input.attn_metadata.prefill_wrapper = \
                    self.flashinfer_prefill_wrapper
                if model_input.attn_metadata.use_cuda_graph:
                    assert False
                    batch_size = model_input.input_tokens.shape[0]
                    model_input.attn_metadata.decode_wrapper = self.graph_runners[
                        model_input.
                        virtual_engine][batch_size].flashinfer_decode_wrapper
                else:
                    model_input.attn_metadata.decode_wrapper = \
                        self.flashinfer_decode_wrapper

                # if step == 0:
                model_input.attn_metadata.begin_forward()
                # if step != num_steps - 1:
                # else:

            # Currently cuda graph is only supported by the decode phase.
            assert model_input.attn_metadata is not None
            prefill_meta = model_input.attn_metadata.prefill_metadata
            decode_meta = model_input.attn_metadata.decode_metadata
            if prefill_meta is None and decode_meta.use_cuda_graph:
                assert False
                assert model_input.input_tokens is not None
                graph_batch_size = model_input.input_tokens.shape[0]
                model_executable = (
                    self.graph_runners[virtual_engine][graph_batch_size])
            else:
                model_executable = self.model

            multi_modal_kwargs = model_input.multi_modal_kwargs or {}
            # print(f'\t\t\t------ON step {step} model executable')
            # print('--')
            hidden_states = model_executable(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                kv_caches=kv_caches,
                attn_metadata=model_input.attn_metadata,
                intermediate_tensors=intermediate_tensors,
                **multi_modal_kwargs,
            )
            # print(f'\t\t\t------ON step {step} compute logits')

            # Compute the logits.
            if get_pp_group().is_last_rank:
                logits = self.model.compute_logits(hidden_states,
                                                model_input.sampling_metadata)
                # print(f'\t\t\t------ON step {step} sample')

                if self.is_driver_worker:
                    # Sample the next token.
                    outputs.append(
                        self.model.sample(
                            logits=logits,
                            sampling_metadata=model_input.sampling_metadata,
                        ))
                if self.return_hidden_states:
                    raise ValueError("return_hidden_states is not supported for MultiStepModelRunner.")

            # TODO(will) take a close look at this logic for TP
            if not self.is_driver_worker:
                return []

            # Prepare the inputs for the next step.
            if step != num_steps - 1:
                assert num_steps > 1
                # print(f'\t\t\t------ON step {step} update_model_input')
                if get_pp_group().is_last_rank:
                    # broadcast the sampled token to all pp stages
                    # print('broadcasting from last rank')
                    get_pp_group().broadcast_tensor_dict(
                        {"sampled_token_ids":outputs[-1].sampled_token_ids},
                        src=get_pp_group().last_rank)
                    # print('after broadcasting from last rank')

                    model_input = self._advance_step(model_input, outputs[-1])
                else:
                    # get next token from broadcast
                    # print('before  recv broadcasting from non last rank')
                    output_dict = get_pp_group().broadcast_tensor_dict(
                        src=get_pp_group().last_rank)
                    # print('after   recv broadcasting from non last rank')
                    out = SamplerOutput([], sampled_token_ids=output_dict["sampled_token_ids"])
                    model_input = self._advance_step(model_input, out)
                # model_input = self.update_model_input(model_input, outputs[-1])
                # model_input = self._advance_step_gpu(model_input, outputs[-1])
            else:
                # if model_input.sampling_metadata.skip_sampler_cpu_output:
                # print('final step, building outputs')
                # pass
                if get_pp_group().is_last_rank:
                    self.pythonize_outputs(model_input, outputs[:-1])
                
        if not get_pp_group().is_last_rank:
            return hidden_states

        if not self.is_driver_worker:
            return []

        return outputs


    def pythonize_outputs(self, 
    model_input: ModelInputForGPUWithSamplingMetadata,
                          outputs: List[SamplerOutput]) -> List[SamplerOutput]:
        for output in outputs:
            # samples generation should have been skipped
            assert not output.outputs 

            # print shapes
            # print('output.sampled_token_ids', output.sampled_token_ids.shape)
            # print('output.logprobs', output.logprobs.shape)
            samples = torch.empty(*output.sampled_token_ids.shape,
                          dtype=output.sampled_token_ids.dtype,
                          device="cpu",
                          pin_memory=True)
            logprobs = torch.empty(*output.logprobs.shape,
                                dtype=output.logprobs.dtype,
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

            samples = samples.tolist()
            # logprobs = logprobs.tolist()
            # print('samples', samples)
            # print('logprobs', logprobs)


            # from vllm.model_executor.layers.sampler import _get_logprobs


            # output.sampled_token_ids = output.sampled_token_ids.cpu()
            # token_ids = output.sampled_token_ids.tolist()
            sampling_metadata = model_input.sampling_metadata

            for (seq_group, sample_result) in zip(sampling_metadata.seq_groups,
                                            samples):
                seq_ids = seq_group.seq_ids
                # next_token_ids, parent_ids = sample_result
                next_token_ids= sample_result
                parent_ids = [0]
                seq_outputs: List[SequenceOutput] = []
                for parent_id, next_token_id in zip(parent_ids,
                                                            next_token_ids):
                    # print('SequenceOutput', seq_ids[parent_id], next_token_id)
                    seq_outputs.append(
                        SequenceOutput(
                            seq_ids[parent_id], 
                            next_token_id, {next_token_id:Logprob(logprob=4)}))
                # print('CompletionSequenceGroupOutput', seq_outputs)
                output.outputs.append(
                    CompletionSequenceGroupOutput(seq_outputs, None))

    def _advance_step(
        self,
        model_input: ModelInputForGPUWithSamplingMetadata,
        last_output: SamplerOutput,
    ) -> ModelInputForGPUWithSamplingMetadata:
        """Advance the model input for the next step."""

        # Append the output token to the sequence data.
        assert self.cached_seq_group_metadata_list is not None
        for seq_group_metadata, sequence_group_outputs in zip(
                self.cached_seq_group_metadata_list, last_output.outputs):
            seq_group_metadata.is_prompt = False

            for seq_output in sequence_group_outputs.samples:
                seq = seq_group_metadata.seq_data[seq_output.parent_seq_id]

                token_id = seq_output.output_token
                token_logprob = seq_output.logprobs[token_id]

                seq.append_token_id(token_id, token_logprob.logprob)
                seq.update_num_computed_tokens(1)

        sampled_tokens = last_output.sampled_token_ids.flatten()
        model_input.input_tokens[:] = sampled_tokens
        model_input.input_positions.add_(1)

        # used for calculating blocks and slot mapping for decode
        model_input.attn_metadata.seq_start_loc.add_(
            model_input.attn_metadata.query_start_loc)

        # input_positions = input_positions[:num_tokens_to_generate]
        # input_positions = model_input.attn_metadata.seq_start_loc[1:]
        # input_positions = self.cached_seq_lens
        input_positions = model_input.attn_metadata.seq_lens_tensor
        block_index = input_positions.floor_divide(self.block_size).to(
            torch.long)
        # print('block_table', self.cached_block_tables)
        # print('block_index', block_index)
        block_number = model_input.attn_metadata.block_tables.gather(
            1,
            block_index.clamp_min_(0).unsqueeze(1)).squeeze(1)
        block_offset = input_positions.remainder(self.block_size)
        model_input.attn_metadata.slot_mapping[:] = (
            block_number * self.block_size + block_offset)
        # slot_m = (block_number * self.block_size + block_offset)
        # model_input.seq_lens.add_(1)
        input_positions.add_(1)

        # print('input_positions', input_positions)
        block_table_bound = input_positions.div(self.block_size)
        # print('block_table_bound', block_table_bound)
        block_table_bound = block_table_bound.ceil().to(torch.long)
        # print('block_table_bound', block_table_bound)

        max_block_table_bound = model_input.attn_metadata.block_tables.shape[1]
        range_tensor = torch.arange(max_block_table_bound,
                                    device=self.device).expand(
                                        (block_table_bound.shape[0], -1))
        paged_kv_indices = model_input.attn_metadata.block_tables.masked_select(
            range_tensor < block_table_bound.unsqueeze(-1))
        # [1:] is to leave the zero at first element
        model_input.attn_metadata.paged_kv_indptr[1:] = torch.cumsum(
            block_table_bound, dtype=torch.int, dim=0)

        model_input.attn_metadata.paged_kv_last_page_len.remainder_(
            self.block_size).add_(1)
        model_input.sampling_metadata.reuse_sampling_tensors = True

        # build the next step's attn_metadata
        # args coming from new_model_input are still on CPU.
        next_attn_metadata = dataclasses.replace(
            model_input.attn_metadata,
            num_prefills=0,
            # slot_mapping=slot_m,
            slot_mapping=model_input.attn_metadata.slot_mapping,
            paged_kv_indices=paged_kv_indices,
            paged_kv_indptr=model_input.attn_metadata.paged_kv_indptr,
            paged_kv_last_page_len=model_input.attn_metadata.paged_kv_last_page_len,
            seq_start_loc=model_input.attn_metadata.seq_start_loc,
            query_start_loc=model_input.attn_metadata.query_start_loc,
        )

        # return a manually built model_input
        # args coming from new_model_input are still on CPU.
        return dataclasses.replace(
            model_input,
            input_tokens=model_input.input_tokens,
            sampling_metadata=model_input.sampling_metadata,
            input_positions=model_input.input_positions,
            attn_metadata=next_attn_metadata,
        )
