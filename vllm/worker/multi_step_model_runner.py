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
from vllm.distributed import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.sequence import (Logprob, CompletionSequenceGroupOutput, IntermediateTensors, 
                           SamplerOutput, SequenceGroupMetadata, SequenceOutput)
from vllm.worker.model_runner import (ModelInputForGPUWithSamplingMetadata,
                                      ModelRunner)
from vllm.attention.backends.flashinfer import NUM_FLASHINFER_WORKSPACE_BUFFERS

logger = init_logger(__name__)

@dataclasses.dataclass
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
    pythonized_sampler_output: Optional[SamplerOutput] = None
    pythonized: bool = False

    def pythonize(self, input_metadata: ModelInputForGPUWithSamplingMetadata,
                  copy_stream: torch.cuda.Stream) -> SamplerOutput:
        """Pythonize the output. Blocking."""
        if not self.pythonized:
            self.pythonized_sampler_output = (
                self._pythonize_sampler_output_wait_on_event(
                    input_metadata, copy_stream))
            self.pythonized = True
        return self.pythonized_sampler_output

    def maybe_pythonize(
            self, input_metadata: ModelInputForGPUWithSamplingMetadata,
            copy_stream: torch.cuda.Stream) -> Optional[SamplerOutput]:
        """Pythonize the output if ready, else return None. Non-blocking."""
        if not self.pythonized:
            self.pythonized_sampler_output, self.pythonized = (
                self._pythonize_sampler_output_if_event_ready(
                    input_metadata, copy_stream))
        return self.pythonized_sampler_output

    def _pythonize_sampler_output_wait_on_event(
            self, input_metadata: ModelInputForGPUWithSamplingMetadata,
            copy_stream: torch.cuda.Stream) -> SamplerOutput:
        self.sampler_output_ready_event.synchronize()
        with torch.cuda.stream(copy_stream):
            return pythonize_sampler_output(input_metadata, self.sampler_output)

    def _pythonize_sampler_output_if_event_ready(
            self, input_metadata: ModelInputForGPUWithSamplingMetadata,
            copy_stream: torch.cuda.Stream) -> Tuple[Optional[SamplerOutput], bool]:
        if self.sampler_output_ready_event.query():
            with torch.cuda.stream(copy_stream):
                return pythonize_sampler_output(input_metadata, self.sampler_output), True
        return None, False


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

        self._copy_stream = torch.cuda.Stream()

        self.flashinfer_decode_workspace_buffers = [None] * NUM_FLASHINFER_WORKSPACE_BUFFERS
        self.flashinfer_decode_wrappers = [None] * NUM_FLASHINFER_WORKSPACE_BUFFERS
        self.flashinfer_prefill_workspace_buffers = [None] * NUM_FLASHINFER_WORKSPACE_BUFFERS
        self.flashinfer_prefill_wrappers = [None] * NUM_FLASHINFER_WORKSPACE_BUFFERS

        self.step_cuda_events = [torch.cuda.Event(blocking=True)] * NUM_FLASHINFER_WORKSPACE_BUFFERS

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

        if self.lora_config:
            raise ValueError("LoRA is not supported for MultiStepModelRunner.")
        
        if get_pp_group().is_last_rank:
            self.model.sampler.include_gpu_probs_tensor = True

        virtual_engine = model_input.virtual_engine
        model_outputs: List[ModelOutput] = []
        current_stream = torch.cuda.current_stream()

        # print(f'------doing {num_steps} steps------')
        model_input.sampling_metadata.skip_sampler_cpu_output = True
        for step in range(num_steps):
            # print(f'\t\t\t------ON step {step} -----')
            idx = step % NUM_FLASHINFER_WORKSPACE_BUFFERS

            # print('skip_sampler_cpu_output', model_input.sampling_metadata.skip_sampler_cpu_output)
            # print('=====self.attn_backend.get_name()', self.attn_backend.get_name())
            if self.attn_backend.get_name() == "flashinfer":
                assert model_input.attn_metadata is not None
                assert model_input.input_tokens is not None
                # if self.flashinfer_decode_workspace_buffers[0] is None:
                if self.flashinfer_decode_wrappers[idx] is None:
                    if self.flashinfer_decode_workspace_buffers[0] is None:
                        self.flashinfer_decode_workspace_buffers[0] = torch.empty(
                            FLASHINFER_WORKSPACE_BUFFER_SIZE,
                            dtype=torch.uint8,
                            device=self.device)
                        self.flashinfer_prefill_workspace_buffers[0] = torch.empty(
                            FLASHINFER_WORKSPACE_BUFFER_SIZE,
                            dtype=torch.uint8,
                            device=self.device)
                    self.flashinfer_decode_wrappers[idx] = \
                        BatchDecodeWithPagedKVCacheWrapper(
                        self.flashinfer_decode_workspace_buffers[0], "NHD")
                    self.flashinfer_prefill_wrappers[idx] = \
                        BatchPrefillWithPagedKVCacheWrapper(
                        self.flashinfer_prefill_workspace_buffers[0], "NHD")

                model_input.attn_metadata.prefill_wrapper = \
                    self.flashinfer_prefill_wrappers[idx]
                if model_input.attn_metadata.use_cuda_graph:
                    # assert False
                    batch_size = model_input.input_tokens.shape[0]
                    model_input.attn_metadata.decode_wrapper = self.graph_runners[
                        model_input.
                        virtual_engine][batch_size].flashinfer_decode_wrapper
                else:
                    model_input.attn_metadata.decode_wrapper = \
                        self.flashinfer_decode_wrappers[idx]

                model_input.attn_metadata.begin_forward()

            # Currently cuda graph is only supported by the decode phase.
            assert model_input.attn_metadata is not None
            prefill_meta = model_input.attn_metadata.prefill_metadata
            decode_meta = model_input.attn_metadata.decode_metadata
            if prefill_meta is None and decode_meta.use_cuda_graph:
                # assert False
                assert model_input.input_tokens is not None
                graph_batch_size = model_input.input_tokens.shape[0]
                model_executable = (
                    self.graph_runners[virtual_engine][graph_batch_size])
            else:
                model_executable = self.model

            multi_modal_kwargs = model_input.multi_modal_kwargs or {}

            if True:
                # We do this here to take advantage of async execution:
                # at this point, the main GPU stream will be occupied with
                # the previous forward pass, leaving CPU idle. Therefore,
                # we can pythonize the outputs that are ready essentially
                # for free.
                if get_pp_group().is_last_rank:
                    if self.is_driver_worker:
                        for o in model_outputs:
                            o.maybe_pythonize(model_input,
                                              self._copy_stream)
            # print(f'\t\t\t------ON step {step} model executable')
            intermediate_tensors = None
            if not get_pp_group().is_first_rank:
                pass
                # intermediate_tensors = IntermediateTensors(
                #     get_pp_group().recv_tensor_dict())
            hidden_states = model_executable(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                kv_caches=kv_caches,
                attn_metadata=model_input.attn_metadata,
                intermediate_tensors=intermediate_tensors,
                **multi_modal_kwargs,
            )
            if not get_pp_group().is_last_rank:
                # output is IntermediateTensors
                pass
                # get_pp_group().send_tensor_dict(hidden_states.tensors)
                # return [None]
            self.step_cuda_events[idx] = torch.cuda.Event(blocking=True)
            self.step_cuda_events[idx].record(current_stream)
            # print(f'\t\t\t------ON step {step} compute logits')

            # Compute the logits.
            if get_pp_group().is_last_rank:
                logits = self.model.compute_logits(hidden_states,
                                                model_input.sampling_metadata)
                # print(f'\t\t\t------ON step {step} sample')

                if self.is_driver_worker:
                    # Sample the next token.
                    output = self.model.sample(
                                logits=logits,
                                sampling_metadata=model_input.sampling_metadata,
                            )

                    output_ready_event = torch.cuda.Event()
                    output_ready_event.record(current_stream)
                    # prev_parameters_tensors = (output.sampling_parameters_tensors,
                    #                            output.sampling_token_tensors)
                    output = ModelOutput(output, output_ready_event,
                                        pythonized = not model_input.sampling_metadata.skip_sampler_cpu_output)
                    model_outputs.append(output)
                if self.return_hidden_states:
                    raise ValueError("return_hidden_states is not supported for MultiStepModelRunner.")
            

            # Prepare the inputs for the next step.
            if step != num_steps - 1:
                # print(f'\t\t\t------ON step {step} update_model_input')
                out = self._get_sampled_token_ids(model_outputs)

                # FIXME debugging cuda event synchronization
                # self.step_cuda_events[(idx+1)%NUM_FLASHINFER_WORKSPACE_BUFFERS].synchronize()
                self.step_cuda_events[(idx)%NUM_FLASHINFER_WORKSPACE_BUFFERS].synchronize()
                # model_input = self._advance_step_flashinfer_gpu(model_input, out, idx+1)
                model_input = self._advance_step(model_input, out)

        if not get_pp_group().is_last_rank:
            return hidden_states

        if not self.is_driver_worker:
            return []

        for output in model_outputs:
            output.pythonize(model_input, self._copy_stream)

        # return model_outputs
        return [output.sampler_output for output in model_outputs]
    
    def _get_sampled_token_ids(self, model_outputs):
        if get_pp_group().is_last_rank:
            # broadcast the sampled token to all pp stages
            if self.is_driver_worker:
                last_output = model_outputs[-1].sampler_output
                # print('broadcasting from last rank driver')
                pp_group = get_pp_group()
                pp_group.broadcast_tensor_dict(
                    {"sampled_token_ids":last_output.sampled_token_ids},
                    src=pp_group.world_size - 1)

                    #src=get_pp_group().last_rank)
                # pp_group.barrier()
                get_tp_group().broadcast_tensor_dict(
                    {"sampled_token_ids":last_output.sampled_token_ids})
                    # src=get_tp_group().first_rank)
                out = last_output
                # print('done broadcasting from last rank driver')
            else:
                # print('broadcasting from last rank worker')
                output_dict = get_tp_group().broadcast_tensor_dict()
                    # src=get_tp_group().first_rank)
                out = SamplerOutput([], sampled_token_ids=output_dict["sampled_token_ids"])
                # print('done broadcasting from last rank worker')

            # model_input = self._advance_step(model_input, out)
        else:
            # get next token from broadcast
            # print('before  recv broadcasting from non last rank')
            if self.is_driver_worker:
                pp_group = get_pp_group()
                output_dict = pp_group.broadcast_tensor_dict(
                    src=pp_group.world_size - 1)
                    # src=get_pp_group().last_rank)
                # get_pp_group().barrier()
                get_tp_group().broadcast_tensor_dict(
                    {"sampled_token_ids":output_dict["sampled_token_ids"]})
            else:
                output_dict = get_tp_group().broadcast_tensor_dict()
            # print('after   recv broadcasting from non last rank')
            out = SamplerOutput([], sampled_token_ids=output_dict["sampled_token_ids"])
        return out

    def _advance_step(
        self,
        model_input: ModelInputForGPUWithSamplingMetadata,
        last_output: SamplerOutput,
    ) -> ModelInputForGPUWithSamplingMetadata:
        """Advance the model input for the next step."""

        return self._advance_step_flashinfer_gpu(model_input, last_output)
        # return self._advance_step_torch(model_input, last_output)



    def _advance_step_torch(
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
        query_len = len(model_input.query_lens)
        model_input.input_tokens[:query_len] = sampled_tokens
        model_input.input_positions.add_(1)

        # used for calculating blocks and slot mapping for decode
        # model_input.attn_metadata.seq_start_loc.add_(
        #     model_input.attn_metadata.query_start_loc)
        model_input.attn_metadata.seq_start_loc.add_(1)


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

        seq_lens = list(map(lambda x: x + 1, model_input.seq_lens))
        bounds = list(map(lambda x: -(x//-self.block_size), seq_lens))
        model_input.attn_metadata.paged_kv_indptr_cpu[1:] = torch.cumsum(
            torch.tensor(bounds, device='cpu'), dtype=torch.int, dim=0)
        # model_input.attn_metadata.paged_kv_last_page_len_cpu[(idx)%NUM_FLASHINFER_WORKSPACE_BUFFERS].remainder_(self.block_size).add_(1)
        model_input.attn_metadata.paged_kv_last_page_len_cpu.remainder_(self.block_size).add_(1)

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

    def _advance_step_flashinfer_gpu(
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
        query_len = len(model_input.query_lens)
        model_input.input_tokens[:query_len] = sampled_tokens
        # print('sampled_tokens.shape', sampled_tokens.shape)
        # print('model_input.input_tokens.shape', model_input.input_tokens.shape)

        num_seqs = len(model_input.seq_lens)
        num_queries = len(model_input.query_lens)
        # print('num_seqs', num_seqs)
        # print('num_queries', num_queries)
        # assert num_seqs == num_queries
        attn_metadata = model_input.attn_metadata
        # Update GPU tensors
        attn_metadata.seq_start_loc = None
        attn_metadata.query_start_loc = None
        # print('BEFORE KERNEL indptr_buf ', model_input.attn_metadata.decode_wrapper._paged_kv_indptr_buf)
        # print data ptr
        # print('buffer.data_ptr', attn_metadata.decode_wrapper._paged_kv_indptr_buf.shape)
        # print('paged_kv_indices.data_ptr', attn_metadata.paged_kv_indices.shape)
        # print('paged_kv_indptr.data_ptr', attn_metadata.paged_kv_indptr.data_ptr())
        # print('paged_kv_last_page_len.data_ptr', attn_metadata.paged_kv_last_page_len.data_ptr())
        # print('block_table_bound.data_ptr', attn_metadata.block_table_bound.data_ptr())
        # print('paged_kv_last_page_len', attn_metadata.paged_kv_last_page_len)
        # print('paged_kv_last_page_len.device', attn_metadata.paged_kv_last_page_len.device)
        # inp = dict(
        #     num_seqs=num_seqs,
        #     num_queries=num_queries,
        #     block_size=self.block_size,
        #     input_tokens=model_input.input_tokens.shape,
        #     sampled_token_ids=model_input.input_tokens.shape,
        #     input_positions=model_input.input_positions.shape,
        #     seq_lens=attn_metadata.seq_lens_tensor.shape,
        #     slot_mapping=attn_metadata.slot_mapping.shape,
        #     block_tables=attn_metadata.block_tables.shape,
        #     paged_kv_indices=attn_metadata.paged_kv_indices.shape,
        #     paged_kv_indptr=attn_metadata.paged_kv_indptr.shape,
        #     paged_kv_last_page_len=attn_metadata.paged_kv_last_page_len.shape,
        #     block_table_bound=attn_metadata.block_table_bound.shape)
        # print(inp)

        # print('first sync')
        # torch.cuda.synchronize(self.device)
        # print('after first sync')
        ops.advance_step_flashinfer(
            num_seqs=num_seqs,
            num_queries=num_queries,
            block_size=self.block_size,
            input_tokens=model_input.input_tokens,
            sampled_token_ids=model_input.input_tokens,
            input_positions=model_input.input_positions,
            seq_lens=attn_metadata.seq_lens_tensor,
            slot_mapping=attn_metadata.slot_mapping,
            block_tables=attn_metadata.block_tables,
            # paged_kv_indices=attn_metadata.cached_paged_kv_indices,
            paged_kv_indices=attn_metadata.paged_kv_indices,
            paged_kv_indptr=attn_metadata.paged_kv_indptr,
            paged_kv_last_page_len=attn_metadata.paged_kv_last_page_len,
            block_table_bound=attn_metadata.block_table_bound)
        # print('seoncd sync')
        # torch.cuda.synchronize(self.device)
        # print('after seoncd sync')

        # print('AFTER KERNEL========')
        # print('buffer.data_ptr', attn_metadata.decode_wrapper._paged_kv_indptr_buf.data_ptr())
        # print('paged_kv_indices.data_ptr', attn_metadata.paged_kv_indices.data_ptr())
        # print('paged_kv_indptr.data_ptr', attn_metadata.paged_kv_indptr.data_ptr())
        # print('paged_kv_last_page_len.data_ptr', attn_metadata.paged_kv_last_page_len.data_ptr())
        # print('block_table_bound.data_ptr', attn_metadata.block_table_bound.data_ptr())

        # print('dumping tensors')
        # print('paged_kv_indices', attn_metadata.paged_kv_indices)
        # print('paged_kv_indptr', attn_metadata.paged_kv_indptr)
        # print('paged_kv_last_page_len', attn_metadata.paged_kv_last_page_len)
        # print('block_table_bound', attn_metadata.block_table_bound)

        # print('print indptr_buf 4.4', model_input.attn_metadata.decode_wrapper._paged_kv_indptr_buf)
        seq_lens = list(map(lambda x: x + 1, model_input.seq_lens))
        bounds = list(map(lambda x: -(x//-self.block_size), seq_lens))

        # we update next step's attn_metadata
        # model_input.attn_metadata.paged_kv_indptr_cpu[(idx)%NUM_FLASHINFER_WORKSPACE_BUFFERS][1:] = torch.cumsum(
        model_input.attn_metadata.paged_kv_indptr_cpu[1:] = torch.cumsum(
            torch.tensor(bounds, device='cpu'), dtype=torch.int, dim=0)
        # model_input.attn_metadata.paged_kv_last_page_len_cpu[(idx)%NUM_FLASHINFER_WORKSPACE_BUFFERS].remainder_(self.block_size).add_(1)
        model_input.attn_metadata.paged_kv_last_page_len_cpu.remainder_(self.block_size).add_(1)
        # print('print indptr_buf 4.5', model_input.attn_metadata.decode_wrapper._paged_kv_indptr_buf)

        new_model_input = self._model_input_cls(
            seq_lens=seq_lens,
            input_tokens=model_input.input_tokens,
            input_positions=model_input.input_positions,
            attn_metadata=attn_metadata,
            query_lens=model_input.query_lens,
            lora_mapping=model_input.lora_mapping,
            lora_requests=model_input.lora_requests,
            multi_modal_kwargs=model_input.multi_modal_kwargs,
            sampling_metadata=model_input.sampling_metadata,
            is_prompt=False,
        )

        # Ensure we skip CPU samples
        # assert new_model_input.sampling_metadata.skip_sampler_cpu_output is True
        # We can reuse sampling tensors since every decode iteration is the same
        new_model_input.sampling_metadata.reuse_sampling_tensors = True

        return new_model_input

def pythonize_sampler_output(model_input: ModelInputForGPUWithSamplingMetadata,
                             output: SamplerOutput) -> SamplerOutput:
    
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