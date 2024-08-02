from vllm.worker.worker import Worker
from dataclasses import dataclass
from vllm.worker.worker import WorkerInput
from vllm.worker.model_runner import (ModelInputForGPUWithSamplingMetadata,
                                      ModelRunnerInputBase)
from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.distributed import broadcast_tensor_dict, get_pp_group
from typing import Tuple, Optional, List, Union
from vllm.worker.worker_base import IntermediateTensors
import dataclasses
from dataclasses import field
import torch
import gc

from vllm.worker.multi_step_model_runner import (
    ModelInputForGPUWithMultiStepMetadata, ModelOutput)


@dataclass
class MultiStepState:
    worker_input: WorkerInput
    model_input: ModelInputForGPUWithMultiStepMetadata


class MultiStepWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pipeline_parallel_size = self.parallel_config.pipeline_parallel_size
        self.multi_step_states: List[
            Optional[MultiStepState]] = [None] * pipeline_parallel_size
        self.temp_output = None

    def prepare_input(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> Optional[Tuple[ModelInputForGPUWithMultiStepMetadata, WorkerInput]]:
        """
        Depending on the current state of the request and multi step worker,
        this method may skip the normal _prepare_model_input and
        _prepare_worker_input methods and instead used cached values.
        """
        if self.is_driver_worker:
            if execute_model_req is None:
                if self.do_metadata_broadcast:
                    # This signals that there's no more requests to process for
                    # now. All workers are running infinite loop with
                    # broadcast_tensor_dict, and it stops the loop when the
                    # driver broadcasts an empty input. Send an empty input to
                    # notify all other workers to stop their execution loop.
                    broadcast_tensor_dict({}, src=0)
                return None

            virtual_engine = execute_model_req.virtual_engine
            if execute_model_req.is_first_multi_step:
                model_input, worker_input = self._get_driver_input_and_broadcast(
                    execute_model_req)
                model_input = ModelInputForGPUWithMultiStepMetadata(
                    **model_input.__dict__,
                    current_step=execute_model_req.current_step,
                    outputs=[],
                    is_multi_step=True,
                    is_last_step=execute_model_req.is_last_step,
                    is_first_multi_step=True,
                )
                self.multi_step_states[virtual_engine] = MultiStepState(
                    worker_input, model_input)
            else:
                # get_tp_group().broadcast_object({'virtual_engine': virtual_engine}, src=0)
                multi_step_state = self.multi_step_states[virtual_engine]
                model_input = multi_step_state.model_input
                worker_input = multi_step_state.worker_input

                assert model_input.current_step == execute_model_req.current_step
                model_input.current_step = execute_model_req.current_step
                model_input.is_last_step = execute_model_req.is_last_step
                model_input.is_first_multi_step = False
                # model_input = dataclasses.replace(
                #     model_input,
                #     current_step=execute_model_req.current_step,
                #     outputs=multi_step_state.model_input.outputs,
                #     is_multi_step=True,
                #     is_last_step=execute_model_req.is_last_step,
                #     is_first_multi_step=False,
                # )
                if self.do_metadata_broadcast:
                    broadcast_data = worker_input.as_broadcastable_tensor_dict(
                    )
                    broadcast_data.update(
                        model_input.as_broadcastable_tensor_dict())
                    broadcast_tensor_dict(broadcast_data, src=0)
                self.multi_step_states[virtual_engine] = MultiStepState(
                    worker_input, model_input)
                assert isinstance(model_input,
                                  ModelInputForGPUWithMultiStepMetadata)
        else:
            assert False
            model_input, worker_input = self._get_worker_input_from_broadcast()
            if self.model_input.is_first_multi_step:
                model_input = ModelInputForGPUWithMultiStepMetadata(
                    **model_input.__dict__,
                    num_steps=0,
                    outputs=[],
                    is_multi_step=True,
                    is_last_step=False,
                )
                virtual_engine = worker_input.virtual_engine
                self.multi_step_states[virtual_engine] = MultiStepState(
                    worker_input, model_input)
            else:
                # output = get_tp_group().broadcast_object({'virtual_engine': virtual_engine}, src=0)
                # virtual_engine = output['virtual_engine']
                multi_step_state = self.multi_step_states[virtual_engine]
                model_input = multi_step_state.model_input
                worker_input = multi_step_state.worker_input
                assert isinstance(model_input,
                                  ModelInputForGPUWithMultiStepMetadata)
                assert model_input.num_steps == worker_input.num_steps
                # input should be cached and ready to go already.
                self.multi_step_states[virtual_engine] = MultiStepState(
                    worker_input, model_input)

        assert model_input is not None
        assert worker_input is not None
        return model_input, worker_input

    def _handle_pipeline_parallel_output(
            self, model_input: ModelInputForGPUWithMultiStepMetadata,
            output: Union[IntermediateTensors, List[SamplerOutput]]) -> None:
        """
        Need to handle the output of the model differently in the case of
        MultiStepWorker.  
        Only the driver worker of the last rank samples the next token ids. This
        needs to be broadcasted to all other ranks so that they are able to
        update the next step's input metadata inplace.  
        """

        num_seqs = len(model_input.query_lens)
        if not get_pp_group().is_last_rank:
            # output is IntermediateTensors
            get_pp_group().send_tensor_dict(output.tensors)

            # recieve broadcast from last rank
            # print('receiving broadcast from last rank')
            self.temp_output = torch.empty((num_seqs, 1),
                                            dtype=torch.long,
                                            device=self.device)
            # output = torch.empty((num_seqs, 1),
            #                      dtype=torch.long,
            #                      device=self.device)
            get_pp_group().broadcast(
                self.temp_output,
                src=self.parallel_config.pipeline_parallel_size - 1,
                async_op=True)
            model_input.add_sampler_output(sampler_output=SamplerOutput(
                outputs=[], sampled_token_ids=self.temp_output), )

            # output = get_pp_group().broadcast_tensor_dict(
            #     src=self.parallel_config.pipeline_parallel_size - 1)
            # model_input.add_sampler_output(sampler_output=SamplerOutput(
            #     outputs=[], sampled_token_ids=output["sampled_token_ids"]), )
            # print('non last rank broadcast received')
        else:
            # make sure we are not last step
            # broadcast to other ranks
            # print('last rank, broadcasting')
            get_pp_group().broadcast(
                model_input.outputs[-1].sampler_output.sampled_token_ids,
                src=self.parallel_config.pipeline_parallel_size - 1,
                async_op=True)
            # get_pp_group().broadcast_tensor_dict(
            #     {
            #         "sampled_token_ids":
            #         model_input.outputs[-1].sampler_output.sampled_token_ids
            #     },
            #     src=self.parallel_config.pipeline_parallel_size - 1)
            # print('last rank broadcasted')
