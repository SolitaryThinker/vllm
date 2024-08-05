from vllm.worker.worker import Worker
from dataclasses import dataclass
from vllm.worker.worker import WorkerInput
from vllm.worker.model_runner_base import BroadcastableModelInput
from vllm.worker.model_runner import (ModelInputForGPUWithSamplingMetadata,
                                      ModelRunnerInputBase)
from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.distributed import broadcast_tensor_dict, get_tp_group, get_pp_group, get_world_group
from typing import Tuple, Optional, List, Union
from vllm.worker.worker_base import IntermediateTensors
import dataclasses
from dataclasses import field
import torch
import gc

from vllm.worker.multi_step_model_runner import (
    MutableModelInputForGPUWithMultiStepMetadata, ModelOutput)


@dataclass
class MultiStepState:
    worker_input: WorkerInput
    model_input: MutableModelInputForGPUWithMultiStepMetadata


class MultiStepWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pipeline_parallel_size = self.parallel_config.pipeline_parallel_size
        self.multi_step_states: List[
            Optional[MultiStepState]] = [None] * pipeline_parallel_size
        self.temp_output = None

    def _get_driver_input_and_broadcast(
        self, execute_model_req: ExecuteModelRequest
    ) -> Tuple[BroadcastableModelInput, WorkerInput]:
        """
        Get the driver input and broadcast it to other workers.
        """
        assert self.is_driver_worker
        virtual_engine = execute_model_req.virtual_engine
        is_first_multi_step = execute_model_req.is_first_multi_step
        if True:
            worker_input: WorkerInput = self.prepare_worker_input(
                execute_model_req=execute_model_req)
            model_input: BroadcastableModelInput = (
                self.model_runner.prepare_model_input(
                    execute_model_req.seq_group_metadata_list,
                    execute_model_req.virtual_engine,
                    execute_model_req.finished_requests_ids))
        # else:
        # print('using CACHED in driver')
        # multi_step_state = self.multi_step_states[virtual_engine]
        # model_input2 = multi_step_state.model_input
        # worker_input2 = multi_step_state.worker_input

        model_input.is_first_multi_step = is_first_multi_step
        model_input.is_last_step = execute_model_req.is_last_step
        # model_input2.is_first_multi_step = is_first_multi_step
        # model_input2.is_last_step = execute_model_req.is_last_step

        # if not is_first_multi_step:
        #     print('filling in last_sampled_token_ids')
        #     assert len(model_input.outputs) > 0
        #     #  only broadcast the last output to save bandwidth
        #     model_input2.last_sampled_token_ids = model_input.outputs[-1].sampler_output.sampled_token_ids

        if self.do_metadata_broadcast:
            broadcast_data = worker_input.as_broadcastable_tensor_dict()
            broadcast_data.update(model_input.as_broadcastable_tensor_dict())
            broadcast_tensor_dict(broadcast_data, src=0)

            # broadcast_data = worker_input2.as_broadcastable_tensor_dict()
            # broadcast_data.update(model_input2.as_broadcastable_tensor_dict())
            # brradcast_tensor_dict(broadcast_data, src=0)

        return model_input, worker_input

    def prepare_input(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> Optional[Tuple[MutableModelInputForGPUWithMultiStepMetadata, WorkerInput]]:
        """
        Depending on the current state of the request and multi step worker,
        this method may skip the normal _prepare_model_input and
        _prepare_worker_input methods and instead used cached values.
        """
        if self.is_driver_worker:
            if execute_model_req is None:
                if self.do_metadata_broadcast:
                    # assert False
                    # This signals that there's no more requests to process for
                    # now. All workers are running infinite loop with
                    # broadcast_tensor_dict, and it stops the loop when the
                    # driver broadcasts an empty input. Send an empty input to
                    # notify all other workers to stop their execution loop.
                    # broadcast_tensor_dict({}, src=0)
                    print('broadcasting empty object')
                    get_tp_group().broadcast_object({}, src=0)
                return None

            virtual_engine = execute_model_req.virtual_engine
            print('first first broad in driver', virtual_engine, execute_model_req.is_first_multi_step)
            get_tp_group().broadcast_object(
                {'is_first_multi_step': execute_model_req.is_first_multi_step,
                 'is_last_step': execute_model_req.is_last_step,
                    'virtual_engine': virtual_engine,}, src=0)
            if execute_model_req.is_first_multi_step:
                print('driver get_driver_input_and_broadcast')
                model_input, worker_input = self._get_driver_input_and_broadcast(
                    execute_model_req)
                assert isinstance(model_input, MutableModelInputForGPUWithMultiStepMetadata)
                # print('driver get_driver_input_and_broadcast', type(model_input))
                # print('driver get_driver_input_and_broadcast', model_input.is_multi_step)
                # print('driver get_driver_input_and_broadcast:')
                # for k,v in model_input.__dict__.items():
                #     print(k, v)
                #     print('---')
                # print('driver query_lens', len(model_input.query_lens))
                # model_input.is_last_step V= execute_model_req.is_last_step
                # model_input.is_first_multi_step = True
                # model_input.num_queries = len(model_input.query_lens)
                # model_input = MutableModelInputForGPUWithMultiStepMetadata(
                #     **model_input.__dict__,
                #     current_step=execute_model_req.current_step,
                #     outputs=[],
                #     is_multi_step=True,
                #     is_last_step=execute_model_req.is_last_step,
                #     is_first_multi_step=True,
                #     num_queries=len(model_input.query_lens),
                # )
                self.multi_step_states[virtual_engine] = MultiStepState(
                    worker_input, model_input)
            else:

                model_input, worker_input = self._get_driver_input_and_broadcast(
                    execute_model_req)
                # get_tp_group().broadcast_object({'virtual_engine': virtual_engine}, src=0)
                # multi_step_state = self.multi_step_states[virtual_engine]
                # model_input = multi_step_state.model_input
                # worker_input = multi_step_state.worker_input

                # assert model_input.current_step == execute_model_req.current_step
                # model_input.is_multi_step = True
                # model_input.current_step = execute_model_req.current_step
                # model_input.is_last_step = execute_model_req.is_last_step
                # model_input.is_first_multi_step = False

                # if self.do_metadata_broadcast:

                #     broadcast_data = worker_input.as_broadcastable_tensor_dict()
                #     broadcast_data.update(model_input.as_broadcastable_tensor_dict())
                #     broadcast_tensor_dict(broadcast_data, src=0)

                # model_input = dataclasses.replace(
                #     model_input,
                #     current_step=execute_model_req.current_step,
                #     outputs=multi_step_state.model_input.outputs,
                #     is_multi_step=True,
                #     is_last_step=execute_model_req.is_last_step,
                #     is_first_multi_step=False,
                # )
                # if self.do_metadata_broadcast:
                #     broadcast_data = worker_input.as_broadcastable_tensor_dict(
                #     )
                #     broadcast_data.update(
                #         model_input.as_broadcastable_tensor_dict())
                #     broadcast_tensor_dict(broadcast_data, src=0)
                # self.multi_step_states[virtual_engine] = MultiStepState(
                #     worker_input, model_input)
                assert isinstance(model_input,
                                  MutableModelInputForGPUWithMultiStepMetadata)
        else:
            print('not driver')
            out = get_tp_group().broadcast_object()
            print('not driver got borad', out)
            # when the driver wants to pause the TP worker loops.
            if not out:
                return None
            print('not driver: out:', out)
            is_first_multi_step = out['is_first_multi_step']
            is_last_step = out['is_last_step']
            ve = out['virtual_engine']
            # print(f'not driver: is_first_multi_step: {is_first_multi_step}, ve: {ve}')
            if is_first_multi_step: 
                print('not deriver get_driver_input_and_broadcast')
                model_input, worker_input = self._get_worker_input_from_broadcast()
                assert isinstance(model_input, MutableModelInputForGPUWithMultiStepMetadata)
                print('is_multi_step:', model_input.is_multi_step)
                model_input.is_last_step = is_last_step
                model_input.is_first_multi_step = True

                # model_input2, worker_input2 = self._get_worker_input_from_broadcast()


                # print('checking stuff', model_input.query_lens)
                # print('checking stuff', model_input)
                # model_input.is_last_step = False
                # model_input.num_steps = 0
                # print('model_input.outputs:', model_input.outputs)
                # model_input = MutableModelInputForGPUWithMultiStepMetadata(
                #     **model_input.__dict__,
                #     num_steps=0,
                #     outputs=[],
                #     is_multi_step=True,
                #     is_last_step=False,
                # )
                virtual_engine = worker_input.virtual_engine
                assert virtual_engine == ve
                self.multi_step_states[virtual_engine] = MultiStepState(
                    worker_input, model_input)
            else:
                print('not deriver using cache')
                # output = get_tp_group().broadcast_object({'virtual_engine': virtual_engine}, src=0)
                # virtual_engine = output['virtual_engine']
                # print('model_input.outputs:', model_input.outputs)
                model_input, worker_input = self._get_worker_input_from_broadcast()
                assert isinstance(model_input, MutableModelInputForGPUWithMultiStepMetadata)

                # assert model_input.last_sampled_token_ids is not None
                # model_input.add_sampler_output(
                #     sampler_output=SamplerOutput([], 
                #                                  sampled_token_ids=model_input.last_sampled_token_ids))

                # model_input2, worker_input2 = self._get_worker_input_from_broadcast()

                # multi_step_state = self.multi_step_states[ve]
                # model_input = multi_step_state.model_input
                # worker_input = multi_step_state.worker_input
                model_input.is_last= is_last_step
                model_input.is_first_multi_step = False
                # assert isinstance(model_input,
                #                   MutableModelInputForGPUWithMultiStepMetadata)
                # assert model_input.num_steps == worker_input.num_steps
                # input should be cached and ready to go already.
                self.multi_step_states[ve] = MultiStepState(
                    worker_input, model_input)

        assert model_input is not None
        assert worker_input is not None
        return model_input, worker_input

    def _handle_pipeline_parallel_output_old(
            self, model_input: MutableModelInputForGPUWithMultiStepMetadata,
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

    """
    def _handle_pipeline_parallel_output(
            self, model_input: MutableModelInputForGPUWithMultiStepMetadata,
            output: Union[IntermediateTensors, List[SamplerOutput]]) -> None:
        Need to handle the output of the model differently in the case of
        MultiStepWorker.  
        Only the driver worker of the last rank samples the next token ids. This
        needs to be broadcasted to all other ranks so that they are able to
        update the next step's input metadata inplace.  

        if get_pp_group().is_last_rank and self.is_driver_worker:
            # make sure we are not last step
            # broadcast to other ranks
            print('last rank, broadcasting')
            # if get_pp_group().is_last_rank:
            print('last sampled_token_ids:', model_input.outputs[-1].sampler_output.sampled_token_ids.shape)
            get_tp_group().broadcast(
                model_input.outputs[-1].sampler_output.sampled_token_ids,
                # src=self.parallel_config.pipeline_parallel_size - 1,
                src=0,
                async_op=True)
            print('last rank broadcasted')
            # get_pp_group().broadcast_tensor_dict(
            #     {
            #         "sampled_token_ids":
            #         model_input.outputs[-1].sampler_output.sampled_token_ids
            #     },
            #     src=self.parallel_config.pipeline_parallel_size - 1)
            # print('last rank broadcasted')
        else:

            # num_seqs = model_input.input_positions.shape[0]
            # print('shape:', model_input.input_positions.shape)
            # print(type(model_input))
            num_seqs = model_input.num_queries
            print('num_seqs:', num_seqs)
            if not get_pp_group().is_last_rank:
                # output is IntermediateTensors
                get_pp_group().send_tensor_dict(output.tensors)

            # recieve broadcast from last rank
            print('receiving broadcast from last rank')
            # self.temp_output = torch.empty((num_seqs, 1),
            #                                 dtype=torch.long,
            #                                 device=self.device)
            output = torch.empty((num_seqs, 1),
                                 dtype=torch.long,
                                 device=self.device)
            get_tp_group().broadcast(
                output,
                src=0,
                async_op=True)
            print('recieved broadcast')

            # print('output:', output)
            model_input.add_sampler_output(sampler_output=SamplerOutput(
                outputs=[], sampled_token_ids=output), )

            # output = get_pp_group().broadcast_tensor_dict(
            #     src=self.parallel_config.pipeline_parallel_size - 1)
            # model_input.add_sampler_output(sampler_output=SamplerOutput(
            #     outputs=[], sampled_token_ids=output["sampled_token_ids"]), )
            # print('non last rank broadcast received')
"""