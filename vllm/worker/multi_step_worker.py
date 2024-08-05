from vllm.worker.worker import Worker
from dataclasses import dataclass
from vllm.worker.worker import WorkerInput
from vllm.worker.model_runner_base import BroadcastableModelInput
from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.distributed import broadcast_tensor_dict, get_tp_group, get_pp_group, get_world_group
from typing import Tuple, Optional, List, Union
from vllm.worker.worker_base import IntermediateTensors
from dataclasses import field
import torch

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
        else:
            multi_step_state = self.multi_step_states[virtual_engine]
            model_input = multi_step_state.model_input
            worker_input = multi_step_state.worker_input
            assert model_input is not None

        model_input.is_first_multi_step = is_first_multi_step
        model_input.is_last_step = execute_model_req.is_last_step

        # if not is_first_multi_step:
            # print('filling in last_sampled_token_ids', len(model_input.outputs))
            # assert len(model_input.outputs) > 0
            #  only broadcast the last output to save bandwidth
            # model_input.last_sampled_token_ids = model_input.outputs[-1].sampler_output.sampled_token_ids

        if self.do_metadata_broadcast:
            broadcast_data = worker_input.as_broadcastable_tensor_dict()
            broadcast_data.update(model_input.as_broadcastable_tensor_dict())
            broadcast_tensor_dict(broadcast_data, src=0)

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
                    broadcast_tensor_dict({}, src=0)
                return None

            virtual_engine = execute_model_req.virtual_engine

            model_input, worker_input = self._get_driver_input_and_broadcast(
                execute_model_req)
            assert isinstance(model_input, MutableModelInputForGPUWithMultiStepMetadata)
            if execute_model_req.is_first_multi_step:
                # print('driver get_driver_input_and_broadcast')
                self.multi_step_states[virtual_engine] = MultiStepState(
                    worker_input, model_input)
        else:
            broadcast_data = self._get_worker_input_from_broadcast()
            if broadcast_data is None:
                return None
            model_input, worker_input = broadcast_data

            assert isinstance(model_input, MutableModelInputForGPUWithMultiStepMetadata)

            if model_input.is_first_multi_step: 
                model_input.is_last_step = model_input.is_last_step
                model_input.is_first_multi_step = True

                virtual_engine = worker_input.virtual_engine
                assert virtual_engine == model_input.frozen_model_input.virtual_engine
                self.multi_step_states[virtual_engine] = MultiStepState(
                    worker_input, model_input)
            else:

                # assert model_input.last_sampled_token_ids is not None
                # model_input.add_sampler_output(
                #     sampler_output=SamplerOutput([], 
                #                                  sampled_token_ids=model_input.last_sampled_token_ids))

                model_input.is_last_step = model_input.is_last_step
                model_input.is_first_multi_step = False
                self.multi_step_states[model_input.frozen_model_input.virtual_engine] = MultiStepState(
                    worker_input, model_input)

        assert model_input is not None
        assert worker_input is not None
        return model_input, worker_input
