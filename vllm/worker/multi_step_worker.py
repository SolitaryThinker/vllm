from vllm.worker.worker import Worker
from dataclasses import dataclass
from vllm.worker.worker import WorkerInput
from vllm.worker.model_runner import (ModelInputForGPUWithSamplingMetadata,
                                      ModelRunnerInputBase, ModelOutput)
from vllm.sequence import ExecuteModelRequest
from vllm.distributed import broadcast_tensor_dict
from typing import Tuple, Optional, List
import dataclasses
from dataclasses import field
import torch


# @dataclass(frozen=True)
class ModelInputForGPUWithMultiStepMetadata(
        ModelInputForGPUWithSamplingMetadata):
    # current_step: int = 0
    outputs: List[ModelOutput] = field(default_factory=list)
    step_cuda_events: List[torch.cuda.Event] = field(
        default_factory=lambda: [torch.cuda.Event(blocking=True)] * 2)
    is_multi_step: bool = False
    is_last_step: bool = False

    def __init__(self, *args, **kwargs):
        self.current_step = kwargs.pop('current_step', 0)
        self.outputs = kwargs.pop('outputs', [])
        self.is_multi_step = kwargs.pop('is_multi_step', False)
        self.is_last_step = kwargs.pop('is_last_step', False)
        self.step_cuda_events = [torch.cuda.Event(blocking=True)] * 2
        super().__init__(*args, **kwargs)

    def record_step_event(self):
        self.step_cuda_events[self.current_step %
                              2] = torch.cuda.Event(blocking=True)
        self.step_cuda_events[self.current_step % 2].record()

    def wait_previous_step(self):
        self.step_cuda_events[(self.current_step + 1) % 2].wait()

    # @property
    # def is_multi_step(self):
    #     return self.seq_group_metadata_list[0].is_first_multi_step


@dataclass
class MultiStepState:
    worker_input: WorkerInput
    model_input: ModelRunnerInputBase


class MultiStepWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pipeline_parallel_size = self.parallel_config.pipeline_parallel_size
        self.multi_step_states: List[
            Optional[MultiStepState]] = [None] * pipeline_parallel_size

    # def _get_driver_input_and_broadcast(
    #     self, execute_model_req: ExecuteModelRequest
    # ) -> Tuple[ModelRunnerInputBase, WorkerInput]:
    #     """
    #     Get the driver input and broadcast it to other workers.
    #     """
    #     assert self.is_driver_worker
    #     worker_input: WorkerInput = self.prepare_worker_input(
    #         execute_model_req=execute_model_req)
    #     model_input: ModelRunnerInputBase = (
    #         self.model_runner.prepare_model_input(
    #             execute_model_req.seq_group_metadata_list,
    #             execute_model_req.virtual_engine,
    #             execute_model_req.finished_requests_ids))

    #     if self.do_metadata_broadcast:
    #         broadcast_data = worker_input.as_broadcastable_tensor_dict()
    #         broadcast_data.update(model_input.as_broadcastable_tensor_dict())
    #         broadcast_tensor_dict(broadcast_data, src=0)

    #     return model_input, worker_input

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
                )
                self.multi_step_states[virtual_engine] = MultiStepState(
                    worker_input, model_input)
            else:
                # model_input, worker_input = self._get_driver_input_and_broadcast(execute_model_req)
                multi_step_state = self.multi_step_states[virtual_engine]
                model_input = multi_step_state.model_input
                assert model_input.current_step == execute_model_req.current_step
                model_input = dataclasses.replace(
                    model_input,
                    current_step=execute_model_req.current_step,
                    outputs=multi_step_state.model_input.outputs,
                    is_multi_step=True,
                    is_last_step=execute_model_req.is_last_step,
                )
                worker_input = multi_step_state.worker_input
                self.multi_step_states[virtual_engine] = MultiStepState(
                    worker_input, model_input)
                assert isinstance(model_input,
                                  ModelInputForGPUWithMultiStepMetadata)
        else:
            assert False
            if self.multi_step_states[virtual_engine].is_first_multi_step:
                model_input, worker_input = self._get_worker_input_from_broadcast(
                )
                model_input = ModelInputForGPUWithMultiStepMetadata(
                    **model_input.__dict__,
                    num_steps=worker_input.num_steps,
                    outputs=[],
                    is_multi_step=True,
                    is_last_step=execute_model_req.is_last_step,
                )
                virtual_engine = worker_input.virtual_engine
                self.multi_step_states[virtual_engine] = MultiStepState(
                    worker_input, model_input)
            else:
                multi_step_state = self.multi_step_states[virtual_engine]
                model_input = multi_step_state.model_input
                worker_input = multi_step_state.worker_input
                assert isinstance(model_input,
                                  ModelInputForGPUWithMultiStepMetadata)
                assert model_input.num_steps == worker_input.num_steps
                # input should be cached and ready to go already.

        assert model_input is not None
        assert worker_input is not None
        return model_input, worker_input
