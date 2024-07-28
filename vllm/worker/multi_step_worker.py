from vllm.model_runner.model_runner import ModelInputForGPUWithSamplingMetadata
from vllm.worker.worker import Worker
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelInputForGPUWithMultiStepMetadata(ModelInputForGPUWithSamplingMetadata):
    num_steps: int = 0
    

class MultiStepWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_step_state = MultiStepState()

    def _get_driver_input_and_broadcast(
        self,
        execute_model_req: ExecuteModelRequest
    ) -> Tuple[ModelRunnerInputBase, WorkerInput]:
        """
        Get the driver input and broadcast it to other workers.
        """
        assert self.is_driver_worker
        worker_input: WorkerInput = self.prepare_worker_input(
            execute_model_req=execute_model_req)
        model_input: ModelRunnerInputBase = (
            self.model_runner.prepare_model_input(
                execute_model_req.seq_group_metadata_list,
                execute_model_req.virtual_engine,
                execute_model_req.finished_requests_ids))

        if self.do_metadata_broadcast:
            broadcast_data = worker_input.as_broadcastable_tensor_dict()
            broadcast_data.update(
                model_input.as_broadcastable_tensor_dict())
            broadcast_tensor_dict(broadcast_data, src=0)

        return model_input, worker_input

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

            if execute_model_req.is_first_multi_step:
                return self._get_driver_input_and_broadcast(
                    execute_model_req)
            else:
                return self._get_driver_input_and_broadcast(
                    execute_model_req)
                pass
                # TODO: implement the case where the request is not the first
                # multi-step request.
                # input should be cached and ready to go already.
        else:

            if self.multi_step_state.is_first_multi_step:
                return self._get_worker_input_from_broadcast()
            else:
                return self._get_worker_input_from_broadcast()
                pass
                # input should be cached and ready to go already.

        return model_input, worker_input
