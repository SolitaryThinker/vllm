import copy
import weakref
from typing import Dict, List, Optional, Tuple

import torch

from vllm.sequence import (ExecuteModelRequest, SamplerOutput, SequenceData,
                           SequenceGroupMetadata)
from vllm.worker.worker import Worker


class MultiStepWorker(Worker):
    """The MultiStepWorker is equivalent to a Worker except that it allows
    multiple forward passes in a single call, assuming the scheduler has
    allocated enough space to store the additional KV. This reduces overhead
    by invoking the scheduler less.

    The MultiStepWorker does not support cache swap operations, or beam search.
    Cache swap operations do not require large modifications. On the other hand,
    beam search requires memory allocations during sequence forks and thus
    requires more thought for MultiStepWorker support.
    """

    def set_include_gpu_probs_tensor(self) -> None:
        # Need include_gpu_probs_tensor for multi_step_worker
        self.model_runner.model.sampler.include_gpu_probs_tensor = True

    def execute_model(
        self, execute_model_req: ExecuteModelRequest
    ) -> Optional[List[SamplerOutput]]:
        if execute_model_req is not None:
            self._raise_if_unsupported(execute_model_req)
            execute_model_req.num_steps = execute_model_req.num_lookahead_slots + 1

        return super().execute_model(execute_model_req)

    def _raise_if_unsupported(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> None:
        """MultiStepWorker does not yet implement support for cache swap
        operations or beam search.
        """
        if any([
                execute_model_req.blocks_to_swap_in,
                execute_model_req.blocks_to_swap_out,
                execute_model_req.blocks_to_copy
        ]):
            raise NotImplementedError(
                "MultiStepWorker does not support cache operations")

        if any(
                len(seq_group_metadata.seq_data.keys()) != 1
                for seq_group_metadata in
                execute_model_req.seq_group_metadata_list):
            raise NotImplementedError(
                "MultiStepWorker does not support beam search.")
