# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/utils.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import json
import os
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.logger import init_logger

from .parallel_state import get_cpu_world_group, get_local_rank

logger = init_logger(__name__)


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> Sequence[torch.Tensor]:
    """ Split a tensor along its last dimension.

        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                     in memory.

        Returns:
            A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # NOTE: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


# code partly borrowed from
# https://github.com/turboderp/exllamav2/blob/1c67f97f3d2a968605a9c31ab791a05c85bb7879/exllamav2/compat.py#L10
# License: MIT
def _can_actually_p2p(idx_a, idx_b):
    dev_i = f"cuda:{idx_a}"
    dev_j = f"cuda:{idx_b}"
    a = torch.randn(5, device=dev_i) + 123.0
    b = a.to(dev_j)
    c = b.to(dev_i)
    return torch.all(a == c).cpu().item()


# why do we need this cache?
# 1. we can have runtime checks for P2P access, where every process checks
#  P2P access to all other GPUs. Unfortunately, the test might cost many
#  (world_size * world_size) cuda context, and reduce the memory available
#  for the model. see https://github.com/vllm-project/vllm/issues/3821
# 2. alternatively, we can have a p2p map that is generated by the master
#  process and broadcasted to all other processes. This still requires
#  #world_size of cuda context, belonging to the master process, on each GPU.
# 3. we can have a cache file, that records the p2p access status. The first
#  time the master process checks the p2p access, it will generate the cache
#  file, at the cost of #world_size of cuda context. Later on, all processes
#  can read the cache file to check the p2p access status without any cost of
#  additional cuda context.
# Note that the cache file is suffixed by the CUDA_VISIBLE_DEVICES, so that we
#  can have different cache files for different CUDA_VISIBLE_DEVICES settings,
#  e.g. used by different vllm engines. The device id in the cache file is a
#  **local** device id, i.e. from 0 to num_dev-1, where num_dev is the number
#  of visible devices in the vllm engine.
_gpu_p2p_access_cache: Optional[Dict[str, bool]] = None


def gpu_p2p_access_check(i: int, j: int) -> bool:
    """Check if GPU i can access GPU j."""

    # if the cache variable is already calculated,
    # read from the cache instead of checking it again
    global _gpu_p2p_access_cache
    if _gpu_p2p_access_cache is not None:
        return _gpu_p2p_access_cache[f"{i}->{j}"]

    is_distributed = dist.is_initialized()

    num_dev = torch.cuda.device_count()
    cuda_visible_devices = envs.CUDA_VISIBLE_DEVICES
    if cuda_visible_devices is None:
        cuda_visible_devices = ",".join(str(i) for i in range(num_dev))
    VLLM_CONFIG_ROOT = envs.VLLM_CONFIG_ROOT
    path = os.path.expanduser(
        f"{VLLM_CONFIG_ROOT}/vllm/gpu_p2p_access_cache_for_{cuda_visible_devices}.json"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if (not is_distributed or get_local_rank() == 0) \
        and (not os.path.exists(path)):
        # only the local master process (with local_rank == 0) can
        #  enter this block to calculate the cache
        logger.info("generating GPU P2P access cache for in %s", path)
        cache = {}
        for _i in range(num_dev):
            for _j in range(num_dev):
                # on some platforms, P2P support might be buggy and we need
                # additional checks. See also:
                # https://github.com/vllm-project/vllm/issues/2728
                cache[f"{_i}->{_j}"] = torch.cuda.can_device_access_peer(
                    _i, _j) and _can_actually_p2p(_i, _j)
        with open(path, "w") as f:
            json.dump(cache, f, indent=4)
    if is_distributed:
        cpu_world_group = get_cpu_world_group()
        dist.barrier(cpu_world_group)
    logger.info("reading GPU P2P access cache from %s", path)
    with open(path, "r") as f:
        cache = json.load(f)
    _gpu_p2p_access_cache = cache
    return _gpu_p2p_access_cache[f"{i}->{j}"]


def get_pp_indices(num_hidden_layers: int, pp_rank: int,
                   pp_size: int) -> Tuple[int, int]:

    layers_per_partition = divide(num_hidden_layers, pp_size)
    start_layer = pp_rank * layers_per_partition
    end_layer = start_layer + layers_per_partition
    return (start_layer, end_layer)
    if pp_size <= 2:
        layers_per_partition = divide(num_hidden_layers, pp_size)
        start_layer = pp_rank * layers_per_partition
        end_layer = start_layer + layers_per_partition
    else:
        # uneven partitioning to account for extra work done by the first and last rank
        layers_per_partition = divide(num_hidden_layers, pp_size)
        #pp_size -2 will each take a layer from the first and last rank
        first_rank_layers = layers_per_partition - (pp_size - 2) * 2
        last_rank_layers = layers_per_partition - (pp_size - 2) * 2
        layers_per_partition += 4

        if pp_rank == 0:
            start_layer = 0
            end_layer = start_layer + first_rank_layers
        elif pp_rank == pp_size - 1:
            start_layer = num_hidden_layers - last_rank_layers
            end_layer = num_hidden_layers
        else:
            start_layer = first_rank_layers + (pp_rank - 1) * layers_per_partition
            end_layer = start_layer + layers_per_partition
        print('pp_rank:', pp_rank, 'start_layer:', start_layer, 'end_layer:', end_layer)
            
    return (start_layer, end_layer)
