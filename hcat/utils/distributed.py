"""
For code used in distributed training.
"""
from typing import Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor


def set_sharing_strategy(new_strategy=None):
    """
    https://pytorch.org/docs/stable/multiprocessing.html
    https://discuss.pytorch.org/t/how-does-one-setp-up-the-set-sharing-strategy-strategy-for-multiprocessing/113302
    https://stackoverflow.com/questions/66426199/how-does-one-setup-the-set-sharing-strategy-strategy-for-multiprocessing-in-pyto
    """
    from sys import platform

    if new_strategy is not None:
        mp.set_sharing_strategy(new_strategy=new_strategy)
    else:
        if platform == 'darwin':  # OS X
            # only sharing strategy available at OS X
            mp.set_sharing_strategy('file_system')
        else:
            # ulimit -n 32767 or ulimit -n unlimited (perhaps later do try catch to execute this increase fd limit)
            mp.set_sharing_strategy('file_descriptor')

def use_file_system_sharing_strategy():
    """
    when to many file descriptor error happens

    https://discuss.pytorch.org/t/how-does-one-setp-up-the-set-sharing-strategy-strategy-for-multiprocessing/113302
    """
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def setup_process(rank, world_size, port, backend='gloo'):
    """
    Initialize the distributed environment (for each process).

    gloo: is a collective communications library (https://github.com/facebookincubator/gloo). My understanding is that
    it's a library/API for process to communicate/coordinate with each other/master. It's a backend library.

    export NCCL_SOCKET_IFNAME=eth0
    export NCCL_IB_DISABLE=1

    https://stackoverflow.com/questions/61075390/about-pytorch-nccl-error-unhandled-system-error-nccl-version-2-4-8

    https://pytorch.org/docs/stable/distributed.html#common-environment-variables
    """
    import torch.distributed as dist
    import os
    import torch

    if rank != -1:  # -1 rank indicates serial code
        print(f'setting up rank={rank} (with world_size={world_size})')
        # MASTER_ADDR = 'localhost'
        MASTER_ADDR = '127.0.0.1'
        # set up the master's ip address so this child process can coordinate
        os.environ['MASTER_ADDR'] = MASTER_ADDR
        print(f"{MASTER_ADDR}")
        os.environ['MASTER_PORT'] = port
        print(f"{port}")

        # - use NCCL if you are using gpus: https://pytorch.org/tutorials/intermediate/dist_tuto.html#communication-backends
        if torch.cuda.is_available():
            # unsure if this is really needed
            # os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
            # os.environ['NCCL_IB_DISABLE'] = '1'
            backend = 'nccl'
        print(f'{backend}')
        # Initializes the default distributed process group, and this will also initialize the distributed package.
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        # dist.init_process_group(backend, rank=rank, world_size=world_size)
        # dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        print(f'--> done setting up rank={rank}')

def cleanup(rank):
    """ Destroy a given process group, and deinitialize the distributed package """
    # only destroy the process distributed group if the code is not running serially
    if rank != -1:  # -1 rank indicates serial code
        dist.destroy_process_group()

def get_batch(batch: Tuple[Tensor, Tensor], rank) -> Tuple[Tensor, Tensor]:
    x, y = batch
    if torch.cuda.is_available():
        x, y = x.to(rank), y.to(rank)
    else:
        # I don't think this is needed...
        # x, y = x.share_memory_(), y.share_memory_()
        pass
    return x, y

def test_setup():
    print('test_setup')
    port = find_free_port()
    world_size = 2
    mp.spawn(setup_process, args=(world_size, port), nprocs=2)
    print('successful test_setup!')


if __name__ == '__main__':
    test_setup()