import os
import io
from contextlib import contextmanager
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_dist(rank, local_rank, world_size, master_addr, master_port):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def broadcast_file_exists(path: str) -> bool:


    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return os.path.isfile(path)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    flag = torch.zeros(1, dtype=torch.long, device=device)
    if dist.get_rank() == 0:
        flag[0] = 1 if os.path.isfile(path) else 0
    dist.broadcast(flag, src=0)
    return flag.item() == 1


def read_file_dist(path):


    if dist.is_initialized() and dist.get_world_size() > 1:

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")


        size = torch.LongTensor(1).to(device)
        if dist.get_rank() == 0:
            with open(path, "rb") as f:
                data = f.read()
            data = torch.ByteTensor(
                torch.UntypedStorage.from_buffer(data, dtype=torch.uint8)
            ).to(device)
            size[0] = data.shape[0]

        dist.broadcast(size, src=0)
        if dist.get_rank() != 0:
            data = torch.ByteTensor(size[0].item()).to(device)

        dist.broadcast(data, src=0)

        data = data.cpu().numpy().tobytes()
        data = io.BytesIO(data)
        return data
    else:
        with open(path, "rb") as f:
            data = f.read()
        data = io.BytesIO(data)
        return data


def unwrap_dist(model):


    if isinstance(model, DDP):
        return model.module
    return model


@contextmanager
def master_first():


    if not dist.is_initialized():
        yield
    else:
        if dist.get_rank() == 0:
            yield
            dist.barrier()
        else:
            dist.barrier()
            yield


@contextmanager
def local_master_first():


    if not dist.is_initialized():
        yield
    else:
        if dist.get_rank() % torch.cuda.device_count() == 0:
            yield
            dist.barrier()
        else:
            dist.barrier()
            yield
