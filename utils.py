# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist

from timm.utils.agc import adaptive_clip_grad
import math


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save({'state_dict_ema':checkpoint}, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

# def init_distributed_mode(args):
#     if args.dist_on_itp:
#         args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
#         args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
#         args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
#         args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
#         os.environ['LOCAL_RANK'] = str(args.gpu)
#         os.environ['RANK'] = str(args.rank)
#         os.environ['WORLD_SIZE'] = str(args.world_size)
#     elif 'SLURM_PROCID' in os.environ:
#         args.rank = int(os.environ['SLURM_PROCID'])
#         args.gpu = int(os.environ['SLURM_LOCALID'])
#         args.world_size = int(os.environ['SLURM_NTASKS'])
#         os.environ['RANK'] = str(args.rank)
#         os.environ['LOCAL_RANK'] = str(args.gpu)
#         os.environ['WORLD_SIZE'] = str(args.world_size)

#         node_list = os.environ['SLURM_NODELIST']
#         addr = subprocess.getoutput(
#             f'scontrol show hostname {node_list} | head -n1')
#         if 'MASTER_ADDR' not in os.environ:
#             os.environ['MASTER_ADDR'] = addr
#     elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         args.rank = int(os.environ["RANK"])
#         args.world_size = int(os.environ['WORLD_SIZE'])
#         args.gpu = int(os.environ['LOCAL_RANK'])
#     else:
#         print('Not using distributed mode')
#         args.distributed = False
#         return

#     args.distributed = True

#     torch.cuda.set_device(args.gpu)
#     args.dist_backend = 'nccl'
#     print('| distributed init (rank {}): {}, gpu {}'.format(
#         args.rank, args.dist_url, args.gpu), flush=True)
#     torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
#                                          world_size=args.world_size, rank=args.rank, timeout=datetime.timedelta(seconds=20888))
#     torch.distributed.barrier()
#     # assert torch.distributed.is_initialized()
#     setup_for_distributed(args.rank == 0)


# if 'pos_embed' in state_dict:
def interpolate_pos_embed(model, state_dict):
    pos_embed_checkpoint = state_dict['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    # class_token and dist_token are kept unchanged
    if orig_size != new_size:
        print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        state_dict['pos_embed'] = new_pos_embed

# def get_cls_idx(n_tokens, n_cls):
#     L = n_tokens // (n_cls + 1)

#     token_idx = torch.cat([
#         torch.arange(L * n_cls).view(n_cls, -1),
#         torch.arange(n_tokens, n_tokens + n_cls).unsqueeze(-1)
#         ], dim=1).contiguous().view(-1)
#     token_idx = torch.cat([token_idx, torch.arange(L * n_cls, n_tokens)])

#     cls_pos = torch.arange(L, L * (n_cls + 1) + n_cls, L + 1)

#     return token_idx, cls_pos

def get_cls_idx(H, W, n_cls, cross=False):
    n_tokens = H * W
    L = n_tokens // (n_cls + 1)
    token_idx = torch.cat([
        torch.arange(L * n_cls).view(n_cls, -1),
        torch.arange(n_tokens, n_tokens + n_cls).unsqueeze(-1)
        ], dim=1).contiguous().view(-1)
    token_idx = torch.cat([token_idx, torch.arange(L * n_cls, n_tokens)])
    cls_pos = torch.arange(L, L * (n_cls + 1) + n_cls, L + 1)
    if not cross:
        return token_idx, cls_pos
    cross_idx = torch.arange(n_tokens + n_cls)
    p_img = token_idx < n_tokens
    cross_idx[p_img] = token_idx[p_img][torch.arange(n_tokens).view(H, W).T.flatten()]
    cross_idx[p_img] += cross_idx[p_img] // L
    cross_idx[n_cls * L + n_cls + L:] -= 1
    return token_idx, cls_pos, cross_idx


# def get_cross_idx(H, W, token_idx):
#     cross_idx = torch.arange(len(token_idx))
#     cross_idx[token_idx >= H * W] = token_idx[token_idx >= H * W]
#     image_idx = token_idx[token_idx < H * W]
#     image_idx = image_idx[torch.arange(H * W).view(H, W).T.flatten()]
#     cross_idx[token_idx < H * W] = image_idx
#     return cross_idx

class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
            self,
            loss,
            optimizer,
            clip_grad=None,
            clip_mode='norm',
            parameters=None,
            create_graph=False,
            need_update=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if need_update:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

def dispatch_clip_grad(parameters, value: float, mode: str = 'norm', norm_type: float = 2.0):
    """ Dispatch to gradient clipping method

    Args:
        parameters (Iterable): model parameters to clip
        value (float): clipping value/factor/norm, mode dependant
        mode (str): clipping mode, one of 'norm', 'value', 'agc'
        norm_type (float): p-norm, default 2.0
    """
    if mode == 'norm':
        torch.nn.utils.clip_grad_norm_(parameters, value, norm_type=norm_type)
    elif mode == 'value':
        torch.nn.utils.clip_grad_value_(parameters, value)
    elif mode == 'agc':
        adaptive_clip_grad(parameters, value, norm_type=norm_type)
    else:
        assert False, f"Unknown clip mode ({mode})."

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def adjust_weight_decay(optimizer, epoch, args):
    wd = args.weight_decay_end + (args.weight_decay - args.weight_decay_end) * 0.5 * \
            (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group["weight_decay"] = wd
    return wd

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, need_update=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if need_update:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == 'inf':
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm