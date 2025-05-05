from config import GPTConfig
from model import GPT2
from warmup import get_lr
from dataloader import DataLoaderLite
import os
import time
import torch
import time
import math

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device(f'cuda:{ddp_local_rank}')
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"using device: {device.type}")



total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 8 # micro batch size
T = 512 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
 
dataloader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, master_process=master_process, num_processes=ddp_world_size)
# # get a small sample of text to train on
# with open('input.txt', 'r') as f:
#     text = f.read()

# text = text[:1000]  # Truncate to 1000 characters
# enc = tiktoken.get_encoding("gpt2")
# tokens = enc.encode(text)

# B, T = 4, 32  # Batch size and sequence length
# buf = torch.tensor(tokens[:B * T + 1])  # Buffer for the batch

# x = buf[:-1].view(B, T)  # Input tensor of shape (B, T)
# y = buf[1:].view(B, T)  # Target tensor of shape (B, T)






model = GPT2(GPTConfig())
model.to(device)
# model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

# optimize the model

model.train()



max_lr = 6e-4 
min_lr = max_lr * 0.1 
warmup_steps = 10 
max_steps = 50 

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, master_process=master_process, device=device)

# torch.set_float32_matmul_precision('high') # TF32 matmul precision

for step in range(max_steps):  # Number of epochs
    t0 = time.time()
    optimizer.zero_grad()


    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = dataloader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping

    lr = get_lr(step, 
                max_lr=max_lr, 
                min_lr=min_lr, 
                warmup_steps=warmup_steps, 
                max_steps=max_steps)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize()  # Wait for all CUDA kernels to finish
    t1 = time.time()
    dt = (t1 - t0)  # seconds per step
    tokens_processed = dataloader.B * dataloader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    # dt ms
    if master_process:
        print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

if ddp:
     destroy_process_group()