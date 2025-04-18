from config import GPTConfig
from model import GPT2
from dataloader import DataLoaderLite
import torch
import time



dataloader = DataLoaderLite(8, 512)
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


# initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2(GPTConfig())
model.to(device)

# optimize the model

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for epoch in range(50):  # Number of epochs
    t0 = time.time()
    optimizer.zero_grad()
    x, y = dataloader.next_batch()
    x = x.to(device)
    y = y.to(device)
    logits, loss = model(x, y)
    #import code; code.interact(local=locals())
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()  # Wait for all CUDA kernels to finish
    t1 = time.time()
    dt = (t1 - t0) * 1000  # Convert to milliseconds
    print(f"Epoch {epoch}: loss = {loss.item()}", f"dt = {dt:.2f} ms")
