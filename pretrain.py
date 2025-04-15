from config import GPTConfig
from model import GPT2
import torch
import tiktoken

# get a small sample of text to train on
with open('input.txt', 'r') as f:
    text = f.read()

text = text[:1000]  # Truncate to 1000 characters
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(text)

B, T = 4, 32  # Batch size and sequence length
buf = torch.tensor(tokens[:B * T + 1])  # Buffer for the batch

x = buf[:-1].view(B, T)  # Input tensor of shape (B, T)
y = buf[1:].view(B, T)  # Target tensor of shape (B, T)


# initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2(GPTConfig())
model.to(device)


