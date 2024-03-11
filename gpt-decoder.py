import torch

import torch.nn as nn

from torch.nn import functional as F


# Global parameters
BATCH_SIZE = 8  # number of sequences to process in parallel
BLOCK_SIZE = 32  # maximum context length for generating
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LR = 3e-4
EVAL_ITERS = 200
N_EMBD = 128
N_FF = 4 * N_EMBD
N_HEAD = 6
N_LAYER = 2
DROPOUT = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)

# load data
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# unique characters in dataset
chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)

# character to integer mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder
decode = lambda l: ''.join([itos[i] for i in l])  # decoder

# split data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    "Generate a small batch of data of inputs x and targets y."
    data = train_data if split == 'train' else val_data
    idx = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE, ))
    x = torch.stack([data[i:BLOCK_SIZE + i] for i in idx])
    y = torch.stack([data[i + 1:BLOCK_SIZE + i + 1] for i in idx])
    x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()  # tell PyTorch no need to allocate memory for backpropagation
def estimate_loss():
    out = {}
    model.eval()  # turns off model weight update
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()
    model.train()  # turns on model weight update

    return out


class Head(nn.Module):
    "One head of self-attention."


    def __init__(self, HEAD_SIZE):
        super().__init__()
        self.key = nn.Linear(N_EMBD, HEAD_SIZE, bias=False)
        self.query = nn.Linear(N_EMBD, HEAD_SIZE, bias=False)
        self.value = nn.Linear(N_EMBD, HEAD_SIZE, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)


    def forward(self, x):
        # Input of size (B, T, C)
        # Output of size (B, T, HS)
        B, T, C = x.shape
        k = self.key(x)  # (B, T, HS)
        q = self.query(x)  # (B, T, HS)

        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, HS) @ (B, HS, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # Perform weighted aggregation of values
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, HS) --> (B, T, HS)

        return out


class MultiHeadAttention(nn.Module):
    "Multiple heads of self-attenion in parallel."


    def __init__(self, NUM_HEADS, HEAD_SIZE):
        super().__init__()
        self.heads = nn.ModuleList([Head(HEAD_SIZE) for _ in range(NUM_HEADS)])
        self.proj = nn.Linear(HEAD_SIZE * NUM_HEADS, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)


    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))

        return out


class FeedForward(nn.Module):
    "Simple linear layer followed by a nonlinearity."


    def __init__(self, N_EMBD):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, N_FF),
            nn.ReLU(),
            nn.Linear(N_FF, N_EMBD),
            nn.Dropout(DROPOUT)
        )


    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    "Transformer block: communication followed by computation."


    def __init__(self, N_EMBD, N_HEAD):
        super().__init__()
        HEAD_SIZE = N_EMBD // N_HEAD
        self.sa = MultiHeadAttention(N_HEAD, HEAD_SIZE)
        self.ffwd = FeedForward(N_EMBD)
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)


    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


class GPTDecoder(nn.Module):
    "Generatively Pretrained Transformer (GPT) decoder language model."


    def __init__(self):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)


    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensors of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T, C); C == channel
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, VOCAB_SIZE)

        if targets is None:
            loss = None
        else:
            # reshape logits and targets to conform with F.cross_entropy expected input size
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to last BLOCK_SIZE tokens
            idx_cond = idx[:, -BLOCK_SIZE:]

            # Get the predictions
            logits, loss = self(idx_cond)

            # Focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)

        return idx


model = GPTDecoder()
m = model.to(device)
print(f'{sum(p.numel() for p in m.parameters()) / 1e6}M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for iter in range(MAX_ITERS):

    # every once in a while evaluate the loss on the train and val sets
    if iter % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()  # backpropagation
    optimizer.step()  # update weights

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
