import torch
import matplotlib.pyplot as plt
from model import GPTLanguageModel
import re
import os


# load dataset
!wget -O german_kafka.txt https://www.gutenberg.org/cache/epub/22367/pg22367.txt

# clean dataset
with open("german_kafka.txt", "r", encoding="utf-8-sig") as f:
    raw_text = f.read()

print(raw_text[:1000])
print("Raw length:", len(raw_text))

start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK DIE VERWANDLUNG ***"
end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK DIE VERWANDLUNG ***"

start = raw_text.find(start_marker)
end = raw_text.find(end_marker)

if start != -1:
    raw_text = raw_text[start + len(start_marker):]

if end != -1:
    raw_text = raw_text[:end]

text = raw_text.strip()

print("Cleaned length:", len(text))
print(text[:1000])



text = text.replace("\r\n", "\n")
text = text.replace("\r", "\n")

text = re.sub(r"\n{3,}", "\n\n", text)
text = re.sub(r"[ \t]+", " ", text)

text = text.strip()

print("Final cleaned length:", len(text))
print(text[:1000])

# tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s]

def decode(tokens):
    return "".join([itos[i] for i in tokens])

print("Vocabulary size:", vocab_size)

# train/val split
data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print("Train tokens:", len(train_data))
print("Validation tokens:", len(val_data))

# get_batch()
batch_size = 32
block_size = 128

max_iters = 2500
eval_interval = 250
eval_iters = 100

learning_rate = 3e-4

n_embd = 128
n_head = 4
n_layer = 3
dropout = 0.2

def get_batch(split):
    data_split = train_data if split == "train" else val_data

    ix = torch.randint(len(data_split) - block_size, (batch_size,))

    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])

    x = x.to(device)
    y = y.to(device)

    return x, y

# estimate_loss()
@torch.no_grad()
def estimate_loss(model):
    out = {}

    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()

    return out
# create model
# optimizer
model = GPTLangModel(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=128,
    n_head=4,
    n_layer=4,
    dropout=0.1
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
train_losses = []
val_losses = []
steps = []

for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)

        train_losses.append(losses["train"])
        val_losses.append(losses["val"])
        steps.append(iter)

        print(
            f"step {iter}: "
            f"train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# save loss plot
os.makedirs("outputs", exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(steps, train_losses, label="Train loss")
plt.plot(steps, val_losses, label="Validation loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

# Save plot
plt.savefig("outputs/loss_curve.png")

plt.show()

# save model checkpoint
checkpoint = {
    "model_state_dict": model.state_dict(),
    "stoi": stoi,
    "itos": itos,
    "vocab_size": vocab_size,
    "block_size": block_size,
}

torch.save(checkpoint, "outputs/german_gpt_checkpoint.pt")

print("Model checkpoint saved!")

# save generated samples
with open("outputs/samples.txt", "w", encoding="utf-8") as f:
    for i in range(3):
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        tokens = generate(model, context, max_new_tokens=500)
        text_out = decode(tokens[0].tolist())

        f.write(f"\n--- Sample {i+1} ---\n")
        f.write(text_out + "\n")

