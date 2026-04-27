import torch
from model import GPTLangModel, generate


device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load("german_gpt_checkpoint.pt", map_location=device)

stoi = checkpoint["stoi"]
itos = checkpoint["itos"]
vocab_size = checkpoint["vocab_size"]
block_size = checkpoint["block_size"]

def encode(s):
    return [stoi[c] for c in s]

def decode(tokens):
    return "".join([itos[i] for i in tokens])

model = GPTLangModel(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=128,
    n_head=4,
    n_layer=3,
    dropout=0.2
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

prompt = "Gregor"
context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

output = generate(model, context, max_new_tokens=500)

print(decode(output[0].tolist()))
