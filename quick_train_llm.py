"""Train local LLM — compact config for fast convergence on CPU."""
import sys, os, time
sys.path.insert(0, '.')
import torch
from torch.utils.data import DataLoader, TensorDataset
from soma_mythos_ehra.arc3.local_coder import ARCDomainLLM, ARCCoderConfig
from soma_mythos_ehra.arc3.trajectory_tokenizer import TOKEN_MAP

d = torch.load('checkpoints/local_llm_dataset.pt')
input_ids = d['input_ids']
target_ids = d['target_ids']
print(f'Full dataset: {input_ids.shape}')

# Use subset
n = min(3000, input_ids.shape[0])
input_ids = input_ids[:n]
target_ids = target_ids[:n]

# Compact model: 3M params instead of 15M
config = ARCCoderConfig(
    vocab_size=TOKEN_MAP.vocab_size,
    d_model=128,
    n_layer=4,
    n_head=4,
    max_seq_len=256,
    dropout=0.05,
)
model = ARCDomainLLM(config)

dataset = TensorDataset(input_ids, target_ids)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

best_loss = float('inf')
start = time.time()
for epoch in range(20):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inp, tgt in loader:
        optimizer.zero_grad()
        logits, loss = model(inp, targets=tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        mask = tgt != 0
        correct += ((preds == tgt) & mask).sum().item()
        total += mask.sum().item()
    scheduler.step()
    avg_loss = total_loss / len(loader)
    acc = correct / total if total > 0 else 0
    elapsed = time.time() - start
    print(f'Epoch {epoch+1}/20: loss={avg_loss:.4f}, acc={acc:.2%}, {elapsed:.0f}s')
    if avg_loss < best_loss:
        best_loss = avg_loss
        model.save('checkpoints/local_arc_llm.pt')

fsize = os.path.getsize('checkpoints/local_arc_llm.pt') / 1e6
print(f'Done. Best loss: {best_loss:.4f}. Model: {fsize:.1f}MB')
