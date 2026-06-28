"""Debug why heuristics aren't matching specific puzzles."""
import json
import torch
from pathlib import Path
from soma_mythos_ehra.arc3.transforms import apply_fill_holes, apply_tile
from soma_mythos_ehra.arc3.objects import connected_component_labeling

# Test 00d62c1b - hole filling
path = Path('ARC-AGI/data/training/00d62c1b.json')
with open(path) as fh:
    data = json.load(fh)
p = data['train'][0]
inp = torch.tensor(p['input'], dtype=torch.long)
out = torch.tensor(p['output'], dtype=torch.long)

print("Task 00d62c1b - Hole filling test:")
pred = apply_fill_holes(inp)
print(f"  Input: {inp.tolist()}")
print(f"  Expected: {out.tolist()}")
print(f"  Got:      {pred.tolist()}")
print(f"  Match: {torch.equal(pred, out)}")

# Test 05269061 - tiling
path = Path('ARC-AGI/data/training/05269061.json')
with open(path) as fh:
    data = json.load(fh)
p = data['train'][0]
inp = torch.tensor(p['input'], dtype=torch.long)
out = torch.tensor(p['output'], dtype=torch.long)

print("\nTask 05269061 - Tiling test:")
print(f"  Input shape: {inp.shape}")
print(f"  Output shape: {out.shape}")
# The input has a 3x3 pattern in top-left, output tiles it
tile = inp[:3, :3]
print(f"  Tile (3x3): {tile.tolist()}")
pred = apply_tile(tile, 3, 3)
print(f"  Tiled (9x9): {pred.tolist()}")
print(f"  Expected: {out.tolist()}")
print(f"  Match: {torch.equal(pred, out)}")

# Test 08ed6ac7 - component coloring
path = Path('ARC-AGI/data/training/08ed6ac7.json')
with open(path) as fh:
    data = json.load(fh)
p = data['train'][0]
inp = torch.tensor(p['input'], dtype=torch.long)
out = torch.tensor(p['output'], dtype=torch.long)

print("\nTask 08ed6ac7 - Component coloring test:")
labels = connected_component_labeling(inp)
print(f"  Input: {inp.tolist()}")
print(f"  Labels: {labels.tolist()}")
print(f"  Num components: {int(labels.max().item())}")
print(f"  Expected: {out.tolist()}")
