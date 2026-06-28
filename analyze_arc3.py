"""Visualize specific puzzle pairs to understand patterns."""
import json
from pathlib import Path

def print_grid(grid, label=""):
    print(f"{label}:")
    for row in grid:
        print("  " + " ".join(str(v) for v in row))

# Look at 00d62c1b - new color appears
path = Path('ARC-AGI/data/training/00d62c1b.json')
with open(path) as fh:
    data = json.load(fh)
p = data['train'][0]
print("="*60)
print("Task 00d62c1b - New color (4) appears in output")
print_grid(p['input'], "Input")
print_grid(p['output'], "Output")

# Look at 05269061 - background removed
path = Path('ARC-AGI/data/training/05269061.json')
with open(path) as fh:
    data = json.load(fh)
p = data['train'][0]
print("\n" + "="*60)
print("Task 05269061 - Background (0) removed from output")
print_grid(p['input'], "Input")
print_grid(p['output'], "Output")

# Look at 08ed6ac7 - single color expands
path = Path('ARC-AGI/data/training/08ed6ac7.json')
with open(path) as fh:
    data = json.load(fh)
p = data['train'][0]
print("\n" + "="*60)
print("Task 08ed6ac7 - Color 5 becomes multiple colors")
print_grid(p['input'], "Input")
print_grid(p['output'], "Output")
