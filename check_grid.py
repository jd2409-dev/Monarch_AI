import json

f = open(r'C:\Users\Jaydan\ARC-AGI-3-Agents\recordings\ls20-9607627b.Monarch_AI.100.d99d8f2d-cd9a-44f0-91da-e21accddca8a.recording.jsonl')
lines = f.readlines()

frame = json.loads(lines[0])['data']['frame'][0]

# Show full 21x21 around agent
r, c = 32, 20
r0, r1 = max(0, r-10), min(64, r+11)
c0, c1 = max(0, c-10), min(64, c+11)
print(f"Agent region ({r0}:{r1}, {c0}:{c1}):")
for row in range(r0, r1):
    vals = []
    for col in range(c0, c1):
        v = frame[row][col]
        vals.append(str(v).rjust(3))
    print(f"  Row {row:2d}: {' '.join(vals)}")

# What does the overall grid look like?
print("\nOverall grid value distribution:")
for v in sorted(set(frame[r][c] for r in range(64) for c in range(64))):
    count = sum(1 for r in range(64) for c in range(64) if frame[r][c] == v)
    print(f"  Value {v:2d}: {count:5d} cells")

# Check if there's a goal (value 3 in our sim, but in ARC it might be different)
# Look for value 0 cells - these are empty/traversable
v0 = [(r,c) for r in range(64) for c in range(64) if frame[r][c] == 0]
print(f"\nValue 0 (empty) cells: {v0}")

# Show top/bottom/left/right boundaries
print("\nTop row:", frame[0][:20])
print("Bottom row:", frame[63][:20])
print("Left col:", [frame[r][0] for r in range(20)])
