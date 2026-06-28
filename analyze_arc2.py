"""Deep analysis of spatial transformation puzzles."""
import json
from pathlib import Path

for f in ['00d62c1b', '025d127b', '045e512c', '05269061', '05f2a901', '06df4c85', '08ed6ac7']:
    path = Path(f'ARC-AGI/data/training/{f}.json')
    with open(path) as fh:
        data = json.load(fh)
    train = data['train']
    print(f"\n{'='*60}")
    print(f"Task: {f} ({len(train)} train pairs)")
    for i, p in enumerate(train[:3]):
        inp, out = p['input'], p['output']
        ih, iw = len(inp), len(inp[0])
        oh, ow = len(out), len(out[0])
        
        # Count colors
        inp_colors = set(v for row in inp for v in row)
        out_colors = set(v for row in out for v in row)
        
        # Check if rotation
        is_rot = False
        if ih == oh and iw == ow:
            # Try 90 deg rotation
            rot90 = [[inp[ih-1-r][c] for r in range(ih)] for c in range(iw)]
            if rot90 == out:
                is_rot = True
                print(f"  Pair {i}: {ih}x{iw} ROTATE_90")
            # Try flip H
            flip_h = [row[::-1] for row in inp]
            if flip_h == out:
                is_rot = True
                print(f"  Pair {i}: {ih}x{iw} FLIP_H")
            # Try flip V
            flip_v = inp[::-1]
            if flip_v == out:
                is_rot = True
                print(f"  Pair {i}: {ih}x{iw} FLIP_V")
        
        if not is_rot:
            # Check if color mapping exists
            mapping = {}
            consistent = True
            for r in range(min(ih, oh)):
                for c in range(min(iw, ow)):
                    iv, ov = inp[r][c], out[r][c]
                    if iv in mapping:
                        if mapping[iv] != ov:
                            consistent = False
                    else:
                        mapping[iv] = ov
            if ih == oh and iw == ow and consistent:
                print(f"  Pair {i}: {ih}x{iw} COLOR_MAP {mapping}")
            else:
                print(f"  Pair {i}: {ih}x{iw}->{oh}x{ow} UNKNOWN (colors: {inp_colors}->{out_colors})")
