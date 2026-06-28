"""Analyze what transformation types are needed for training puzzles."""
import json
from pathlib import Path

for f in ['007bbfb7', '00d62c1b', '017c7c7b', '025d127b', '045e512c',
          '0520fde7', '05269061', '05f2a901', '06df4c85', '08ed6ac7']:
    path = Path(f'ARC-AGI/data/training/{f}.json')
    with open(path) as fh:
        data = json.load(fh)
    train = data['train']
    pairs_info = []
    for p in train[:2]:
        inp, out = p['input'], p['output']
        ih, iw = len(inp), len(inp[0])
        oh, ow = len(out), len(out[0])
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
            pairs_info.append(f'{ih}x{iw}->{oh}x{ow} COLOR_MAP')
        elif ih != oh or iw != ow:
            pairs_info.append(f'{ih}x{iw}->{oh}x{ow} RESIZE')
        else:
            pairs_info.append(f'{ih}x{iw}->{oh}x{ow} SPATIAL')
    print(f'{f}: {" | ".join(pairs_info)}')
