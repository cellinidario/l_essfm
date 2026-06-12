import json

with open('03_15x80km_93GBd.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        old = "def peak(nos, kind, ns, params=None, edc=False):\n    S = build_system(CFG[nos])\n    return curve(S, data, Pgrid, kind, ns, params, edc=edc).max()"
        new = "def peak(nos, kind, ns, params=None, edc=False):\n    S = build_system(CFG[nos])\n    ts = False if kind in ('ideal', 'edc') else None\n    return curve(S, data, Pgrid, kind, ns, params, edc=edc, total_steps=ts).max()"
        source = source.replace(old, new)
        cell['source'] = [source]

with open('03_15x80km_93GBd.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Fixed peak function in notebook 03!")
