import json

with open('03_15x80km_93GBd.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        if "make_bw(S_eval, 300)" in source:
            source = source.replace("make_bw(S_eval, 300)", "make_bw(S_eval, 20, total_steps=False)")
        if "make_bw(Strain, 300)" in source:
            source = source.replace("make_bw(Strain, 300)", "make_bw(Strain, 20, total_steps=False)")
        if "train_ldbp(Strain, 300" in source:
            source = source.replace("train_ldbp(Strain, 300", "train_ldbp(Strain, 20")
        if "target_taps=target, n_blocks=200)" in source:
            source = source.replace("target_taps=target, n_blocks=200)", "target_taps=target, n_blocks=200, total_steps=False)")
            
        cell['source'] = [source]

with open('03_15x80km_93GBd.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("LDBP in 03 successfully fixed to 20 steps per span!")
