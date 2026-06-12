import json

with open('02_170km_186GBd.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Replace 50 with 100 for make_bw and train_ldbp
        source = source.replace("make_bw(S_eval, 50)", "make_bw(S_eval, 100)")
        source = source.replace("make_bw(Strain, 50)", "make_bw(Strain, 100)")
        source = source.replace("train_ldbp(Strain, 50", "train_ldbp(Strain, 100")
        source = source.replace("complexity_ldbp(50", "complexity_ldbp(100")
        
        # Replace initialization [55]*M with [21]*M
        source = source.replace("[55]*M", "[21]*M")
        
        # Replace LDBP_POINTS
        source = source.replace(
            "LDBP_POINTS = [('prune45',[45],45), ('p41',[41],41), ('p37',[37],37), ('p35',[35],35)]",
            "LDBP_POINTS = [('prune17',[17],17), ('p15',[15],15), ('p13',[13],13), ('p11',[11],11), ('v10',[11,9],10)]"
        )
        
        cell['source'] = [source]

with open('02_170km_186GBd.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook 02 switched to 100 total steps and initialization 21!")
