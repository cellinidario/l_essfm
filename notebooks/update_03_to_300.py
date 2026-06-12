import json

with open('03_15x80km_93GBd.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Replace 225 with 300
        source = source.replace("make_bw(S_eval, 225)", "make_bw(S_eval, 300)")
        source = source.replace("make_bw(Strain, 225)", "make_bw(Strain, 300)")
        source = source.replace("train_ldbp(Strain, 225", "train_ldbp(Strain, 300")
        source = source.replace("complexity_ldbp(225", "complexity_ldbp(300")
        
        # Replace initialization [25]*M with [21]*M
        source = source.replace("[25]*M", "[21]*M")
        
        # Replace LDBP_POINTS
        source = source.replace(
            "LDBP_POINTS = [('prune17',[17],17), ('p15',[15],15), ('p13',[13],13)]",
            "LDBP_POINTS = [('prune17',[17],17), ('p15',[15],15), ('p13',[13],13), ('p11',[11],11), ('v10',[11,9],10)]"
        )
        
        cell['source'] = [source]

with open('03_15x80km_93GBd.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook 03 switched to 300 total steps (20/span) and initialization 21!")
