import json

with open('03_15x80km_93GBd.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Replace 150 with 225 in make_bw, train_ldbp, and complexity_ldbp
        source = source.replace("make_bw(S_eval, 150)", "make_bw(S_eval, 225)")
        source = source.replace("make_bw(Strain, 150)", "make_bw(Strain, 225)")
        source = source.replace("train_ldbp(Strain, 150", "train_ldbp(Strain, 225")
        source = source.replace("complexity_ldbp(150", "complexity_ldbp(225")
        
        # Replace initialization [35]*M with [25]*M
        source = source.replace("[35]*M", "[25]*M")
        
        # Replace LDBP_POINTS
        source = source.replace(
            "LDBP_POINTS = [('p25',[25],25), ('p23',[23],23), ('p21',[21],21)]",
            "LDBP_POINTS = [('prune17',[17],17), ('p15',[15],15), ('p13',[13],13)]"
        )
        
        cell['source'] = [source]

with open('03_15x80km_93GBd.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook 03 switched to option A (M=225, prune to 13-17)!")
