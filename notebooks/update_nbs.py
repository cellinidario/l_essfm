import json

def update_notebook(filepath, updates):
    with open(filepath, 'r') as f:
        nb = json.load(f)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            for old, new in updates:
                cell['source'] = [line.replace(old, new) for line in cell['source']]
    with open(filepath, 'w') as f:
        json.dump(nb, f, indent=1)

# Notebook 02
updates_02 = [
    ("train_ldbp(St, 50, p, [taps]*make_bw(St,50).model_steps)", "train_ldbp(St, 50, p, [55]*make_bw(St,50).model_steps, iters=15000, P_dbm=7.0, lr=5e-4, target_taps=taps, n_blocks=200)"),
    ("LDBP_PRUNE = {'p11':11, 'p13':13, 'p15':15, 'prune17':17}", "LDBP_PRUNE = {'p35':35, 'p37':37, 'p41':41, 'prune45':45}")
]
update_notebook('02_170km_186GBd.ipynb', updates_02)
print("Updated 02")

# Notebook 03
updates_03 = [
    ("bw_l = make_bw(Sn2, 50)", "bw_l = make_bw(Sn2, 150)"),
    ("make_bw(St,50)", "make_bw(St,150)"),
    ("train_ldbp(St, 50, p, [taps]*make_bw(St,150).model_steps)", "train_ldbp(St, 150, p, [35]*make_bw(St,150).model_steps, iters=15000, P_dbm=7.0, lr=5e-4, target_taps=taps, n_blocks=200)"),
    ("LDBP_PRUNE = {'p11':11, 'p13':13, 'p15':15, 'prune17':17}", "LDBP_PRUNE = {'p21':21, 'p23':23, 'p25':25}"),
    ("complexity_ldbp(50,", "complexity_ldbp(150,")
]
update_notebook('03_15x80km_93GBd.ipynb', updates_03)
print("Updated 03")
