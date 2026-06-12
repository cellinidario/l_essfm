import json

with open('02_170km_186GBd.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Change P_dbm and lr for train_ldbp
        source = source.replace("P_dbm=7.0, lr=5e-4", "P_dbm=12.5, lr=1e-4")
        
        cell['source'] = [source]

with open('02_170km_186GBd.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook 02 LDBP fixed with P_dbm=12.5 and lr=1e-4!")
