import json

with open('02_170km_186GBd.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        source = source.replace("P_dbm=12.5", "P_dbm=12.0")
        
        cell['source'] = [source]

with open('02_170km_186GBd.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook 02 LDBP fixed with P_dbm=12.0!")
