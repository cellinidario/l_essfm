import json

with open('02_170km_186GBd.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Change initialization [21]*M to [41]*M
        source = source.replace("[21]*M", "[41]*M")
        
        cell['source'] = [source]

with open('02_170km_186GBd.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook 02 LDBP fixed with init 41!")
