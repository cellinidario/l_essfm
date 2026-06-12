import json
import re

with open('02_170km_186GBd.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Force 41 taps
        source = re.sub(r'\[\d+\]\*M', '[41]*M', source)
        # Force extremely conservative learning rate
        source = re.sub(r'lr=1e-4', 'lr=1e-5', source)
        source = re.sub(r'lr=5e-4', 'lr=1e-5', source)
        
        cell['source'] = [source]

with open('02_170km_186GBd.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook 02 LDBP fixed with init 41 and lr=1e-5!")
