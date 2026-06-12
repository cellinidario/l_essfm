import json

with open('01_170km_93GBd.ipynb', 'r') as f:
    nb01 = json.load(f)

def clear(nb):
    for cell in nb['cells']:
        if 'outputs' in cell:
            cell['outputs'] = []
        if 'execution_count' in cell:
            cell['execution_count'] = None

# Create 02
nb02 = json.loads(json.dumps(nb01))
clear(nb02)
for cell in nb02['cells']:
    source = "".join(cell['source'])
    source = source.replace("170 km / 93 GBaud", "170 km / 186 GBaud")
    source = source.replace("SCENARIO = 'scenario_170km_93GBd'", "SCENARIO = 'scenario_170km_186GBd'")
    source = source.replace("L_KM, R_GBD = 170e3, 93e9", "L_KM, R_GBD = 170e3, 186e9")
    source = source.replace("peak(1.125, 'ideal', 100)", "peak(1.125, 'ideal', 400)")
    source = source.replace("peak(2.0,   'ideal', 100)", "peak(2.0,   'ideal', 400)")
    source = source.replace(
        "CFG  = {1.125: f'config/{SCENARIO}.ini', 2.0: 'config/_test_n2.ini'}",
        "N2CFG = f'config/{SCENARIO}_n2.ini'\nif not os.path.exists(N2CFG):\n    import re\n    _s = open(f'config/{SCENARIO}.ini').read()\n    _s = re.sub(r'(?m)^digital oversampling = .*','digital oversampling = 2', _s)\n    open(N2CFG,'w').write(_s)\nCFG  = {1.125: f'config/{SCENARIO}.ini', 2.0: N2CFG}"
    )
    if "LDBP_POINTS =" in source:
        source = source.replace(
            "LDBP_POINTS = [('prune17',[17],17), ('p15',[15],15), ('p13',[13],13),\n               ('p11',[11],11), ('v10',[11,9],10)]",
            "LDBP_POINTS = [('prune45',[45],45), ('p41',[41],41), ('p37',[37],37), ('p35',[35],35)]"
        )
    if "train_ldbp(Strain, 50, p, [21]*M" in source:
        source = source.replace("[21]*M", "[55]*M")
    cell['source'] = [source]

with open('02_170km_186GBd.ipynb', 'w') as f:
    json.dump(nb02, f, indent=1)

# Create 03
nb03 = json.loads(json.dumps(nb01))
clear(nb03)
for cell in nb03['cells']:
    source = "".join(cell['source'])
    source = source.replace("170 km / 93 GBaud (OFC/CLEO): reproduce Fig. 1b", "15 x 80 km / 93 GBaud (long-haul)")
    source = source.replace("SCENARIO = 'scenario_170km_93GBd'", "SCENARIO = 'scenario_15x80km_93GBd'")
    source = source.replace("L_KM, R_GBD = 170e3, 93e9", "L_KM, R_GBD = 1200e3, 93e9")
    source = source.replace("peak(1.125, 'ideal', 100)", "peak(1.125, 'ideal', 50)")
    source = source.replace("peak(2.0,   'ideal', 100)", "peak(2.0,   'ideal', 50)")
    source = source.replace(
        "CFG  = {1.125: f'config/{SCENARIO}.ini', 2.0: 'config/_test_n2.ini'}",
        "N2CFG = f'config/{SCENARIO}_n2.ini'\nif not os.path.exists(N2CFG):\n    import re\n    _s = open(f'config/{SCENARIO}.ini').read()\n    _s = re.sub(r'(?m)^digital oversampling = .*','digital oversampling = 2', _s)\n    open(N2CFG,'w').write(_s)\nCFG  = {1.125: f'config/{SCENARIO}.ini', 2.0: N2CFG}"
    )
    if "def _train_cfg" in source and "re.sub" in source:
        source = source.replace(
            "s = re.sub(r'(?m)^modulation = .*', 'modulation = Gaussian', s)",
            "s = re.sub(r'(?m)^modulation = .*', 'modulation = Gaussian', s)\n    s = re.sub(r'(?m)^forward steps per span = .*', 'forward steps per span = 100', s)"
        )
    if "def _ldbp_cfg(" in source:
        source = source.replace(
            "s = re.sub(r'(?m)^modulation = .*', 'modulation = Gaussian', s)",
            "s = re.sub(r'(?m)^modulation = .*', 'modulation = Gaussian', s)\n    s = re.sub(r'(?m)^forward steps per span = .*', 'forward steps per span = 100', s)"
        )
        
    source = source.replace("bw_eval = make_bw(S_eval, 50)", "bw_eval = make_bw(S_eval, 150)")
    source = source.replace("make_bw(Strain, 50)", "make_bw(Strain, 150)")
    source = source.replace("train_ldbp(Strain, 50", "train_ldbp(Strain, 150")
    source = source.replace("complexity_ldbp(50", "complexity_ldbp(150")
    source = source.replace("[21]*M", "[35]*M")
        
    if "LDBP_POINTS =" in source:
        source = source.replace(
            "LDBP_POINTS = [('prune17',[17],17), ('p15',[15],15), ('p13',[13],13),\n               ('p11',[11],11), ('v10',[11,9],10)]",
            "LDBP_POINTS = [('p25',[25],25), ('p23',[23],23), ('p21',[21],21)]"
        )
    cell['source'] = [source]

with open('03_15x80km_93GBd.ipynb', 'w') as f:
    json.dump(nb03, f, indent=1)

print("Notebooks 02 and 03 have been successfully cloned and adapted from 01.")
