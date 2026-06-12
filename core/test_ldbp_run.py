import os
import sys
import torch
import numpy as np
import configparser

sys.path.append('/home/dario/l_essfm/core')
from system import build_system, make_bw
from train import train_ldbp
from ldbp_eval import load_ldbp_params, ldbp_backprop

RESDIR = '/home/dario/l_essfm/results/scenario_170km_93GBd'
os.makedirs(RESDIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CFG = {2.0: '/home/dario/l_essfm/config/scenario_170km_93GBd.ini'}
def _ldbp_cfg(logarithmic=True):
    import re
    s = open(CFG[2.0]).read()
    s = s.replace('data symbols per block = 262144', 'data symbols per block = 1024')
    s = re.sub(r'(?m)^modulation = .*', 'modulation = Gaussian', s)
    s = re.sub(r'(?m)^digital oversampling = .*', 'digital oversampling = 2.0', s)
    if logarithmic:
        s = re.sub(r'(?m)^step size method = .*', 'step size method = logarithmic', s)
    out = '/home/dario/l_essfm/config/_ldbp_train.ini'
    open(out, 'w').write(s)
    return out

def _ldbp_cfg_eval(logarithmic=True):
    import re
    s = open(CFG[2.0]).read()
    s = re.sub(r'(?m)^modulation = .*', 'modulation = Gaussian', s)
    s = re.sub(r'(?m)^digital oversampling = .*', 'digital oversampling = 2.0', s)
    if logarithmic:
        s = re.sub(r'(?m)^step size method = .*', 'step size method = logarithmic', s)
    out = '/home/dario/l_essfm/config/_ldbp_eval.ini'
    open(out, 'w').write(s)
    return out

Strain = build_system(_ldbp_cfg())
M = make_bw(Strain, 50).model_steps
target = [17]

print(f'training LDBP prune17 (init 21 -> [17], log step) ...', flush=True)

# Build pruning events correctly as per patch_notebook_3.py logic
iters = 15000
import random
random.seed(42)
cd_filter_delay = [(21-1)//2] * M
target_delay = [(17-1)//2] * M
prune_order = []
max_len = max(cd_filter_delay)
min_len = min(target_delay)
for i in range(max_len-min_len+1):
    for NN in range(M):
        if cd_filter_delay[NN] >= max_len-i and target_delay[NN] < max_len-i:
            prune_order.append(NN)
random.shuffle(prune_order)
pruning_steps = len(prune_order)
pruning_schedule = np.ceil(np.ceil(2.0**(-np.arange(pruning_steps, 0, -1))*iters))
pruning_schedule = np.ceil(pruning_schedule + np.arange(pruning_steps)*iters/8/pruning_steps)
prune_events = []
for i, p_it in enumerate(pruning_schedule):
    prune_events.append((int(p_it), prune_order[i]))

train_ldbp(Strain, 50, f'{RESDIR}/ldbp_prune17/parameters.csv', [21]*M, iters=iters, P_dbm=7.0, lr=5e-4,
           target_taps=target, n_blocks=200, prune_events=prune_events)
