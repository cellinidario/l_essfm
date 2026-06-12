import sys
import numpy as np
sys.path.append('/home/dario/l_essfm/core')
import torch
from system import build_system, make_bw
from ldbp_torch import LdbpModel
from train import gen_dualpol, equalize_loss

d = {"number of polarizations": "2", "modulation": "64-QAM",
     "combine half-steps": "yes", "cd alpha": "1", "nl alpha": "1",
     'forward step size method': 'linear', 'forward split step method': 'symmetric'}
import configparser
cfg = configparser.ConfigParser(d)
cfg.read('/home/dario/l_essfm/config/scenario_170km_93GBd.ini')
cfg['system parameters']['digital oversampling'] = '2.0'
cfg['L-ESSFM parameters']['steps per span'] = '50'
cfg['L-ESSFM parameters']['split step method'] = 'asymmetric'
cfg['data generation']['data symbols per block'] = '1024'
with open('/home/dario/l_essfm/config/_tmp.ini', 'w') as f: cfg.write(f)

S = build_system('/home/dario/l_essfm/config/_tmp.ini')
S['nl_param'] = make_bw(S, 50).nl_param

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def forward_mock(S, P):
    from system import forward
    return forward(S, P)
    
ys, xs, P = gen_dualpol(forward_mock, S, 1, 9.0, 42)

model = LdbpModel(S, 50, device, [21]*51)
model.eval()
with torch.no_grad():
    loss = equalize_loss(model, ys.to(device), xs.to(device), P, S, ldbp=True)
    snr = -10.0 * np.log10(loss.item())
    print(f"Init SNR without pruning (21 taps): {snr:.3f} dB")

model.prune_to([17]*51)
with torch.no_grad():
    loss = equalize_loss(model, ys.to(device), xs.to(device), P, S, ldbp=True)
    snr = -10.0 * np.log10(loss.item())
    print(f"Init SNR with prune17 (17 taps): {snr:.3f} dB")

