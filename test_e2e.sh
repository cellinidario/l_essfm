#!/bin/bash
# End-to-end validation of the NEW l_essfm software (training + simulation from
# scratch) vs the exercise Fig.1b. One sample per method, split across BOTH
# oversamplings. Full settings: training Gaussian 15000 iter; simulation on the
# true 262144-symbol QAM forward.
#
#   n=1.125 : EDC,  ESSFM Ns=10,  L-ESSFM Ns=10
#   n=2     : Ideal DBP (800),  OSSFM Ns=10
cd /home/dario/l_essfm
set +e
ITER=15000
NS=10
FWD125=/home/dario/ldbp2/forward_resultsMAX.npy   # n=1.125 forward (Dario)
RES=results/e2e
mkdir -p $RES

# build a TRAINING config (Gaussian, short block) from a test config + overrides
make_train_cfg () {  # base_cfg out_cfg ns mode  (mode: lessfm|essfm)
    local base=$1 out=$2 ns=$3 mode=$4
    python3 - "$base" "$out" "$ns" "$mode" <<'PY'
import sys, re
base, out, ns, mode = sys.argv[1:5]
t = open(base).read()
t = t.replace('modulation = 64-QAM', 'modulation = Gaussian')
t = re.sub(r'data symbols per block = \d+', 'data symbols per block = 1024', t)
t = re.sub(r'(?m)^steps per span = .*', f'steps per span = {ns}', t)  # ^ avoids 'forward steps per span'
if mode == 'essfm':
    t = t.replace('tied Kerr parameters = no', 'tied Kerr parameters = yes')
    t = re.sub(r'optimize lengths = .*', 'optimize lengths = no', t)
    t = re.sub(r'nl filter length = \d+', 'nl filter length = 9', t)  # essfmNc(10)+1 @93GBd
    if 'optimize splitting ratio' not in t:
        t = t.replace('optimize lengths = no', 'optimize lengths = no\noptimize splitting ratio = yes')
open(out, 'w').write(t)
PY
}

train () {  # cfg logdir -> echoes params path
    local cfg=$1 log=$2
    python3 core/lessfm.py "[7]" 0.002 $ITER --config_path=$cfg --logdir=$log 2>/dev/null | grep -iE "iter $ITER" | tail -1 1>&2
    ls -t $log/P\[7\]*/*/parameters.csv | head -1
}

echo "===== E2E TEST (new l_essfm, Ns=$NS, both oversamplings) ====="

# ---------- n=1.125: train ESSFM and L-ESSFM ----------
CFG125=config/scenario_170km_93GBd.ini
echo "--- [n=1.125] training ESSFM (tied+rho) ---"
make_train_cfg $CFG125 $RES/essfm125.ini $NS essfm
PE=$(train $RES/essfm125.ini $RES/essfm125_log); cp "$PE" $RES/essfm125_params.csv
echo "--- [n=1.125] training L-ESSFM (lengths+NLPR) ---"
make_train_cfg $CFG125 $RES/lessfm125.ini $NS lessfm
PL=$(train $RES/lessfm125.ini $RES/lessfm125_log); cp "$PL" $RES/lessfm125_params.csv

# ---------- simulate on the true forward ----------
echo "--- simulation on true forward ---"
python3 - <<PY 2>/dev/null
import sys; sys.path.insert(0,'core'); import numpy as np
from system import build_system
from backprop import curve
fwd=np.load('$FWD125',allow_pickle=True).item()
data={round(10*np.log10(Pw/1e-3),4):[(np.array(y),np.array(x)) for (y,x) in r] for Pw,r in fwd.items()}
Pg=np.array(sorted(data))
# n=1.125 system
S1=build_system('$CFG125')
print('[n=1.125] EDC        :', round(curve(S1,data,Pg,'edc',1,edc=True).max(),3), ' (Fig.1b 18.018)')
print('[n=1.125] ESSFM:10   :', round(curve(S1,data,Pg,'lessfm',10,'$RES/essfm125_params.csv').max(),3), ' (Fig.1b 18.701)')
print('[n=1.125] L-ESSFM:10 :', round(curve(S1,data,Pg,'lessfm',10,'$RES/lessfm125_params.csv').max(),3), ' (Fig.1b 18.878)')
# n=2 system: same forward (analog OS=7), digital os=2 read from config
S2=build_system('config/_test_n2.ini')
print('[n=2]     Ideal:800  :', round(curve(S2,data,Pg,'ideal',800).max(),3), ' (Fig.1b 19.14)')
print('[n=2]     OSSFM:10   :', round(curve(S2,data,Pg,'ossfm',10).max(),3), ' (Fig.1b ~18.70)')
PY
echo "===== E2E_DONE ====="
