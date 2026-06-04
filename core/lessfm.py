#========================================================#
# Learned-Enhanced Split-Step Fourier Method (L-ESSFM) for Digital Backpropagation
# Dario Cellini (Dario.Cellini@santannapisa.it)
#========================================================#
#
# ONE parametric training script that replaces the old l-essfm{,2,3,4}.py copies
# (which differed ONLY in the GVD-length initialization). The behaviour is set
# entirely from the [L-ESSFM parameters] section of the .ini:
#
#   optimize lengths = yes|no
#       yes -> L-ESSFM: every per-step GVD length is a free trainable Variable,
#              initialized from the backward power profile (exponential init),
#              with the total dispersion constrained (last step derived).
#       no  -> uniform step lengths L/Ns (ESSFM/SSFM-like).
#   tied Kerr parameters = yes|no
#       yes -> ESSFM: a SINGLE NLPR (Kerr) filter shared across all steps.
#       no  -> L-ESSFM: an independent NLPR filter per step.
#   optimize splitting ratio = yes|fixed|no   (ESSFM only, with tied Kerr)
#       yes   -> one shared splitting ratio rho is a trained Variable.
#       fixed -> rho held at 'rho value' (used for grid-search / reproduction).
#       no    -> no rho (symmetric uniform split).
#   nl filter length = <Nc+1>   one-sided NLPR taps (mirrored to 2N-1 at apply).
#
# So:  L-ESSFM = (optimize lengths=yes, tied Kerr=no)
#      ESSFM   = (optimize lengths=no,  tied Kerr=yes, optimize splitting ratio=yes)
#      OSSFM   = (optimize lengths=no,  tied Kerr=yes, nl filter length=1)
#
# Key fix (vs the legacy code): for the tied-Kerr ESSFM the shared NLPR filter is
# scaled per step by nl_param[NN]/mean(nl_param) (signed mean), otherwise one
# filter cannot serve steps whose nonlinear strength differs by up to ~1700x.
# The matched filter is built in frequency (see core/rrc.py / system.py).
#
#========================================================#
# imports and constants {{{
#========================================================#
import tensorflow as tf
import sys # sys.exit()
import warnings
import os # os.path.exists(), os.environ['v'], os.makedirs()
import numpy as np
import scipy as sp # sp.fft(), sp.linalg.solve(), sp.optimize.fsolve(), sp.special.erf()
from scipy import signal
import time # time.gmtime(), time.strftime()
import math # math.isnan()
import random # random.shuffle()
import threading # threading.Thread()
import multiprocessing
import argparse as ap
import configparser
import shutil # shutil.copyfile(src, dst)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # avoid TF logging output for GPU devices etc. 
sp.set_printoptions(precision = 4, suppress = True)
np.set_printoptions(precision = 4)

# constants
co_h = 6.6260657e-34
co_c0 = 299792458
co_lambda = 1550.0e-9
co_dB = 10.0*np.log10(np.exp(1.0))
nu = co_c0/co_lambda
dB_conv = 4.342944819032518

# }}}
#========================================================#
# functions {{{
#========================================================#
def rrcosine(rolloff, delay, OS):
	""" Root-raised cosine filter for pulse shaping
	Args:
		rolloff: between 0 and 1
		delay: in symbols 
		OS: oversampling factor (samples per symbol)
	
	Returns:
		A vector of length 2*(OS*delay)+1
	"""
	rrcos = np.zeros(2*round(delay*OS)+1)
	rrcos[round(delay*OS)] = 1 + rolloff*(4/np.pi-1)
	for i in range(1,round(delay*OS)+1):
		t = i/OS
		if(t == 1/4/rolloff):
			val = rolloff/np.sqrt(2)*((1+2/np.pi)*np.sin(np.pi/(4*rolloff)) + (1-2/np.pi)*np.cos(np.pi/(4*rolloff)))
		else:
			val = (np.sin(np.pi*t*(1-rolloff)) + 4*rolloff*t*np.cos(np.pi*t*(1+rolloff))) / (np.pi*t*(1-(4*rolloff*t)**2))
		rrcos[round(delay*OS)+i] = val
		rrcos[round(delay*OS)-i] = val
	return rrcos / np.sqrt(np.sum(rrcos**2))

def get_fvec(N,fs):
	return np.concatenate((np.linspace(0,N//2-1,N//2), np.linspace(-N//2,-1,N//2))) * fs/N

def get_optimizer():
	if optimizer == "adam":
		opt = tf.train.AdamOptimizer(learning_rate, float(conf_train['adam_A']), float(conf_train['adam_B']))
	elif optimizer == "rmsprop":
		opt = tf.train.RMSPropOptimizer(learning_rate, float(conf_train['rmsprop_A']), float(conf_train['rmsprop_B']))
	elif optimizer == "adadelta":
		opt = tf.train.AdadeltaOptimizer(learning_rate, float(conf_train['adadelta_A']))
	elif optimizer == "adagrad":
		opt = tf.train.AdagradOptimizer(learning_rate, float(conf_train['adagrad_A']))
	else:
		raise ValueError("wrong optimizer string: optimizer = '"+optimizer+"'")
	return opt

def complex_multiply(x,y):
	"""
	Args:
		x: Tensor of shape = [batch_size, N, 2]
		y: Tensor of shape = [batch_size, N, 2]
	
	Returns:
		A Tensor of shape = [batch_size, N, 2]
	"""
	xr = x[:,:,0]
	xi = x[:,:,1]
	yr = y[:,:,0]
	yi = y[:,:,1]
	return tf.stack([xr*yr-xi*yi, xr*yi+xi*yr], axis=2)

def tf_real_filter(init_coeffs, opt=False):
	""" Create arbitrary FIR filter with real coefficients
	
	Args:
		init_coeffs: real numpy array of shape [filter_length,] with initial filter coefficients
		opt: (Optional) True for variable, False for constant, default=False
	
	Returns: 
		A Tensor with shape = [filter_length]
	"""
	
	if opt == True:
		h = tf.Variable(init_coeffs, dtype=tf.float32)
	else:
		h = tf.constant(init_coeffs, tf.float32)
	
	return h

def periodically_extend(x, M):
	""" Extends a numpy vector of length N to length M>N by periodically copying the elements """
	N = x.shape[0]
	y = np.zeros(M, dtype=x.dtype)
	for i in range(M):
		y[i] = x[i%N]
	return y

def line2array(line):
	""" Converts a string of comma-separated numbers to numpy array """
	return np.array([float(v) for v in line.strip().split(",")])

def effective_length(length, alpha_lin):
	if alpha_lin == 0:
		return length
	else:
		return (1-np.exp(-alpha_lin*length))/alpha_lin

class ssfm_parameters:
	""" Handles parameters related to the split-step Fourier method (SSFM)
	
	Initialization is performed with a dictionary that should have the following keys:
		step_size_method
			logarithmic
			linear 
			step_size 
			predefined 
		StPS: steps per span (only for logarithmic and linear)
		adjusting_factor: recommended is 0.4 (only for logarithmic)
		ssfm_method
			symmetric: linear->nonlinear->linear 
			asymmetric: linear->nonlinear
		combine_half_steps: wether to combine half-steps of adjacent spans (only for symmetric)
		alpha: attenuation parameter; should be 0 for less steps than spans
		beta2: dispersion parameter
		gamma: nonlinear parameter
		Nsp: number of spans
		Lsp: span length [m]
		fsamp: sampling frequency
		Nsamp: length of the assumed FFT
		direction: +1 for forward, -1 for backpropagation
	
	Computed attributes:
		model_steps
		cd_length
		nl_param
		nl_length (not used)
	 
	Usage example: 
	
	bw = ssfm_parameters(parameter_dict)
	for NN in range(bw.model_steps):
		u = sp.ifft(bw.get_cd_filter_freq(NN)*sp.fft(u))
		u = u*np.exp(1J*bw.nl_param[NN]*np.abs(u)**2)
	"""
	
	def __init__(self, opts):
		self.__dict__.update(opts) # converts all dictionary entries to attributes 
		
		alpha_lin = self.alpha/(10*np.log10(np.exp(1)))
		Nsp = self.Nsp
		Lsp = self.Lsp
		direction = self.direction
		
		if direction == +1 and self.Nsp > 1:
			raise ValueError("forward propagation valid only for 1 span")
		
		if self.step_size_method == 'logarithmic':
			if 'adjusting_factor' not in opts:
				self.adjusting_factor = 0.4 # 0: linear, 1: very logarithmic
		
		if 'combine_half_steps' not in opts:
			self.combine_half_steps = True
		
		if self.step_size_method == 'step_size': # used only for subband processing
			step_size = self.step_size
			Ltot = Lsp*Nsp
			model_steps = int(np.floor(Ltot/step_size)+1)
			last_step_size = Ltot - (model_steps-1)*step_size
			
			cd_length = step_size*np.ones(model_steps)
			cd_length[model_steps-1] = last_step_size
			
			tmp = np.mod(np.cumsum(cd_length), Lsp)
			len_before = np.zeros(model_steps)
			len_after = np.zeros(model_steps)
			amplifier_location = np.zeros(model_steps)
			for NN in range(1, model_steps):
				if(tmp[NN-1] > tmp[NN]):
					amplifier_location[NN] = 1
					len_after[NN] = tmp[NN]
					len_before[NN] = cd_length[NN] - len_after[NN]
			amplifier_location[0] = 1
			amplifier_location[-1] = 0
			
			nl_length = np.zeros(model_steps)
			eff_len_before = np.zeros(model_steps)
			for NN in range(model_steps):
				if (amplifier_location[NN] == 1) and (NN != 0):
					h = len_after[NN]
					eff_len_before[NN] = effective_length(len_before[NN],np.abs(alpha_lin))
				else:
					h = cd_length[NN]
				nl_length[NN] = effective_length(h,np.abs(alpha_lin))
		else:
			StPS = self.StPS
			# ====================================================== #
			# compute step sizes for one span
			# ====================================================== #
			if self.step_size_method == 'logarithmic':
				alpha_adj = self.adjusting_factor*alpha_lin
				delta = (1-np.exp(-alpha_adj*Lsp))/StPS
				if(direction == -1):
					nn = np.arange(StPS)+1    # 1,2,...,StPS
				else:
					nn = StPS-np.arange(StPS) # StPS,...,2,1
				step_size = -1/(alpha_adj) * np.log((1-(StPS-nn+1)*delta)/(1-(StPS-nn)*delta))
			elif self.step_size_method == "linear":
				step_size = Lsp/StPS*np.ones(StPS)
			else:
				raise ValueError("wrong step_size_method given (should be 'linear' or 'logarithmic'): "+self.step_size_method)
			# ====================================================== #
			# compute cd_length, nl_length, amplifier_location
			# ====================================================== #
			if self.ssfm_method == "symmetric":
				if self.combine_half_steps == True:
					model_steps = Nsp*StPS+1
					cd_length = np.zeros(model_steps)
					nl_length = np.zeros(model_steps)
					for NN in range(Nsp):
						for MM in range(StPS):
							cd_length[NN*StPS+MM] = step_size[MM]/2 + step_size[(MM+StPS-1)%StPS]/2
							nl_length[NN*StPS+MM] = step_size[MM]
					cd_length[0] = step_size[0]/2
					cd_length[model_steps-1] = step_size[StPS-1]/2
					
					amplifier_location = np.zeros(model_steps)
					amplifier_location[:-1:StPS] = 1
				else:
					model_steps = Nsp*(StPS+1)
					cd_length = np.concatenate([[step_size[0]/2], (step_size[0:-1]+step_size[1:])/2, [step_size[-1]/2]])
					cd_length = np.tile(cd_length, Nsp)
					nl_length = np.concatenate([step_size, [0]])
					nl_length = np.tile(nl_length, Nsp)
					
					amplifier_location = np.zeros(model_steps)
					amplifier_location[::StPS+1] = 1
			elif self.ssfm_method == "asymmetric":
				model_steps = Nsp*StPS
				cd_length = np.zeros(model_steps)
				nl_length = np.zeros(model_steps)
				for NN in range(Nsp):
					for MM in range(StPS):
						cd_length[NN*StPS+MM] = step_size[MM]
						nl_length[NN*StPS+MM] = effective_length(step_size[MM], np.abs(alpha_lin))
				
				amplifier_location = np.zeros(model_steps)
				amplifier_location[::StPS] = 1
			else:
				raise ValueError("wrong split step method given (should be 'symmetric' or 'asymmetric'): "+self.ssfm_method)
		# ====================================================== #
		# compute attenuation and nl_param
		# ====================================================== #
		nl_param = direction*self.gamma*nl_length
		
		attenuation = np.exp(-direction*alpha_lin*cd_length/2)
		#cd_length[0] = step_size[0]
		#cd_length[model_steps-1] = step_size[StPS-1]
		for NN in range(model_steps):
			if direction == -1 and amplifier_location[NN] == 1:
				attenuation[NN] = attenuation[NN] * np.exp(direction*alpha_lin*Lsp/2)
		
		# re-normalize nl_param
		for NN in range(model_steps):
			nl_param[NN] = nl_param[NN]*np.prod(attenuation[0:NN+1:])**2
		
		if self.step_size_method == "step_size":
			for NN in range(model_steps):
				if amplifier_location[NN] == 1:
					nl_param[NN] = nl_param[NN] + direction*self.gamma*eff_len_before[NN]
		
		self.model_steps = model_steps
		self.cd_length = cd_length
		self.nl_length = nl_length
		self.nl_param = nl_param
		
		N = self.Nsamp
		self.fvec = np.concatenate((np.linspace(0,N//2-1,N//2), np.linspace(-N//2,-1,N//2))) * self.fsamp/N
	
	def get_cd_filter_freq(self, NN):
		return (self.beta2/2)*(2*np.pi*self.fvec)**2*(self.direction*self.cd_length[NN])

def ordered_direct_product(A,B):
	p = A.shape[0]
	q = B.shape[0]
	n = A.shape[1]
	m = B.shape[1]
	
	C = np.zeros([p*q,n+m])
	for i in range(q):
		C[i*q:(i+1)*q:, :n:] = A[i,:]
	for i in range(p):
		C[i*q:(i+1)*q:, n::] = B
	return C

def QAM(M):
    Msqrt = (np.sqrt(M)).astype(np.int)
    if Msqrt**2 != M:
        raise ValueError("M has to be of the form M=4^m where m>0")
    x_pam = np.expand_dims(-(Msqrt-2*np.arange(start=1, stop=Msqrt+1)+1), axis=1)
    x_qam = ordered_direct_product(x_pam, x_pam)
    const = x_qam[:,0] + 1j * x_qam[:,1]
    return const/np.sqrt(np.mean(np.abs(const)**2))

# }}}
#========================================================#
# parse function arguments {{{
#========================================================#
parser = ap.ArgumentParser("python3 l-essfm2.py")
parser.description = "Learned-Enhanced Split-Step Fourier Method (L-ESSFM)"
parser.add_argument("P", help="set of training powers in dB, e.g., [5] or [5,6,7]")
parser.add_argument("Lr", help="learning rate, e.g., 0.01")
parser.add_argument("iter", help="gradient descent iterations, e.g., 1000")
parser.add_argument("-c", "--config_path", help="path to configuration file (default is l-essfm_config.ini)", default="l-essfm_config.ini")
parser.add_argument("-l", "--logdir", help="directory for log files (default is log)", default="log")
parser.add_argument("-t", "--timing", help="time the forward propagation", action="store_true")

args = parser.parse_args()
args_dict = vars(args) # converts to a dictionary

opt_list="P,Lr,iter".split(",")
arg_str = ""
for i in range(len(opt_list)):
	arg_str += opt_list[i]
	arg_str += args_dict[opt_list[i]]
	if(i != len(opt_list)-1):
		arg_str += "_"

config_path = args.config_path
P_dB_r = np.asarray(eval(args.P))
P_W_r = pow(10, P_dB_r/10)*1e-3
iterations = int(args.iter)
learning_rate = float(args.Lr)

# }}}
#========================================================#
# read config file {{{
#========================================================#
defaults = {
	# system
	"sigma scaling"             : "1",
	"modulation"                : "16-QAM",
	# LDBP
	"combine half-steps"        : "yes",
	"optimize lengths"       	: "no",
	"complex Kerr parameters"   : "no",
	"tied Kerr parameters"      : "no",
	"less steps than spans"     : "no",
	"cd alpha"                  : "1",
	"nl alpha"                  : "1",
	# training
	"adam_A"                    : "0.9", # decay for running average of the gradient
	"adam_B"                    : "0.999", # decay for running average of the square of the gradient
	"rmsprop_A"                 : "0.9",
	"rmsprop_B"                 : "0.1",
	"adadelta_A"                : "0.1",
	"adagrad_A"                 : "0.1",
	# data
	'forward step size method'  : 'logarithmic',
	'forward split step method' : 'symmetric'
}

config = configparser.ConfigParser(defaults)

config_folder, config_file = os.path.split(config_path)
print("configuration file name: '"+config_file+"'")

if not os.path.exists(config_path):
	raise RuntimeError("config file in '"+config_file+"' does not exist")
config.read(config_path)

# system parameters
conf_sys      = config['system parameters']
Lsp           = conf_sys.getfloat('span length [km]')*1.0e3
alpha         = conf_sys.getfloat('alpha [dB/km]')*1.0e-3
gamma         = conf_sys.getfloat('gamma [1/W/km]')*1.0e-3
noise_figure  = conf_sys.getfloat('amplifier noise figure [dB]')
sigma_scaling = conf_sys.getfloat('sigma scaling')
Nsp           = conf_sys.getint('number of spans')
fsym          = conf_sys.getfloat('symbol rate [Gbaud]')*1.0e9
modulation    = conf_sys['modulation']
rolloff       = conf_sys.getfloat('RRC roll-off')
delay         = conf_sys.getint('RRC delay')
lp_bandwidth  = conf_sys.getfloat('low-pass filter bandwidth [GHz]')*1.0e9
Nsym          = conf_sys.getint('data symbols per block')
OS_a          = conf_sys.getint('analog oversampling')
OS_d          = conf_sys.getfloat('digital oversampling')
Nch           = conf_sys.getint('number of channels')
spacing  	  = conf_sys.getfloat('channel spacing [GHz]')*1.0e9
if config.has_option('system parameters', 'D [ps/nm/km]'):
	D     = conf_sys.getfloat('D [ps/nm/km]')*1.0e-6
	beta2 = -D*co_lambda**2/(2*np.pi*co_c0)
else:
	beta2 = conf_sys.getfloat('beta2 [ps^2/km]')*1.0e-27

# L-ESSFM parameters
conf_essfm            = config['L-ESSFM parameters']
step_size_method_bw   = conf_essfm['step size method']
ssfm_method_bw        = conf_essfm['split step method']
combine_half_steps    = conf_essfm.getboolean('combine half-steps')
cd_opt                = conf_essfm.getboolean('optimize lengths')
# ESSFM mode: uniform step lengths + ONE shared splitting ratio rho (NLPR position
# within each step). 'yes' -> rho is a trained tf.Variable; 'fixed' -> rho held at
# 'rho value' from config (used for grid-search); absent/'no' -> disabled.
_sr = conf_essfm.get('optimize splitting ratio', fallback='no').strip().lower()
opt_rho               = _sr in ('yes', 'true', 'fixed')
rho_fixed             = _sr == 'fixed'
rho_value             = conf_essfm.getfloat('rho value', fallback=0.5) if rho_fixed else 0.5
cd_alpha              = conf_essfm.getfloat('cd alpha')
tied_Kerr         	  = conf_essfm.getboolean('tied Kerr parameters')
nl_alpha              = conf_essfm.getfloat('nl alpha')
nl_filter_length      = conf_essfm.getint('nl filter length')
less_steps_than_spans = conf_essfm.getboolean('less steps than spans')

# training parameters
conf_train       = config['training']
minibatch_size   = conf_train.getint('minibatch size')
optimizer        = conf_train['optimizer']
summary_interval = conf_train.getint('summary writing interval')
SAVE_FILE        = conf_train.getboolean('save results to file')

# data generation
conf_data           = config['data generation']
StPS_fw             = conf_data.getint('forward steps per span')
step_size_method_fw = conf_data['forward step size method']
ssfm_method_fw      = conf_data['forward split step method']
QMAX                = conf_data.getint('number of queue elements') # number of queue elements
QBSIZE              = conf_data.getint('generation batch size') # batch size to pupulate the queue
NPROC               = conf_data.getint('number of parallel processors') # number of processors used to populate the queue
REPF                = conf_data.getint('data replication factor') # replication factor for data

if np.abs(OS_d*16-round((OS_d*16))) > 0:
	raise ValueError('For simplicity, choose OS_d s.t. OS_d*16 is integer')
    # e.g., 1, 1.0625, 1.125, 1.1875, 1.25, ..., 2

# derived parameters
L = Lsp*Nsp
Gain = 10.0**(alpha*Lsp/10.0)
sef = 10.0**(noise_figure/10.0)/2.0#/(1.0-1.0/Gain)
alpha_lin = alpha / dB_conv
N0 = sigma_scaling*Nsp*(np.exp(alpha_lin*Lsp)-1.0)*co_h*nu*sef
sigma2 = N0 * fsym * OS_a
Nsamp_a = Nsym*OS_a
Nsamp_d = round(Nsym*OS_d)
fsamp_a = fsym*OS_a
fsamp_d = fsym*OS_d
f_a = get_fvec(Nsamp_a, fsamp_a)
f_d = get_fvec(Nsamp_d, fsamp_d)

if "QAM" in modulation:
    splitstr = modulation.split("-")
    modulation_order = int(splitstr[0])
    modulation = "QAM"

print("total memory of the data queue: {} MB".format(QMAX*64*(Nsamp_d+Nsym)/8/1e6))

# }}}
#========================================================#
# forward propagation generative model {{{
#========================================================#
ps_filter_tx_coeffs = rrcosine(rolloff, delay, OS_a) # pulse shaping filter
ps_filter_tx_length = 2*(OS_a*delay)+1
ps_filter_tx_delay = OS_a*delay # delay in samples

# pre-compute frequency responses
ps_tmp = np.concatenate((ps_filter_tx_coeffs, np.zeros(Nsamp_a-ps_filter_tx_length)))
ps_tmp = np.roll(ps_tmp, -ps_filter_tx_delay)
ps_filter_tx_freq = sp.fft(ps_tmp, n=Nsamp_a)
lp_filter_freq = (abs(f_a) <= lp_bandwidth/2).astype(float)

if modulation == "QAM":
    const = QAM(modulation_order)

ssfm_opts = {
    "alpha": alpha,
    "beta2": beta2,
    "gamma": gamma,
    "Nsp": 1,
    "Lsp": Lsp,
    "fsamp": fsamp_a,
    "Nsamp": Nsamp_a,
    "step_size_method": step_size_method_fw,
    "ssfm_method": ssfm_method_fw,
    "StPS": StPS_fw,
    "direction": 1
}

fw = ssfm_parameters(ssfm_opts)

def forward_propagation():
    """
    Returns:
        y: received signal (shape = [Nsamp_d, 2], separate real and imaginary part)
        x: symbol vector (shape = [Nsym], complex)
        P: launch power (in W)
    """
    np.random.seed() # new seed is necessary for multiprocessor 
    P = P_W_r[np.random.randint(P_W_r.shape[0])] # get random launch power
    # [SOURCE] random points from the signal constellation
    if modulation == "QAM":
        x = const[np.random.randint(const.shape[0], size=[Nch, Nsym])]
    elif modulation == "Gaussian":
        x = (np.random.normal(0,1,size=[Nch, Nsym]) + 1j*np.random.normal(0,1,size=[Nch, Nsym]))/np.sqrt(2)
    else:
        raise ValueError("wrong modulation format: " + modulation)
    # [MODULATION] upsample + pulse shaping
    x_up = np.zeros([Nch, Nsamp_a], dtype=np.complex64)
    x_up[:, ::OS_a] = x*np.sqrt(OS_a)
    u = sp.ifft(sp.fft(x_up)*ps_filter_tx_freq)*np.sqrt(P)
	
    u_wdm = np.zeros([1, Nsamp_a], dtype=np.complex64)
    for NN in range(Nch):
        freq_shift = (NN - Nch//2)*spacing
        phase_shift = np.exp(1j*2*np.pi*freq_shift*np.arange(Nsamp_a)/fsamp_a)
        u_wdm += u[NN, :]*phase_shift

	# [CHANNEL] simulate forward propagation
    for NN in range(Nsp): # enter a span
		# add noise, NOTE: amplifier gain (u = u*np.exp(alpha_lin*Lsp/2.0)) is absorbed in nl_param
        u_wdm += np.sqrt(sigma2/2/Nsp)*(np.random.randn(1,Nsamp_a) + 1j*np.random.randn(1,Nsamp_a))
        for MM in range(fw.model_steps): # enter a segment
            u_wdm = sp.ifft(sp.fft(u_wdm)*np.exp(1j*fw.get_cd_filter_freq(MM)))
            u_wdm *= np.exp(1j*fw.nl_param[MM]*np.abs(u_wdm)**2)
    # [RECEIVER] low-pass filter + downsample
    Y = sp.fft(u_wdm)*lp_filter_freq
    Y = Y[0, :Nsamp_d] + Y[0, -Nsamp_d:]
    y = sp.ifft(Y)/OS_a*OS_d
    y = np.stack([np.real(y), np.imag(y)], axis=1)
    return y, x[Nch//2, :], P

if args.timing == True:
	print("")
	print("timing the forward propation ...")
	t = time.time()
	_,_,_ = forward_propagation()
	elapsed = time.time()-t
	print("{0:.2f} seconds to generate 1 input/output data pair".format(elapsed))
	print("Generating approx. {0:.0f} input/output data pairs per seconds".format(NPROC*REPF/elapsed))
	sys.exit("")

# }}}
#========================================================#
# compute step sizes for DBP {{{
#========================================================#
ssfm_opts = {}
ssfm_opts['beta2'] = beta2
ssfm_opts['gamma'] = gamma
ssfm_opts['fsamp'] = fsamp_d
ssfm_opts['Nsamp'] = Nsamp_d
ssfm_opts['step_size_method'] = step_size_method_bw
ssfm_opts['ssfm_method'] = ssfm_method_bw
ssfm_opts['combine_half_steps'] = combine_half_steps
ssfm_opts['direction'] = -1

if less_steps_than_spans == False:
	ssfm_opts['alpha'] = alpha
	ssfm_opts['Nsp'] = Nsp
	ssfm_opts['Lsp'] = Lsp
	ssfm_opts['StPS'] = int(conf_essfm['steps per span'])
else:
	ssfm_opts['alpha'] = alpha
	ssfm_opts['Nsp'] = int(conf_essfm['total steps'])
	ssfm_opts['Lsp'] = Lsp*Nsp/int(conf_essfm['total steps'])
	ssfm_opts['StPS'] = 1

bw = ssfm_parameters(ssfm_opts)

# }}}
#========================================================#
# define tunable parameters {{{
#========================================================#
no_filter = np.zeros(nl_filter_length, dtype=np.float32)
no_filter[0] = nl_alpha

step_size = ssfm_opts['Lsp']/ssfm_opts['StPS']
cd_rho = [1.0]
cd_sum = 0.0

if tied_Kerr == True:
	nl_filter_all = tf.Variable(no_filter, dtype=tf.float32)

length = {}
nl_filter = {}

# ====== EXPONENTIAL length init (Strada B, physics-motivated) ======
# The length[NN] multipliers start from the backward power profile (equal-NL per step),
# so training starts near the physical optimum -> robust convergence.
# length[NN] is a multiplier of the nominal cd_length; sum of multipliers = number of steps.
def _exp_length_multipliers():
	Ns_eq = ssfm_opts['StPS']*ssfm_opts['Nsp']          # numero di step (NL)
	Lspan = ssfm_opts['Lsp']; alpha_l = alpha/dB_conv    # 1/m
	# equal-NL boundaries forward over ONE span, then backward
	def span_steps(Nstep):
		z=[0.0]
		for i in range(1,Nstep):
			frac=i/Nstep
			zi=-np.log(1-frac*(1-np.exp(-alpha_l*Lspan)))/alpha_l
			z.append(zi)
		z.append(Lspan)
		return np.diff(z)[::-1]   # backward: long->short
	# for multi-span, repeat the profile per span
	steps_phys = np.concatenate([span_steps(ssfm_opts['StPS']) for _ in range(ssfm_opts['Nsp'])])
	# physical cd per model_step (symmetric+combine): borders = half step
	M = bw.model_steps
	cd_phys = np.zeros(M)
	cd_phys[0] = steps_phys[0]/2
	for i in range(1, M-1):
		cd_phys[i] = (steps_phys[i-1]+steps_phys[i])/2
	cd_phys[M-1] = steps_phys[-1]/2
	# multiplier = physical cd / nominal cd
	mult = cd_phys / bw.cd_length
	return mult.astype(np.float32)

if opt_rho == True:
	# ESSFM: uniform steps + ONE shared optimized splitting ratio rho in (0,1).
	# length[NN] is a MULTIPLIER of bw.cd_length (which already encodes the nominal
	# symmetric split, incl. half-step borders). rho=0.5 -> multiplier 1.0 everywhere
	# (= the symmetric uniform model). rho shifts each NLPR: the CD segment just
	# before a NLPR scales by 2*rho, the one just after by 2*(1-rho); tied across
	# all steps. With combine_half_steps the interior segments combine before+after
	# of adjacent NLPRs -> net multiplier 1.0; only the two borders feel rho.
	# rho fixed (grid-search) -> constant; else a trained Variable.
	# CONVENTION (MATLAB/practice, Dario: rho_single-step ~0.9): first GVD border =
	# rho*L, last = (1-rho)*L, so the NLPR sits near the END of each step (where the
	# dispersion of the long first portion has already accumulated). The paper writes
	# it mirrored as (1-rho)L/rho L — Stella likely swapped rho<->(1-rho) paper-vs-code;
	# the physical config is identical. We follow the MATLAB/practice convention.
	if rho_fixed:
		rho_var = tf.constant([rho_value], dtype=tf.float32)
	else:
		rho_var = tf.Variable([0.5], dtype=tf.float32)
	M = bw.model_steps
	for NN in range(M):
		if NN == 0:
			length[NN] = 2.0*rho_var           # first half-step: rho*step
		elif NN == M-1:
			length[NN] = 2.0*(1.0 - rho_var)   # last half-step: (1-rho)*step
		else:
			length[NN] = tf.constant([1.0], dtype=tf.float32)
elif cd_opt == True:
	exp_mult = _exp_length_multipliers()
	# Exponential init + TOTAL-dispersion CONSTRAINT: the first model_steps-1 steps are
	# free variables initialized with the exponential profile; the LAST step is derived
	# to preserve the total sum (as in the original design, which guarantees sum=L).
	# Target multiplier sum (with halved borders) = sum(exp_mult with borders/2).
	target_sum = exp_mult[0]/2 + float(np.sum(exp_mult[1:-1])) + exp_mult[-1]/2
	for NN in range(bw.model_steps):
		if NN == 0:
			length[NN] = tf.Variable([exp_mult[0]], dtype=tf.float32)
			cd_sum += length[NN]/2
		elif NN < bw.model_steps-1:
			length[NN] = tf.Variable([exp_mult[NN]], dtype=tf.float32)
			cd_sum += length[NN]
		else:
			length[NN] = 2*(target_sum - cd_sum)   # last step derived (constrains the sum)
else:
	for NN in range(bw.model_steps):
		length[NN] = tf.constant(cd_rho, dtype=tf.float32)

for NN in range(bw.model_steps):
	# nonlinear parameters
	if NN < bw.model_steps-1:
		if tied_Kerr == True:
			nl_filter[NN] = nl_filter_all
		else:
			nl_filter[NN] = tf.Variable(no_filter, dtype=tf.float32)

# matched filter 
ps_filter = rrcosine(rolloff, delay, OS_d)
ps_filter_length = 2*round(OS_d*delay)+1
ps_filter_delay = round(OS_d*delay) # delay in samples
ps_rx_tmp = np.concatenate((ps_filter, np.zeros(Nsamp_d-ps_filter_length)))
ps_rx_tmp = np.roll(ps_rx_tmp, -ps_filter_delay)
ps_filter_rx_freq = sp.fft(ps_rx_tmp)

# }}}
#========================================================#
# build the computation graph in TensorFlow {{{
#========================================================#
print("building the TensorFlow graph ", end='', flush=True)

y_enq = tf.placeholder(tf.float32, shape=[None, Nsamp_d, 2])
x_enq = tf.placeholder(tf.complex64, shape=[None, Nsym])
P_enq = tf.placeholder(tf.float32, shape=[None, 1])

min_after_dequeue = int(conf_data["minimum elements after dequeue"]) # at least this many elements must remain after dequeue

myq = tf.RandomShuffleQueue(QMAX, min_after_dequeue, dtypes=[tf.float32, tf.complex64, tf.float32], shapes=[[Nsamp_d, 2], [Nsym], [1]])
enqueue_op = myq.enqueue_many([y_enq, x_enq, P_enq])
dummy_dequeue = myq.dequeue_many(QBSIZE*NPROC*REPF)
y,x,P_W = myq.dequeue_many(minibatch_size)

# [L-ESSFM], signals have shape = [batch_size, N, 2] (if complex) or [batch_size, N] (if real)
for NN in range(bw.model_steps):
	y = tf.complex(y[:,:,0],y[:,:,1])
	print('.', end='', flush=True)
	# linear step
	Y = tf.signal.fft(y)
	cd_freq = tf.exp(tf.complex(0.0, bw.get_cd_filter_freq(NN)*length[NN]/cd_alpha))
	Y *= cd_freq
	y = tf.signal.ifft(Y)
	y = tf.stack([tf.real(y), tf.imag(y)], axis=2)
	
	if NN < bw.model_steps-1:
		# nonlinear step, includes possible filtering of {|y_i|^2}
		ysq = tf.reduce_sum(tf.square(y), axis=2)
		Ysq = tf.signal.rfft(ysq)
		# ESSFM tied NLPR: a SINGLE shared filter encodes the normalized NLPR SHAPE;
		# each step is scaled by its own nonlinear strength nl_param[NN] (MATLAB xi(i)*C).
		# We scale by the per-step nl_param relative to the mean |nl_param| so the shared
		# filter stays O(1) (its init magnitude) and no step blows up at high Ns, where
		# nl_param[NN]/nl_param[0] would reach ~1700x. The L-ESSFM branch (free per-step
		# filters) keeps scale=1: each Variable absorbs its own nl_param during training.
		if opt_rho and tied_Kerr:
			nl_ref = float(np.mean(bw.nl_param[:-1]))  # signed mean: keeps scale=+1 at Ns=1
			nl_step_scale = bw.nl_param[NN]/nl_ref
		else:
			nl_step_scale = 1.0
		nl_time = -nl_filter[NN]*nl_step_scale/nl_alpha
		nl_time = tf.concat([tf.reverse(nl_time[1:], axis=[0]), nl_time, tf.zeros(Nsamp_d-2*nl_filter_length+1)], axis=0)
		nl_time = tf.roll(nl_time, -nl_filter_length+1, axis=0)
		nl_freq = tf.signal.rfft(nl_time)
		Ysq_filtered = Ysq*nl_freq
		ysq_filtered = tf.signal.irfft(Ysq_filtered)
		y = complex_multiply(y, tf.stack([tf.cos(ysq_filtered), tf.sin(ysq_filtered)], axis=2))
	
# matched filter
y = tf.complex(y[:,:,0],y[:,:,1])
Y = tf.signal.fft(y)*ps_filter_rx_freq
# downsample
a = Y[:, :Nsym-Nsamp_d//2]
b1 = Y[:, Nsym-Nsamp_d//2:Nsym//2]
b2 = Y[:, Nsamp_d//2:Nsamp_d-Nsym//2]
c1 = Y[:, Nsym//2:Nsamp_d//2]
c2 = Y[:, Nsamp_d-Nsym//2:3*Nsamp_d//2-Nsym]
d = Y[:, 3*Nsamp_d//2-Nsym:]
Y = tf.concat([a, b1+b2, c1+c2, d], axis=1)/OS_d
y = tf.signal.ifft(Y)/tf.complex(tf.sqrt(P_W), 0.0)/np.sqrt(OS_d)
# mean phase rotation removal
tmp = tf.reduce_sum(tf.conj(x)*y, 1, keepdims=True)
phi_cpr = -tf.atan2(tf.imag(tmp),tf.real(tmp))
x_hat = y*tf.exp(tf.complex(0.0, phi_cpr))

mean_squared_error = tf.reduce_mean(tf.square(tf.abs(x-x_hat)))
effective_snr = -10.0*tf.log(mean_squared_error+1e-12)/tf.log(10.0)

print("")
print("calling optimizer ... ", end="", flush=True)
optimizer = get_optimizer()
train = optimizer.minimize(mean_squared_error)

# compute total number of tunable parameters
total_parameters = 0
for variable in tf.trainable_variables():
	shape = variable.get_shape() # shape is an array of tf.Dimension
	variable_parameters = 1
	for dim in shape:
		variable_parameters *= dim.value
	total_parameters += variable_parameters

print("done, total tunable parameters: {}".format(total_parameters))

# }}}
#========================================================#
# start session {{{
#========================================================#
tf.summary.scalar("effective_snr", effective_snr)
tf.summary.scalar("data_queue_size", myq.size())
summary = tf.summary.merge_all()

init_op = tf.global_variables_initializer()
sess = tf.Session()

# create log dir
logdir = args.logdir+"/"+arg_str
logdir += "/" + time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime())

if not os.path.exists(logdir):
	os.makedirs(logdir)
else:
	raise RuntimeError("log directory \'" + logdir + "\' already exists")

print("name of the log directory: " + logdir)

# copy the .ini file to log folder
shutil.copyfile(config_path, logdir+"/"+config_file)
summary_writer = tf.summary.FileWriter(logdir, sess.graph)
sess.run(init_op) # run the OP that initializes global variables

# }}}
#========================================================#
# populate the data queue {{{
#========================================================#
def forward_propagation_batch(ignore_arg):
	y_read = np.zeros([QBSIZE, Nsamp_d, 2], np.float32)
	x_read = np.zeros([QBSIZE, Nsym], np.complex64)
	P_read = np.zeros([QBSIZE, 1], np.float32)
	for i in range(QBSIZE):
		y_read[i,:,:], x_read[i,:], P_read[i,:] = forward_propagation()
	return y_read, x_read, P_read

def populate_queue(sess, enqueue_op, coord):
	m = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(m)
	while not coord.should_stop():
		results = pool.map(forward_propagation_batch, [0]*NPROC)
		y_batch = np.zeros([QBSIZE*NPROC*REPF, Nsamp_d, 2], np.float32)
		x_batch = np.zeros([QBSIZE*NPROC*REPF, Nsym], np.complex64)
		P_batch = np.zeros([QBSIZE*NPROC*REPF, 1], np.float32)
		for j in range(REPF):
			for i in range(NPROC):
				off = j*QBSIZE*NPROC
				y_batch[i*QBSIZE+off:(i+1)*QBSIZE+off,:,:] = results[i][0]
				x_batch[i*QBSIZE+off:(i+1)*QBSIZE+off,:]   = results[i][1]
				P_batch[i*QBSIZE+off:(i+1)*QBSIZE+off,:]   = results[i][2]
		sess.run(enqueue_op, feed_dict={y_enq: y_batch, x_enq: x_batch, P_enq: P_batch})

coord = tf.train.Coordinator()
t = threading.Thread(target=populate_queue, args=(sess, enqueue_op, coord))
t.start()

# }}}
#========================================================#
# optimization routine {{{
#========================================================#
# inital values 
mse_tmp, snr_tmp = sess.run([mean_squared_error, effective_snr])
print("---------------------------------------------")
print("initial: MSE = {0:.6f}, effective SNR = {1:.3f} dB".format(mse_tmp, snr_tmp))
print("elements in the data queue: {}".format(sess.run(myq.size())))

# write initial summary
sstr = sess.run(summary)
summary_writer.add_summary(sstr, 0)
summary_writer.flush()

# gradient descent
start = time.time()

pruned = 0
for i in range(1,iterations+1):
	_, mse_tmp, snr_tmp, sstr = sess.run([train, mean_squared_error, effective_snr, summary]) # 1 step in gradient descent
	if(math.isnan(mse_tmp)):
		print("nan detected, exiting optimization loop")
		break
	# summary
	if i%summary_interval == 0 or i==iterations:
		summary_writer.add_summary(sstr, i)
		summary_writer.flush()
		print("iter {0}: MSE = {1:.6f}, effective SNR = {2:.3f} dB, summary written".format(i, mse_tmp, snr_tmp))
	
end = time.time()

print("requesting stop")
coord.request_stop()

queue_size = sess.run(myq.size())
if(QMAX - queue_size < QBSIZE*NPROC*REPF):
	print("dummy dequeue")
	sess.run(dummy_dequeue) # otherwise the threads hang at enqueue_op 

print("joining threads")
coord.join([t]) # wait for threads to terminate

opt_time = end-start
print("total optimization time: {0:.1f}s".format(opt_time))
print("processing approx. {0:.0f} input/output data pairs per second".format(iterations*minibatch_size/opt_time))

# }}}
#========================================================#
# save results to csv file {{{
#========================================================#
if SAVE_FILE == True:
	print("saving optimized parameters ... ", end="", flush=True)
	
	f=open(logdir+'/parameters.csv', 'ab') # a: append, b: binary mode
	f.truncate(0)
	
	length_print = sess.run(length)
	nl_filter_print = sess.run(nl_filter)
	
	for NN in range(bw.model_steps):
		tmp = np.transpose(length_print[NN]/cd_alpha)
		if NN == 0 or NN == bw.model_steps-1:
			tmp /= 2
		np.savetxt(f, tmp, delimiter=',')
		if NN < bw.model_steps-1:
			# For the ESSFM tied filter, fold the per-step nl_param scaling into the
			# SAVED filter so each step gets its correctly-scaled NLPR (the test reads
			# one filter per step, with no scaling of its own). Same nl_ref=mean(|nl_param|)
			# normalization as training. For L-ESSFM (free per-step filters) scale=1.
			if opt_rho and tied_Kerr:
				nl_ref = float(np.mean(bw.nl_param[:-1]))  # signed mean: keeps scale=+1 at Ns=1
				step_scale = bw.nl_param[NN]/nl_ref
			else:
				step_scale = 1.0
			np.savetxt(f, np.transpose(nl_filter_print[NN]*step_scale/nl_alpha), delimiter=',')
	f.close()
	print("done")
else:
	print("nothing is saved ...")

# }}}
#========================================================#