import tomli
import logging
from functools import partial
from scipy.io import loadmat, savemat
import pyrfdpd.volterra as volterra
import pyrfdpd.visa.collect_signal as visa_collect_signal
import pyrfdpd.visa.down_signal as visa_down_signal
from pyrfdpd.utils import metrics, plot, align
import pyrfdpd



configfile = "gmp.toml"

with open("config/" + "common.toml", "rb") as f:
    common_dict = tomli.load(f)
with open("config/" + configfile, "rb") as f:
    config_dict = tomli.load(f)

test_name = config_dict["title"]

logger = logging.getLogger(test_name)
logger.setLevel(logging.DEBUG)
log_file = logging.FileHandler('log/' + test_name + '.log')
log_file.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_file.setFormatter(formatter)
logger.addHandler(log_file)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(formatter)
logger.addHandler(console)

# Data configuration
data_file = common_dict["data"]["data_path"] + common_dict["data"]["data_file"]
result_file = common_dict["data"]["data_path"] + test_name + '_' + common_dict["data"]["data_file"]
data = loadmat(data_file)
xorg = data[common_dict["data"]["data_name"]].reshape(-1)
xorg = xorg / max(abs(xorg))

# Instrument configuration
sg_ip = common_dict["instrument"]["signal_generator"]["ip"]
sa_ip = common_dict["instrument"]["spectrum_analyzer"]["ip"]
sg_brand = common_dict["instrument"]["signal_generator"]["brand"]
sa_brand = common_dict["instrument"]["spectrum_analyzer"]["brand"]

fc = common_dict["instrument"]["configuration"]["center_frequency"]
fs = common_dict["instrument"]["configuration"]["sampling_frequency"]
pow = common_dict["instrument"]["configuration"]["power"]
att = common_dict["instrument"]["configuration"]["attenuation"]

# Model configuration
model = config_dict["model"]["model_name"]
iteration = config_dict["model"]["hyperparameters"]["iteration"]
if model == "MP":
    M = config_dict["model"]["memory_depth"]
    K = config_dict["model"]["nonlinear_order"]
    ratio = config_dict['model']['hyperparameters']['ratio']
    model_e = partial(volterra.mp.MP_e, M = M, K = K, ratio=ratio)
    model_v = partial(volterra.mp.MP_v, M = M, K = K)
elif model == "GMP":
    M = config_dict["model"]["memory_depth"]
    K = config_dict["model"]["nonlinear_order"]
    L = config_dict["model"]["lagging_depth"]
    ratio = config_dict['model']['hyperparameters']['ratio']
    # model_e = partial(volterra.gmp.GMP_e, M = M, K = K, L = L, ratio=ratio)
    # model_v = partial(volterra.gmp.GMP_v, M = M, K = K, L = L)
else:
    pass

logger.info(f"{'-'*30}")
logger.info(f"Start Testing --- " + test_name)
logger.info(f"{'-'*30}")

# Initial test

# visa_down_signal(sg_brand, xorg, fc, fs, pow, sg_ip, logger=logger)
yraw = visa_collect_signal(sa_brand, fc, fs, att, sa_ip, logger=logger)
yorg = align.align(xorg, yraw)
pa_input, pa_output = xorg.copy(), yorg.copy()

# DPD iteration
for idx in range(iteration):
    logger.debug(f"Start the {idx+1}th iteration")
    cc = model_e(pa_output, pa_input)
    pa_input = model_v(xorg, cc)
    visa_down_signal(sg_brand, pa_input, fc, fs, pow, sg_ip, logger=logger)
    pa_output = visa_collect_signal(sa_brand, fc, fs, att, sa_ip, logger=logger)
    pa_output = align.align(xorg, pa_output)
logger.debug("DPD done!")

# Save results and plots
mdict = {"xorg": xorg, "yorg": yorg, "wDPD": pa_output}
savemat(result_file, mdict)

logger.info("The preformance before DPD:")
metrics.nmse(xorg, yorg, logger)
metrics.acpr(yorg, fs, 40e6, 40e6, logger)
plot.amam(xorg, yorg, "tests/figures/" + test_name + " amam wo DPD.png")
plot.ampm(xorg, yorg, "tests/figures/" + test_name + " ampm wo DPD.png")

logger.info("The preformance after DPD:")
metrics.nmse(xorg, pa_output, logger)
metrics.acpr(pa_output, fs, 40e6, 40e6, logger)
plot.amam(xorg, pa_output, "tests/figures/" + test_name + " amam w DPD.png")
plot.ampm(xorg, pa_output, "tests/figures/" + test_name + " ampm w DPD.png")
plot.psd(
    {"original input": xorg, "output w/o DPD": yorg, "output w DPD": pa_output},
    fs=fs,
    filename="tests/figures/" + test_name + " psd.png",
)