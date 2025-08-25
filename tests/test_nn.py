import torch
import torch.nn as nn
import torch.optim as optim
import tomli
import logging
from functools import partial
from scipy.io import loadmat, savemat
import pyrfdpd.nn as dpdnn
from pyrfdpd.nn import rvtdnn,trainer
import pyrfdpd.visa as visa
from pyrfdpd.utils import metrics, plot, align
# import pyrfdpd.nn as dpdnn
import pyrfdpd.utils
# import pyrfdpd.nn
# import Model.Orth_NN as NN

# model = NN.Orth_Basis_DVR_NN(1,2)

configfile = "rvtdnn.toml"
# tests/config/common.toml
with open("../tests/config/" + "common.toml", "rb") as f:
    common_dict = tomli.load(f)
with open("../tests/config/" + configfile, "rb") as f:
    config_dict = tomli.load(f)

test_name = config_dict["title"]

logger = logging.getLogger(test_name)
logger.setLevel(logging.DEBUG)
log_file = logging.FileHandler("../tests/log/" + test_name + ".log")
log_file.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log_file.setFormatter(formatter)
logger.addHandler(log_file)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(formatter)
logger.addHandler(console)

# Data configuration
data_file = '../tests/data/signal_20M_in.mat'
# # data_file = common_dict["data"]["data_path"] + common_dict["data"]["data_file"]
# result_file = (
#     common_dict["data"]["data_path"]
#     + test_name
#     + "_"
#     + common_dict["data"]["data_file"]
# )
result_file = '../tests/data/signal_20M_out.mat'
x_data = loadmat(data_file)
xorg = x_data['x_sync_01'].reshape(-1)
xorg = xorg / max(abs(xorg))
y_data = loadmat(result_file)
yorg = y_data['y_sync_01'].reshape(-1)
yorg = yorg / max(abs(yorg))
# Instrument configuration
# sg_ip = common_dict["instrument"]["signal_generator"]["ip"]
# sa_ip = common_dict["instrument"]["spectrum_analyzer"]["ip"]
# sg_brand = common_dict["instrument"]["signal_generator"]["brand"]
# sa_brand = common_dict["instrument"]["spectrum_analyzer"]["brand"]
#
# fc = common_dict["instrument"]["configuration"]["center_frequency"]
# fs = common_dict["instrument"]["configuration"]["sampling_frequency"]
# pow = common_dict["instrument"]["configuration"]["power"]
# att = common_dict["instrument"]["configuration"]["attenuation"]

# Model configuration
network = config_dict["model"]["model_name"]
train_ratio = config_dict["model"]["hyperparameters"]["train_ratio"]
iteration = config_dict["model"]["hyperparameters"]["iteration"]
lr = config_dict["model"]["hyperparameters"]["learning_rate"]
batch_size = config_dict["model"]["hyperparameters"]["batch_size"]
lossFcn = eval(config_dict["model"]["hyperparameters"]["loss_function"])
optimizer = eval(config_dict["model"]["hyperparameters"]["optimizer"])
epochs = config_dict["model"]["hyperparameters"]["epochs"]
if network == "RVTDNN":
    M = config_dict["model"]["memory_depth"]
    hidden_layers = config_dict["model"]["hidden_layers"]
    activation = config_dict["model"]["activation"]
    layers = [2 * (M + 1)] + hidden_layers + [2]
    net = rvtdnn.RVTDNN(layers,activation)
    # net = dpdnn.rvtdnn.RVTDNN(layers, activation)
    dataset = partial(dpdnn.rvtdnn.Dataset, memory=M)
elif network == "ARVTDNN":
    M = config_dict["model"]["memory_depth"]
    order = config_dict["model"]["nonlinear_order"]
    order_memory = config_dict["model"]["nonlinear_order_memory"]
    layers = (
        [2 * (M + 1) + (order_memory + 1) * order]
        + config_dict["model"]["hidden_layers"]
        + [2]
    )
    activation = config_dict["model"]["activation"]
    net = dpdnn.arvtdnn.ARVTDNN(layers, activation)
    dataset = partial(
        dpdnn.arvtdnn.Dataset, memory=M, order=order, order_memory=order_memory
    )
elif network == "R2TDNN":
    M = config_dict["model"]["memory_depth"]
    hidden_layers = config_dict["model"]["hidden_layers"]
    activation = config_dict["model"]["activation"]
    layers = [2 * (M + 1)] + hidden_layers + [2]
    net = dpdnn.r2tdnn.R2TDNN(layers, activation)
    dataset = partial(dpdnn.r2tdnn.Dataset, memory=M)
else:
    # You can add your own network here.
    pass
my_trainer = trainer.Trainer(
    net, test_name, lr, batch_size, lossFcn, optimizer, tensorboard=True, logger=logger
)
logger.info(f"{'-'*30}")
logger.info(f"Start Testing --- " + test_name)
logger.info(f"{'-'*30}")

# Initial test
# visa.down_signal(sg_brand, xorg, fc, fs, pow, sg_ip, logger=logger)
# yraw = visa.collect_signal(sa_brand, fc, fs, att, sa_ip, logger=logger)
yorg = align.align(xorg,yorg)
pa_input, pa_output = torch.from_numpy(xorg).clone(), torch.from_numpy(yorg).clone()

# Full data set for prediction
org_data_set = dataset(
    pa_input=pa_input,
    pa_output=pa_output,
    train_ratio=1,
    train=True,
    inverse=False,
)

# DPD iteration
# for idx in range(iteration):
#     logger.debug(f"Start the {idx+1}th iteration")
training_set = dataset(
    pa_input=pa_input,
    pa_output=pa_output,
    train_ratio=train_ratio,
    train=True,
    inverse=True,
)
validation_set = dataset(
    pa_input=pa_input,
    pa_output=pa_output,
    train_ratio=train_ratio,
    train=False,
    inverse=True,
)
my_trainer.train(training_set, validation_set, epochs=epochs)
pa_input = my_trainer.predict(org_data_set)
# visa.down_signal(
#     sg_brand, pa_input.numpy().copy(), fc, fs, pow, sg_ip, logger=logger
# )
# pa_output = visa.collect_signal(sa_brand, fc, fs, att, sa_ip, logger=logger)
# pa_output = torch.from_numpy(align.align(xorg, pa_output)).clone()
pa_input = pa_input.numpy()
# Save results and plots
pa_output = pa_output.numpy()
torch.save(net.state_dict(), "net/" + test_name + ".pth")
mdict = {"xorg": xorg, "yorg": yorg, "wDPD": pa_output}
savemat(result_file, mdict)

fs = 100e6
logger.info("The preformance before DPD:")
metrics.nmse(xorg, yorg, logger)
metrics.acpr(yorg, fs, 40e6, 20e6, logger)
plot.amam(xorg, yorg, "figures/" + test_name + " amam wo DPD.png")
plot.ampm(xorg, yorg, "figures/" + test_name + " ampm wo DPD.png")

logger.info("The preformance after DPD:")
metrics.nmse(xorg, pa_output, logger)
metrics.acpr(pa_output, fs, 40e6, 40e6, logger)
plot.amam(xorg, pa_output, "figures/" + test_name + " amam w DPD.png")
plot.ampm(xorg, pa_output, "figures/" + test_name + " ampm w DPD.png")
plot.psd(
    {"original input": xorg, "output w/o DPD": yorg},
    fs=fs,
    filename="figures/" + test_name + " psd.png",
)
plot.psd(
    {"original input": xorg,"DPD signal":pa_input},
    fs=fs,filename='figures/DPD.png'
)