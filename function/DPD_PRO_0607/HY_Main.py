import datetime
from Dataloader import *
import tensorflow as tf
import numpy as np
import NeuralNetwork as NN
from scipy.io import loadmat
from scipy.io import savemat
import UserFunction as uf

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu  Comment out this line to use GPU, and uncomment it to use CPU
##列出你所有的物理GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus, True)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data = loadmat('GaN_2p4G_32p1dBm_100_307.mat')
pa_input = data['pa_input']
pa_output = data['pa_output']

train_number = len(pa_input)//8*3
test_number = len(pa_input)//8*5

memory_depth = 5
seed = 42

model1 = NN.RVTDNN(1)
tf.keras.backend.set_floatx('float64') # 平均性能差不多，float64 batch>1024更好

for lr in [0.01]: # 0.01 for adam
    for dr in [0.95]: # 95/97 差不多
        learning_rate_schedules = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr,decay_steps=1000, decay_rate=dr,staircase=True)
        for batch_size in [2048]:  # num_epochs//4
            for num_epochs in [batch_size]: # /2 *1 *2 略有提升
                # for model in [NN.RVTDNN(20),NN.RVTDNN(30),NN.RVTDNN(40)]:
                for neu in [30]: # 12，48 好于 6，36 , out = out, n3 = n3
                    # model = NN.RVTDBNN(30)
                    model = NN.RVTDNN(neu, mem=memory_depth, label_dim=0)
                    uf.seed_tensorflow(seed)
                    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedules)
                    data_loader = My_dataloader(model1.name,pa_input,pa_output,memory_depth,train_number,test_number, env_order = (1,2,3), input_label = 0, nk = 4, seq_length = 5)
                    hparam = uf.make_hparam_string(model, lr, dr, batch_size, num_epochs, seed, model.dense.units)
                    summary_writer = tf.summary.create_file_writer('D://PycharmProjects/DPD/20220812/' + hparam)
                    print('Start running for %s' % hparam)
                    trained_model = uf.train_model(data_loader, model, summary_writer, batch_size, num_epochs, optimizer, hparam, 0, lr)
                    weights = trained_model.get_weights()
                    data = savemat(
                        'E://ShareCache (2)\用户文档\郁煜铖01\Matlab\Function and Script\TT_TEST\model1/' + 'HY_MODEL_%d%d' % (seed,neu), {'data_nn9': weights})

''' run tensorboard 
activate TF2.1
tensorboard --logdir=D:\PycharmProjects\DPD\logs

'''
