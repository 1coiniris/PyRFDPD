import numpy as np
import datetime
import tensorflow as tf
import pandas as pd
from scipy.io import loadmat
from scipy.io import savemat
from openpyxl import load_workbook
import os

# XXX = loadmat('data\polydata.mat')
# XX = XXX['XX1']

#  自定义函数

'''模型训练函数(单双带兼容)：集成了显示误差变化，tensorboard记录，nmse记录到csv等功能'''

def train_model(data_loader, model, summary_writer, batch_size, num_epochs, optimizer, hparam, L1=0, lr = 0.01):
    tf.compat.v1.reset_default_graph() ## 清除计算图 作用 ：每次运行网络的名字不会多一个_1,参考 https://blog.csdn.net/duanlianvip/article/details/98626111
    num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
    Loss_mat = np.zeros([1, num_batches])
    starttime = datetime.datetime.now()
    # X, y = data_loader.get_batch(batch_size)
    # y_pred = model(X)
    # weights = model.get_weights()
    # savemat('E://ShareCache (2)\用户文档\郁煜铖01\Matlab\Function and Script\SS_TEST\model/' + 'Recovered-model.mat', {'data_nn9': weights})
    # print("Model recovered successfullly!")
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.mean_squared_error(y_true=y, y_pred=y_pred)
            # loss2 = tf.keras.losses.mean_squared_error(y_true=y[:, 2:4], y_pred=y_pred[:, 2:4])
            # regular1 = tf.reduce_mean(tf.square(model.variables[0])) + tf.reduce_mean(tf.square(model.variables[1])) + \
            #            tf.reduce_mean(tf.square(model.variables[2])) + tf.reduce_mean(tf.square(model.variables[3])) + \
            #            tf.reduce_mean(tf.square(model.variables[4])) + tf.reduce_mean(tf.square(model.variables[5]))
            # regular2 = tf.reduce_mean(tf.square(model.variables[6])) + tf.reduce_mean(tf.square(model.variables[7]))
            # regular1 = tf.reduce_mean(tf.abs(model.variables[0])) + tf.reduce_mean(tf.abs(model.variables[1]))

            # regular2 = tf.reduce_mean(tf.abs(model.variables[6])) + tf.reduce_mean(tf.abs(model.variables[7]))
            # loss = tf.reduce_mean(loss)  + L1 * regular1
            loss = tf.reduce_mean(loss)
            if batch_index % 20 == 0:
                print("batch %d: loss %f" % (batch_index, loss.numpy()))
                Loss_mat[0,batch_index] = loss.numpy()
                with summary_writer.as_default():  # 指定记录器
                    tf.summary.scalar("loss", loss, step=batch_index)  # 将当前损失函数的值写入记录器
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    endtime = datetime.datetime.now()

    mean_squared_error = tf.keras.metrics.MeanSquaredError()
    num_batches = int(data_loader.num_test_data // batch_size)
    for batch_index in range(num_batches):
        start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
        y_pred = model.predict(data_loader.test_data[start_index: end_index])
        mean_squared_error.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
    print("mse: %f" % mean_squared_error.result())

    num_dim = data_loader.train_label.shape[1]
    nmse = np.zeros([1,num_dim])
    nmse[0,0 : num_dim//2] = complexNMSE(model, data_loader.train_data, data_loader.train_label)
    nmse[0,num_dim//2 : num_dim] = complexNMSE(model, data_loader.test_data, data_loader.test_label)

    # print("mse: %f" % mean_squared_error.result())
    for i in range(num_dim):          # 用循环是为了保证输出格式是每行一个，用*输出多维矩阵是可以不输出两边的[]
        print(nmse[0,i])  # 注，由于精度为single，sometimes结果与matlab中计算NMSE存在0.01%左右的差别，不影响结果
    print(endtime - starttime)
    model.summary()
    c = pd.DataFrame(nmse, index = [hparam], columns = None)
    c.to_csv('SB_NMSE.csv', mode='a', header=None)

    lr = lr*1000
    savemat('E://ShareCache (2)\用户文档\郁煜铖01\Matlab\Function and Script\SS_TEST\model1/' + 'LOSS_RMS_%d_dr1.mat'%lr, {
        'loss': Loss_mat})

    return model

'''基础函数：IO设置函数'''

def ap_io(x, k):
    dim = len(x)
    n = np.zeros([dim, 3 * k])
    abo = abs(x)
    ang = np.angle(x)
    for i in range(k):
        abs_1 = np.zeros([dim, 1])
        ang_1 = np.zeros([dim, 1])
        abs_1[i:dim] = abo[0:dim - i]
        ang_1[i:dim] = ang[0:dim - i]
        ang_cos = np.cos(ang_1)
        ang_sin = np.sin(ang_1)
        n[:, i:i + 1] = abs_1
        n[:, 2 * i + k:2 * i + k + 1] = ang_cos
        n[:, 2 * i + 1 + k:2 * i + 1 + k + 1] = ang_sin
    return n

def ap_io_v2(x, k):
    dim = len(x)
    n = np.zeros([dim, 3 * k])
    abo = abs(x)
    ang = np.angle(x)
    for i in range(k):
        abs_1 = np.zeros([dim, 1])
        ang_1 = np.zeros([dim, 1])
        abs_1[i:dim] = abo[0:dim - i]
        ang_1[i:dim] = ang[0:dim - i]
        ang_cos = np.cos(ang_1)
        ang_sin = np.sin(ang_1)
        n[:, i:i + 1] = abs_1
        n[:, i + k: i + k + 1] = ang_cos
        n[:, i + 2 * k:i + 2 * k + 1] = ang_sin
    return n

def iq_io(x, k):
    dim = len(x)
    n = np.zeros([dim, 2 * k])
    re = np.real(x)
    im = np.imag(x)
    for i in range(k):
        re_1 = np.zeros([dim, 1])
        im_1 = np.zeros([dim, 1])
        re_1[i:dim] = re[0:dim - i]
        im_1[i:dim] = im[0:dim - i]
        n[:, i:i + 1] = re_1
        n[:, i + k:i + k + 1] = im_1
    return n

def mp_io(x, j, k, l):  # x输入或输出变量 j记忆深度+1 k非线性阶数 l非线性相隔
    dim = len(x)
    #x = x / max(abs(x))
    n = np.zeros([dim, j * (k+1)], dtype=complex)
    ind = 0
    for i in range(j):
        x1 = np.zeros([dim, 1], dtype=complex)
        x1[i:dim] = x[0:dim - i]
        for m in range(k+1):
            p = l * m
            # n[:, i * k + m:i * k + m + 1] = x1 * (abs(x1) ** p)
            n[:, ind:ind + 1] = x1 * (abs(x1) ** p)
            ind = ind+1
    return n

def mp_io_2d(x01, x02, M, K, p):  # x输入或输出变量 j记忆深度+1 k非线性阶数 l非线性相隔
    dim = len(x01)
    #x01 = x01 / max(abs(x01)) #归一化需谨慎，动态模型会出问题
    #x02 = x02 / max(abs(x02))
    n = np.zeros([dim, (K//p+2)*(K//p+1)*(M)//2], dtype=complex)
    ind = 0
    for i in range(M):
        x01_d1 = np.zeros([dim, 1], dtype=complex)
        x01_d1[i:dim] = x01[0:dim - i]
        x02_d1 = np.zeros([dim, 1], dtype=complex)
        x02_d1[i:dim] = x02[0:dim - i]
        for k in range(0,K+1,p):
            for j in range(0,1,p):
                # n[:, (K//p+2)*(K//p+1)*i//2+(k//p+1)*(k//p)//2+j//p:(K//p+2)*(K//p+1)*i//2+(k//p+2)*(k//p)//2+j//p+1] = x01_d1 * (abs(x01_d1)**(k-j)) *(abs(x02_d1)**j)
                n[:, ind:ind+1] = x01_d1 * (abs(x01_d1) ** (k - j)) * (abs(x02_d1) ** j)
                ind = ind + 1
    for i in range(M):
        x01_d1 = np.zeros([dim, 1], dtype=complex)
        x01_d1[i:dim] = x01[0:dim - i]
        x02_d1 = np.zeros([dim, 1], dtype=complex)
        x02_d1[i:dim] = x02[0:dim - i]
        for k in range(0,K+1,p):
            for j in range(p,k+1,p):
                # n[:, (K//p+2)*(K//p+1)*i//2+(k//p+1)*(k//p)//2+j//p:(K//p+2)*(K//p+1)*i//2+(k//p+2)*(k//p)//2+j//p+1] = x01_d1 * (abs(x01_d1)**(k-j)) *(abs(x02_d1)**j)
                n[:, ind:ind+1] = x01_d1 * (abs(x01_d1) ** (k - j)) * (abs(x02_d1) ** j)
                ind = ind + 1
    return n

'''基础函数：模型IO函数'''

'单双带兼容'
def RVTDNN_IO(pa_input, pa_output, m, trainnum, testnum, env_order = (), input_label = 0, output_type = 0):  # x输入或输出变量 j记忆深度+1 k非线性阶数 l非线性相隔

    band_num = pa_output.shape[1]
    inputn = iq_io(pa_input[:,0:1], m)
    outputn = iq_io(pa_output[:,0:1], 1)

    input_am = ap_io(pa_input[:,0:1], m)
    input_am_0 = input_am[:, 0:m]
    for i in env_order:
        input_am_temp = input_am_0 ** i
        inputn = np.concatenate((inputn, input_am_temp), axis=1)
    '多带'
    for i in range(1,band_num):
        inputn_temp = iq_io(pa_input[:,i:i+1], m)
        outputn_temp = iq_io(pa_output[:,i:i+1], 1)

        input_am = ap_io(pa_input[:,i:i+1], m)
        input_am_0 = input_am[:, 0:m]
        for i in env_order:
            input_am_temp = input_am_0 ** i
            inputn = np.concatenate((inputn_temp, input_am_temp), axis=1)
        inputn = np.concatenate((inputn, inputn_temp), axis=1)
        outputn = np.concatenate((outputn, outputn_temp), axis=1)
    '特殊情况'
    if output_type == 'm-out':
        outputn = np.column_stack((outputn, outputn))
    elif output_type == 'dynamic':
        inputn = np.column_stack((inputn, input_label))

    [input_train, output_train, input_test, output_test] = train_test_div(inputn, outputn, trainnum, testnum, 10)

    return [input_train, input_test, output_train, output_test]

def VDTDNN_IO(pa_input, pa_output, m, trainnum, testnum, input_label = 0, output_type = 0):  # x输入或输出变量 j记忆深度+1 k非线性阶数 l非线性相隔
    # dim = len(pa_input)
    inputn = ap_io(pa_input, m)
    outputn = iq_io(pa_output, 1)

    if output_type == 'm-out':
        outputn = np.column_stack((outputn, outputn))
    elif output_type == 'dynamic':
        data_dim = len(pa_input)
        label_dim = len(np.transpose(input_label))
        print("The number of labels: %d" % label_dim)

        inputn1 = inputn
        inputn = np.zeros([data_dim, 3 * m + label_dim])
        inputn[:, 0:m] = inputn1[:, 0:m]
        inputn[:, m: m + label_dim] = input_label
        inputn[:, m + label_dim:3 * m + label_dim] = inputn1[:, m:3 * m]

    [input_train, output_train, input_test, output_test] = train_test_div(inputn, outputn, trainnum, testnum, 10)

    return [input_train, input_test, output_train, output_test]

def PANN_IO(pa_input, pa_output, m, k, trainnum, testnum, input_label = 0, output_type = 0):   # Made the input and output for Polynomial-assisted neural network
    # m: memory depth+1 k nonlinear order
    inputn = ap_io(pa_input, m)
    outputn = iq_io(pa_output, 1)

    if output_type == 'm-out':
        outputn = np.column_stack((outputn, outputn))
    elif output_type == 'dynamic':
        data_dim = len(pa_input)
        label_dim = len(np.transpose(input_label))
        print("The number of labels: %d" % label_dim)

        inputn1 = inputn
        inputn = np.zeros([data_dim, 3 * m + label_dim])
        inputn[:, 0:m] = inputn1[:, 0:m]
        inputn[:, m: m + label_dim] = input_label
        inputn[:, m + label_dim:3 * m + label_dim] = inputn1[:, m:3 * m]

    [input_train, output_train, input_test, output_test] = train_test_div(inputn, outputn, trainnum, testnum, 10)

    outputn_mp = mp_io(pa_output, 1, 1, 1)
    inputn_mp = mp_io(pa_input, m, k, 1)
    # inputn_mp = XX
    print(inputn_mp.shape)
    [input_train_mp, output_train_mp, input_test_mp, output_test_mp] = train_test_div(inputn_mp, outputn_mp, trainnum,
                                                                                     testnum, 10)

    input_train1 = np.column_stack((input_train, input_train_mp.real, input_train_mp.imag))
    input_test1 = np.column_stack((input_test, input_test_mp.real, input_test_mp.imag))
    return [input_train1, input_test1, output_train, output_test]

def RVRNN_IO(pa_input, pa_output, m, seq_length, trainnum, testnum):  # x输入或输出变量 j记忆深度+1 k非线性阶数 l非线性相隔
    dim = len(pa_input)
    inputn = iq_io(pa_input, m)
    outputn = iq_io(pa_output, 1)

    input_padding = np.zeros([seq_length - 1, 2 * m])
    inputn = np.row_stack((input_padding, inputn))
    inputn_1 = []
    for index in range(dim):
        inputn_1.append(inputn[index:index + seq_length, :])
    inputn = np.array(inputn_1)
    # print(inputn.shape)
    [input_train, output_train, input_test, output_test] = train_test_div(inputn, outputn, trainnum, testnum, 210)

    return [input_train, input_test, output_train, output_test]

def RVTDCNN_IO(pa_input, pa_output, m, trainnum, testnum):  # x输入或输出变量 j记忆深度+1 k非线性阶数 l非线性相隔
    dim = len(pa_input)
    inputn = iq_io(pa_input, 1)
    outputn = iq_io(pa_output, 1)
    input_am = ap_io(pa_input, 1)
    input_am_1 = input_am[:, 0:1]
    input_am_2 = input_am_1 ** 2
    input_am_3 = input_am_1 ** 3
    inputn = np.concatenate((inputn, input_am_1, input_am_2, input_am_3), axis=1)

    input_padding = np.zeros([m - 1, 5])
    inputn = np.row_stack((input_padding, inputn))
    inputn_1 = []
    for index in range(dim):
        inputn_1.append(inputn[index:index + m, :])
    inputn = np.array(inputn_1)

    inputn = np.expand_dims(inputn, axis=3)

    [input_train, output_train, input_test, output_test] = train_test_div(inputn, outputn, trainnum, testnum, 10)

    return [input_train, input_test, output_train, output_test]

def VDTDNN_DUAL_IO(pa_input1, pa_input2, pa_output1, pa_output2, m, trainnum, testnum, input_label = 0, cutdata = 10):  # x输入或输出变量 j记忆深度+1 k非线性阶数 l非线性相隔
    # dim = len(pa_input)
    inputn1 = ap_io(pa_input1, m)
    inputn2 = ap_io(pa_input2, m)
    inputn = np.concatenate((inputn1, inputn2), axis=1)  # 合并
    outputn1 = iq_io(pa_output1, 1)
    outputn2 = iq_io(pa_output2, 1)
    outputn = np.concatenate((outputn1, outputn2), axis=1)
    if (np.max(input_label) != 0):
        inputn = np.concatenate((inputn, input_label), axis=1)
    [input_train, output_train, input_test, output_test] = train_test_div(inputn, outputn, trainnum, testnum, cutdata)

    return [input_train, input_test, output_train, output_test]

def PANN_DUAL_IO(pa_input, pa_output, m, k, trainnum, testnum, input_label = 0, cutdata = 10, out = 2):  # x输入或输出变量 j记忆深度+1 k非线性阶数 l非线性相隔 cutdata去掉开头的几个数据（其实没什么影响
    # dim = len(pa_input)
    pa_input1 = pa_input[:, 0:1]
    pa_input2 = pa_input[:, 1:2]
    pa_output1 = pa_output[:, 0:1]
    pa_output2 = pa_output[:, 1:2]

    inputn1 = ap_io(pa_input1, m)
    inputn2 = ap_io(pa_input2, m)
    inputn = np.concatenate((inputn1, inputn2), axis=1)
    outputn1 = iq_io(pa_output1, 1)
    outputn2 = iq_io(pa_output2, 1)
    outputn = np.concatenate((outputn1, outputn2), axis=1)

    if (out == 1 or out == 3):
        pa_output3 = pa_output[:, 2:3]
        outputn3 = iq_io(pa_output3, 1)
        if (out == 1):
            outputn = outputn3
        else:
            outputn = np.concatenate((outputn, outputn3), axis=1)

    if (input_label.any != 0):
        inputn = np.concatenate((inputn, input_label), axis=1)
        label_dim = len(np.transpose(input_label))
        print("The number of labels: %d" % label_dim)
    [input_train, output_train, input_test, output_test] = train_test_div(inputn, outputn, trainnum, testnum, cutdata)


    outputn_mp1 = mp_io(pa_output1,1, 1, 1)
    inputn_mp1 = mp_io_2d(pa_input1,pa_input2, m, k, 1)
    outputn_mp2 = mp_io(pa_output2, 1, 1, 1)
    inputn_mp2 = mp_io_2d(pa_input2,pa_input1, m, k, 1)
    # inputn_mp = np.concatenate((inputn_mp1, inputn_mp2), axis=1)
    # outputn_mp = np.concatenate((outputn_mp1, outputn_mp2), axis=1)
    [input_train_mp1, output_train_mp, input_test_mp1, output_test_mp] = train_test_div(inputn_mp1, outputn_mp1,
                                                                                        trainnum, testnum, cutdata)  # 12*2*2
    [input_train_mp2, output_train_mp, input_test_mp2, output_test_mp] = train_test_div(inputn_mp2, outputn_mp2,
                                                                                        trainnum, testnum, cutdata)  # 12*2*2
    input_train1 = np.column_stack(
        (input_train, input_train_mp1.real, input_train_mp1.imag, input_train_mp2.real, input_train_mp2.imag))
    input_test1 = np.column_stack(
        (input_test, input_test_mp1.real, input_test_mp1.imag, input_test_mp2.real, input_test_mp2.imag))
    return [input_train1, input_test1, output_train, output_test]

'''基础函数：模型子函数'''

def PhaseRecovery_new(x, phase, mem, band = 1, phase2 = 0):
    I = x[:, 0: 2]
    Q = x[:, 2: 4]

    for i in range(1,mem):
        I_temp = x[:, i*4+0:i*4+2]
        Q_temp = x[:, i*4+2:i*4+4]
        I = tf.concat((I, I_temp), axis=1)
        Q = tf.concat((Q, Q_temp), axis=1)

    out_i = I * phase
    out_q = Q * phase
    out_i = tf.reduce_sum(out_i, 1, keepdims=True)
    out_q = tf.reduce_sum(out_q, 1, keepdims=True)
    y = tf.concat((out_i, out_q), axis=1)

    if (band==2):
        for i in range(mem,2*mem):
            # print(i)
            I_temp = x[:, i * 4 + 0:i * 4 + 2]
            Q_temp = x[:, i * 4 + 2:i * 4 + 4]
            if i == mem:
                I = I_temp
                Q = Q_temp
            else:
                I = tf.concat((I, I_temp), axis=1)
                Q = tf.concat((Q, Q_temp), axis=1)
        # print(I.shape)
        # print(phase2.shape)
        out_i = I * phase2
        out_q = Q * phase2
        out_i = tf.reduce_sum(out_i, 1, keepdims=True)
        out_q = tf.reduce_sum(out_q, 1, keepdims=True)
        y = tf.concat((y, out_i, out_q), axis=1)


    return y

'''基础函数：训练子函数'''

def my_loss(x, y):  #自定义loss function 有需要可以使用
    # dim = len(x)
    # y_pred = tf.zeros([dim, 1])
    # y_true = tf.zeros([dim, 1])
    # y_pred[:, 1] = x[:, 0] * 32 + x[:, 1] * 16 + x[:, 2] * 8 + x[:, 3] * 4 + x[:, 4] * 2 + x[:, 5]
    # y_true[:, 1] = y[:, 0] * 32 + y[:, 1] * 16 + y[:, 2] * 8 + y[:, 3] * 4 + y[:, 4] * 2 + y[:, 5]
    loss = 0.7 * tf.reduce_mean(tf.reduce_mean(tf.math.squared_difference(
        x[:, 0] * 1 / 2 + x[:, 1] * 1 / 4 + x[:, 2] * 1 / 8 + x[:, 3] * 1 / 16 + x[:, 4] * 1 / 32 + x[:, 5] * 1 / 64,
        y[:, 0] * 1 / 2 + y[:, 1] * 1 / 4 + y[:, 2] * 1 / 8 + y[:, 3] * 1 / 16 + y[:, 4] * 1 / 32 + y[:,
                                                                                                    5] * 1 / 64))) + 0.3 * tf.reduce_mean(
        tf.reduce_mean(tf.math.squared_difference(x, y)))
    return loss

def NMSE_YU(y_t,y_m):
    x = y_t / np.sqrt(np.mean(y_t * np.conj(y_t)))
    #print(np.mean(y_t * np.conj(y_t)))
    y = y_m / np.sqrt(np.mean(y_m * np.conj(y_m)))
    error = x - y
    nmse = 10 * np.log10(np.mean((abs(error)) ** 2) / np.mean((abs(x)) ** 2))
    #print(np.mean((abs(x)) ** 2))
    return nmse

def complexNMSE(model, train_data, y_label):

    num_data = y_label.shape[1]//2
    nmse = np.zeros([num_data])
    y_pred_all = model.predict(train_data)
    y_pred_all = np.array(y_pred_all)
    for i in range(num_data):
        y_pred_all_i = y_pred_all[:, 2*i]
        y_pred_all_q = y_pred_all[:, 2*i+1]
        y_pred_all_c = y_pred_all_i + y_pred_all_q * 1j
        y_i = y_label[:, 2*i]
        y_q = y_label[:, 2*i+1]
        y_c = y_i + y_q * 1j

        nmse[i] = NMSE_YU(y_pred_all_c,y_c)

    # savemat('E://ShareCache (2)\用户文档\郁煜铖01\Matlab\Function and Script\SS_TEST\model/' + 'out1.mat', {
    #      'out': y_pred_all})

    return nmse

def complexNMSE_dual(train_data, model, y_label):

    y_pred_all = model.predict(train_data)

    y_pred_all_i = y_pred_all[:, 0]
    y_pred_all_q = y_pred_all[:, 1]
    y_pred_all_c1 = y_pred_all_i + y_pred_all_q * 1j
    y_i = y_label[:, 0]
    y_q = y_label[:, 1]
    y_c = y_i + y_q * 1j

    x = y_pred_all_c1 / np.sqrt(np.mean(y_pred_all_c1 * np.conj(y_pred_all_c1)))
    y = y_c / np.sqrt(np.mean(y_c * np.conj(y_c)))
    error = x - y
    nmse1 = 10 * np.log10(np.mean((abs(error)) ** 2) / np.mean((abs(y)) ** 2))

    y_pred_all_i = y_pred_all[:, 2]
    y_pred_all_q = y_pred_all[:, 3]
    y_pred_all_c2 = y_pred_all_i + y_pred_all_q * 1j
    y_i = y_label[:, 2]
    y_q = y_label[:, 3]
    y_c = y_i + y_q * 1j

    x = y_pred_all_c2 / np.sqrt(np.mean(y_pred_all_c2 * np.conj(y_pred_all_c2)))
    y = y_c / np.sqrt(np.mean(y_c * np.conj(y_c)))
    error = x - y
    nmse2 = 10 * np.log10(np.mean((abs(error)) ** 2) / np.mean((abs(y)) ** 2))

    # data0 = train_data
    # data1 = model(train_data,3)
    # data2 = model(train_data,1)
    # data3 = model(train_data,2)
    # savemat('E://ShareCache (2)\用户文档\郁煜铖01\Matlab\Function and Script\TT_TEST\data/' + 'out.mat', {
    #     'out': data1,'out1': data2,'out2': data3,'in':data0})

    return nmse1, nmse2

'''功能函数'''

def MP_e(pa_input, pa_output, j, k, l, N):  # x输入或输出变量 j记忆深度+1 k非线性阶数 l非线性相隔
    inputn = mp_io(pa_input, j, k, l)
    inputn = inputn[j-1:N,:]
    outputn = pa_output[j-1:N,:]
    print(inputn.shape)
    print(outputn.shape)
    coeff = np.dot(np.linalg.pinv(inputn) , outputn) # Python内乘号是元素相乘，矩阵相乘要使用np.dot
    output_m = np.dot( inputn, coeff)
    nmse = NMSE_YU(outputn,output_m)
    return coeff, nmse

def train_test_div(inputn, outputn, m, k, n):
    input_train = inputn[n:m + n, :]
    output_train = outputn[n:m + n, :]
    input_test = inputn[m + n:m + k + n, :]
    output_test = outputn[m + n:m + k + n, :]
    return [input_train, output_train, input_test, output_test]

'''功能函数：部分参考网上'''

def seed_tensorflow(seed=42): # 固定随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    # random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1' # `pip install tensorflow-determinism` first,使用与tf>2.1
    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    #session_conf = tf.compat.v1.ConfigProto(  # 应该是适合于 1.x 版本的
    #    intra_op_parallelism_threads=1,
    #    inter_op_parallelism_threads=1
    #)
    #sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    #tf.compat.v1.keras.backend.set_session(sess)

def make_hparam_string(model,learning_rate,dr,batch_size, iter_num,seed,unit = 0):
    #fc_param = 'fc=2' if use_two_fc else 'fc=1'
    current_time = datetime.datetime.now().strftime("%d%H%M")  # 记录到tensorboard
    model = model.name
    return '%s,%s,lr_%s,dr_%s,bs_%sk,in_%sk_%s_%s' % (model,unit,learning_rate,dr,batch_size/1000, iter_num/1000,seed,current_time)


