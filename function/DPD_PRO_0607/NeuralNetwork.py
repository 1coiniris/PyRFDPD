import tensorflow as tf
import numpy as np
from Layer import My_dense
from Layer import GroupwiseLayer_new
from Layer import Complexdenselayer
from Layer import Complexdenselayer_dual
import UserFunction as uf
from binary_ops import binary_tanh as binary_tanh_op
from binary_layers import BinaryDense, Clip

# 自定义神经网络


'''1D-模型 Note:部分模型通过输入label_dim转变为动态模型'''

class RVTDNN(tf.keras.Model):
    def __init__(self,unit, mem = 5 , label_dim = 0, band = 1):
        super().__init__()
        self.dense = tf.keras.layers.Dense(units=unit, activation=tf.nn.relu)
        self.dense1 = tf.keras.layers.Dense(units=unit, activation=tf.nn.relu)
        self.dense20 = tf.keras.layers.Dense(units=2*band)
    def call(self, inputs):
        x = self.dense(inputs)
        x = self.dense1(x)
        x = self.dense20(x)  # [batch_size, 10]
        return x

class VDTDNN(tf.keras.Model):
    def __init__(self,unit,mem = 5, label_dim = 0):
        super().__init__()
        self.unit = unit
        self.memory = mem
        self.la = label_dim
        self.dense = tf.keras.layers.Dense(units=self.unit, activation=tf.nn.relu)
        self.dense1 = tf.keras.layers.Dense(units=self.unit, activation=tf.nn.relu)
        self.groupwiselayer = GroupwiseLayer_new(units=4, mem=self.memory)

    def call(self, inputs):  # [batch_size, 28, 28, 1]

        input_am = inputs[:, 0: self.memory + self.la]  # 前3行 为幅度值
        input_phase = inputs[:, self.memory + self.la : 3 * self.memory + self.la]  # 之间6行 为相位值的cos和sin值
        x = self.dense(input_am)
        x = self.dense1(x)
        per_neuron = self.unit // self.memory

        x1 = x[:, 0 :  per_neuron]        # 相位恢复网络的输入
        x1 = tf.concat((x1, inputs[:, 0: 1]), axis=1)  # 附加线性项
        for i in range(1,self.memory):
            x_temp = x[:, i * per_neuron : (i + 1) * per_neuron]        # 相位恢复网络的输入
            x_temp = tf.concat((x_temp, inputs[:, i : i + 1]), axis=1)  # 附加线性项
            x1 = tf.concat((x1, x_temp), axis=1)

        x = self.groupwiselayer(x1)  # [batch_size, 10]
        output = uf.PhaseRecovery_new(x, input_phase, self.memory)

        return output

class PANN(tf.keras.Model):
    def __init__(self, unit, mem = 5, label_dim = 0):
        super().__init__()
        self.unit = unit
        self.memory = mem
        self.la = label_dim
        self.dense = tf.keras.layers.Dense(units=self.unit, activation=tf.nn.relu)
        self.dense1 = tf.keras.layers.Dense(units=self.unit, activation=tf.nn.relu)
        self.groupwiselayer = GroupwiseLayer_new(units=4, mem=self.memory)
        self.complexdenselayer = Complexdenselayer()

    def call(self, inputs):  # [batch_size, 28, 28, 1]
        input_am = inputs[:, 0: self.memory + self.la]  # 前3行 为幅度值
        input_phase = inputs[:, self.memory + self.la: 3 * self.memory + self.la]  # 之间6行 为相位值的cos和sin值
        x = self.dense(input_am)
        x = self.dense1(x)
        per_neuron = self.unit // self.memory

        x1 = x[:, 0:  per_neuron]  # 相位恢复网络的输入
        x1 = tf.concat((x1, inputs[:, 0: 1]), axis=1)  # 附加线性项
        for i in range(1, self.memory):
            x_temp = x[:, i * per_neuron: (i + 1) * per_neuron]  # 相位恢复网络的输入
            x_temp = tf.concat((x_temp, inputs[:, i: i + 1]), axis=1)  # 附加线性项
            x1 = tf.concat((x1, x_temp), axis=1)

        x = self.groupwiselayer(x1)  # [batch_size, 10]
        output1 = uf.PhaseRecovery_new(x, input_phase, self.memory)

        # input_am1 = inputs[:, 3 * self.memory + self.la:3 * self.memory  + self.la + self.poly]  # 后40行 为幅度值
        input_am1 = inputs[:, 3 * self.memory + self.la: inputs.shape[-1]]  # 后40行 为幅度值

        output2 = self.complexdenselayer(input_am1)
        output = output1 + output2
        return output
        # out = tf.concat((output2, output), axis=1)
        # return out # 这种输出方式同时输出多个模块的输出，使用时还需要修改Dotaloader中的output_type 因为很少使用，所以用手动注释代替if判断

class RVRNN(tf.keras.Model):
    def __init__(self,unit,batch_size=32, seq_length=40):
        super().__init__()
        # self.seq_length = seq_length
        # self.batch_size = batch_size
        # self.cell = tf.keras.layers.LSTMCell(units=unit)
        self.lstm = tf.keras.layers.LSTM(units=unit)
        self.dense = tf.keras.layers.Dense(units=16)
        self.dense1 = tf.keras.layers.Dense(units=2)

    def call(self, inputs):
        # batch_s = inputs.shape[0]
        # state = self.cell.get_initial_state(batch_size=batch_s, dtype=tf.float32)   # 获得 RNN 的初始状态
        # for t in range(self.seq_length):
        #     output, state = self.cell(inputs[:, t, :], state)   # 通过当前输入和前一时刻的状态，得到输出和当前时刻的状态
        output = self.lstm(inputs)
        y = self.dense(output)
        y = self.dense1(y)
        return y

    def predict(self, inputs, temperature=1.):
        y = self(inputs)  # 调用训练好的RNN模型，预测下一个字符的概率分布
        return y

class RVTDCNN(tf.keras.Model): # m=5, m=7, tanh-relu, relu-relu,
    def __init__(self,unit):
        super().__init__()
        self.filters = 2
        kernel_size = [5,4]
        self.conv1 = tf.keras.layers.Conv2D(
            filters = self.filters,  # 卷积层神经元（卷积核）数目
            kernel_size = kernel_size,  # 感受野大小
            padding='same',  # padding策略（valid 或 same）
            activation=tf.nn.tanh  # 激活函数
        )
        self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 5 * self.filters,))
        self.dense = tf.keras.layers.Dense(units=unit, activation=tf.nn.tanh)
        self.dense1 = tf.keras.layers.Dense(units=unit, activation=tf.nn.tanh)
        self.dense2 = tf.keras.layers.Dense(units=2)
    def call(self, inputs):
        x = self.conv1(inputs)  # [batch_size, 3, 3, 2]
        x = self.flatten(x)  # [batch_size, 3 * 3 * 2]
        x = self.dense(x)  # [batch_size, unit]
        x = self.dense1(x)  # [batch_size, unit]
        output = self.dense2(x)  # [batch_size, 2]
        return output

'''2D-模型 Note:部分模型通过输入label_dim转变为动态模型'''

class VDTDNN_2D(tf.keras.Model):
    def __init__(self,unit,mem = 3, label_dim = 0):
        super().__init__()
        self.unit = unit
        self.memory = mem
        self.la = label_dim
        # self.dense = tf.keras.layers.Dense(units=10, activation=tf.nn.relu)
        self.dense = tf.keras.layers.Dense(units=self.unit, activation=tf.nn.relu)
        self.dense1 = tf.keras.layers.Dense(units=self.unit, activation=tf.nn.relu)
        self.groupwiselayer = GroupwiseLayer_new(units=4, mem=2 * self.memory)
        # self.groupwiselayer = GroupwiseLayer_6(units=4)

    def call(self, inputs):  # [batch_size, 28, 28, 1]

        input_am1 = inputs[:, 0:self.memory]  # 前3行 为幅度值
        input_am2 = inputs[:, 3 * self.memory: 4 * self.memory]  # 前3行 为幅度值
        input_label = inputs[:, 6 * self.memory: 6 * self.memory + self.la]  # 前3行 为幅度值
        input_am = tf.concat((input_am1, input_am2, input_label), axis=1)

        input_phase1 = inputs[:, self.memory: 3 * self.memory]  # 之间6行 为相位值的cos和sin值
        input_phase2 = inputs[:, 4 * self.memory: 6 * self.memory]  # 之间6行 为相位值的cos和sin值
        # input_phase = tf.concat((input_phase1, input_phase2), axis=1)

        input_am0 = inputs[:, 6 * self.memory + self.la:inputs.shape[-1]]  # 后40行 为幅度值
        # input_am0 = inputs[:, 18:270]  # 后40行 为幅度值
        x = self.dense(input_am)
        x = self.dense1(x)

        per_neuron = self.unit // (2 * self.memory)
        for i in range(2 * self.memory):
            x_temp = x[:, i * per_neuron: (i + 1) * per_neuron]  # 5个相位恢复网络的输入
            x_temp = tf.concat((x_temp, input_am[:, i: i + 1]), axis=1)  # 附加线性项
            if i == 0:
                x1 = x_temp
                # print(x.shape)
            else:
                x1 = tf.concat((x1, x_temp), axis=1)
        x = self.groupwiselayer(x1)  # [batch_size, 10]

        output = uf.PhaseRecovery_new(x, input_phase1, self.memory, 2, input_phase2)

        return output

class PANN_2D(tf.keras.Model):
    def __init__(self, unit, mem = 3, label_dim = 0, poly = 40, out = 2, n3 = 1): #out 3 输出模式or not
        super().__init__()
        self.memory = mem
        self.la = label_dim
        self.unit = unit[1]
        self.out = out
        self.n3 = n3
        # self.dense = tf.keras.layers.Dense(units=unit, activation=tf.nn.relu)
        self.dense = My_dense(units=unit[0],mem=self.memory)
        self.dense1 = My_dense(units=unit[1],mem=self.memory)
        # self.dense2 = My_dense(units=unit[2], mem=self.memory)
        self.groupwiselayer = GroupwiseLayer_new(units=4, mem=2*self.memory)
        self.complexdenselayer = Complexdenselayer_dual()

    def call(self, inputs, select = 0):  # [batch_size, 28, 28, 1]
        input_am1 = inputs[:, 0:self.memory]  # 前3行 为幅度值
        input_am2 = inputs[:, 3 * self.memory: 4 * self.memory]  # 前3行 为幅度值
        input_label = inputs[:, 6 * self.memory: 6 * self.memory + self.la]  # 前3行 为幅度值
        input_am = tf.concat((input_am1, input_am2, input_label), axis=1)
        input_phase1 = inputs[:, self.memory: 3 * self.memory]  # 之间6行 为相位值的cos和sin值
        input_phase2 = inputs[:, 4 * self.memory: 6 * self.memory]  # 之间6行 为相位值的cos和sin值
        # input_phase = tf.concat((input_phase1, input_phase2), axis=1)
        input_am0 = inputs[:, 6 * self.memory + self.la:inputs.shape[-1]]  # 后40行 为幅度值

        if (self.out >= 1):

            x = self.dense(input_am)
            x = self.dense1(x)
            # x = self.dense2(x)
            per_neuron = self.unit // (2*self.memory)
            for i in range(2*self.memory):
                x_temp = x[:, i * per_neuron: (i + 1) * per_neuron]  # 5个相位恢复网络的输入
                x_temp = tf.concat((x_temp, input_am[:, i: i + 1]), axis=1)  # 附加线性项
                if i == 0:
                    x1 = x_temp
                        # print(x.shape)
                else:
                    x1 = tf.concat((x1, x_temp), axis=1)
            # x1 = x
            x = self.groupwiselayer(x1)  # [batch_size, 10]

            output1 = uf.PhaseRecovery_new(x, input_phase1, self.memory, 2, input_phase2)
            output2 = self.complexdenselayer(input_am0)
            output = output1 + output2

        if (self.out == 1 or self.out == 3): # 1 or 3输出模式，单双带混合

            x = self.dense(input_am, out = self.out)
            x = self.dense1(x, out=self.out, n3=self.n3)
            # x = self.dense2(x, out=self.out, n3=self.n3)
            per_neuron = self.unit // (2 * self.memory)

            for i in range(2 * self.memory):
                x_temp = x[:, i * per_neuron: i * per_neuron + self.n3]  # 取前n3个神经元进行groupwise
                x_temp = tf.concat((x_temp, input_am[:, i: i + 1]), axis=1)  # 附加线性项
                if i == 0:
                    x1 = x_temp
                    # print(x.shape)
                else:
                    x1 = tf.concat((x1, x_temp), axis=1)
            x = self.groupwiselayer(x1, out = self.out, n3 = self.n3)  # [batch_size, 10]

            output1 = uf.PhaseRecovery_new(x, input_phase1, self.memory)
            output2 = self.complexdenselayer(input_am0, out = 3)
            output3 = output1 + output2
            if (self.out == 1):
                output = output3
            elif (self.out == 3):
                output = tf.concat((output, output3), axis=1)
            # print(output.shape)
        # if (select == 1):
        #     output = np.array(output1)
        # elif (select == 2):
        #     output = np.array(output2)
        # elif (select == 3):
        #     output = np.array(output)
        return output

    def getoutput1(self, inputs):  # [batch_size, 28, 28, 1]
        # print(inputs.dtype)
        # print(inputs.shape)
        input_am1 = inputs[:, 0:self.memory]  # 前3行 为幅度值
        input_am2 = inputs[:, 3 * self.memory: 4 * self.memory]  # 前3行 为幅度值
        input_label = inputs[:, 6 * self.memory: 6 * self.memory+self.la]  # 前3行 为幅度值
        input_am = tf.concat((input_am1, input_am2, input_label), axis=1)

        input_phase1 = inputs[:, self.memory: 3 * self.memory]  # 之间6行 为相位值的cos和sin值
        input_phase2 = inputs[:, 4 * self.memory: 6 * self.memory]  # 之间6行 为相位值的cos和sin值
        #input_phase = tf.concat((input_phase1, input_phase2), axis=1)

        input_am0 = inputs[:, 6 * self.memory+self.la:inputs.shape[-1]]  # 后40行 为幅度值
        #input_am0 = inputs[:, 18:270]  # 后40行 为幅度值
        x = self.dense(input_am)
        x = self.dense1(x)
        # input_am = tf.cast(input_am, tf.float32)
        per_neuron = self.unit // (2*self.memory)
        for i in range(2*self.memory):
            x_temp = x[:, i * per_neuron: (i + 1) * per_neuron]  # 5个相位恢复网络的输入
            x_temp = tf.concat((x_temp, input_am[:, i: i + 1]), axis=1)  # 附加线性项
            if i == 0:
                x1 = x_temp
                # print(x.shape)
            else:
                x1 = tf.concat((x1, x_temp), axis=1)
        x = self.groupwiselayer(x1)  # [batch_size, 10]

        output1 = uf.PhaseRecovery_new(x, input_phase1, self.memory, 2, input_phase2)
        output2 = self.complexdenselayer(input_am0)
        output = output1 + output2
        # print(output1.shape)
        # print(output2.shape)
        # print(output.shape)
        return output1

    def getoutput2(self, inputs):  # [batch_size, 28, 28, 1]
        # print(inputs.dtype)
        # print(inputs.shape)
        input_am1 = inputs[:, 0:self.memory]  # 前3行 为幅度值
        input_am2 = inputs[:, 3 * self.memory: 4 * self.memory]  # 前3行 为幅度值
        input_label = inputs[:, 6 * self.memory: 6 * self.memory+self.la]  # 前3行 为幅度值
        input_am = tf.concat((input_am1, input_am2, input_label), axis=1)

        input_phase1 = inputs[:, self.memory: 3 * self.memory]  # 之间6行 为相位值的cos和sin值
        input_phase2 = inputs[:, 4 * self.memory: 6 * self.memory]  # 之间6行 为相位值的cos和sin值
        #input_phase = tf.concat((input_phase1, input_phase2), axis=1)

        input_am0 = inputs[:, 6 * self.memory+self.la:inputs.shape[-1]]  # 后40行 为幅度值
        #input_am0 = inputs[:, 18:270]  # 后40行 为幅度值
        x = self.dense(input_am)
        x = self.dense1(x)
        input_am = tf.cast(input_am, tf.float32)
        per_neuron = self.unit // (2*self.memory)
        for i in range(2*self.memory):
            x_temp = x[:, i * per_neuron: (i + 1) * per_neuron]  # 5个相位恢复网络的输入
            x_temp = tf.concat((x_temp, input_am[:, i: i + 1]), axis=1)  # 附加线性项
            if i == 0:
                x1 = x_temp
                # print(x.shape)
            else:
                x1 = tf.concat((x1, x_temp), axis=1)
        x = self.groupwiselayer(x1)  # [batch_size, 10]

        output1 = uf.PhaseRecovery_new(x, input_phase1, self.memory, 2, input_phase2)
        output2 = self.complexdenselayer(input_am0)
        output = output1 + output2
        return output2

'''动态模型-已整合到正常模型中'''


'''BNN - TO BE Continued'''

class RVTDBNN(tf.keras.Model):
    def __init__(self,num_unit,H = 1.):
        super().__init__()
        # self.dense1 = BinaryDense(num_unit, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=false)
        self.dense = BinaryDense(num_unit, H=H, use_bias=False)#, activation=binary_tanh_op)
        #self.act1=binary_tanh
        # self.bm = tf.keras.layers.BatchNormalization(trainable=True,momentum=0.9)
        # self.re = tf.keras.layers.ReLU()
        self.dense2 = BinaryDense(num_unit, H=H, use_bias=False)#, activation=binary_tanh_op)
        # self.dense3 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        # self.dense4 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense20 = BinaryDense(2, H=H, use_bias=False)

    def call(self, inputs):
        x = self.dense(inputs)  # [batch_size, 15]
        #x = self.act1(x)
        # x = self.bm(x,training=True)
        # x = self.re(x)
        x = self.dense2(x)  # [batch_size, 15]
        # x = self.dense3(x)
        # x = self.dense4(x)
        # x = self.dense3(x)
        x = self.dense20(x)  # [batch_size, 10]
        output = x
        return output

















