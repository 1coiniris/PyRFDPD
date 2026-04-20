import tensorflow as tf
import numpy as np

#  自定义层
class My_dense(tf.keras.layers.Layer):
    def __init__(self, units, mem = 3):
        super().__init__()
        self.units = units
        self.memory = mem

    def build(self, input_shape):     # 这里 input_shape 是第一次运行call()时参数inputs的形状

        self.shape = input_shape[-1]
        self.w = self.add_weight(name='w',
            shape=[input_shape[-1], self.units], initializer=tf.compat.v1.keras.initializers.glorot_uniform)
        self.b = self.add_weight(name='b',
            shape=[self.units], initializer=tf.zeros_initializer())

    def call(self, inputs, out = 2, n3 = 0):
        if (out == 2):
            y = tf.matmul(inputs, self.w) + self.b

        elif (out == 1 or out == 3): # out = 3时，仅第一个带的输入和信号label进入全连接层
            if (n3 == 0) :
                y1 = tf.matmul(inputs[:, 0:self.memory], self.w[0:self.memory,:])
                y2 = tf.matmul(inputs[:, 2*self.memory:self.shape], self.w[2*self.memory:self.shape,:])
                y = y1 + y2 + self.b
            else:
                y1 = tf.matmul(inputs[:, 0:self.memory * n3], self.w[0:self.memory * n3, :])
                y = y1 + self.b

        y = tf.nn.leaky_relu(y)

        return y

class GroupwiseLayer_new(tf.keras.layers.Layer):
    def __init__(self, units, mem):
        super().__init__()
        self.units = units
        self.memory = mem

    def build(self, input_shape):  # 这里 input_shape 是第一次运行call()时参数inputs的形状
        units_1 = self.units
        self.input_1 = input_shape[-1]//self.memory#         改
        self.w = self.add_weight(name='w1',
                                  shape=[input_shape[-1], units_1], initializer=tf.zeros_initializer)
        # self.b = self.add_variable(name='b',
        #     shape=[self.units], initializer=tf.zeros_initializer())

    def call(self, inputs, out = 2, n3 = 1):
        if (out == 2):
            y = tf.matmul(inputs[:, 0 :  self.input_1], self.w[ 0 : self.input_1, :])
            for i in range(1, self.memory):
                y_temp = tf.matmul(inputs[:, i * self.input_1 : (i + 1) * self.input_1], self.w[ i * self.input_1 : (i + 1) * self.input_1, :])
                y = tf.concat((y, y_temp), axis=1)
        elif (out == 1 or out == 3):
            total_n = n3+1
            y = tf.matmul(inputs[:, 0:n3], self.w[0:n3, :]) + tf.matmul(inputs[:, n3:n3+1], self.w[self.input_1-1:self.input_1, :])
            for i in range(1, self.memory):
                y_temp = tf.matmul(inputs[:, i * total_n: i * total_n + n3],self.w[i * self.input_1: i * self.input_1 + n3, :]) \
                         + tf.matmul(inputs[:, i * total_n + n3: i * total_n + n3 + 1],self.w[(i + 1) * self.input_1 - 1: (i + 1) * self.input_1, :])
                y = tf.concat((y, y_temp), axis=1)
        return y

class Complexdenselayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # self.units = units

    def build(self, input_shape):  # 这里 input_shape 是第一次运行call()时参数inputs的形状

        self.input_shapediv2 = input_shape[-1]
        self.input_shapediv4 = input_shape[-1] // 2
        self.w = self.add_weight(name='w',
                                 shape=[self.input_shapediv2, 1], initializer=tf.zeros_initializer)

        # self.b = self.add_variable(name='b',
        #     shape=[self.units], initializer=tf.zeros_initializer())

    def call(self, inputs):
        y_pred_1 = tf.matmul(inputs, self.w)
        w2 = self.w[0:  self.input_shapediv4, 0:1]
        w3 = -self.w[self.input_shapediv4: self.input_shapediv2, 0:1]
        w4 = tf.concat((w3, w2), axis=0)          #为了符合复数的规范，Q的权重有I的权重重新排列得到
        y_pred_2 = tf.matmul(inputs, w4)
        y_pred = tf.concat((y_pred_1, y_pred_2), axis=1)

        return y_pred

class Complexdenselayer_dual(tf.keras.layers.Layer): # to be detleted
    def __init__(self):
        super().__init__()
        # self.units = units

    def build(self, input_shape):  # 这里 input_shape 是第一次运行call()时参数inputs的形状

        self.input_shapediv2 = input_shape[-1] // 2
        self.input_shapediv4 = input_shape[-1] // 4
        self.w = self.add_weight(name='w',
                                 shape=[self.input_shapediv2, 1], initializer=tf.zeros_initializer)

        self.ww = self.add_weight(name='ww',
                                 shape=[self.input_shapediv2, 1], initializer=tf.zeros_initializer())

        # self.b = self.add_variable(name='b',
        #     shape=[self.units], initializer=tf.zeros_initializer())

    def call(self, inputs, out = 2):
        input_0 = inputs[:, 0: 1 * self.input_shapediv2]
        input_1 = inputs[:, 1 * self.input_shapediv2: 2 * self.input_shapediv2]

        if (out==2):
            y_pred_1 = tf.matmul(input_0, self.w)
            w2 = self.w[0:  self.input_shapediv4, 0:1]
            w3 = -self.w[self.input_shapediv4: self.input_shapediv2, 0:1]
            w4 = tf.concat((w3, w2), axis=0)
            y_pred_2 = tf.matmul(input_0, w4)

            y_pred_3 = tf.matmul(input_1, self.ww)
            ww2 = self.ww[0:  self.input_shapediv4, 0:1]
            ww3 = -self.ww[self.input_shapediv4: self.input_shapediv2, 0:1]
            ww4 = tf.concat((ww3, ww2), axis=0)
            y_pred_4 = tf.matmul(input_1, ww4)
            y_pred = tf.concat((y_pred_1, y_pred_2, y_pred_3, y_pred_4), axis=1)

            # k1 = 13
            # k2 = 30
            #
            # y_pred_1 = tf.matmul(input_0[:, k1:k2], self.w[k1:k2, :]) + tf.matmul(input_0[:, k1+30:k2+30], self.w[k1+30:k2+30, :])
            # w2 = self.w[0:  self.input_shapediv4, 0:1]
            # w3 = -self.w[self.input_shapediv4: self.input_shapediv2, 0:1]
            # w4 = tf.concat((w3, w2), axis=0)
            # y_pred_2 = tf.matmul(input_0[:, k1:k2], w4[k1:k2, :]) + tf.matmul(input_0[:, k1+30:k2+30], w4[k1+30:k2+30, :])
            #
            # y_pred_3 = tf.matmul(input_1[:, k1:k2], self.ww[k1:k2, :]) + tf.matmul(input_1[:, k1+30:k2+30], self.ww[k1+30:k2+30, :])
            # ww2 = self.ww[0:  self.input_shapediv4, 0:1]
            # ww3 = -self.ww[self.input_shapediv4: self.input_shapediv2, 0:1]
            # ww4 = tf.concat((ww3, ww2), axis=0)
            # y_pred_4 = tf.matmul(input_1[:, k1:k2], ww4[k1:k2, :]) + tf.matmul(input_1[:, k1+30:k2+30], ww4[k1+30:k2+30, :])
            #
            # y_pred = tf.concat((y_pred_1, y_pred_2, y_pred_3, y_pred_4), axis=1)

        elif (out==1 or out == 3):
            # ind = tf.make_ndarray([1, 2, 4, 7])
            # index = tf.make_ndarray([ind, ind + 10, ind + 20, ind + 30, ind + 40, ind + 50])
            # index = tf.reshape(index, 24)
            # index = tf.convert_to_tensor([1,2,4,7,11,12,14,17,21,22,24,27,31,32,34,37,41,42,44,47,51,52,54,57])
            # index = [1:4]

            y_pred_1 = tf.matmul(input_0[:,0:12], self.w[0:12,:]) + tf.matmul(input_0[:,30:42], self.w[30:42,:])
            w2 = self.w[0:  self.input_shapediv4, 0:1]
            w3 = -self.w[self.input_shapediv4: self.input_shapediv2, 0:1]
            w4 = tf.concat((w3, w2), axis=0)
            y_pred_2 = tf.matmul(input_0[:,0:12], w4[0:12,:]) + tf.matmul(input_0[:,30:42], w4[30:42,:])
            y_pred = tf.concat((y_pred_1, y_pred_2), axis=1)

        return y_pred # to be deleted


