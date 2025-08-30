import datetime
import tensorflow as tf
import numpy as np
import NeuralNetwork as NN
from scipy.io import loadmat
from scipy.io import savemat
import UserFunction as uf
'''
model_type:模型类别
pa_input, pa_output
m, nk: 记忆深度+1，非线性深度（部分模型）
train_num, test_num: 返回的训练数据和测试数据数量
input_label： 输入模型的标签，仅适用于动态模型
seq_length： 序列长度，仅适用于RNN
output_type：输出类别，正常不需要，多输出设置为 ‘m-out’ ，动态模型设置为 ‘dynamic’
out: 输出个数，例：单双带融合模型out设置为3
'''

class My_dataloader():
    def __init__(self,model_type,pa_input,pa_output,memory,train_num, test_num, env_order = (), input_label = 0, nk = 4, seq_length = 40, output_type = 0, out = 2): # 类的构造形参
        if model_type == 'rvtdnn':
            self.train_data, self.test_data, self.train_label, self.test_label = uf.RVTDNN_IO(pa_input, pa_output, memory, train_num, test_num, env_order, input_label, output_type)
        elif model_type == 'vdtdnn':
            self.train_data, self.test_data, self.train_label, self.test_label = uf.VDTDNN_IO(pa_input, pa_output,memory, train_num,test_num, input_label,output_type)
        elif model_type == 'pann':
            self.train_data, self.test_data, self.train_label, self.test_label = uf.PANN_IO(pa_input, pa_output, memory, nk, train_num, test_num, input_label, output_type)
        elif model_type == 'rvrnn':
            self.train_data, self.test_data, self.train_label, self.test_label = uf.RVRNN_IO(pa_input, pa_output, memory, seq_length, train_num, test_num)
        elif model_type == 'rvtdcnn':
            self.train_data, self.test_data, self.train_label, self.test_label = uf.RVTDCNN_IO(pa_input, pa_output, memory, train_num,test_num)
        elif model_type == 'vdtdnn_2d':
            self.train_data, self.test_data, self.train_label, self.test_label = uf.VDTDNN_DUAL_IO(pa_input[:,0:1], pa_input[:,1:2], pa_output[:,0:1], pa_output[:,1:2], memory,train_num, test_num, input_label)
        elif model_type == 'pann_2d':
            self.train_data, self.test_data, self.train_label, self.test_label = uf.PANN_DUAL_IO(pa_input, pa_output, memory,nk,train_num, test_num, input_label, out = out)

        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

        print(self.train_data.shape)
        print(self.train_label.shape)
        print(self.test_data.shape)
        print(self.test_label.shape)

    def get_batch(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_label[index]

