import torch
import torch.nn as nn
from torch.autograd import Variable
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 


plt.rcParams['font.sans-serif']=['SimHei']  # 使得plot可以显示中文
plt.rcParams['axes.unicode_minus'] = False

######################################################
#########           数据加载和处理            #########
######################################################

# ------------------------数据读入---------------------
fileData = pd.read_csv('.\\data\\melt_data.csv', dtype=np.float32, header=None)  # 读入数据
wholeData = fileData.values


# ----------------------定义训练数据范围 ----------------------
trx_start = 0
trx_datastep = 2000

# ----------------------定义测试数据范围 ----------------------
tex_start = 2000
tex_datastep = 500

# ----------------------设置训练参数-------------------
EPOCH = 1000      # 循环次数
BATCH_SIZE = 10
LR = 0.001        # 学习速率

# ----------------------数据格式转换-------------------
trX, teX = wholeData[trx_start:trx_start + trx_datastep , :8], wholeData[tex_start: tex_start + tex_datastep, :8]

Xtrain = trX.astype(np.float32)
Xtrain = torch.Tensor(Xtrain)

Xtest = teX.astype(np.float32)
Xtest = torch.Tensor(Xtest)

Xtest_row, Xtest_list = Xtest.shape


######################################################
#########      自编码器模型与网络构建         #########
######################################################

# ----------------------网络构建-----------------------
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(       # 编码网络层
            nn.Linear(Xtest_list, 16), 
            nn.Tanh(),
            nn.Linear(16, 32), 
            nn.Tanh(),
            nn.Linear(32, 64), 
            nn.Tanh(),
            nn.Linear(64, 128),
        )
 
        self.decoder = nn.Sequential(       # 解码网络层
            nn.Linear(128,64), 
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32,16),
            nn.Tanh(),
            nn.Linear(16, Xtest_list),
            nn.Sigmoid()
        )
 
    def forward(self, x):                   # 前向传递层
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = AutoEncoder()                 # 模型实例化
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)   # 优化器
loss_func = nn.BCELoss()                    # 均方误差损失

summary(autoencoder,(700, 13),device="cpu") # 显示网络信息
######################################################
#########  函数：     添加高斯噪声            #########
######################################################
def add_noise(data):
    noise = torch.randn(Xtrain.size()) * 0.2
    noise_data = data + noise
    return Variable(noise_data)


######################################################
#########  函数：     l1与l2正则项           #########
######################################################
def Regularization(reg_model, reg_type=1, reg_lambda =0.01):
    if reg_type == 1 :
        L1_reg = torch.tensor(0., requires_grad=True)
        for name, param in reg_model.named_parameters():
            if 'weight' in name:
                L1_reg = L1_reg + reg_lambda*torch.norm(param, 1)
        return L1_reg
    elif reg_type == 2 :
        L2_reg = torch.tensor(0., requires_grad=True)
        for name, param in reg_model.named_parameters():
            if 'weight' in name:
                L2_reg = L2_reg + reg_lambda*torch.norm(param, 2)
        return L2_reg

######################################################
#########          自编码器模型训练           #########
######################################################


loss_history = []                                   #训练集损失函数记录数组
loss_history_val = []                               #测试集损失函数记录数组
for epoch in range(1,EPOCH+1):
    Xtrain_noise = add_noise(Xtrain)
    encoded, decoded = autoencoder(Xtrain_noise)    # 训练集经过网络并返回编码层和解码层数据
    encoded_val, decoded_val = autoencoder(Xtest)   # 测试集经过网络并返回编码层和解码层数据
    
    # loss = loss_func(decoded, Xtrain) + Regularization(autoencoder, 1, 0.0001) + Regularization(autoencoder, 2, 0.001)             # 计算训练集损失(均方误差损失)
    # loss_val = loss_func(decoded_val, Xtest) + Regularization(autoencoder, 1, 0.0001) + Regularization(autoencoder, 2, 0.001)      # 计算测试集损失(均方误差损失)

    loss = loss_func(decoded, Xtrain)               # 计算训练集损失(均方误差损失)
    loss_val = loss_func(decoded_val, Xtest)        # 计算测试集损失(均方误差损失)

    optimizer.zero_grad()                           # 每次迭代清空上一次的梯度
    loss.backward()                                 # 反向传播
    optimizer.step()                                # 更新梯度

    loss_history.append(loss.data.numpy())          # 保存训练集损失数据
    loss_history_val.append(loss_val.data.numpy())  # 保存测试集损失数据

    if epoch % 10 == 0:                             # 训练过程打印
        print('Epoch [{}/{}] | train loss:{:.8f} | test loss: {:.8f}' .format(epoch, EPOCH, loss.data.numpy(), loss_val.data.numpy()  ) )


torch.save(autoencoder.state_dict(), 'E:\PycharmCode\Autoencoder_DAE_tanh(Ascending dimension)\model\\autoencoder_DAE_tanh_1000.pkl') #模型保存


######################################################
#########               数据打印              #########
######################################################
font_title = {'family' : 'Times New Roman', 'weight': 'normal', 'size': 30}
font_lable = {'family' : 'Times New Roman', 'weight': 'normal', 'size': 16}

_, decoded_train = autoencoder(Xtrain)              #解码训练集数据
_, decoded_test = autoencoder(Xtest)                #解码测试集数据

test_show   = Xtest.cpu().detach().numpy()          #测试集原始数据
decode_show = decoded_test.cpu().detach().numpy()   #测试集解码数据

decoded_train = decoded_train.cpu().detach().numpy()


# ---------------------损失函数可视化-------------------
mumber_plt = 1
plt.figure(mumber_plt)
plt.plot(loss_history)
plt.plot(loss_history_val, color = 'red',linestyle = '-.')
plt.legend(labels= ["loss_train","loss_test"], loc = 'upper right')
plt.xlabel('S t e a p', font_lable)
plt.ylabel('A m p l i t u d e', font_lable)
plt.title("LossFunction", font_title)

# ---------------------原始数据可视化-------------------
mumber_plt = mumber_plt + 1
plt.figure(mumber_plt)
plt.subplots_adjust(hspace=0.4)
plt.subplot(7,2,1),  plt.plot(wholeData[0:trx_datastep, 0]),  plt.plot(decoded_train[:,0],  color = 'red', linewidth = 1.0, linestyle = '--'),
plt.subplot(7,2,2),  plt.plot(wholeData[0:trx_datastep, 1]),  plt.plot(decoded_train[:,1],  color = 'red', linewidth = 1.0, linestyle = '--'), 
plt.subplot(7,2,3),  plt.plot(wholeData[0:trx_datastep, 2]),  plt.plot(decoded_train[:,2],  color = 'red', linewidth = 1.0, linestyle = '--'),
plt.subplot(7,2,4),  plt.plot(wholeData[0:trx_datastep, 3]),  plt.plot(decoded_train[:,3],  color = 'red', linewidth = 1.0, linestyle = '--'),
plt.subplot(7,2,5),  plt.plot(wholeData[0:trx_datastep, 4]),  plt.plot(decoded_train[:,4],  color = 'red', linewidth = 1.0, linestyle = '--'),
plt.subplot(7,2,6),  plt.plot(wholeData[0:trx_datastep, 5]),  plt.plot(decoded_train[:,5],  color = 'red', linewidth = 1.0, linestyle = '--'),
plt.subplot(7,2,7),  plt.plot(wholeData[0:trx_datastep, 6]),  plt.plot(decoded_train[:,6],  color = 'red', linewidth = 1.0, linestyle = '--'),

plt.subplot(7,2,8),  plt.plot(wholeData[0:trx_datastep, 7]),  plt.plot(decoded_train[:,7],  color = 'red', linewidth = 1.0, linestyle = '--'),
plt.subplot(7,2,9),  plt.plot(wholeData[0:trx_datastep, 8]),  plt.plot(decoded_train[:,8],  color = 'red', linewidth = 1.0, linestyle = '--'),
plt.subplot(7,2,10), plt.plot(wholeData[0:trx_datastep, 9]),  plt.plot(decoded_train[:,9],  color = 'red', linewidth = 1.0, linestyle = '--'),
plt.subplot(7,2,11), plt.plot(wholeData[0:trx_datastep, 10]), plt.plot(decoded_train[:,10], color = 'red', linewidth = 1.0, linestyle = '--'),
plt.subplot(7,2,12), plt.plot(wholeData[0:trx_datastep, 11]), plt.plot(decoded_train[:,11], color = 'red', linewidth = 1.0, linestyle = '--'),
plt.subplot(7,2,13), plt.plot(wholeData[0:trx_datastep, 12]), plt.plot(decoded_train[:,12], color = 'red', linewidth = 1.0, linestyle = '--'),
plt.suptitle('training data and decoding data',)
plt.tight_layout()


# ------------pytorch测试数据与解码数据对比---------------
mumber_plt = mumber_plt + 1
plt.figure(mumber_plt)
plt.subplot(7,2,1),  plt.plot(test_show[:,0]),  plt.plot(decode_show[:,0],  color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据1")
plt.subplot(7,2,2),  plt.plot(test_show[:,1]),  plt.plot(decode_show[:,1],  color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据2")
plt.subplot(7,2,3),  plt.plot(test_show[:,2]),  plt.plot(decode_show[:,2],  color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据3")
plt.subplot(7,2,4),  plt.plot(test_show[:,3]),  plt.plot(decode_show[:,3],  color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据4")
plt.subplot(7,2,5),  plt.plot(test_show[:,4]),  plt.plot(decode_show[:,4],  color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据5")
plt.subplot(7,2,6),  plt.plot(test_show[:,5]),  plt.plot(decode_show[:,5],  color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据6")
plt.subplot(7,2,7),  plt.plot(test_show[:,6]),  plt.plot(decode_show[:,6],  color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据7")

plt.subplot(7,2,8),  plt.plot(test_show[:,7]),  plt.plot(decode_show[:,7],  color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据8")
plt.subplot(7,2,9),  plt.plot(test_show[:,8]),  plt.plot(decode_show[:,8],  color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据9")
plt.subplot(7,2,10), plt.plot(test_show[:,9]),  plt.plot(decode_show[:,9],  color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据10")
plt.subplot(7,2,11), plt.plot(test_show[:,10]), plt.plot(decode_show[:,10], color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据11")
plt.subplot(7,2,12), plt.plot(test_show[:,11]), plt.plot(decode_show[:,11], color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据12")
plt.subplot(7,2,13), plt.plot(test_show[:,12]), plt.plot(decode_show[:,12], color = 'red', linewidth = 1.0, linestyle = '--')
plt.title("数据13")


plt.show()

