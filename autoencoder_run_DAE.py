import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd 


plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

######################################################
#########           数据加载和处理            #########
######################################################

# ------------------------数据读入---------------------
fileData = pd.read_csv(r'./data/meltadata.txt', sep="\t", dtype=np.float32, header=None)
wholeData = fileData.values

# ----------------------定义训练数据范围 ----------------------
trx_start = 0
trx_datastep = 700

# ----------------------定义测试数据范围 ----------------------
tex_start = 0
tex_datastep = 1050

# ----------------------定义学习速率 ----------------------
LR = 0.001

# ----------------------定义正则化系数 ----------------------
lambda1 = 0.02

# ----------------------数据格式转换-------------------
trX, teX = wholeData[trx_start:trx_start + trx_datastep , :13], wholeData[tex_start: tex_start + tex_datastep, :13]

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
        self.encoder = nn.Sequential(                           # 编码网络层
            nn.Linear(Xtest_list, 16), 
            nn.Tanh(),
            nn.Linear(16, 32), 
            nn.Tanh(),
            nn.Linear(32, 64), 
            nn.Tanh(),
            nn.Linear(64, 128),

        )
 
        self.decoder = nn.Sequential(                           # 解码网络层
            nn.Linear(128,64), 
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32,16),
            nn.Tanh(),
            nn.Linear(16, Xtest_list),
            nn.Sigmoid()
        )

    def forward(self, x):                                       # 前向传递层
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = AutoEncoder()                                     # 模型实例化
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)   # 优化器
loss_func = nn.BCELoss()                                        # 交叉熵损失
loss_reconstruction = nn.MSELoss()                              # 误差重构中的损失函数

autoencoder.load_state_dict(torch.load('E:\PycharmCode\Autoencoder_DAE_tanh(Ascending dimension)\model\\autoencoder_DAE_tanh_1000.pkl'))   # 加载模型


######################################################
######### 函数：      计算置信区间函数        #########
######################################################
def confidence(data, c=0.95 ): 
    # ddof取值为1是因为在统计学中样本的标准偏差除的是(N-1)而不是N，统计学中的标准偏差除的是N
    # SciPy中的std计算默认是采用统计学中标准差的计算方式
    mean, std = data.mean(), data.std(ddof=1) # 计算均值和标准差
    # print(mean, std)
    # 计算置信区间
    # 默认0.95的置信水平
    lower, higher = stats.norm.interval(c, loc=mean, scale=std)
    print(lower, higher)
    return lower, higher

######################################################
######### 函数：    误差重构（梯度下降）       #########
######################################################
F=[]
def restructure(Xtest):
    _, decoded_teX_update = autoencoder(Xtest)     # 测试集经过网络并返回编码层和解码层数据

    flag = 0                                                        # 定义标签 判断是否是第一次进入循环
    echo = 0
    loss_val_history = []
    loss_val = loss_reconstruction(decoded_teX_update, Xtest)

    while echo-20< 0 or loss_val.data.numpy() > 0.015:
        if flag == 0 :
            f = Variable(torch.zeros(1,13), requires_grad=True)
            Xtest_update = Xtest

        Xtest_update = Xtest - torch.matmul(torch.ones(tex_datastep, 1), f)  # 更新重构原始数据  x = x - f
        f_history = f

        l1_regularization = lambda1 * torch.norm(f, 1)  # 求重构误差f的L1范数
        _, decoded_teX_update = autoencoder(Xtest_update)  # 求输入变量重构后的解码数据

        loss_val = loss_reconstruction(Xtest_update, decoded_teX_update)+ l1_regularization  # 在误差函数中加入f的L1正则项，使得f稀疏
        loss_val_history.append(loss_val.data.numpy())

        # 打印循环次数，偏差，损失函数
        # print('echo:| ', echo, 'F:\n', f_history.data.numpy(), '\nchange loss: %.8f' % loss_val.data.numpy(), "\n")

        weight1 = torch.ones(loss_val.size())  # 对误差loss_val构建权重为1的权重矩阵
        loss_val_backward_first = torch.autograd.grad(loss_val, f, weight1, create_graph= True)    # 误差函数loss_val对f求一阶导数
        
        f = f - 0.1*loss_val_backward_first[0]   # 梯度下降更新偏差

        flag = 1
        echo = echo + 1
        if echo >= 200:
            break
    return f_history  # 返回重构误差


######################################################
#########      误差重构求解（两种方法）       #########
######################################################

# 方法①（选取一段数据求解这段数据的整体重构误差）
# f_history = restructure(Xtest)


# 方法②（选取一段数据对每个数据点进行误差重构）
for i in range(Xtest_row):
    F.extend(restructure(Xtest[i,:]).data.numpy())
    print('Epoch [{}/{}]' .format(i, Xtest_row))
F = np.asarray(F)


######################################################
#########   故障检测(计算H2和SPE的置信区间)   #########
######################################################
encoded_trX, decoded_trX = autoencoder(Xtrain)                  # 训练集经过网络并返回编码层和解码层数据
encoded_teX, decoded_teX = autoencoder(Xtest)                   # 训练集经过网络并返回编码层和解码层数据

# ----------------------记录维度与创建数组-------------------
H2_trX_shape, _ = encoded_trX.shape
SPE_trX_shape, _ = decoded_trX.shape

H2_teX_shape, _ = encoded_teX.shape
SPE_teX_shape, _ = decoded_teX.shape


H2_trX_history = []
SPE_trX_history = []

H2_teX_history = []
SPE_teX_history = []

H2_rec_history = []
SPE_rec_history = []

# ----------------------计算H2统计量与其置信限------------------
# 公式：H2 = x.T*x
for i in range(H2_trX_shape):
    H2 = torch.matmul(encoded_trX[i, :].T, encoded_trX[i, :])
    H2_trX_history.append(H2.detach().numpy())
H2_trX_history = np.array(H2_trX_history)
l_H2_trX, h_H2_trX = confidence(H2_trX_history, 0.99)

for i in range(H2_teX_shape):
    H2 = torch.matmul(encoded_teX[i, :].T, encoded_teX[i, :])
    H2_teX_history.append(H2.detach().numpy())
H2_teX_history = np.array(H2_teX_history)
l_H2_teX, h_H2_teX = confidence(H2_teX_history, 0.99)

# np.savetxt(".\\SVDD\\SVDD\\data\\H2_teX_history.csv", H2_teX_history , delimiter=',')

# ----------------------计算SPE统计量与其置信限------------------
# 公式：SPE=(x-x').T*(x-x')
# for i in range(SPE_trX_shape):
#     SPE = torch.matmul((Xtrain-decoded_trX)[i, :].T, (Xtrain-decoded_trX)[i, :])
#     SPE_trX_history.append(SPE.detach().numpy())
# SPE_trX_history = np.array(SPE_trX_history)
# l_SPE_trX, h_SPE_trX = confidence(SPE_trX_history, 0.99)

# for i in range(SPE_teX_shape):
#     SPE = torch.matmul((Xtest-decoded_teX)[i, :].T, (Xtest-decoded_teX)[i, :])
#     SPE_teX_history.append(SPE.detach().numpy())
# SPE_teX_history = np.array(SPE_teX_history)
# l_SPE_teX, h_SPE_teX = confidence(SPE_teX_history, 0.99)

# ----------------------加入协方差(逆)------------------
# 公式：SPE=(x-x').T*E^(-1)*(x-x')
# for i in range(SPE_trX_shape):
#     x_trX = ((Xtrain-decoded_trX)[i, :].T).detach().numpy().reshape(13,1)
#     means = np.mean(x_trX,axis = 0)
#     mean_dataMat = x_trX  - means
#     cov_trX = 1/13*(np.dot(mean_dataMat,mean_dataMat.T))
#     cov_trX = np.linalg.inv(cov_trX)  #逆
#     E_trX = torch.from_numpy(cov_trX)
#     SPE_temp = torch.matmul((Xtrain-decoded_trX)[i, :].T, E_trX)
#     SPE = torch.matmul(SPE_temp,(Xtrain-decoded_trX)[i, :])
#     SPE_trX_history.append(SPE.detach().numpy())
# SPE_trX_history = np.array(SPE_trX_history)
# l_SPE_trX, h_SPE_trX = confidence(SPE_trX_history, 0.99)

# ----------------------加入协方差(伪逆)------------------
# 公式：SPE=(x-x').T*E^(-1)*(x-x')
# for i in range(SPE_teX_shape): 
#     x_teX = ((Xtest-decoded_teX)[i, :].T).detach().numpy().reshape(13,1)
#     means = np.mean(x_teX,axis = 0)
#     mean_dataMat = x_teX  - means
#     cov_teX = 1/13*(np.dot(mean_dataMat,mean_dataMat.T))
#     cov_teX = np.linalg.pinv(cov_teX) #伪逆
#     E_teX = torch.from_numpy(cov_teX)
#     SPE_temp = torch.matmul((Xtest-decoded_teX)[i, :].T, E_teX)
#     SPE = torch.matmul(SPE_temp,(Xtest-decoded_teX)[i, :])
#     SPE_teX_history.append(SPE.detach().numpy())
# SPE_teX_history = np.array(SPE_teX_history)
# l_SPE_teX, h_SPE_teX = confidence(SPE_teX_history, 0.99)

# ----------------------变量误差减去均值除以方差的平方和------------------
# 公式：统计量 = sum([((x-x')-men(x-x'))/E]^2)
for i in range(SPE_teX_shape): 
    x_teX = ((Xtest-decoded_teX)[i, :].T).detach().numpy().reshape(13,1)
    means = np.mean(x_teX,axis = 0)
    var = np.var(x_teX)
    mean_dataMat = x_teX  - means
    mean_dataMat = mean_dataMat/var
    SPE = np.sum(mean_dataMat**2)
    SPE_teX_history.append(SPE)
SPE_teX_history = np.array(SPE_teX_history)
l_SPE_teX, h_SPE_teX = confidence(SPE_teX_history, 0.99)

# np.savetxt(".\\SVDD\\SVDD\\data\\SPE_teX_history.csv", SPE_teX_history , delimiter=',')

# for i in range(H2_rec_shape):
#     H2 = torch.matmul(encoded_rec[i, :].T, encoded_rec[i, :])
#     H2_rec_history.append(H2.detach().numpy())
# H2_rec_history = np.array(H2_rec_history)

# for i in range(SPE_rec_shape):
#     SPE = torch.matmul((decoded_rec1-decoded_rec)[i, :].T, (decoded_rec1-decoded_rec)[i, :])
#     SPE_rec_history.append(SPE.detach().numpy())
# SPE_rec_history = np.array(SPE_rec_history)



######################################################
#########               数据打印              #########
######################################################
font_title = {'family' : 'Times New Roman', 'weight': 'normal', 'size': 13}
font_lable = {'family' : 'Times New Roman', 'weight': 'normal', 'size': 10}

encode_show_trX = encoded_trX.cpu().detach().numpy()        # 训练集编码数据
test_show = Xtest.cpu().detach().numpy()                    # 测试集原始数据
decode_show_teX = decoded_teX.cpu().detach().numpy()        # 测试集解码数据

# Xtest_update_show = Xtest_update.data.numpy()


# ---------------------损失函数可视化-------------------
mumber_plt = 1

# plt.figure(mumber_plt)
# plt.plot(loss_val_history)
# plt.title("Loss value in the process of error reconstruction")
# # plt.ylim(0, 0.2)


# # ------------pytorch测试数据与解码数据对比---------------
# mumber_plt = mumber_plt + 1
# plt.figure(mumber_plt)
# plt.subplot(7,2,1), plt.plot(test_show[:,0]), plt.plot(decoded_show_rec[:,0] , color = 'red', linewidth = 1.0, linestyle = '--')
# plt.title("数据0",),plt.ylim(0, 1)
# plt.subplot(7,2,2), plt.plot(test_show[:,1]), plt.plot(decoded_show_rec[:,1] , color = 'red', linewidth = 1.0, linestyle = '--')
# plt.title("数据1"),plt.ylim(0, 1)
# plt.subplot(7,2,3), plt.plot(test_show[:,2]), plt.plot(decoded_show_rec[:,2] , color = 'red', linewidth = 1.0, linestyle = '--')
# plt.title("数据2"),plt.ylim(0, 1)
# plt.subplot(7,2,4), plt.plot(test_show[:,3]), plt.plot(decoded_show_rec[:,3] , color = 'red', linewidth = 1.0, linestyle = '--')
# plt.title("数据3"),plt.ylim(0, 1)
# plt.subplot(7,2,5), plt.plot(test_show[:,4]), plt.plot(decoded_show_rec[:,4] , color = 'red', linewidth = 1.0, linestyle = '--')
# plt.title("数据4"),plt.ylim(0, 1)
# plt.subplot(7,2,6), plt.plot(test_show[:,5]), plt.plot(decoded_show_rec[:,5] , color = 'red', linewidth = 1.0, linestyle = '--'),
# plt.title("数据5"),plt.ylim(0, 1)
# plt.subplot(7,2,7), plt.plot(test_show[:,6]), plt.plot(decode_show_teX[:,6] , color = 'red', linewidth = 1.0, linestyle = '--')
# plt.title("数据6"),plt.ylim(0, 1)

# plt.subplot(7,2,8), plt.plot(test_show[:,7]), plt.plot(decoded_show_rec[:,7] , color = 'red', linewidth = 1.0, linestyle = '--')
# plt.title("数据7"),plt.ylim(0, 1)
# plt.subplot(7,2,9), plt.plot(test_show[:,8]), plt.plot(decoded_show_rec[:,8] , color = 'red', linewidth = 1.0, linestyle = '--')
# plt.title("数据8"),plt.ylim(0, 1)
# plt.subplot(7,2,10), plt.plot(test_show[:,9]), plt.plot(decoded_show_rec[:,9] , color = 'red', linewidth = 1.0, linestyle = '--')
# plt.title("数据9"),plt.ylim(0, 1)
# plt.subplot(7,2,11), plt.plot(test_show[:,10]), plt.plot(decoded_show_rec[:,10] , color = 'red', linewidth = 1.0, linestyle = '--')
# plt.title("数据10"),plt.ylim(0, 1)
# plt.subplot(7,2,12), plt.plot(test_show[:,11]), plt.plot(decoded_show_rec[:,11] , color = 'red', linewidth = 1.0, linestyle = '--')
# plt.title("数据11"),plt.ylim(0, 1)
# plt.subplot(7,2,13), plt.plot(test_show[:,12]), plt.plot(decoded_show_rec[:,12] , color = 'red', linewidth = 1.0, linestyle = '--')
# plt.title("数据12"),plt.ylim(0, 1)

# # ------------pytorch重建数据--------------
# mumber_plt = mumber_plt + 1
# plt.figure(mumber_plt)
# plt.subplot(7,2,1), plt.plot(test_show[:,0]), plt.plot(decode_show_teX[:,0] , color = 'red', linewidth = 1.0, linestyle = '--')
# plt.title("数据0",),plt.ylim(0, 1)
# plt.subplot(7,2,2), plt.plot(test_show[:,1]), plt.plot(decode_show_teX[:,1] , color = 'red', linewidth = 1.0, linestyle = '--')
# plt.title("数据1"),plt.ylim(0, 1)
# plt.subplot(7,2,3), plt.plot(test_show[:,2]), plt.plot(decode_show_teX[:,2] , color = 'red', linewidth = 1.0, linestyle = '--')
# plt.title("数据2"),plt.ylim(0, 1)
# plt.subplot(7,2,4), plt.plot(test_show[:,3]), plt.plot(decode_show_teX[:,3] , color = 'red', linewidth = 1.0, linestyle = '--')
# plt.title("数据3"),plt.ylim(0, 1)
# plt.subplot(7,2,5), plt.plot(test_show[:,4]), plt.plot(decode_show_teX[:,4] , color = 'red', linewidth = 1.0, linestyle = '--')
# plt.title("数据4"),plt.ylim(0, 1)
# plt.subplot(7,2,6), plt.plot(test_show[:,5]), plt.plot(decode_show_teX[:,5] , color = 'red', linewidth = 1.0, linestyle = '--'),
# plt.title("数据5"),plt.ylim(0, 1)
# plt.subplot(7,2,7), plt.plot(test_show[:,6]), plt.plot(decode_show_teX[:,6] , color = 'red', linewidth = 1.0, linestyle = '--')
# plt.title("数据6"),plt.ylim(0, 1)

# plt.subplot(7,2,8), plt.plot(test_show[:,7]), plt.plot(decode_show_teX[:,7] , color = 'red', linewidth = 1.0, linestyle = '--')
# plt.title("数据7"),plt.ylim(0, 1)
# plt.subplot(7,2,9), plt.plot(test_show[:,8]), plt.plot(decode_show_teX[:,8] , color = 'red', linewidth = 1.0, linestyle = '--')
# plt.title("数据8"),plt.ylim(0, 1)
# plt.subplot(7,2,10), plt.plot(test_show[:,9]), plt.plot(decode_show_teX[:,9] , color = 'red', linewidth = 1.0, linestyle = '--')
# plt.title("数据9"),plt.ylim(0, 1)
# plt.subplot(7,2,11), plt.plot(test_show[:,10]), plt.plot(decode_show_teX[:,10] , color = 'red', linewidth = 1.0, linestyle = '--')
# plt.title("数据10"),plt.ylim(0, 1)
# plt.subplot(7,2,12), plt.plot(test_show[:,11]), plt.plot(decode_show_teX[:,11] , color = 'red', linewidth = 1.0, linestyle = '--')
# plt.title("数据11"),plt.ylim(0, 1)
# plt.subplot(7,2,13), plt.plot(test_show[:,12]), plt.plot(decode_show_teX[:,12] , color = 'red', linewidth = 1.0, linestyle = '--')
# plt.title("数据12"),plt.ylim(0, 1)


# # # ------------pytorch重构数据显示---------------
mumber_plt = mumber_plt + 1
plt.figure(mumber_plt)
plt.subplot(7, 2, 1)
plt.plot(F[:, 0])
plt.subplot(7, 2, 2)
plt.plot(F[:, 1])
plt.subplot(7, 2, 3)
plt.plot(F[:, 2])
plt.subplot(7, 2, 4)
plt.plot(F[:, 3])
plt.subplot(7, 2, 5)
plt.plot(F[:, 4])
plt.subplot(7, 2, 6)
plt.plot(F[:, 5])
plt.subplot(7, 2, 7)
plt.plot(F[:, 6])
plt.subplot(7, 2, 8)
plt.plot(F[:, 7])
plt.subplot(7, 2, 9)
plt.plot(F[:, 8])
plt.subplot(7, 2, 10)
plt.plot(F[:, 9])
plt.subplot(7, 2, 11)
plt.plot(F[:, 10])
plt.subplot(7, 2, 12)
plt.plot(F[:, 11])
plt.subplot(7, 2, 13)
plt.plot(F[:, 12],color = 'black')
# plt.legend(labels= ["data1","data2","data3","data4","data5","data6","data7","data8","data9",
#                     "data10","data11","data12","data13"], loc = 'upper left')
# plt.xlabel('Data point', font_lable)
# plt.ylabel('Amplitude', font_lable)

# x=[1,2,3,4,5,6,7,8,9,10,11,12,13]
# plt.bar(x, f_history.data.numpy()[0,:])
# plt.axis([0.5,13.5,-1,1])
# plt.xlabel('Process data', font_lable)
# plt.ylabel('Amplitude', font_lable)
# # ------------pytorc H2与SPE显示--------------
# #H2显示
# mumber_plt = mumber_plt + 1
# plt.figure(mumber_plt)
# plt.subplot(2,1,1)
# plt.title("H2_trX")
# plt.ylim(0,150)
# plt.plot(H2_trX_history)
# plt.plot([0, len(H2_trX_history)], [h_H2_trX, h_H2_trX], color = 'red')

# plt.subplot(2,1,2)
# plt.title("SPE_trX")
# plt.ylim(0,1)
# plt.plot(SPE_trX_history)
# plt.plot([0, len(SPE_trX_history )], [h_SPE_trX, h_SPE_trX], color = 'red')

# #SPE显示
mumber_plt = mumber_plt + 1
plt.figure(mumber_plt)

plt.subplot(2,1,1)
plt.ylim(0,150)
plt.plot(H2_teX_history)
plt.plot([0, len(H2_teX_history)], [h_H2_teX, h_H2_teX], color = 'red')
plt.legend(labels= ["Original test set","confidence limit 99%"], loc = 'upper left')
plt.xlabel('Process data', font_lable)
plt.ylabel('Amplitude', font_lable)
plt.title("H^2 Statistics", font_title)

plt.subplot(2,1,2)
plt.plot(SPE_teX_history)
plt.plot([0, len(SPE_teX_history )], [h_SPE_teX, h_SPE_teX], color = 'red')
plt.legend(labels= ["Original test set","confidence limit 99%"], loc = 'upper left')
plt.xlabel('Process data', font_lable)
plt.ylabel('Amplitude', font_lable)
plt.title("SPE Statistics", font_title)

plt.show()

