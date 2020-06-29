# 终极版 Edited by SiJia

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib
matplotlib.use('Agg')   # 不显示绘图
import pickle
import time  
import shutil 
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt   
import platform
import pandas as pd 
import keras
from keras import initializers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Convolution2D as Conv2D
from keras.layers import MaxPooling2D
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import platform
# 绘图包
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

config = tf.ConfigProto()
config.gpu_options.allow_growth=True 
tf.compat.v1.keras.backend.set_session(tf.Session(config=config))
tf.disable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"]='2'

# 保存图片
# 画图参数
Leftp=0.18
Bottomp=0.18
Widthp=0.88-Leftp
Heightp=0.9-Bottomp
pos=[Leftp,Bottomp,Widthp,Heightp]

# 保存图片 pltm为图片，fntmp为图片名称
def mySaveFig(pltm, fntmp,fp=0,ax=0,isax=0,iseps=0,isShowPic=0):
    if isax==1: # 是否需要坐标轴
        pltm.rc('xtick',labelsize=18)
        pltm.rc('ytick',labelsize=18)
        ax.set_position(pos, which='both')
    fnm='%s.png'%(fntmp)
    pltm.savefig(fnm)
    if iseps: # 保存为eps格式
        fnm='%s.eps'%(fntmp)
        pltm.savefig(fnm, format='eps', dpi=600)
    if fp!=0:  # 保存为pdf格式
        fp.savefig("%s.pdf"%(fntmp), bbox_inches='tight')
    if isShowPic==1: # 是否展示图片
        pltm.show() 
    elif isShowPic==-1:
        return
    else:
        pltm.close()

# 创建目录
def mkdir(fn):
    if not os.path.isdir(fn):
        os.mkdir(fn)
#
#def get_y_func(xs):
#    tmp = np.sin(20*xs)
#    return tmp
# 变量作为字典形式，储存所有信息
R_variable={}  ### used for saved all parameters and data


R_variable['function'] = 'f(x(t))=∑cos(10*tj)+sin(5*tj)'#'#'y=cos(4*x1)+cos(4*x2)'#'y = sin(2*x)'#'y=cos(4*x1)+cos(4*x2)' # 要拟合的函数

# 自变量应变量的维数，隐藏层神经元的个数设置
R_variable['input_dim']=60
R_variable['output_dim']=1
R_variable['t_input_dim']=3

# 模型的参数设置
### setup for activation function
R_variable['type']='低维嵌入高维 sin(2x+20x)_dnn_tanh'
R_variable['seed']=0
R_variable['actfun']='tanh'###  0: ReLU; 1: Tanh; 3:sin;4: x**5,, 5: sigmoid  6 sigmoid derivate 7.sRelu 激活函数，可供选择

# 模型训练测试的数据集&batch的大小设置
# 自变量应变量的维数，隐藏层神经元的个数设置
R_variable['input_dim']=60
R_variable['output_dim']=1
R_variable['t_input_dim']=3
R_variable['hidden_units']=[200,200,200]

# 真实函数
# (x1,x2,……), f(x(t)=∑cos(10*tj)+sin(5*tj)
def get_y_func(t):
    tmp=0
    for ii in range(R_variable['t_input_dim']):
        tmp+=np.cos(10*t[:,ii:ii+1])+np.sin(5*t[:,ii:ii+1])
    return tmp

R_variable['train_size']=1400  ### training size
R_variable['batch_size']=128     #R_variable['train_size'] # int(np.floor(R_variable['train_size'])) ### batch size
R_variable['test_size']=600  ### test size
# 模型训练测试的数据集设置
R_variable['x_start']=-np.pi/2  #math.pi*3 ### start point of input
R_variable['x_end']=np.pi/2  #6.28/4 #math.pi*3  ### end point of input
# 模型训练测试的输入
if R_variable['input_dim']==1:
    R_variable['inputs'] =np.reshape(np.linspace(R_variable['x_start'], R_variable['x_end'], num=(R_variable['test_size']+R_variable['train_size']),endpoint=True),[R_variable['test_size']+R_variable['train_size'],1]) 
else:
    R_variable['inputs']=np.random.rand(R_variable['test_size']+R_variable['train_size'],R_variable['input_dim'])*(R_variable['x_end']-R_variable['x_start'])+R_variable['x_start']

random_inputs = np.random.permutation(R_variable['inputs'])
R_variable['train_inputs'] = random_inputs[:R_variable['train_size']]
R_variable['test_inputs'] = random_inputs[R_variable['train_size']:]

# 模型训练测试应得到的真实输出
R_variable['y_true_test']= get_y_func(R_variable['test_inputs'])
R_variable['y_true_train']=get_y_func(R_variable['train_inputs'])

# 模型训练测试的数据集&batch的大小设置
R_variable['epoch'] = 1000  # 100
R_variable['batch_size'] = 50  #50
R_variable['batch_num'] = 5000 #5000
R_variable['learning_rate'] = 1e-4 #1e-4
R_variable['tol']=1e-6
R_variable['breakstandard'] = 0.96

# 模型训练测试的数据集设置
R_variable['t_start']=0  
R_variable['t_end']=1 
# 模型训练测试的输入
if R_variable['t_input_dim']==1:
    R_variable['inputs'] =np.reshape(np.linspace(R_variable['t_start'], R_variable['t_end'], num=(R_variable['test_size']+R_variable['train_size']),endpoint=True),[R_variable['test_size']+R_variable['train_size'],1]) 
else:
    R_variable['inputs']=np.random.rand(R_variable['test_size']+R_variable['train_size'],R_variable['t_input_dim'])*(R_variable['t_end']-R_variable['t_start'])+R_variable['t_start']

# xi = cos(i)t_[i/(d/d_in)]
random_inputs = np.random.permutation(R_variable['inputs'])
R_variable['t_train_inputs'] = random_inputs[:R_variable['train_size']]
R_variable['t_test_inputs'] = random_inputs[R_variable['train_size']:]

R_variable['train_inputs'] = np.zeros((R_variable['train_size'], R_variable['input_dim']))
R_variable['test_inputs'] = np.zeros((R_variable['test_size'], R_variable['input_dim']))

d = R_variable['input_dim'] / R_variable['t_input_dim'] 

for i in range(R_variable['train_size']):
    for j in range(R_variable['input_dim']):
        R_variable['train_inputs'][i,j] = np.cos(j)*R_variable['t_train_inputs'][i, int(j/d)]

for i in range(R_variable['test_size']):
    for j in range(R_variable['input_dim']):
        R_variable['test_inputs'][i,j] = np.cos(j)*R_variable['t_test_inputs'][i, int(j/d)]

# 模型训练测试应得到的真实输出
R_variable['y_true_test']= get_y_func(R_variable['t_test_inputs'])
R_variable['y_true_train']=get_y_func(R_variable['t_train_inputs'])

### mkdir a folder to save all output
BaseDir = 'fitnd/'
mkdir(BaseDir)

# 为同时跑多个程序
def mk_newfolder(): 
    subFolderName = '%s_%s'%(R_variable['type'] ,int(np.absolute(np.random.normal([1])*100000))//int(1))  # 以随机数字命名生成文件夹，防止重复
    FolderName = '%s%s/'%(BaseDir,subFolderName) # 生成的output都储存在这个目录下
    mkdir(FolderName)
    R_variable['FolderName']=FolderName ### folder for save images
    # 若非windows客户
    if not platform.system()=='Windows':
        shutil.copy(__file__,'%s%s'%(FolderName,os.path.basename(__file__)))
    return FolderName


class model():
    def __init__(self):
        def plotloss():
            plt.figure()
            ax = plt.gca()
            y1 = R_variable['loss_test']
            y2 = R_variable['loss_train']
            plt.plot(y1,'ro',label='Test')
            plt.plot(y2,'g*',label='Train')
            # ax.set_xscale('log')
            ax.set_yscale('log')                
            plt.legend(fontsize=18)
            plt.xlabel('Epoch',fontsize=15)
            plt.title('loss',fontsize=15)
            fntmp = '%sloss'%(self.FolderName)
            mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)

        # 保存文件       
        def savefile():
            # 序列化变量R, 需要的话可以load出来
            with open('%s/objs.pkl'%(self.FolderName), 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(R_variable, f, protocol=4)

            # 保存变量R参数长度小于等于20的 
            text_file = open("%s/Output.txt"%(self.FolderName), "w")
            for para in R_variable:
                if np.size(R_variable[para])>20:
                    continue
                text_file.write('%s: %s\n'%(para,R_variable[para])) 
            text_file.close()

            # 保存loss到csv中
            da = pd.DataFrame(R_variable['loss_train'])
            da.to_csv(self.FolderName + "loss_train" + ".csv", header=False, columns=None)
            db = pd.DataFrame(R_variable['loss_test'])
            db.to_csv(self.FolderName + "loss_test" + ".csv", header=False, columns=None)
 
        # 记录误差值，L2，在每次画loss前更新(以防中期停止程序)
        def gapReocord():
            R_variable['final_train_loss'] = R_variable['loss_train'][-1]
            R_variable['final_test_loss'] = R_variable['loss_test'][-1]
                
        # 储存误差
        R_variable['loss_test']=[]
        R_variable['loss_train']=[]

        # s-relu
        def custom_activation(x):
            return K.relu(-(x-1))*K.relu(x)

        get_custom_objects().update({'srelu': Activation(custom_activation)})

        # 记时，创建新目录
        self.t0=time.time()
        self.FolderName = mk_newfolder()

        self.model = Sequential()
        self.model.add(Dense(800, kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0)))
        self.model.add(Activation(R_variable['actfun']))

        self.model.add(Dense(512, kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=1)))
        self.model.add(Activation(R_variable['actfun']))
        # self.model.add(Activation(custom_activation, name='SpecialActivation3'))

        self.model.add(Dense(64, kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=2)))
        self.model.add(Activation(R_variable['actfun']))
        # self.model.add(Activation(custom_activation, name='SpecialActivation4'))

        self.model.add(Dense(1, kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=3)))

        self.model.compile(loss='mse',
                    optimizer=tf.keras.optimizers.Adam(learning_rate=R_variable['learning_rate']))

        self.history = self.model.fit(R_variable['train_inputs'], R_variable['y_true_train'],
                        batch_size=R_variable['batch_size'],
                        epochs=R_variable['epoch'],
                        verbose=1,
                        validation_data=(R_variable['test_inputs'], R_variable['y_true_test']))

        score1 = self.model.evaluate(R_variable['train_inputs'], R_variable['y_true_train'], verbose=0)
        score2 = self.model.evaluate(R_variable['test_inputs'], R_variable['y_true_test'], verbose=0)
        R_variable['loss_train'] = self.history.history['loss']
        R_variable['loss_test'] = self.history.history['val_loss']
        round_time = time.time()
        R_variable['use_time'] = round_time - self.t0
        savefile()
        gapReocord()
        plotloss()

        print("Program ends. ")
        print("Train loss is: %3.3f" % (score1))
        print("Test loss is: %3.3f" % (score2))
        print("The program have been running for %ds." % (time.time()-self.t0))

model1=model()
