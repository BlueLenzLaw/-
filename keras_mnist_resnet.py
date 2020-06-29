# coding:utf-8
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import tensorflow.compat.v1 as tf
import matplotlib
matplotlib.use('Agg')   # 不显示绘图
import pickle
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from setup_mnist import MNIST
import time
import shutil 
import keras
from keras import initializers
from keras.models import Sequential
from keras.utils.generic_utils import get_custom_objects
from keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Add, ZeroPadding2D, BatchNormalization, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.utils.np_utils import *
from keras.optimizers import SGD
from keras.models import Model
from keras.models import load_model
from keras.callbacks import EarlyStopping
import keras.backend as K
from matplotlib.pyplot import imshow
import scipy.misc
from keras.initializers import glorot_uniform
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.data_utils import get_file
from keras.utils import layer_utils
from keras.preprocessing import image
from keras.utils.generic_utils import get_custom_objects
import platform

tf.disable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"]='1'

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

# 变量作为字典形式，储存所有信息
R_variable={}  ### used for saved all parameters and data

# 模型的参数设置
### setup for activation function
R_variable['type']='mnist_resnet_srelu'
R_variable['seed']=0
R_variable['actfun']='srelu'###  0: ReLU; 1: Tanh; 3:sin;4: x**5,, 5: sigmoid  6 sigmoid derivate 7.sRelu 激活函数，可供选择


# 模型训练测试的数据集设置
data = MNIST()
R_variable['train_inputs'] = data.train_data
R_variable['y_true_train'] = data.train_labels
R_variable['test_inputs'] = data.test_data
R_variable['y_true_test'] = data.test_labels
# print(train_x[:,:,:,0].shape)
R_variable['graph_shape'] = R_variable['train_inputs'].shape[1:]
R_variable['label_shape'] = R_variable['y_true_train'][1:]

# 模型训练测试的数据集&batch的大小设置
R_variable['train_size']= data.train_data.shape[0] ### training size
R_variable['test_size']=data.test_data[0] ### test size
R_variable['tol']=1e-6
R_variable['breakstandard'] = 0.96

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

        def plotacc():
            plt.figure()
            ax = plt.gca()
            y1 = R_variable['acc_test']
            y2 = R_variable['acc_train']
            plt.plot(y1,'ro',label='Test')
            plt.plot(y2,'g*',label='Train')
            # ax.set_xscale('log')
            # ax.set_yscale('log')                
            plt.legend(fontsize=18)
            plt.xlabel('Epoch',fontsize=15)
            plt.title('accuracy',fontsize=15)
            fntmp = '%saccuracy'%(self.FolderName)
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
            dc = pd.DataFrame(R_variable['acc_train'])
            dc.to_csv(self.FolderName + "acc_train" + ".csv", header=False, columns=None)
            dd = pd.DataFrame(R_variable['acc_test'])
            dd.to_csv(self.FolderName + "acc_test" + ".csv", header=False, columns=None)
 
        # 记录误差值，L2，在每次画loss前更新(以防中期停止程序)
        def gapReocord():
            R_variable['final_train_loss'] = R_variable['loss_train'][-1]
            R_variable['final_test_loss'] = R_variable['loss_test'][-1]
            R_variable['final_train_acc'] = R_variable['acc_train'][-1]
            R_variable['final_test_acc'] = R_variable['acc_test'][-1]
                
        # 储存误差
        R_variable['loss_test']=[]
        R_variable['loss_train']=[]
        R_variable['acc_test']=[]
        R_variable['acc_train']=[]

        # s-relu
        def custom_activation(x):
            return K.relu(-(x-1))*K.relu(x)

        get_custom_objects().update({'srelu': Activation(custom_activation)})

        # 记时，创建新目录
        self.t0=time.time()
        self.FolderName = mk_newfolder()
        
        def Resnet18_block(X, filters, s ,stage, block):
            """
            Implementation of the identity block as defined in Figure 4
            Arguments:
            X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
            f -- integer, specifying the shape of the middle CONV's window for the main path
            filters -- python list of integers, defining the number of filters in the CONV layers of the main path
            stage -- integer, used to name the layers, depending on their position in the network
            block -- string/character, used to name the layers, depending on their position in the network
            Returns:
            X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
            """
            # defining name basis
            conv_name_base = 'res' + str(stage) + block + '_branch'
            bn_name_base = 'bn' + str(stage) + block + '_branch'
            # Retrieve Filters
            F1, F2 = filters
            # Save the input value. You'll need this later to add back to the main path. 
            X_shortcut = X
            if s != 1:
                X_shortcut = Conv2D(filters = F1, kernel_size = (1, 1), strides = (s,s), padding = 'same', name = conv_name_base + '2c')(X_shortcut)
                X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X_shortcut)
                X_shortcut = Activation(R_variable['actfun'])(X_shortcut)       
            # First component of main path
            X = Conv2D(filters = F1, kernel_size = (3, 3), strides = (s,s), padding = 'same', name = conv_name_base + '2a')(X)
            X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
            X = Activation(R_variable['actfun'])(X)
            ### START CODE HERE ###
            # Second component of main path (≈3 lines)
            X = Conv2D(filters = F2, kernel_size = (3, 3), strides = (1,1), padding = 'same', name = conv_name_base + '2b')(X)
            X = BatchNormalization(axis=3, name = bn_name_base + '2b')(X)
            X = Activation(R_variable['actfun'])(X)
            # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
            X = layers.add([X, X_shortcut])
            X = Activation(R_variable['actfun'])(X)
            ### END CODE HERE ###
            return X

        def ResNet18(input_shape = (28,28,1),classes=10):
            X_input = Input(input_shape)
            #X1 = ZeroPadding2D((3,3))(X_input)
            #stage 1 
            X1 = Conv2D(64,(3,3),strides = (1,1),name = 'conv1',padding='same', kernel_initializer = glorot_uniform(seed=0))(X_input)
            X3 = BatchNormalization(axis = 3, name = 'bn_conv1')(X1)
            X2 = Activation(R_variable['actfun'])(X3)
            X32 = Dropout(0.1)(X2)
            #stage2
            X3 = Resnet18_block(X32, filters=[64,64],s=1, stage = 2, block = 'a')
            X4 = Resnet18_block(X3, filters=[64,64],s=1, stage = 2, block = 'b')
            X33 = Dropout(0.1)(X4)
            #Stage3
            X5 = Resnet18_block(X33, filters=[128,128],s=2, stage = 3, block = 'a')
            X6 = Resnet18_block(X5, filters=[128,128],s=1, stage = 3, block = 'b')
            X34 = Dropout(0.1)(X6)
            #Stage4
            X7 = Resnet18_block(X34, filters=[256,256],s=2, stage = 4, block = 'a')
            X8 = Resnet18_block(X7, filters=[256,256],s=1, stage = 4, block = 'b')
            X35 = Dropout(0.1)(X8)
            #Stage5
            X9 = Resnet18_block(X35, filters=[1024,1024],s=2, stage = 5, block = 'a')
            X10 = Resnet18_block(X9, filters=[1024,1024],s=1, stage = 5, block = 'b')
            X19 = Resnet18_block(X10, filters=[1024,1024],s=1, stage = 5, block = 'c')    
            X11 = AveragePooling2D(pool_size=(4,4))(X19)
            X12 = Dropout(0.1)(X11)

            X23 = Flatten()(X12)
            X24 = Dense(1024,activation=R_variable['actfun'], kernel_initializer = glorot_uniform(seed=1),name='dense1')(X23)
            X25 = Dense(1024,activation=R_variable['actfun'], kernel_initializer = glorot_uniform(seed=2),name='dense2')(X24)
            X26 = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=3))(X25)

            model = Model(inputs = X_input, outputs = X26, name='ResNet18')

            return model
        
        self.model = ResNet18(input_shape = (28,28,1),classes=10)
        self.sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=1e-6, nesterov=True, name='SGD')
        self.adam1= tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.85, beta_2=0.999, epsilon=1e-07, amsgrad=False,name='Adam1')
        self.adam2 = tf.keras.optimizers.Adam(
                learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
                name='Adam2'
                )
        self.adam3 = tf.keras.optimizers.Adam(
                learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
                name='Adam3'
                )

        self.adam4 = tf.keras.optimizers.Adam(
                learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
                name='Adam4'
                )


        self.adam5 = tf.keras.optimizers.Adam(
                learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
                name='Adam4'
                )

        def fn(correct, predicted):
            return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                        logits=predicted)

        self.model.compile(optimizer=self.adam2,
                    loss= tf.keras.losses.CategoricalCrossentropy(),
                    metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=2)

        self.history1 = self.model.fit(R_variable['train_inputs'], R_variable['y_true_train'],
                            epochs=50, 
                            batch_size=256,
                            validation_data=(R_variable['test_inputs'], R_variable['y_true_test']),
                            shuffle=True)

        R_variable['loss_train'] = self.history1.history['loss']
        R_variable['loss_test'] = self.history1.history['val_loss']
        R_variable['acc_train']= self.history1.history['accuracy']
        R_variable['acc_test']= self.history1.history['val_accuracy']
        round_time = time.time()
        R_variable['use_time'] = round_time - self.t0
        savefile()
        gapReocord()
        plotloss()
        plotacc()

        self.model.compile(optimizer=self.adam4,
                    loss= tf.keras.losses.CategoricalCrossentropy(),
                    metrics=['accuracy'])

        self.history2 = self.model.fit(R_variable['train_inputs'], R_variable['y_true_train'],
                            epochs=25, 
                            batch_size=256,
                            validation_data=(R_variable['test_inputs'], R_variable['y_true_test']),
                            shuffle=True,
                            callbacks=[early_stopping])

        R_variable['loss_train'] += self.history2.history['loss']
        R_variable['loss_test'] += self.history2.history['val_loss']
        R_variable['acc_train']+= self.history2.history['accuracy']
        R_variable['acc_test'] += self.history2.history['val_accuracy']
        round_time = time.time()
        R_variable['use_time'] = round_time - self.t0
        savefile()
        gapReocord()
        plotloss()
        plotacc()

        self.model.compile(optimizer=self.adam5,
                    loss= tf.keras.losses.CategoricalCrossentropy(),
                    metrics=['accuracy'])

        self.history3 = self.model.fit(R_variable['train_inputs'], R_variable['y_true_train'],
                            epochs=25, 
                            batch_size=256,
                            validation_data=(R_variable['test_inputs'], R_variable['y_true_test']),
                            shuffle=True,
                            callbacks=[early_stopping])

        
        R_variable['loss_train'] += self.history3.history['loss']
        R_variable['loss_test'] += self.history3.history['val_loss']
        R_variable['acc_train'] += self.history3.history['accuracy']
        R_variable['acc_test'] += self.history3.history['val_accuracy']
        round_time = time.time()
        R_variable['use_time'] = round_time - self.t0
        savefile()
        gapReocord()
        plotloss()
        plotacc()

        score1 = self.model.evaluate(R_variable['train_inputs'], R_variable['y_true_train'], verbose=0)
        score2 = self.model.evaluate(R_variable['test_inputs'], R_variable['y_true_test'], verbose=0)

        print("Program ends. ")
        print("Train accuracy is: %3.3f, Train loss is: %3.3f" % (score1[1], score1[0]))
        print("Test accuracy is: %3.3f, Test loss is: %3.3f" % (score2[1], score2[0]))
        print("The program have been running for %ds." % (time.time()-self.t0))

model1=model()
