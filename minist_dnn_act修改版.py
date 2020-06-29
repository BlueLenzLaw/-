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
from setup_mnist import MNIST
# 绘图包
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

tf.disable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"]='1'

# 创建目录
def mkdir(fn):
    if not os.path.isdir(fn):
        os.mkdir(fn)

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

# 变量作为字典形式，储存所有信息
R_variable={}  ### used for saved all parameters and data

# 模型的参数设置
### setup for activation function
R_variable['type']='mnist_dnn_relu'
R_variable['seed']=0
# np.random.seed(R_variable['seed'])
tf.set_random_seed(R_variable['seed'])
R_variable['ActFuc']='relu'   ###  0: ReLU; 1: Tanh; 3:sin;4: x**5,, 5: sigmoid  6 sigmoid derivate 7.sRelu 激活函数，可供选择

### initialization standard deviation
R_variable['keep_rate']=1 # 对于dropout来说保留多少, =1时即全部保留
R_variable['loss_type']=0 # loss是否加l2
R_variable['flag'] = 0 #是否能达到0.96精度

# 模型训练测试的数据集设置
data = MNIST()
R_variable['train_inputs'] = data.train_data[:40000,:,:,:]
R_variable['y_true_train'] = data.train_labels[:40000]
R_variable['test_inputs'] = data.test_data
R_variable['y_true_test'] = data.test_labels
# print(train_x[:,:,:,0].shape)
R_variable['graph_shape'] = R_variable['train_inputs'].shape[1:]
R_variable['label_shape'] = R_variable['y_true_train'][1:]

# 模型训练测试的数据集&batch的大小设置
R_variable['epoch'] = 100  # 100
R_variable['batch_size'] = 50  #50
R_variable['batch_num'] = 5000 #5000
R_variable['learning_rate'] = 1e-4 #1e-4
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
        # 误差图      
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

        # 记时，创建新目录
        self.t0=time.time()
        self.FolderName = mk_newfolder()
        self.x = tf.placeholder(tf.float32, [None].extend(R_variable['graph_shape']), name='x')
        self.y0 = tf.placeholder(tf.float32, shape=[None].extend(R_variable['label_shape']), name='y0')

        dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y0))
        dataset = dataset.shuffle(20).batch(R_variable['batch_size']).repeat()
        itetator = dataset.make_initializable_iterator()
        data_element = itetator.get_next()

        def weight_variable(shape,name=None):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=name)
        def bias_variable(shape,name=None):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name)
        def activation_fun(x, name=None):
            if (R_variable['ActFuc'] == 'relu'):
                z = tf.nn.relu(x, name=name)
            if (R_variable['ActFuc'] == 'tanh'):
                z = tf.nn.tanh(x, name=name)
            if (R_variable['ActFuc'] == 'srelu'):
                z = tf.nn.relu(-(x-1))*tf.nn.relu(x)
            return z

        # 全连接层
        x_flat = tf.reshape(self.x, [-1, R_variable['graph_shape'][0]*R_variable['graph_shape'][1]], name='x_flat')
        W_fc1 = weight_variable([R_variable['graph_shape'][0]*R_variable['graph_shape'][1], 800], name = 'W_fc1')
        b_fc1 = bias_variable([800], name = 'b_fc1')
        h_fc1 = activation_fun(tf.matmul(x_flat, W_fc1) + b_fc1, name='h_fc1')

        # 全连接层
        W_fc2 = weight_variable([800, 800], name = 'W_fc2')
        b_fc2 = bias_variable([800], name = 'b_fc2')
        h_fc2 = activation_fun(tf.matmul(h_fc1, W_fc2) + b_fc2, name='h_fc2')

        # 全连接层
        W_fc3 = weight_variable([800, 512], name = 'W_fc3')
        b_fc3 = bias_variable([512], name = 'b_fc3')
        h_fc3 = activation_fun(tf.matmul(h_fc2, W_fc3) + b_fc3, name='h_fc3')

        # 全连接层
        W_fc4 = weight_variable([512, 64], name = 'W_fc4')
        b_fc4 = bias_variable([64], name = 'b_fc4')
        h_fc4 = activation_fun(tf.matmul(h_fc3, W_fc4) + b_fc4, name='h_fc4')

        # softmax
        W_fc5 = weight_variable([64, 10], name='W_fc5')
        b_fc5 = bias_variable([10], name='b_fc3')
        y_pre = tf.add(tf.matmul(h_fc4, W_fc5), b_fc5, name='y_pre')
        self.y = tf.nn.softmax(y_pre, name='y')

        # loss func
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y0))
        # train aim
        self.train = tf.train.AdamOptimizer(learning_rate=R_variable['learning_rate']).minimize(self.cross_entropy)
        # accuracy
        self.result = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y0, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.result, tf.float32))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True #服务器跑，可忽略
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(itetator.initializer, feed_dict={self.x: R_variable['train_inputs'], self.y0: R_variable['y_true_train']})

        for e in range(R_variable['epoch']):
            for s in range(R_variable['batch_num']):
                finished_batch = e * R_variable['batch_num'] + s + 1
                # training
                x_batch, y_batch = self.sess.run(data_element)
                self.sess.run(self.train, feed_dict={self.x: x_batch, self.y0: y_batch})
                acc_Test, loss_Test = self.sess.run([self.accuracy, self.cross_entropy], feed_dict={self.x:R_variable['test_inputs'], self.y0:R_variable['y_true_test']})
                if (acc_Test >= R_variable['breakstandard']):
                    R_variable['uesd batch'] = finished_batch
                    R_variable['uesd time'] = time.time() - self.t0
                    R_variable['flag'] = 1
                    break
                
                if s % 1000 == 0:
                    acc_Train, loss_Train = self.sess.run([self.accuracy, self.cross_entropy], feed_dict={self.x:x_batch, self.y0:y_batch})
                    acc_Test, loss_Test = self.sess.run([self.accuracy, self.cross_entropy], feed_dict={self.x:R_variable['test_inputs'], self.y0:R_variable['y_true_test']})
                    # R_variable['loss_train'].append(loss_Train)
                    # R_variable['loss_test'].append(loss_Test)
                    batch_needed = R_variable['epoch'] * R_variable['batch_num'] - finished_batch
                    round_time = time.time()
                    R_variable['use_time'] = round_time - self.t0
                    time_needed = (round_time-self.t0) / finished_batch * batch_needed
                    print("In epoch: %d, step: %d, Train accuracy is: %3.3f, Train loss is: %3.3f" % (e+1, s, acc_Train, loss_Train))
                    print("Test accuracy is: %3.3f, Test loss is: %3.3f" % (acc_Test, loss_Test))
                    print("The program have been running for %ds, still need %ds" % (round_time-self.t0, time_needed))
                    # savefile()
                    # gapReocord()
    

            acc_Train, loss_Train = self.sess.run([self.accuracy, self.cross_entropy], feed_dict={self.x: R_variable['train_inputs'], self.y0:R_variable['y_true_train']})
            acc_Test, loss_Test = self.sess.run([self.accuracy, self.cross_entropy], feed_dict={self.x: R_variable['test_inputs'], self.y0:R_variable['y_true_test']})
            R_variable['loss_train'].append(loss_Train)
            R_variable['loss_test'].append(loss_Test)
            R_variable['acc_train'].append(acc_Train)
            R_variable['acc_test'].append(acc_Test)
            savefile()
            gapReocord()
            plotloss()
            plotacc()
            if (R_variable['flag'] == 1):
                break

        print("Program ends. ")
        print("Train accuracy is: %3.3f, Train loss is: %3.3f" % (acc_Train, loss_Train))
        print("Test accuracy is: %3.3f, Test loss is: %3.3f" % (acc_Test, loss_Test))
        print("The program have been running for %ds." % (round_time-self.t0))


model1=model()