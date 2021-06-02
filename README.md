#VGG模型实现图片分类

import os
name_dict = {'apple':0,'banana':1,'grape':2,'orange':3,'pear':4}
data_root_path = "data/fruits/" # 数据集所在目录
test_file_path = data_root_path + 'test.txt' #测试集文件路径
train_file_path = data_root_path + 'train.txt' #训练集文件路径
name_data_dict = {} #记录每个类别有那些图片 {apple:[0.jpg,1.jpg.....],banana:[..].....}

#将图片路径存入name_data_list字典中
def save_train_test_file(path,name):
    if name not in name_data_dict: #该类别水果不在字典中，则新建一个键值对
        img_list = []
        img_list.append(path)
        name_data_dict[name] = img_list
    else:
        name_data_dict[name].append(path)

#遍历每个子目录，拼接完整图片路径，并加入到字典

dirs = os.listdir(data_root_path)
for d in dirs:
    full_path = data_root_path + d #拼接完整路径 data/fruits/apple

    if os.path.isdir(full_path): #是否为子目录,如果是则读取其中的图片
        imgs = os.listdir(full_path)
        for img in imgs:
            save_train_test_file(full_path + '/' + img,# data/fruits/apple/0.jpg
                                 d) #以子目录(文件夹)的名字作为键
    else:
        pass

# 遍历字典，划分训练集，测试集
#清空训练集和测试集文件
with open(test_file_path,'w') as f:
    pass
with open(train_file_path,'w') as f:
    pass

for name,img_list in name_data_dict.items():
    i = 0 #计数器,为了实现　90%训练样本　10%测试样本
    num = len(img_list) #获取到每个类别的样本数量
    print("{}:{}张图片".format(name,num))

    for img in img_list: #每个img拿到的是图片的完整路径
        if i % 10 == 0: #写入测试集
            with open(test_file_path,'a') as f:
                #拼成一行 ：　图片路径 \t 类别
                line = "%s\t%d\n" % (img,name_dict[name])
                f.write(line)
        else:
            with open(train_file_path,'a') as f:
                line = '%s\t%d\n' % (img,name_dict[name])
                f.write(line)

        i += 1 #写一个计数器+1

print('数据预处理完成')

################## 模型搭建，训练，保存 ###############

import paddle
import paddle.fluid as fluid
import numpy
import os
import sys
from multiprocessing import cpu_count
import time
import matplotlib.pyplot as plt

def train_mapper(sample):
    '''
    根据传入的一行文本(路径\t类别)，读取相应的图像数据
    :param sample: 元祖 (路径\t类别)
    :return: 图像路径　类别
    '''
    img,label = sample #img路径　label类别
    if not os.path.exists(img):
        print('图像不存在')

    #读取图像数据
    img = paddle.dataset.image.load_image(img)
    #统一图像大小
    img = paddle.dataset.image.simple_transform(im=img, #原始图像数据
                                                resize_size=128,#图像缩放大小
                                                crop_size=128,#裁剪图像大小
                                                is_color=True,#是否为彩色图像
                                                is_train=True)#是否为训练模式：随机裁剪
    #对图像进行归一化
    img = img.astype('float32') / 255.0
    return img,label

# 定义reader,从训练集中读取样本
def train_r(train_list,buffered_size=1024):
    def reader():
        with open(train_list,'r') as f:
            lines = [line.strip() for line in f] #读取所有行，去掉空格
            for line in lines:
                img_path,lab = line.replace('\n','').split('\t')
                yield img_path,int(lab)
    return paddle.reader.xmap_readers(train_mapper,#讲reader读到的数据交给train_mapper进行二次处理
                                      reader,#读取样本的函数
                                      cpu_count(),#线程数量
                                      buffered_size)#缓冲区大小

#测试集读取器
def test_mapper(sample):
    '''
    根据传入的一行文本(路径\t类别)，读取相应的图像数据
    :param sample: 元祖 (路径\t类别)
    :return: 图像路径　类别
    '''
    img,label = sample #img路径　label类别

    #读取图像数据
    img = paddle.dataset.image.load_image(img)
    #统一图像大小
    img = paddle.dataset.image.simple_transform(im=img, #原始图像数据
                                                resize_size=128,#图像缩放大小
                                                crop_size=128,#裁剪图像大小
                                                is_color=True,#是否为彩色图像
                                                is_train=False)#是否为训练模式：随机裁剪
    #对图像进行归一化
    img = img.astype('float32') / 255.0
    return img,label

# 定义reader,从训练集中读取样本
def test_r(test_list,buffered_size=1024):
    def reader():
        with open(test_list,'r') as f:
            lines = [line.strip() for line in f] #读取所有行，去掉空格
            for line in lines:
                img_path,lab = line.split('\t')
                yield img_path,int(lab)
    return paddle.reader.xmap_readers(test_mapper,#讲reader读到的数据交给train_mapper进行二次处理
                                      reader,#读取样本的函数
                                      cpu_count(),#线程数量
                                      buffered_size)#缓冲区大小




#定义reader
BATCH_SIZE = 32 #批次大小
#训练集的reader
trainer_reader = train_r(train_list=train_file_path)
random_train_reader = paddle.reader.shuffle(reader=trainer_reader,
                                            buf_size=1300)
batch_train_reader = paddle.batch(random_train_reader,
                                  batch_size=BATCH_SIZE)

#测试集reader
tester_reader = test_r(test_list=test_file_path)
test_reader = paddle.batch(tester_reader,batch_size=BATCH_SIZE)


#变量
image = fluid.layers.data(name='image',shape=[3,128,128],dtype='float32')
label = fluid.layers.data(name='label',shape=[1],dtype='int64')

#搭建VGG网络模型
def vgg_bn_drop(image,type_size):
    def conv_block(ipt,num_filter,groups,dropouts):
        #卷积块，用于生成卷积操作的
        #创建　conv2d,batch normal,dropout,pool2d
        return fluid.nets.img_conv_group(input=ipt, #输入数据
                                         pool_stride=2,#池化步长值
                                         pool_size=2,#池化区域大小
                                         conv_filter_size=3,#卷积核大小
                                         conv_num_filter= [num_filter] * groups,#卷积核数量
                                         conv_act='relu',#激活函数
                                         conv_with_batchnorm=True)#是否处理批量归一化

    conv1 = conv_block(image,64,2,[0.0,0.0])
    conv2 = conv_block(conv1,128,2,[0.0,0.0])
    conv3 = conv_block(conv2,256,3,[0.0,0.0,0.0])
    conv4 = conv_block(conv3,512,3,[0.0,0.0,0.0])
    conv5 = conv_block(conv4,512,3,[0.0,0.0,0.0])

    drop = fluid.layers.dropout(x=conv5,dropout_prob=0.5)
    fc1 = fluid.layers.fc(input=drop,size=512,act=None)

    #批量归一化
    bn = fluid.layers.batch_norm(input=fc1,act='relu')
    drop2 = fluid.layers.dropout(x=bn,dropout_prob=0.0)

    fc2 = fluid.layers.fc(input=drop2,size=512,act=None)
    predict = fluid.layers.fc(input=fc2,size=type_size,act='softmax')

    return predict


predict = vgg_bn_drop(image=image,type_size=5)

#损失函数 交叉熵
cost = fluid.layers.cross_entropy(input=predict,#预测结果
                                  label=label)#真实标签
avg_cost = fluid.layers.mean(cost)
#准确率
accuracy = fluid.layers.accuracy(input=predict,
                      label=label)

#复制（克隆）一个program,用于模型的评估
test_program = fluid.default_main_program().clone(for_test=True)

#优化器
#VGG模型学习率要调整的非常小
optimizer = fluid.optimizer.Adam(learning_rate=0.000001)
optimizer.minimize(avg_cost)

#执行器
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program()) #初始化
#参数喂入器
feeder = fluid.DataFeeder(feed_list=[image,label],
                          place=place)

model_save_dir = 'model/fruits'#模型保存的路径
costs = [] #记录损失值
accs = [] #记录准确率
batches = [] #记录迭代次数
times = 0

#开始训练
for pass_id in range(50):
    train_cost = 0 #临时变量，记录损失值
    for batch_id,data in enumerate(batch_train_reader()):
        times += 1
        train_cost,train_acc = exe.run(program=fluid.default_main_program(),
                                       feed=feeder.feed(data),
                                       fetch_list=[avg_cost,accuracy])
        if batch_id % 20 == 0:
            print('pass_id:{},batch_id:{},cost:{},acc:{}'.format(pass_id,batch_id,train_cost[0],train_acc[0]))

        accs.append(train_acc[0])
        costs.append(train_cost[0])
        batches.append(times)
    #模型评估,完成一个批次，进行评估
    test_accs = []
    test_costs = []
    for batch_id,data in enumerate(test_reader()):
        test_cost,test_acc = exe.run(program=test_program,
                                       feed=feeder.feed(data),
                                       fetch_list=[avg_cost,accuracy])
        test_accs.append(test_acc[0])
        test_costs.append(test_cost[0])

    test_cost = (sum(test_costs) / len(test_costs))
    test_acc = (sum(test_accs) / len(test_accs))

    print('Test:{},Cost:{},Acc:{}'.format(pass_id,test_cost,test_acc))

#训练结束后，保存模型
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
fluid.io.save_inference_model(dirname=model_save_dir,
                              feeded_var_names=['image'],
                              target_vars=[predict],
                              executor=exe)
print('模型保存成功')

#训练过程可视化
plt.figure('training')
plt.title('training',fontsize=24)
plt.xlabel('iter',fontsize=14)
plt.xlabel('cost/acc',fontsize=14)
plt.plot(batches,costs,color='red',label='Training Cost')
plt.plot(batches,accs,color='green',label='Training Acc')
plt.legend()
plt.grid()
plt.savefig('train.png')
plt.show()

################## 模型加载 预测###############

#定义执行器
place = fluid.CPUPlace()
infer_exe = fluid.Executor(place)
model_save_dir = 'model/fruits/'

#加载图像数据
def load_img(path):
    img = paddle.dataset.image.load_and_transform(path,#路径
                                                  128,#裁剪大小
                                                  128,#缩放大小
                                                  False).astype('float32')#是否训练
    img = img / 255.0
    return img

infer_imgs = [] #存放要预测的图像数据
test_img = 'apple_1.png' #待预测的图片路径
infer_imgs.append(load_img(test_img))
infer_imgs = numpy.array(infer_imgs)

#加载模型
infer_program,feed_target_names,fetch_targets = fluid.io.load_inference_model(model_save_dir,infer_exe)

#执行预测
results = infer_exe.run(infer_program,
                        feed={feed_target_names[0]:infer_imgs},
                        fetch_list=fetch_targets)

print(results)

#对预测结果进行转换
result = numpy.argmax(results[0])
for k,v in name_dict.items():
    if result == v:
        print('预测结果为:',k)

from PIL import Image
img = Image.open(test_img)
plt.imshow(img)
plt.show()
