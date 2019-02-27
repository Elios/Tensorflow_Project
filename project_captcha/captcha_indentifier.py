import tensorflow as tf
import os
from PIL import Image
from nets import nets_factory
import numpy as np
import time

#1.创建dataset
#直接导入
#从TFRecord导入  filenames=['a.tfrecord','b.record']  tf.data.TFRecordDataset(filenames) 

#2.操作dataset
#样本解析  dataset.map(parse_fn) parse_fn为函数
##feature信息 解析基本为写入是的逆过程，需要写入时的信息
##创建解析函数parse_fn  parse_fn(example_proto)  example_proto为序列化后的样本tf_serialized
###创建样本解析字典  存放所有的feature的解析方式，key为feature名，value为feature的解析方式
####解析方式两种  定长tf.FixedLenFeature(shape,dtype.default_value)   不定长tf.VarLenFeature(dtype)
###解析样本parse_example为字典 key为feature名字 value为feature解析值  tf.parse_single_example(example_proto,dicts)
###转变特征 若使用如下两种情况需对值进行转变
####string类型  tf.decode_raw(parsed_feature，type)解码 type要一致
####VarLen解析  视情况需用tf.sparse_tensor_to_dense(SparseTensor)转为DenseTensor
###改变形状 因到此为止特征都是向量，需根据之前的shape信息对feature进行reshape
###返回样本parsed_example
##执行解析函数 new_dataset=dataset.map(parse_fn)
#创建迭代器获取样本  iter=new_dataset.make_one_shot_iterator()
#获取样本 next_element=iter.get_next()下一个样本
#shuffle打乱顺序shuffle_dataset  .shuffle(buffer_size) buffer_size设置为一个大于样本数量的值以充分打乱
#batch  shuffle_dataset.batch(batch_size)
#batch_padding 也可在每个batch中padding  new_dataset.padded_batch()
#epoch  .repeat(num_epochs)指定遍历几遍整个数据集



#需要识别的类型数量
CHAR_SET_LEN = 36
#图片高度
IMAGE_HEIGHT = 60
#图片宽度
IMAGE_WIDTH = 160
#每批次数量
BATCH_SIZE = 30
#tfrecord文件位置
TFRECORD_DIR = 'captcha/tfrecord/train.tfrecords'

x = tf.placeholder(tf.float32,[None,224,224])
y0 = tf.placeholder(tf.float32,[None])
y1 = tf.placeholder(tf.float32,[None])
y2 = tf.placeholder(tf.float32,[None])
y3 = tf.placeholder(tf.float32,[None])

#学习率learning_rate
lr = tf.Variable(0.005,dtype=tf.float32)

#读取tfrecord数据
def read_and_decode(filename):
    #根据文件名生成队列
##    filename_queue = tf.data.Dataset.from_tensor_slices([filename])
    filename_queue = tf.train.string_input_producer([filename])
##    serialized_example = tf.data.TFRecordDataset(filename_queue)
    reader = tf.TFRecordReader()

    _,serialized_example = reader.read(filename_queue)
    #返回文件名和文件
    features = tf.parse_single_example(serialized_example,features={
        'image':tf.FixedLenFeature([],tf.string),
        'label0':tf.FixedLenFeature([],tf.int64),
        'label1':tf.FixedLenFeature([],tf.int64),
        'label2':tf.FixedLenFeature([],tf.int64),
        'label3':tf.FixedLenFeature([],tf.int64)
    })
    #读取图片
    image = tf.decode_raw(features['image'],tf.uint8)
    #tf.train.shuffle_batch必须确定shape
    image = tf.reshape(image,[224,224])
    #图片处理
    image = tf.cast(image,tf.float32) / 255.0
    image = tf.subtract(image,0.5)
    image = tf.multiply(image,2.0)
    #获取label
    label0 = tf.cast(features['label0'],tf.int32)
    label1 = tf.cast(features['label1'],tf.int32)
    label2 = tf.cast(features['label2'],tf.int32)
    label3 = tf.cast(features['label3'],tf.int32)
    return image,label0,label1,label2,label3

if __name__ == '__main__':
    #获取图片和标签
    image,label0,label1,label2,label3 = read_and_decode(TFRECORD_DIR)
    #使用shuffle_batch随机打乱
    #capacity队列容量
    #返回只是op不是数据
##    [image_batch,label0_batch,label1_batch,label2_batch,label3_batch] = [image,label0,label1,label2,label3].shuffle(min_after_dequeue=10000).batch(BATCH_SIZE).repeat(10)
    image_batch,label0_batch,label1_batch,label2_batch,label3_batch = tf.train.shuffle_batch(
        [image,label0,label1,label2,label3],
        batch_size=BATCH_SIZE,
        capacity=50000,
        min_after_dequeue=10000,
        num_threads=1)
    #定义网络结构，使用alexnet网络,weight_decay权值衰减
    train_net = nets_factory.get_network_fn(
        'alexnet_v2',
        num_classes = CHAR_SET_LEN,
        weight_decay = 0.0005,
        is_training = True)

    with tf.Session() as sess:
        #alexnet,input需要[batch_size,height,width,channels]的tensor格式
        X = tf.reshape(x,[BATCH_SIZE,224,224,1])
        #数据输入网络得到输出
        logits0,logits1,logits2,logits3,_ = train_net(X)

        one_hot_labels0 = tf.one_hot(indices=tf.cast(y0,tf.int32),depth=CHAR_SET_LEN)
        one_hot_labels1 = tf.one_hot(indices=tf.cast(y1,tf.int32),depth=CHAR_SET_LEN)
        one_hot_labels2 = tf.one_hot(indices=tf.cast(y2,tf.int32),depth=CHAR_SET_LEN)
        one_hot_labels3 = tf.one_hot(indices=tf.cast(y3,tf.int32),depth=CHAR_SET_LEN)

        loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits0,labels=one_hot_labels0))
        loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits1,labels=one_hot_labels1))
        loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits2,labels=one_hot_labels2))
        loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits3,labels=one_hot_labels3))

        total_loss = (loss0+loss1+loss2+loss3) / 4.0
        
        train = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss)

        correct0 = tf.equal(tf.argmax(one_hot_labels0,1),tf.argmax(logits0,1))
        acc0 = tf.reduce_mean(tf.cast(correct0,tf.float32))

        correct1 = tf.equal(tf.argmax(one_hot_labels1,1),tf.argmax(logits1,1))
        acc1 = tf.reduce_mean(tf.cast(correct1,tf.float32))
        
        correct2 = tf.equal(tf.argmax(one_hot_labels2,1),tf.argmax(logits2,1))
        acc2 = tf.reduce_mean(tf.cast(correct2,tf.float32))

        correct3 = tf.equal(tf.argmax(one_hot_labels3,1),tf.argmax(logits3,1))
        acc3 = tf.reduce_mean(tf.cast(correct3,tf.float32))

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        #创建协调器管理线程
        coord = tf.train.Coordinator()
        #启动QueueRunner，此时文件名队列已进队
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        
        for i in range(1,6001):
            #真正的数据
            img,l0,l1,l2,l3 = sess.run([image_batch,label0_batch,label1_batch,label2_batch,label3_batch])
            sess.run(train,feed_dict={x:img,y0:l0,y1:l1,y2:l2,y3:l3})
            if i % 20 == 0:
                if i % 1000 == 0:
                    sess.run(tf.assign(lr,lr/2))
                    saver.save(sess,'model_cap_iden/captcha_iden.model',global_step=i)
                a0,a1,a2,a3,loss = sess.run([acc0,acc1,acc2,acc3,total_loss],feed_dict={x:img,y0:l0,y1:l1,y2:l2,y3:l3})

                learning_rate = sess.run(lr)

                print('Time:%f Iter:%d  Loss:%.3f  Acc:%.2f,%.2f,%.2f,%.2f  Learning_rate:%.5f' % (time.clock(),i,loss,a0,a1,a2,a3,learning_rate))

                if a0 > 0.98 and a1 > 0.98 and a2 > 0.98 and a3 > 0.98:
                    print('Time:%f Iter:%d  Loss:%.3f  Acc:%.2f,%.2f,%.2f,%.2f  Learning_rate:%.5f' % (time.clock(),i,loss,a0,a1,a2,a3,learning_rate))
                    saver.save(sess,'model_cap_iden/captcha_iden.model',global_step=i)
                    print('completed')
                    break
        #通知其他线程关闭
        coord.request_stop()
        #其他所有线程关闭后，这一函数才返回
        coord.join(threads)

















 










    
