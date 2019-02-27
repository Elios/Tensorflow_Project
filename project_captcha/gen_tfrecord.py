import tensorflow as tf
import os
import sys
from PIL import Image
import numpy as np

#1.打开TFRecord file  writer = tf.python_io.TFRecordWriter('')
#2.创建样本(sample)写入字典 把每个样本中所有feature的信息和值存到字典中，key为feature名，value为feature值
#feature值需要转为指定的类型中的一个int64,float32,string,且输入值必须为list
#处理类型是张量的feature：1、转为list  2、转为string  需加入shape信息作为额外feature  feature名字+_shape
#3.转为tf_features  tf.train.Features(feature=features)
#4.转为tf_examle  tf.train.Example(feature=tf_features)
#5.序列化样本tf_serialized  tf_example.SerializeToString()
#6.写入样本 writer.write(tf_serialized)
#7.关闭TFRecord file    writer.close()

alphabet = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

_NUM_TEST = 100

_RANDOM_SEED = 0

DATASET_DIR = 'captcha/images/'

TFRECORD_DIR = 'captcha/tfrecord/'

def _dataset_exists(ds_dir):
    for split_name in ['train','test']:
        output_filename = os.path.join(ds_dir,split_name+'.tfrecords')
        if not tf.gfile.Exists(output_filename):
            return False
    return True

def _get_filenames_and_classes(ds_dir):
    i_filenames = []
    for filename in os.listdir(ds_dir):
        path = os.path.join(ds_dir,filename)
        i_filenames.append(path)
    return i_filenames

def int64_feature(values):
    if not isinstance(values,(tuple,list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample(i_data,label0,label1,label2,label3):
    return tf.train.Example(features=tf.train.Features(feature={
        'image':bytes_feature(i_data),
        'label0':int64_feature(label0),
        'label1':int64_feature(label1),
        'label2':int64_feature(label2),
        'label3':int64_feature(label3)
    }))


#把数据转为tfrecord格式
def _convert_ds(split_name,filenames,ds_dir):
    assert split_name in ['train','test']
    with tf.Session() as sess:
        output_filename = os.path.join(TFRECORD_DIR,split_name+'.tfrecords')
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            for i,filename in enumerate(filenames):
                try:
                    sys.stdout.write('\r>>Converting image %d/%d' % (i+1,len(filenames)))
                    sys.stdout.flush()
                    image_data = Image.open(filename)
                    #训练时用到224*224  alexnet
                    image_data = image_data.resize((224,224))
                    #灰度化 变为黑白
                    image_data = np.array(image_data.convert('L'))
                    image_data = image_data.tobytes()

                    #获取label 此处为图片名称
                    labels = filename.split('/')[-1][0:4]
                    num_labels = []
                    for j in labels:
                        labels = alphabet.index(j)
                        num_labels.append(int(labels))
                        
                    #生成protocol数据类型
                    example = image_to_tfexample(image_data,num_labels[0],num_labels[1],num_labels[2],num_labels[3])
                    tfrecord_writer.write(example.SerializeToString())
                except IOError as e:
                    print('Error on:',filename)
                    print('Error:',e)
    sys.stdout.write('\n')
    sys.stdout.flush()

if __name__ == '__main__':
    if _dataset_exists(TFRECORD_DIR):
        print('tfrecord已存在')
    else:
        i_filenames = _get_filenames_and_classes(DATASET_DIR)

        training_filenames = i_filenames[_NUM_TEST:]
        testing_filenames = i_filenames[:_NUM_TEST]
        
        _convert_ds('train',training_filenames,DATASET_DIR)
        _convert_ds('test',testing_filenames,DATASET_DIR)

        print('TFRecord has been generated')
    














                    
        












