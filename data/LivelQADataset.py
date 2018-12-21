from __future__ import print_function, unicode_literals
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import sys
sys.path.append("..")
import data.binvox_rw

class LiveIQADataset(object):
    def __init__(self, batch_size=1, shuffle=True, num_epochs=1, crop_shape=[224, 224, 3]):
        self.path_to_train_db = '/home/afan/Reconstruction/3D_GAN/data/train.tfrecord'
        self.path_to_test_db = '/home/afan/Reconstruction/3D_GAN/data/test.tfrecord'

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.crop_shape = crop_shape
        self.num_epochs = num_epochs
        self.means = [103.94, 116.78, 123.68]
        self.shape_dim = 64
        self.azimuth_max = 350
        self.elevation_max = 67
        self.distance = 2.0

    def preprocessing(self, features):
        azimuth = tf.cast(features['azimuth'], tf.float32)
        azimuth = tf.reshape(azimuth, [1])
        elevation = tf.cast(features['elevation'], tf.float32)
        elevation = tf.reshape(elevation, [1])
        distance = tf.cast(features['distance'], tf.float32)
        distance = tf.reshape(distance, [1])

        img = tf.decode_raw(features['image_raw'], tf.uint8)
        img = tf.reshape(img, [256, 256, 3])
        img = tf.to_float(img)
        img = self._mean_image_subtraction(img, self.means, 3)

        shape3D = tf.decode_raw(features['shape3D'], tf.uint8)
        shape3D = tf.reshape(shape3D, [self.shape_dim, self.shape_dim, self.shape_dim, 1])
        shape3D = tf.to_float(shape3D)

        #return img, shape3D, azimuth, elevation, distance
        return img, shape3D

    def decode_tfrecord(self, value):
        features = tf.parse_single_example(value,
                                           features={
                                               'height': tf.FixedLenFeature([], tf.int64),
                                               'width': tf.FixedLenFeature([], tf.int64),
                                               'channel': tf.FixedLenFeature([], tf.int64),
                                               'azimuth': tf.FixedLenFeature([], tf.float32),
                                               'elevation': tf.FixedLenFeature([], tf.float32),
                                               'distance': tf.FixedLenFeature([], tf.float32),
                                               'image_raw': tf.FixedLenFeature([], tf.string),
                                               'shape3D': tf.FixedLenFeature([], tf.string)
                                           })
        return features

    def _mean_image_subtraction(self, image, means, channel):

        image_channels = tf.split(axis=2, num_or_size_splits=channel, value=image)
        for i in range(channel):
            image_channels[i] -= means[i]
        return tf.concat(axis=2, values=image_channels)

    def get_train_dataset(self):
        dataset = tf.data.TFRecordDataset([self.path_to_train_db])
        dataset = dataset.map(self.decode_tfrecord)
        dataset = dataset.map(self.preprocessing)
        dataset = dataset.batch(self.batch_size)
        return dataset

    def get_test_dataset(self):
        dataset = tf.data.TFRecordDataset([self.path_to_test_db])
        dataset = dataset.map(self.decode_tfrecord)
        dataset = dataset.map(self.preprocessing)
        dataset = dataset.batch(self.batch_size)
        return dataset

    def generateTrainTestTF(self, sy_img_path, binvox_dir, train_path, test_path):
        train = open(train_path)
        test = open(test_path)
        train_list = []
        test_list = []
        for line in train:
            train_list.append(line.replace("\n", ""))
        for line in test:
            test_list.append(line.replace("\n", ""))
        self.save(sy_img_path, binvox_dir, './train.tfrecord', train_list)
        self.save(sy_img_path, binvox_dir, './test.tfrecord', test_list)

    def save(self, sy_img_path='', binvox_dir = '', save_path='./train.tfrecord', name_list=[]):
        def _bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _float_feature(value):
            """Returns a float_list from a float / double."""
            return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

        def _int64_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        count = 0
        writer = tf.python_io.TFRecordWriter(save_path)
        #model_namelist = os.listdir(sy_img_path)
        for model_name in name_list:
            count = count + 1
            binvox_path = os.path.join(binvox_dir, model_name + '.binvox')
            imgs_path = os.path.join(sy_img_path, model_name)
            with open(binvox_path, 'rb') as f:
                m = binvox_rw.read_as_3d_array(f)
            m_shape = np.reshape(m.data, (self.shape_dim, self.shape_dim, self.shape_dim, 1))
            m_shape = m_shape.astype(np.uint8)
            print(m_shape.shape)
            m_shape_raw = m_shape.tobytes()

            img_list = os.listdir(imgs_path)
            for img_name in img_list:
                img_path = os.path.join(imgs_path, img_name)
                para = img_name.split('_')
                azimuth = float(para[2])/self.azimuth_max
                elevation = float(para[3])/self.elevation_max
                distance = float(para[5][:-4]) - self.distance
                image = Image.open(img_path)
                width, height = image.size
                image = np.array(image).astype(np.uint8)
                image = image[:, :, :3]
                image_raw = Image.fromarray(image).tobytes()


                feature = {
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'channel': _int64_feature(3),
                    'azimuth': _float_feature(float(azimuth)),
                    'elevation': _float_feature(float(elevation)),
                    'distance': _float_feature(float(distance)),
                    'image_raw': _bytes_feature(image_raw),
                    'shape3D': _bytes_feature(m_shape_raw)
                }

                tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(tf_example.SerializeToString())

            print("write:".ljust(10) + str(count) + "Down!".rjust(4))
        writer.close()

if __name__ == '__main__':
    # dataset = LiveIQADataset()
    # sy_path ='/home/afan/Reconstruction/RenderForTrain/RenderResult/crop_images/03001627'
    # binvox_dir = './ShapeNetVox64'
    # train_path = './train.txt'
    # test_path = './test.txt'
    # dataset.generateTrainTestTF(sy_path, binvox_dir, train_path, test_path)

    dataset = LiveIQADataset()
    data = dataset.get_train_dataset()
    iterator = data.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        for i in range(5):
            print(sess.run(one_element))
