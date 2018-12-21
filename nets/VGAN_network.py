import os, time, itertools
import numpy as np
import matplotlib
import tensorflow as tf

class VGAN_network(object):

    def __init__(self):
        self.n_latent = 200

    def getLatent(self):
        return self.n_latent

    def init(self):
        current_epoch = tf.Variable(0, name="current_epoch")
        return current_epoch

    def lrelu(self, x, th = 0.2):
        return tf.maximum(th * x, x)

    def encoder(self, x, keep_prob=0.5, isTrain=True):
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):  # 64 * 64 * 4
            conv1 = tf.layers.conv2d(x, 64, [11, 11], strides=(4, 4), padding='same', \
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())  # 32 * 32 * 128
            lrelu1 = tf.nn.elu(conv1)

            conv2 = tf.layers.conv2d(lrelu1, 128, [5, 5], strides=(2, 2), padding='same',\
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())  # 16 * 16 *256
            lrelu2 = tf.nn.elu(tf.layers.batch_normalization(conv2, training=isTrain))

            conv3 = tf.layers.conv2d(lrelu2, 256, [5, 5], strides=(2, 2), padding='same',\
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())  # 8 * 8 * 512
            lrelu3 = tf.nn.elu(tf.layers.batch_normalization(conv3, training=isTrain))

            conv4 = tf.layers.conv2d(lrelu3, 512, [5, 5], strides=(2, 2), padding='same',\
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())  # 4 * 4 * 1024
            lrelu4 = tf.nn.elu(tf.layers.batch_normalization(conv4, training=isTrain))

            conv5 = tf.layers.conv2d(lrelu4, 400, [8, 8], strides=(1, 1), padding='valid',\
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())  # 1 * 1 * 32
            lrelu5 = tf.nn.sigmoid(tf.layers.batch_normalization(conv5, training=isTrain))

            x = tf.nn.dropout(lrelu5, keep_prob)
            x = tf.contrib.layers.flatten(x)
            z_mu = tf.layers.dense(x, units = self.n_latent)
            z_sig = 0.5 * tf.layers.dense(x, units = self.n_latent)
            epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], self.n_latent]))
            print(tf.shape(x)[0])
            z = z_mu + tf.multiply(epsilon, tf.exp(z_sig))

            return z, z_mu, z_sig

    def generator(self, x, isTrain=True):
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            # 1st hidden layer
            conv1 = tf.layers.conv3d_transpose(x, 512, [4, 4, 4], strides=(1, 1, 1), padding='valid', use_bias=False,
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())  # (-1, 2, 2, 2, 256)
            lrelu1 = tf.nn.elu(tf.layers.batch_normalization(conv1, training=isTrain))

            # 2nd hidden layer
            conv2 = tf.layers.conv3d_transpose(lrelu1, 256, [4, 4, 4], strides=(2, 2, 2), padding='same', use_bias=False,
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())  # (-1, 4, 4, 4, 128)
            lrelu2 = tf.nn.elu(tf.layers.batch_normalization(conv2, training=isTrain))

            # 3rd hidden layer
            conv3 = tf.layers.conv3d_transpose(lrelu2, 128, [4, 4, 4], strides=(2, 2, 2), padding='same', use_bias=False,
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())  # (-1, 8, 8, 8, 64)
            lrelu3 = tf.nn.elu(tf.layers.batch_normalization(conv3, training=isTrain))

            # 4th hidden layer
            conv4 = tf.layers.conv3d_transpose(lrelu3, 64, [4, 4, 4], strides=(2, 2, 2), padding='same', use_bias=False,
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())  # (-1, 16, 16, 16, 32)
            lrelu4 = tf.nn.elu(tf.layers.batch_normalization(conv4, training=isTrain))

            # output layer
            conv5 = tf.layers.conv3d_transpose(lrelu4, 1, [4, 4, 4], strides=(2, 2, 2), padding='same', use_bias=False,
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())  # (-1, 32, 32, 32, 1)
            o = tf.nn.sigmoid(conv5)

            return o

    def discriminator(self, x, isTrain=True):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):  # (-1, 64, 64, 64, 1)
            # 1st hidden layer
            conv1 = tf.layers.conv3d(x, 64, [4, 4, 4], strides=(2, 2, 2), padding='same', use_bias=False,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())  # (-1, 16, 16, 16, 128)
            lrelu1 = tf.nn.elu(conv1)
            # 2nd hidden layer
            conv2 = tf.layers.conv3d(lrelu1, 128, [4, 4, 4], strides=(2, 2, 2), padding='same', use_bias=False,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())  # (-1, 8, 8, 8, 256)
            lrelu2 = tf.nn.elu(tf.layers.batch_normalization(conv2, training=isTrain))

            # 3rd hidden layer
            conv3 = tf.layers.conv3d(lrelu2, 256, [4, 4, 4], strides=(2, 2, 2), padding='same', use_bias=False,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())  # (-1, 4, 4, 4, 512)
            lrelu3 = tf.nn.elu(tf.layers.batch_normalization(conv3, training=isTrain))

            # output layer
            conv4 = tf.layers.conv3d(lrelu3, 512, [4, 4, 4], strides=(2, 2, 2), padding='same', use_bias=False,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            lrelu4 = tf.nn.elu(tf.layers.batch_normalization(conv4, training=isTrain))

            conv5 = tf.layers.conv3d(lrelu4, 1, [4, 4, 4], strides=(1, 1, 1), padding='valid', use_bias=False,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            o = tf.nn.sigmoid(conv5)

            return o, conv5