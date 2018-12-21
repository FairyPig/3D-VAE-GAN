import os, time, itertools
import numpy as np
import matplotlib
import sys
sys.path.append("..")
import tensorflow as tf
import numpy as np
from nets.VGAN_network import VGAN_network
from data.LivelQADataset import LiveIQADataset

class runVGAN(object):
    @classmethod
    def default_params(cls):
        '''
        this is the default_params to the params for the this model
        please set the your own params to replace the following dict
        :return: a dict with params
        '''
        return {
            'root_dir': "/home/afan/",
            'summary_dir': "../logs/batch128epochs40",
            'save_dir': "../save/tfdatamodel/",
            'orginal_learing_rate': 0.001,
            'decay_steps': 10,
            'decay_rate': 0.1,
            'momentum': 0.9,
            'epochs': 100,
            'crop_size': 1,
            'batch_size': 32,
            'height': 64,
            'width': 64,
            'channels': 3,
            'D_lr' : 5e-5,
            'G_lr' : 1e-4,
            'alpha_1' : 5,
            'alpha_2' : 5e-4

        }

    def __init__(self):
        # get the params to use
        self.params = self.default_params()

        # set the grpah
        self.graph = tf.Graph()

        # set the gpu options
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        self.keep_prob = 0.8
        self.isTrain = True

        # set the sess
        self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options))

        # define the network
        with self.graph.as_default():
            # define the placeholder
            self.placeholders = {}

            # define the operations dict
            self.ops = {}

            # define the data dict
            self.data = {}
            self.get_DataSet()

            # build the network
            self.build_VGAN_net()

            # build the train options
            self.make_train_step()

            # paramater initilizer or restore model
            self.initial_model()

    def get_DataSet(self):
        # get the dataset from the LiveIQA Dataset
        dataset = LiveIQADataset(batch_size=self.params['batch_size'], shuffle=True, num_epochs=self.params['crop_size'],
                                 crop_shape=[self.params['height'], self.params['width'], self.params['channels']])

        # get the train dataset
        train_dataset = dataset.get_train_dataset()
        # get the test dataset
        test_dataset = dataset.get_test_dataset()

        # put it in the params dict
        self.params['train_dataset'] = train_dataset
        self.params['test_dataset'] = test_dataset

        # set the dict of both train and test
        iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

        # define the init op of train step and test step
        self.ops['train_init_op'] = iter.make_initializer(train_dataset)
        self.ops['test_init_op'] = iter.make_initializer(test_dataset)

        # get the shape3D and image from the dataset
        #self.data['image'], self.data['shape3D'], _, _, _ = iter.get_next()
        self.data['image'], self.data['shape3D'] = iter.get_next()

    def build_VGAN_net(self):

        self.net = VGAN_network()
        self.current_epoch = self.net.init()

        self.placeholders['keep_prob'] = self.keep_prob
        self.placeholders['isTrain'] = self.isTrain
        # networks : encoder
        self.ops['encoder'] = self.net.encoder(self.data['image'], self.keep_prob, self.isTrain)
        self.ops['reshape_z'] = tf.reshape(self.ops['encoder'][0], [-1, 1, 1, 1, 200])
        # networks : generator
        self.ops['decoder'] = self.net.generator(self.ops['reshape_z'], self.isTrain)

        # networks : discriminator
        self.ops['dis_real'] = self.net.discriminator(self.data['shape3D'], self.isTrain)
        self.ops['dis_fake'] = self.net.discriminator(self.ops['decoder'], self.isTrain)

        self.ops['reconstruction_loss'] = tf.reduce_sum(tf.squared_difference(tf.reshape(self.ops['decoder'], (-1, 64 * 64 * 64)), tf.reshape(self.data['shape3D'], (-1, 64 * 64 * 64))), 1)
        self.ops['KL_divergence'] = -0.5 * tf.reduce_sum(1.0 + 2.0 * self.ops['encoder'][2] - self.ops['encoder'][1] ** 2 - tf.exp(2.0 * self.ops['encoder'][2]), 1)
        self.ops['mean_KL'] = tf.reduce_sum(self.ops['KL_divergence'])
        self.ops['mean_recon'] = tf.reduce_sum(self.ops['reconstruction_loss'])

        self.ops['VAE_loss'] = tf.reduce_mean(self.params['alpha_1'] * self.ops['KL_divergence'] + self.params['alpha_2'] * self.ops['reconstruction_loss'])

        self.ops['D_loss_real'] = tf.reduce_mean(self.ops['dis_real'][1])
        self.ops['D_loss_fake'] = tf.reduce_mean(self.ops['dis_fake'][1])
        self.ops['D_loss'] = self.ops['D_loss_real'] - self.ops['D_loss_fake']
        self.ops['G_loss'] = -tf.reduce_mean(self.ops['dis_fake'][1])

        self.train_writer = tf.summary.FileWriter(self.params['summary_dir'], self.sess.graph)
        tf.summary.scalar('loss_recon', self.ops['mean_recon'])
        tf.summary.scalar('loss_kl', self.ops['mean_KL'])
        self.ops['merged'] = tf.summary.merge_all()

    def make_train_step(self):
        self.ops['T_vars'] = tf.trainable_variables()
        self.ops['D_vars'] = [var for var in self.ops['T_vars'] if var.name.startswith('discriminator')]
        self.ops['G_vars'] = [var for var in self.ops['T_vars'] if var.name.startswith('generator')]
        self.ops['E_vars'] = [var for var in self.ops['T_vars'] if var.name.startswith('encoder')]

        self.ops['clip'] = [p.assign(tf.clip_by_value(p, -0.5, 0.5)) for p in self.ops['D_vars']]

        # optimizer for each network
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.ops['D_optim'] = tf.train.RMSPropOptimizer(self.params['D_lr']).minimize(-self.ops['D_loss'], var_list=self.ops['D_vars'])
            self.ops['G_optim'] = tf.train.RMSPropOptimizer(self.params['G_lr']).minimize(self.ops['G_loss'], var_list=self.ops['G_vars'])
            self.ops['E_optim'] = tf.train.AdamOptimizer(self.params['G_lr']).minimize(self.ops['VAE_loss'], var_list=self.ops['E_vars'])

    def initial_model(self):
        self.saver = tf.train.Saver()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)

    def train(self):
        total_step = 0
        with self.graph.as_default():
            for epochs in range(self.params['epochs']):
                self.current_epoch = epochs + 1
                self.sess.run(self.ops['train_init_op'])
                while True:
                    try:
                        for _ in range(4):
                            self.sess.run(self.ops['D_optim'])
                            self.sess.run(self.ops['clip'])
                        loss_d_, loss_g_, _VAE_loss, _KL_divergence, _reconstruction_loss, summary, _, _, _ = \
                            self.sess.run([self.ops['D_loss'], self.ops['G_loss'], self.ops['VAE_loss'], self.ops['mean_KL'], self.ops['mean_recon'], self.ops['merged'], self.ops['D_optim'], self.ops['G_optim'], self.ops['E_optim']])
                        self.sess.run(self.ops['clip'])

                        total_step += 1
                        if total_step % 100 == 0:
                            self.train_writer.add_summary(summary, total_step)
                            print("total_step:", total_step)
                            print("D Loss:", loss_d_)
                            print("G Loss:", loss_g_)
                            print("VAE loss:", _VAE_loss)
                            print("KL divergence:", _KL_divergence)
                            print("reconstruction_loss:", _reconstruction_loss)
                            print("###########")
                            if total_step % 1000 == 0:
                                self.saver.save(self.sess, self.params['save_dir'] + 'saved_' + str(total_step) + 'ckpt')

                    except tf.errors.OutOfRangeError:
                        break

model = runVGAN()
model.train()