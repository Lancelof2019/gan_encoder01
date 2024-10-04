import os
import sys
import tensorflow as tf
import pickle
import math
import numpy as np
from sklearn.model_selection import KFold
from hyperopt import hp
from hyperopt import fmin, tpe, Trials, STATUS_OK
from sklearn import model_selection
import hyperopt
import random
import matplotlib.pyplot as plt
import warnings
import time
from datetime import datetime
import logging
import datetime
import copy
import multiprocessing
from functools import partial
import csv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Lambda
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.layers import Layer,Dense,Flatten,Reshape,Input
from tensorflow.keras.layers import BatchNormalization, Activation, Dense
if len(sys.argv) > 1:
    cancer_type = sys.argv[1]  # 接收从命令行传入的第一个参数
else:
    print("Error: No filename provided.")
    sys.exit(1)  # 退出程序，返回非零值表示出错

# 程序继续执行的地方
print(f"Processing for cancer type: {cancer_type}")
max_requirement=20
class inputSmallNetwork(tf.keras.layers.Layer):
    def __init__(self, n_hidden1, n_hidden2, activation, _init):
        super(inputSmallNetwork, self).__init__()
        self.l2 = None
        self.l1 = None
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        # self.is_train = is_train
        # self.concate = self.input_concate
        self.activation = activation

        # self.layer1 = tf.keras.layers.Dense(self.n_hidden1, activation='relu')
        # self.l1_layer = tf.keras.layers.Dense(self.n_input1, kernel_initializer='random_normal', name='layer1')
        # self.l2_layer = tf.keras.layers.Dense(self.n_input2, kernel_initializer='random_normal', name='layer2')
        self.l1_layer = tf.keras.layers.Dense(self.n_hidden1, kernel_initializer=_init, name='layer1')
        self.l2_layer = tf.keras.layers.Dense(self.n_hidden2, kernel_initializer=_init, name='layer2')
        # self.l1=None
        # self.l2=None
        # self.layer2 = tf.keras.layers.Dense(self.n_hidden2, activation='relu')
        # tf.keras.layers.BatchNormalization()(l1, training=bool(self.is_train))

    def call(self, inputs, is_train):
       # print("yes or no")
        #print("In the part of inputSmallNetwork",inputs[0])
        l1 = self.l1_layer(inputs[0])
       # print(type(inputs[0]))
       # print("l1 type is :",type(l1))
        l2 = self.l2_layer(inputs[1])
        #print("we good1?")
        self.is_train = is_train
        l1 = tf.keras.layers.BatchNormalization()(l1, training=bool(self.is_train))
        l2 = tf.keras.layers.BatchNormalization()(l2, training=bool(self.is_train))
        l1 = self.activation(l1)
        l2 = self.activation(l2)

        self.l1 = l1
        self.l2 = l2
        # self.W1 = self.l1_layer.kernel
        # self.W2 = self.l2_layer.kernel
        return self.l1, self.l2

   # def get_weights(self):
       # if not self.l1_layer.built or not self.l2_layer.built:
           # raise ValueError("Weights have not been initialized yet.")
       # return self.l1_layer.kernel, self.l2_layer.kernel, self.l1_layer.bias, self.l2_layer.bias



class encoderNetwork(tf.keras.models.Model):
    def __init__(self, n_hidden1, n_hidden2, n_hiddensh, activation, _init):
        super(encoderNetwork, self).__init__()
        # super().__init__(*args, **kwargs)
        self.l3 = None
        self.ensmallNetwork = inputSmallNetwork(n_hidden1, n_hidden2, activation, _init)
        self.n_hidden3 = n_hiddensh
        # self.l3_layer = tf.keras.layers.Dense(self.n_hidden3, kernel_initializer='random_normal', name='layer3')
        self.l3_layer = tf.keras.layers.Dense(self.n_hidden3, kernel_initializer=_init, name='layer3')

        # l3 = self.l3_layer(tf.concat([self.small_network.l1,self.small_network.l2 ], 1))
        # l1,l2 are bounded in l
        # print("-----layer3 shape", l3.shape)
        # l3 = tf.keras.layers.BatchNormalization()(l3, training=bool(self.is_train))

        # self.concatenate = tf.keras.layers.Concatenate()
        # self.output_layer = tf.keras.layers.Dense(self.concate, activation='softmax')

    def call(self, inputs, is_train):
        ############################################################################
        def sampling(args):
            z_mean, z_log_var = args
            return z_mean + K.exp(0.5 * z_log_var) * K.random_normal(K.shape(z_mean), seed=0)
        ###############################################################################
        self.is_train = is_train
       # print("Ok or not")
        output = self.ensmallNetwork(inputs, self.is_train)
       # print("output is type :",type(output))
       # print("nice output")
        self.W1 = self.ensmallNetwork.l1_layer.kernel
        self.W2 = self.ensmallNetwork.l2_layer.kernel
        self.bias1 = self.ensmallNetwork.l1_layer.bias
        self.bias2 = self.ensmallNetwork.l2_layer.bias

        # self.l3_layer = tf.keras.layers.Dense(self.n_hiddensh, kernel_initializer=self.init, name='layer3')
        l3 = self.l3_layer(tf.concat([output[0], output[1]], 1))
       # l3 = tf.keras.layers.BatchNormalization()(l3, training=bool(self.is_train))

    
        model =  tf.keras.layers.BatchNormalization()(l3, training=bool(self.is_train))
        model = Activation('tanh')(model)
        model = Dense(self.n_hidden3)(model)
       # self.l3 = l3
        self.l3 = model
        self.Wsht = self.l3_layer.kernel
        self.bias3 = self.l3_layer.bias
        
        ########################################################################
        self.z_mean = Dense(self.n_hidden3)(self.l3)
        self.z_log_var = Dense(self.n_hidden3)(self.l3)
        self.z = Lambda(sampling, output_shape=(self.n_hidden3,), name='z')([self.z_mean, self.z_log_var])
        # self.Wsht = self.l3_layer.kernel
        ########################################################################
        return self.l3, self.z_mean, self.z_log_var, self.z

   
    
   # def get_weights(self):
       # if not self.l3_layer.built:
           # raise ValueError("Weights have not been initialized yet.")
       # return self.l3_layer.kernel, self.l3_layer.bias
##################################################### # encoder GAN######################################################



class outputSmallNetwork(tf.keras.layers.Layer):
    def __init__(self, n_input1, n_input2, activation, _init):
        super(outputSmallNetwork, self).__init__()
        self.l5 = None
        self.l6 = None
        self.n_hidden5 = n_input1
        self.n_hidden6 = n_input2
        self.activation = activation
        # self.layer1 = tf.keras.layers.Dense(self.n_hidden1, activation='relu')
        # self.l5_layer = tf.keras.layers.Dense(self.n_hidden5, kernel_initializer='random_normal', name='layer1')
        # self.l6_layer = tf.keras.layers.Dense(self.n_hidden6, kernel_initializer='random_normal', name='layer2')

        self.l5_layer = tf.keras.layers.Dense(self.n_hidden5, kernel_initializer=_init, name='layer5')
        self.l6_layer = tf.keras.layers.Dense(self.n_hidden6, kernel_initializer=_init, name='layer6')
        # self.l1=None
        # self.l2=None
        # self.layer2 = tf.keras.layers.Dense(self.n_hidden2, activation='relu')
        # tf.keras.layers.BatchNormalization()(l1, training=bool(self.is_train))

    def call(self, inputs, is_train):
        #print("we good 2?")
        l5 = self.l5_layer(inputs[0])
        l6 = self.l6_layer(inputs[1])
        self.is_train = is_train
        l5 = tf.keras.layers.BatchNormalization()(l5, training=bool(self.is_train))
        l6 = tf.keras.layers.BatchNormalization()(l6, training=bool(self.is_train))
        l5 = self.activation(l5)
        #print("l5 is type of data:",type(l5))
        l6 = self.activation(l6)

        self.l5 = l5
        self.l6 = l6
        # self.W1t = self.l5_layer.kernel
        # self.W2t = self.l6_layer.kernel
        return self.l5, self.l6
        

   # def get_weights(self):
        #if not self.l6_layer.built or not self.l5_layer.built:
           #raise ValueError("Weights have not been initialized yet.")
       #return self.l5_layer.kernel, self.l6_layer.kernel, self.l5_layer.bias, self.l6_layer.bias

class decoderNetwork(tf.keras.models.Model):
    def __init__(self, n_input1, n_input2, n_hidden1, n_hidden2, activation, _init):
        super(decoderNetwork, self).__init__()

        # self.l4_layer = tf.keras.layers.Dense(n_hidden1 + n_hidden2, kernel_initializer='random_normal',
        # name='layer4')
        self.l4_layer = tf.keras.layers.Dense(n_hidden1 + n_hidden2, kernel_initializer=_init,
                                              name='layer4')
        # self.l5_layis_trainer = tf.keras.layers.Dense(input_n_hidden1, kernel_initializer=self.init, name='layer5')
        # self.l6_layer = tf.keras.layers.Dense(input_n_hidden2, kernel_initializer=self.init, name='layer6')
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_hidden5 = n_input1
        self.n_hidden6 = n_input2
        self.outsmallNetwork = outputSmallNetwork(self.n_hidden5, self.n_hidden6, activation, _init)

    def call(self, inputs, is_train):
        l4 = self.l4_layer(inputs)
        self.is_train = is_train
        # self.is_train = True
        l4 = tf.keras.layers.BatchNormalization()(l4, training=bool(self.is_train))

        self.l4 = l4
       # print("we good 3?" )
        # self.Wsh = self.l4_layer.kernel
        output = tf.split(l4, [self.n_hidden1, self.n_hidden2], 1)

        #print("output is type:",type(output))
        #print("we good 4?")
        l5, l6 = self.outsmallNetwork(output, self.is_train)

        self.Wsh = self.l4_layer.kernel
        self.bias4 = self.l4_layer.bias
        self.W1t = self.outsmallNetwork.l5_layer.kernel
        self.bias5 = self.outsmallNetwork.l5_layer.bias
        self.W2t = self.outsmallNetwork.l6_layer.kernel
        self.bias6 = self.outsmallNetwork.l6_layer.bias

        return l5, l6


    def get_weights(self):
        if not self.l4_layer.built:
            raise ValueError("Weights have not been initialized yet.")
        return self.l4_layer.kernel, self.l4_layer.bias

#class SelfAttention(Layer):
#    def __init__(self, units):
 #       super(SelfAttention, self).__init__()
 #       self.units = units

  #  def build(self, input_shape):
   #     self.W_q = self.add_weight(shape=(input_shape[-1], self.units),
    #                               initializer='glorot_uniform',
    #                               trainable=True)
    #    self.W_k = self.add_weight(shape=(input_shape[-1], self.units),
    #                               initializer='glorot_uniform',
      #                             trainable=True)
     #   self.W_v = self.add_weight(shape=(input_shape[-1], self.units),
      #                             initializer='glorot_uniform',
      #                             trainable=True)
      #  super(SelfAttention, self).build(input_shape)

    #def call(self, inputs):
    #    Q = K.dot(inputs, self.W_q)
    #    K_mat = K.dot(inputs, self.W_k)
    #    V = K.dot(inputs, self.W_v)

        # Scaled dot-product attention
     #   attention_scores = K.batch_dot(Q, K.permute_dimensions(K_mat, (0, 2, 1)))
      #  attention_scores = attention_scores / K.sqrt(K.cast(self.units, dtype=tf.float32))
      #  attention_weights = K.softmax(attention_scores, axis=-1)

       # attention_output = K.batch_dot(attention_weights, V)
       # return attention_output

class Discriminator(tf.keras.models.Model):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()
        self.model = tf.keras.Sequential([
         tf.keras.layers.Dense(latent_dim),
         tf.keras.layers.BatchNormalization(),
         tf.keras.layers.Activation('gelu'),
         tf.keras.layers.Dense(1, activation='sigmoid')
         ])

    def call(self, inputs):
        return self.model(inputs)

        # 检查 input1 和 input2 的形状
           #if len(input1.shape) == 2:  # 如果是2D输入
           #     input1 = tf.expand_dims(input1, axis=-1)  # 扩展为3D
           #if len(input2.shape) == 2:  # 同样处理第二个输入
           #     input2 = tf.expand_dims(input2, axis=-1)



class Discriminatord(tf.keras.models.Model):
    def __init__(self, latent_dim):
        super(Discriminatord, self).__init__()

        # 分别为 input1 和 input2 定义 Dense 层
        self.input1_dense = tf.keras.Sequential([
            tf.keras.layers.Dense(latent_dim),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('gelu')
        ])

        self.input2_dense = tf.keras.Sequential([
            tf.keras.layers.Dense(latent_dim),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('gelu')
        ])

        # 最后的输出层，用于二分类
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')


    def call(self, inputs):
        # 分别处理两个输入
        input1, input2 = inputs

        # 分别通过各自的 Dense 层进行处理
        input1_processed = self.input1_dense(input1)
        input2_processed = self.input2_dense(input2)

        # 合并两个处理后的特征向量
        combined_inputs = tf.concat([input1_processed, input2_processed], axis=0)

        # 通过最后的输出层进行二分类
        return self.output_layer(combined_inputs)

class Autoencoder(tf.keras.models.Model):
    def __init__(self, n_input1, n_input2,  iterator, activation):
        super(Autoencoder, self).__init__()
        self.bias6 = None
        self.bias5 = None
        self.W2t = None
        self.W1t = None
        self.bias4 = None
        self.Wsh = None
        self.Wsht = None
        self.bias3 = None
        self.bias2 = None
        self.bias1 = None
        self.W2 = None
        self.W1 = None
        self.n_hidden2 = None
        self.is_train = None
        self.n_hiddensh = 1
        self._init = None

        self.n_input1 = n_input1
        self.n_input2 = n_input2
        # self.n_hidden1 = n_hidden1
        # self.n_hidden2 = n_hidden2
        # self.optimizer = tf.keras.optimizers.legacy.Adam()
        self.max_epochs = 1500
        self.require_improvement = 20
        self.iterator = iterator
        self.activation = activation
        self.trigger = False
        # self.lamda = 0.13
        # self.alpha = 0.012
        # self.learning_rate = 0.032
        # self.inputData=X_train

    def call(self, inputs, is_train):
        self.sampleInput = inputs
        # print("self._init is:",self._init)

        # print("#################################################################")
        # print(inputs)
        # print("__________________________________________________")
        # print(self.sampleInput)
        # print("#################################################################")
        # self.temp_record = inputs
        # print("The first time of sampleInput",type(sampleInput))

        self.is_train = is_train
        encoded, z_mean, z_log_var, z = self.encoder(inputs, self.is_train)
        decoded = self.decoder(z, self.is_train)

        # self.W1, self.W2,self.bias1,self.bias2= self.encoder.ensmallNetwork.get_weights()
        # self.W2 = self.encoder.ensmallNetwork.
        # self.Wsht,self.bias3= self.encoder.get_weights()
        # self.Wsh,self.bias4= self.decoder.get_weights()
        # self.W1t, self.W2t ,self.bias5,self.bias6= self.decoder.outsmallNetwork.get_weights()

        self.W1 = self.encoder.W1
        self.W2 = self.encoder.W2
        self.bias1 = self.encoder.bias1
        self.bias2 = self.encoder.bias2
        self.Wsht = self.encoder.Wsht
        self.bias3 = self.encoder.bias3

        self.Wsh = self.decoder.Wsh
        self.bias4 = self.decoder.bias4

        self.W1t = self.decoder.W1t
        self.bias5 = self.decoder.bias5
        self.W2t = self.decoder.W2t
        self.bias6 = self.decoder.bias6

        return decoded,z_mean, z_log_var, z
        # self.W2t = self.decoder.outsmallNetwork.W2t
        # print(self.W1)
        #  print("======================================")
        # print(self.W2)
        # print("======================================")
        # print(self.Wsht)
        # print("======================================")
        # print(self.Wsh)
        # print("======================================")
        # print(self.W1t)
        # print("======================================")
        # print(self.W2t)
        # return decoded

    def L1regularization(self, weights):
        return tf.reduce_sum(tf.abs(weights))

    def L2regularization(self, weights, nbunits):
        return math.sqrt(nbunits) * tf.nn.l2_loss(weights)

    def lossfun(self, sampleInput, is_train, init,logger):
        ###########################################################################
        # print("self._init is:", init)
        #print("The length is :",sampleInput[0].shape)
        #print("The type of sampleInput[0]:",type(sampleInput[0]))

        if self.trigger == False:
            self.encoder = encoderNetwork(self.n_hidden1, self.n_hidden2, self.n_hiddensh, self.activation, init)
            #self.encoder , self.z_mean, self.z_log_var, self.z = encoderNetwork(self.n_hidden1, self.n_hidden2, self.n_hiddensh, self.activation, init)
            self.decoder = decoderNetwork(self.n_input1, self.n_input2, self.n_hidden1, self.n_hidden2, self.activation,init)
            self.trigger = True
            
            # print("self._init is:", self._init)
        #logger.info(f"The self.trigger is :{self.trigger}")
        # print("#################################################################")
        # print(inputs)
        # print("__________________________________________________")
        # print(self.sampleInput)
        # print("#################################################################")
        # self.temp_record = inputs
        # print("The first time of sampleInput",type(sampleInput))
        
       # self.attention = SelfAttention(units=self.n_hidden1 + self.n_hidden2)
       # attention_output = self.attention(sampleInput)
        self.is_train = is_train
        
        #self.encoded = self.encoder(sampleInput, self.is_train)
        
        #self.encoded = self.encoder(attention_output, self.is_train)
        
        self.encoded, z_mean, z_log_var, z = self.encoder(sampleInput, self.is_train)
        self.decoded = self.decoder(z, self.is_train)

        # self.W1, self.W2,self.bias1,self.bias2= self.encoder.ensmallNetwork.get_weights()
        # self.W2 = self.encoder.ensmallNetwork.
        # self.Wsht,self.bias3= self.encoder.get_weights()
        # self.Wsh,self.bias4= self.decoder.get_weights()
        # self.W1t, self.W2t ,self.bias5,self.bias6= self.decoder.outsmallNetwork.get_weights()

        self.W1 = self.encoder.W1
        self.W2 = self.encoder.W2
        self.bias1 = self.encoder.bias1
        self.bias2 = self.encoder.bias2
        self.Wsht = self.encoder.Wsht
        self.bias3 = self.encoder.bias3

        self.Wsh = self.decoder.Wsh
        self.bias4 = self.decoder.bias4

        self.W1t = self.decoder.W1t
        self.bias5 = self.decoder.bias5
        self.W2t = self.decoder.W2t
        self.bias6 = self.decoder.bias6
        # self.H = self.encodefun(X1, X2)
        # X1_, X2_ = self.decodefun(self.H)
        # self.get_weights()
        # print("The 2nd time of sampleInput", type(sampleInput))
        # self.compareOutPut = self.call(sampleInput,is_train)
        ###########################################################################
        #self.compareOutPut = self.decoded
        # print("The 3rd time of sampleInput", type(sampleInput))
        sgroup_lasso = self.L2regularization(self.W1, self.n_input1 * self.n_hidden1) + \
                       self.L2regularization(self.W2, self.n_input2 * self.n_hidden2)
        # print(sgroup_lasso.shape)
        # lasso
        lasso = self.L1regularization(self.W1) + self.L1regularization(self.W2) + \
                self.L1regularization(self.Wsh) + self.L1regularization(self.Wsht) + \
                self.L1regularization(self.W1t) + self.L1regularization(self.W2t)
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print("The big value is :", self.L2regularization(self.W1, self.n_input1 * self.n_hidden1))

        error = tf.reduce_mean(tf.square(sampleInput[0] - self.decoded[0])) + tf.reduce_mean(
            tf.square(sampleInput[1] - self.decoded[1]))
        reconstruction_loss = 0.5 * error + 0.5 * self.lamda * (1 - self.alpha) * sgroup_lasso + 0.5 * self.lamda * self.alpha * lasso
        
        self.l3=self.encoder.l3
        
        num_samples=sampleInput[0].shape[0]
        print("The num of samples are:",sampleInput[0].shape[0])
        ######################GAN_loss###########################
        self.discriminator=Discriminator(self.n_hiddensh)
        self.decoder_discriminator=Discriminatord(self.n_hidden1 + self.n_hidden2)
        #############################################################
        z_real = tf.random.normal(shape=(num_samples, self.n_hiddensh))
        z_fake = z
        real_labels = 0.9
        d_real = self.discriminator(z_real)
        d_fake = self.discriminator(z_fake)
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        d_loss_real = cross_entropy(tf.ones_like(d_real)*real_labels, d_real)
        d_loss_fake = cross_entropy(tf.zeros_like(d_fake), d_fake)
        discriminator_loss = 0.5*d_loss_real + 0.5*d_loss_fake
        generator_loss = cross_entropy(tf.ones_like(d_fake), d_fake)
        #total_loss = 0.5*reconstruction_loss + 0.5*generator_loss + 0.5*discriminator_loss
        self.z=z
        fake_output = self.decoded  # decoder 输出生成的假数据
        real_output = sampleInput  # 输入的真实数据a

        #fake_output1, fake_output2 = fake_output  # 将假数据解包
        #real_output1, real_output2 = real_output  # 将真实数据解包
    
        # 判别器对真实数据和生成数据进行判别
        d_real_output = self.decoder_discriminator(real_output)
        d_fake_output = self.decoder_discriminator(fake_output)
    
        # 判别器对 decoder 输出的损失
        d_loss_real_output = cross_entropy(tf.ones_like(d_real_output) * real_labels, d_real_output)
        d_loss_fake_output = cross_entropy(tf.zeros_like(d_fake_output), d_fake_output)
        print("loss shape:",d_loss_fake_output.shape)
        # 判别器针对 decoder 输出的总损失
        discriminator_loss_output = 0.5 * d_loss_real_output + 0.5 * d_loss_fake_output
        print("loss shape:",discriminator_loss_output.shape)
    
        # 生成器针对 decoder 输出的损失
        generator_loss_decoder = cross_entropy(tf.ones_like(d_fake_output), d_fake_output)
       
        ###########################################################################
       # 总损失 = 重建损失 + 对抗损失（生成器和判别器）
        total_loss = 0.5 * reconstruction_loss + 0.25 * (generator_loss + generator_loss_decoder) + \
                 0.25 * (discriminator_loss + discriminator_loss_output)
        
        return total_loss, reconstruction_loss, discriminator_loss, generator_loss, generator_loss_decoder ,discriminator_loss_output

##############################################################################################test#######################
    def testprocess(self, inputs, iterator, params):
        #global _init, optimizer
        # self.batch_size  = params['batch_size']

        #self.n_hidden1 = params['units1']
        #self.n_hidden2 = params['units2']

        self.n_hidden1 = self.n_input1
        self.n_hidden2 = self.n_input2
        self.alpha = params['alpha']
        self.lamda = params['lamda']
        self.learning_rate = params['learning_rate']

        self.require_improvement = 50
        self.max_epochs = 1500

        init = params['initializer']
        if init == 'normal':
            self.init = tf.keras.initializers.RandomNormal(mean=0., stddev=0.01)
        if init == 'uniform':
            self.init = tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01)
        current_time = int(time.time())
        if init == 'He':
            self.init = tf.keras.initializers.HeNormal(seed=current_time)
        if init == 'xavier':
            self.init = tf.keras.initializers.GlorotNormal(seed=current_time)
        self.init=self.init
        #self.init = _init
        opt = params['optimizer']

        if opt == 'SGD':
            # self.optimizer = tf.keras.optimizers.SGD()
            optimizer = tf.keras.optimizers.legacy.SGD()
        if opt == 'adam':
            # self.optimizer = tf.keras.optimizers.Adam()
            optimizer = tf.keras.optimizers.legacy.Adam()
        if opt == 'nadam':
            # self.optimizer = tf.keras.optimizers.Nadam()
            optimizer = tf.keras.optimizers.legacy.Nadam()
        if opt == 'Momentum':
            # self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
            optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=self.learning_rate, momentum=0.9)
        if opt == 'RMSProp':
            # self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
            optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=self.learning_rate)

        self.optimizer = optimizer
        # self._init = _init

        print("H.layer1:", self.n_hidden1, ", H.layer2:", self.n_hidden2)
        print( "lamda", self.lamda,  "alpha:", self.alpha, ", learning_rate:", self.learning_rate)
        print("initializer: ", init, ', optimizer:',  opt)

        # loss, res = self.test(inputs, iterator)
        costs = []  # long memoery
        costs_inter = []
        # for early stopping:
        best_cost = 10000
        #best_cost_cost = 100000
        stop = False
        last_improvement = 0
        epoch = 0
        counter = 0
        #batch_xs1 = inputs[0]
        #batch_xs2 = inputs[1]
        #print(batch_xs1)
        #print(type(inputs))
        while epoch < self.max_epochs and stop == False:
            avg_cost = 0.
            # Loop over all batches
            # for sample in mini_batches:

            self.is_train = True
           # with tf.GradientTape(persistent=True) as tape_ae,tf.GradientTape() as tape_d:

                 #total_loss, reconstruction_loss, discriminator_loss, generator_loss = self.lossfun(inputs, self.is_train, self.init, logger=None)
            with tf.GradientTape(persistent=True) as tape:
                 total_loss, reconstruction_loss, discriminator_loss, generator_loss, generator_loss_decoder, discriminator_loss_output = self.lossfun(inputs, self.is_train, self.init, logger=None)
                 #current_loss = self.lossfun(inputs, self.is_train, self.init, logger=None)

            #ae_gradients = tape_ae.gradient(total_loss, self.trainable_variables)
            #ae_gradients = tape_ae.gradient(total_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
            #d_gradients = tape_d.gradient(discriminator_loss, self.discriminator.trainable_variables)
            
            
            # self.optimizer.apply_gradients(zip(ae_gradients, self.encoder.trainable_variables + self.decoder.trainable_variables))
            #self.optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
            
           

            # gen_gradients = tape_ae.gradient(generator_loss, self.encoder.trainable_variables)
            #self.optimizer.apply_gradients(zip(gen_gradients, self.encoder.trainable_variables)) 
            d_gradients = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
            decoder_d_gradients = tape.gradient(discriminator_loss_output, self.decoder_discriminator.trainable_variables)
            
            
            gen_gradients_encoder = tape.gradient(generator_loss, self.encoder.trainable_variables)
            gen_gradients_decoder = tape.gradient(generator_loss_decoder, self.decoder.trainable_variables)
            
            ae_gradients = tape.gradient(total_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
            
            
            
            
            self.optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
            self.optimizer.apply_gradients(zip(decoder_d_gradients, self.decoder_discriminator.trainable_variables))
            self.optimizer.apply_gradients(zip(gen_gradients_encoder, self.encoder.trainable_variables))
            self.optimizer.apply_gradients(zip(gen_gradients_decoder, self.decoder.trainable_variables))
            self.optimizer.apply_gradients(zip(ae_gradients, self.encoder.trainable_variables + self.decoder.trainable_variables))
            total_loss, reconstruction_loss, discriminator_loss, generator_loss, generator_loss_decoder, discriminator_loss_output = self.lossfun(inputs, self.is_train, self.init, logger=None)
            #avg_cost = cost

            costs_inter.append(total_loss)  #
            costs += costs_inter
            costs_inter = []
            # print("-----------------------------The costs_inter is :----------------------")
            # print(costs_inter)
            # print("-----------------------------The costs_inter is :----------------------")
            #####################################################################################################

            #if cost < best_cost:

             #   best_cost = cost
             #   costs += costs_inter  # costs history of the training set

             #   last_improvement = 0

              #  costs_inter = []

           # else:
            #    last_improvement += 1

           # if last_improvement > self.require_improvement:
                # print("No improvement found during the (self.require_improvement) last iterations, stopping optimization.")
                # Break out from the loop.
            #    stop = True

            epoch += 1
        cmt_scores = self.z
        cmt_scores=cmt_scores.numpy()
        #costs=costs.numpy()
        with open(
            r'../data/python_related/result/community/communityScores_compare_'+str(cancer_type)+"_"+str(iterator)+'_test14_01.csv',
            'w',newline='', encoding='utf-8') as csvfile:
             writer = csv.writer(csvfile, lineterminator='\n')
             [writer.writerow(r) for r in cmt_scores]
        print("--------------------------------------------------------------")
        print(costs)
        print("--------------------------------------------------------------")
        plt.figure()
        plt.plot(costs)
        plt.ylabel('Loss')
        plt.xlabel('Iterations')
        plt.title("Learning rate =" + str(round(self.learning_rate, 9))+"@"+str(cancer_type)+" "+str(self.n_input1)+","+str(self.n_input2)+"@"+str(self.n_hidden1)+","+str(self.n_hidden2)+"\n")
        plt.savefig(r'../figure/loss_curve/getscores_picture_' + str(cancer_type) + "_" + str(iterator) + '_test14_01.png')
        plt.close()



def partialtestProcess(iterator, selected_features, inputhtseq, inputmethy, act):
    #global inputs_tensor, best, n_input1, n_input2
    trials = {}

    fname = r'../data/python_related/result/comm_trials_binary_' + str(
        cancer_type) + "_" + str(iterator) + '_test14.pkl'


    #../data/python_related/result/comm_trials_binary_SKCM_20_test14.pkl

    # fname = r'C:\Users\gklizh\Documents\Workspace\code_and_data19\data\python_related\result\comm_trials_binary_' + str(
    #     cancer_type) + "_" + str(iterator) + '_test10.pkl'
    # with open(fname, 'wb+') as fpkl:
    #     pass
    print('iteration', iterator)
    selected_feat_cmt = selected_features[np.where(selected_features[:, 0] == iterator + 1)[0], :]

    print('first source ...')
    htseq_cmt = selected_feat_cmt[np.where(selected_feat_cmt[:, 1] == 1)[0], :]
    htseq_nbr = len(htseq_cmt)
    htseq_sel_data = inputhtseq[:, htseq_cmt[:, 2].astype(int) - 1]

    print("second source ...")
    methy_cmt = selected_feat_cmt[np.where(selected_feat_cmt[:, 1] == 2)[0], :]
    methy_nbr = len(methy_cmt)
    methy_sel_data = inputmethy[:, methy_cmt[:, 2].astype(int) - 1]

    print("features size of the 1st dataset:", htseq_nbr)
    print("features size of the 2nd dataset:", methy_nbr)

    # n_hidden1 = htseq_nbr# the same as n_input1 below
    # n_hidden2 = methy_nbr#

    ##################################################
    # create logger recorder

    current_time = datetime.datetime.now().strftime("%Y-%m-%d")
    log_filename = f'demoapp.{current_time}_test_{iterator}.log'

    logger = logging.getLogger(f'demoapp_{iterator}')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_filename)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.info(f'Partial process started {iterator}')
    input_data = open(fname, 'rb')

    ##################################################

    if htseq_nbr > 1 and methy_nbr > 1:
        # split dataset to training and test data 80%/20%
        trials = pickle.load(input_data)
        best = trials.best_trial['result']['params']
        X_train1 = htseq_sel_data
        X_train2 = methy_sel_data

        print("The cancer is :"+str(cancer_type))
        print("The community is :"+str(iterator))
        print("-------------------------------------------------------------")
        print(best)
        print("-------------------------------------------------------------")
        sampleInput = [X_train1, X_train2]  # 364x120

        is_train = True
        # print(X_train1)
        n_input1 = X_train1.shape[1]
        n_input2 = X_train2.shape[1]
        print("data 1 shape of col:", n_input1)
        print("data 2 shape of col:", n_input2)


        X_train1 = tf.convert_to_tensor(X_train1, dtype=tf.float32)
        X_train2 = tf.convert_to_tensor(X_train2, dtype=tf.float32)

        inputs_tensor = tf.convert_to_tensor(sampleInput, dtype=tf.float32)
        sae = Autoencoder(n_input1, n_input2, iterator, activation=act)

        # space = {
        #    'units1': hp.choice('units1', range(1, n_hidden1)),
        #    'units2': hp.choice('units2', range(1, n_hidden2)),
        #    'batch_size': hp.choice('batch_size', [16, 8, 4]),
        #    'alpha': hp.choice('alpha', [0, hp.uniform('alpha2', 0, 1)]),
        #    'learning_rate': hp.loguniform('learning_rate', -5, -1),
        #    'lamda': hp.choice('lamda', [0, hp.loguniform('lamda2', -8, -1)]),
        #    'optimizer': hp.choice('optimizer', ["adam", "nadam", "SGD", "Momentum", "RMSProp"]),
        #    'initializer': hp.choice('initializer', ["xavier"]),
        # }
        sae.testprocess(inputs_tensor, iterator, best)

        del htseq_sel_data
        del methy_sel_data
        # del trials
        del sae





if __name__ == '__main__':
    #f = open(
       # r'C:\Users\gklizh\Documents\Workspace\code_and_data12\data\python_related\result\comm_parameter_binary26.txt',
       # 'w+')
    #f.close()

    #fname = r'C:\Users\gklizh\Documents\Workspace\code_and_data12\data\python_related\result\comm_trials_binary_test26.pkl'
   # with open(fname, 'wb+') as fpkl:
    #    pass


    community_index = int(os.getenv('SLURM_ARRAY_TASK_ID', '1')) - 1
    
    feature_path = os.path.join(r'/projappl/project_2010541/data/python_related/data',
                         cancer_type + "_selected_features14.csv")

    selected_features = np.genfromtxt(feature_path,delimiter=',', skip_header=1)
    # log10 (fpkm + 1)

    exp_path = os.path.join(r'/projappl/project_2010541/data/python_related/data',
                            cancer_type + "_exp_intgr14.csv")
    inputhtseq = np.genfromtxt(exp_path,dtype=np.unicode_, delimiter=',', skip_header=1)
    inputhtseq = inputhtseq[:, 1:inputhtseq.shape[1]].astype(float)
    inputhtseq = np.divide((inputhtseq - np.mean(inputhtseq)), np.std(inputhtseq))
    print(inputhtseq.shape)
    mty_path = os.path.join(r'/projappl/project_2010541/data/python_related/data',
                            cancer_type + "_mty_intgr14.csv")
    # methylation β values
    inputmethy = np.genfromtxt(mty_path, dtype=np.unicode_, delimiter=',', skip_header=1)
    inputmethy = inputmethy[:, 1:inputmethy.shape[1]].astype(float)
    inputmethy = np.divide((inputmethy - np.mean(inputmethy)), np.std(inputmethy))
    print(inputmethy.shape)
    #community_num = 21

    act = tf.nn.tanh
    # C:\Program Files\R\R-4.3.3\
    # 使用R的print函数来打印一个消息

    #ro.r('options(stringsAsFactors = F)')
    #ro.r('suppressPackageStartupMessages(library(NbClust))')
    #ro.r('suppressPackageStartupMessages(library(ggplot2))')
    #ro.r('suppressPackageStartupMessages(library(grid))')
    #ro.r('suppressPackageStartupMessages(library(ComplexHeatmap))')
    #ro.r('suppressPackageStartupMessages(library(circlize))')
    #ro.r('suppressPackageStartupMessages(library(tidyverse))')
    #ro.r('suppressPackageStartupMessages(library(maftools))')
    #load_file = 'C:/Users/gklizh/Documents/Workspace/code_and_data12/data/spinglass/melanet_cmt.RData'

    # 将文件路径传递给R环境
    #ro.r(f"load_file <- '{load_file}'")

    # 使用这个变量来读取RDS文件
    #ro.r("melanet_cmt <- readRDS(load_file)")

    # 获取社区的数量
   # ro.r('number_of_communities <- length(melanet_cmt)')

    # 在Python中获取值
    #community_num = ro.r('number_of_communities')[0]
    #print(community_num)


    #num_processes=7
    #num_processes = 8
    #community_num = 25
    # parallel_processing(selected_features, inputhtseq, inputmethy, num_processes, act, community_num)
    #parallel_processing(selected_features, inputhtseq, inputmethy, num_processes, act, community_num)

    #parallel_testprocessing(selected_features, inputhtseq, inputmethy, num_processes, act, community_num)
    #partialProcess(community_index, selected_features, inputhtseq, inputmethy, act)
    
    
    partialtestProcess(community_index, selected_features, inputhtseq, inputmethy, act)
    # with open('script.py', 'r') as file:
    #     script_content = file.read()
    #
    # # 执行读取的内容
    # exec(script_content)
    # tanh activation function

    # trials = {}
    # run the autoencoder for communities
