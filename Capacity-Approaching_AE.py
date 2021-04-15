from __future__ import absolute_import, division, print_function, unicode_literals

from keras.layers import Input, Dense, GaussianNoise, Concatenate, Lambda, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras import backend as K
from uniform_noise import UniformNoise

import os
import argparse

K.clear_session()
import scipy.io as sio
import numpy as np
import tensorflow as tf

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

# custom cross-entropy to allow gradient separation
def categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0.2):
    y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)

    if label_smoothing is not 0:
        smoothing = K.cast_to_floatx(label_smoothing)

        def _smooth_labels():
            num_classes = K.cast(K.shape(y_true)[1], y_pred.dtype)
            return y_true * (1.0 - smoothing) + (smoothing / num_classes)

        y_true = K.switch(K.greater(smoothing, 0), _smooth_labels, lambda: y_true)
    return K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)

# MINE loss
def mine_loss(args):
    t_xy = args[0]
    t_xy_bar = args[1]
    loss = -(K.mean(t_xy) - K.logsumexp(t_xy_bar) + K.log(tf.cast(K.shape(t_xy)[0], tf.float32)))
    return loss

# mixed loss
def customLoss(MI):
  def dice(yTrue, yPred):
      beta = 0.2 # BETA PARAMETER IN THE PAPER, to choose at the beginning
      return categorical_crossentropy(yTrue, yPred) - beta*MI

  return dice

# shuffling for MINE input
def data_generation_mi(data_x, data_y):
    data_xy = np.hstack((data_x, data_y))
    data_y_shuffle = np.take(data_y, np.random.permutation(data_y.shape[0]), axis=0, out=data_y)
    data_x_y = np.hstack((data_x, data_y_shuffle))
    return data_xy, data_x_y

# BLER computation
def compute_BER(x,y):
    sh = x.shape
    p = sh[0]
    ber = 0
    for i in range(p):
        if np.argmax(y[i,:]) != np.argmax(x[i,:]):
            ber = ber +1
    return ber/p

# get a "l2 norm of gradients" tensor
def get_gradient_norm(model):
    with K.name_scope('gradient_norm'):
        grads = K.gradients(model.total_loss, model.trainable_weights)
        norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
    return norm

# shuffling tensor
def shuffleColumns(x):
    return tf.gather(x, tf.random.shuffle(tf.range(tf.shape(x)[0])))

# for parser
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class CAAE():
    def __init__(self, encoder_path, decoder_path, output_directory):
        block_size = 5 # k parameter
        num_enc_inputs = pow(2, block_size)  # M parameter = 2^k
        num_hidden = 2  # 2*N parameter
        R = block_size / num_hidden # rate
        EbN0dB = 7 # training parameter
        N = pow(10, -0.1 * EbN0dB) / (2 * R) # training variance
        K.set_learning_phase(1) # set flag to allow noise layer during training
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.output_directory = output_directory

        # if model is already available, load it
        if os.path.exists(self.encoder_path) and os.path.exists(self.decoder_path):
            self.encoder = load_model(self.encoder_path)
            self.decoder = load_model(self.decoder_path)
            print("Loaded models...")

        else:
            # input shape
            self.num_enc_inputs = num_enc_inputs
            self.num_hidden = num_hidden
            self.num_joint = 2*num_hidden

            # set optimizers
            optimizer = Adam(0.01, 0.5)
            optimizer_MI = Adam(0.01, 0.5)

            # build the transmitter
            self.transmitter = self.build_transmitter()

            # build the receiver
            self.receiver = self.build_receiver()

            # build the mutual information block estimator
            self.discriminator = self.build_discriminator()

            # the transmitter encodes the bits in s
            s_in = Input(shape=(self.num_enc_inputs,))
            x = self.transmitter(s_in)

            # build the channel
            #x_n = Lambda(lambda x: np.sqrt(num_hidden)*K.l2_normalize(x,axis=1))(x) # if CONSTANT POWER
            x_n = BatchNormalization(axis=-1, center=False, scale=False)(x)  # if AVERAGE POWER, use batch normalization

            ch = Lambda(lambda x: x)(x_n) # delta channel, you can set up different channels in a different function, e.g., Rayleigh

            y = GaussianNoise(np.sqrt(N))(ch) # gaussian layer
            #y = UniformNoise(minval=-np.sqrt(3*N), maxval=np.sqrt(3*N)(x_n) # uniform noise layer

            # receiver connection
            s_out = self.receiver(y)

            # build the keras model
            self.encoder = Model(s_in,x_n)

            # set up the tensors for the mutual information block
            T1 = Concatenate(name='network/concatenate_layer_1')([x_n, y])
            y_bar_input = Lambda(lambda x:shuffleColumns(x))(y) # shuffle y input as y_bar
            T2 = Concatenate(name='network/concatenate_layer_2')([x_n, y_bar_input])

            # estimation of joint and marginals
            t_xy = self.discriminator(T1)
            t_x_y = self.discriminator(T2)

            # MINE loss
            loss = Lambda(mine_loss, name='mine_loss')([t_xy, t_x_y])
            output_MI = Lambda(lambda x: -x)(loss)
            self.loss_model = Model(s_in, output_MI)
            self.loss_model.add_loss(loss)
            self.loss_model.compile(optimizer=optimizer_MI)

            # decoder model
            y_in = Input(shape=(self.num_hidden,))
            s_dec = self.receiver(y_in)
            self.decoder = Model(y_in,s_dec)

            # combined model
            self.combined = Model(s_in, s_out)
            loss_model_2 = customLoss(output_MI)
            self.combined.compile(loss=loss_model_2, optimizer=optimizer)


            # model for the estimation of MI at different SNRs
            T1 = Input(shape=(self.num_joint,))
            T2 = Input(shape=(self.num_joint,))
            t_xy = self.discriminator(T1)
            t_x_y = self.discriminator(T2)
            loss = Lambda(mine_loss, name='mine_loss')([t_xy, t_x_y])
            output_MI = Lambda(lambda x: -x)(loss)
            self.loss_model_est = Model([T1, T2], output_MI)
            self.loss_model_est.add_loss(loss)
            self.loss_model_est.compile(optimizer=optimizer_MI)

    def build_transmitter(self):

        model = Sequential()

        model.add(Dense(self.num_enc_inputs, activation="relu", input_dim=self.num_enc_inputs))
        # model.add(Dense(100)) if high rate
        model.add(Dense(self.num_hidden))

        model.summary()

        s_in = Input(shape=(self.num_enc_inputs,))
        x = model(s_in)

        return Model(s_in, x)

    def build_receiver(self):

        model = Sequential()

        model.add(Dense(self.num_enc_inputs, activation="relu", input_dim=self.num_hidden))
        model.add(Dense(self.num_enc_inputs, activation='softmax'))

        model.summary()

        y = Input(shape=(self.num_hidden,))
        s_out = model(y)

        return Model(y, s_out)

    def build_discriminator(self):

        model = Sequential()
        model.add(Dense(200, activation="relu", input_dim=self.num_joint))
        model.add(GaussianNoise(0.3)) # It works only during the training
        model.add(Dense(1))

        model.summary()

        T = Input(shape=(self.num_joint,))
        MI = model(T)

        return Model(T,MI)


    def train(self, epochs, batch_size_AE, batch_size_MI, k , n):

        block_size = k # k parameter
        num_enc_inputs = pow(2, block_size)  # M parameter = 2^k
        p = 1000 # realizations

        alphabet = np.eye(num_enc_inputs, dtype='float32')  # one-hot encoded values
        s_in = np.transpose(np.tile(alphabet, p))
        m = n # to avoid confusion with noise
        R = block_size / m # (half) rate

        EbN0dB = 7
        N = pow(10, -0.1 * EbN0dB) / (2 * R) # noise power

        MI = np.zeros((1,epochs))

        epochs_AE = [10,100,1000] # training epochs for the autoencoder block
        epochs_MI = [10,100,100] # training epochs for the MINE block

        for e in range(3):
            for ij in range(epochs):

                # training the MI estimator
                for epoch in range(epochs_MI[e]):
                    idx = np.random.randint(0, s_in.shape[0], batch_size_MI)
                    s_in_batch = s_in[idx]
                    loss_MI = self.loss_model.train_on_batch(s_in_batch,[])
                    # Plot the progress. if needed
                    # print("%d [MI loss: %f" % (epoch, loss_MI))
                mutual_information = self.loss_model.predict(s_in)
                MI[0,ij] = np.median(mutual_information)

                # traning auto encoder ON BATCH
                loss_AE = np.zeros((epochs_AE[e],))

                for epoch in range(epochs_AE[e]):
                    idx = np.random.randint(0, s_in.shape[0], batch_size_AE)
                    s_in_batch = s_in[idx]
                    loss_AE[epoch] = self.combined.train_on_batch(s_in_batch, s_in_batch)

                print("%d [CAAE_loss: %f, MI_loss: %f]" % (ij, np.median(loss_AE), MI[0,ij])) # median for a stable output

            # if you want to visualize the constellation and the BLER during the training progress on Matlab
            EbN0_dB = range(-14, 29)
            ber = np.zeros((43,))
            j = 0
            p_test = 1000
            s_in_t = np.transpose(np.tile(alphabet, p_test))
            for EbN0 in EbN0_dB:
                N = pow(10, -0.1 * EbN0) / (2 * R)
                mean_t = np.zeros((m,))
                cov_t = np.dot(N, np.eye(m))
                features_r = self.encoder.predict(s_in_t) # get the code
                noise = np.random.multivariate_normal(mean_t, cov_t, p_test * num_enc_inputs) # add channel influence
                features_n = np.add(features_r, noise) # received code
                s_out_t = self.decoder.predict(features_n) # decoded message
                ber[j] = compute_BER(s_in_t,s_out_t)
                j = j + 1
            sio.savemat('BER_%d.mat'%e,{'ber': ber})
            features_r = self.encoder.predict(s_in_t)
            sio.savemat('code_%d.mat'%e, {'code': features_r})

        # save the model
        save_path = self.output_directory + "/Models_AE"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.encoder.save(save_path + "/encoder.h5")
        self.decoder.save(save_path + "/decoder.h5")

    def train_MI(self, epochs_MI, batch_size_MI, k, n):

            block_size = k  # k parameter
            num_enc_inputs = pow(2, block_size)  # M parameter = 2^k
            m = n  # N parameter
            R = block_size/m
            p = 10000
            alphabet = np.eye(num_enc_inputs, dtype='float32')  # One-hot encoded values
            s_in = np.transpose(np.tile(alphabet, p))

            EbN0_dB = range(-14, 19)
            j = 0
            MI = np.zeros((1,33))

            for EbN0 in EbN0_dB:
                N = pow(10, -0.1 * EbN0)/ (2 * R)
                mean_t = np.zeros((m,))
                cov_t = np.dot(N, np.eye(m))
                features_r = self.encoder.predict(s_in)
                noise = np.random.multivariate_normal(mean_t,cov_t,p*num_enc_inputs)
                features_n = np.add(features_r,noise)

                # training the MI estimator
                data_xy, data_x_y = data_generation_mi(features_r, features_n)
                for epoch in range(epochs_MI):
                    idx = np.random.randint(0, s_in.shape[0], batch_size_MI)
                    data_xy_batch = data_xy[idx]
                    data_x_y_batch = data_x_y[idx]
                    self.loss_model_est.train_on_batch([data_xy_batch, data_x_y_batch],[])

                mutual_information = self.loss_model_est.predict([data_xy, data_x_y])
                MI[0,j] = np.median(mutual_information)
                print(MI[0,j])
                j = j+1

            # save the estimated mutual information for Matlab
            sio.savemat('MI_estimation.mat', {'Eb':EbN0_dB, 'MI': MI})

    def test(self, batch_size_AE, k, n):

        block_size = k  # k parameter
        num_enc_inputs = pow(2, block_size)  # M parameter = 2^k
        alphabet = np.eye(num_enc_inputs, dtype='float32')  # One-hot encoded values

        m = n
        R = block_size / m
        p_test = batch_size_AE
        s_in_t = np.transpose(np.tile(alphabet, p_test))
        EbN0_dB = range(-14, 19)
        ber = np.zeros((33,))
        j = 0

        for EbN0 in EbN0_dB:
            N = pow(10, -0.1 * EbN0) / (2 * R)
            mean_t = np.zeros((m,))
            cov_t = np.dot(N, np.eye(m))
            features_r = self.encoder.predict(s_in_t)
            noise = np.random.multivariate_normal(mean_t, cov_t, p_test * num_enc_inputs)
            features_n = np.add(features_r, noise)
            s_out_t = self.decoder.predict(features_n)
            ber[j] = compute_BER(s_in_t, s_out_t)
            j = j + 1
        sio.savemat('BLER_test.mat', {'ber': ber})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_encoder', help='Path to existing generator1 weights file',
                        default="Models_AE/encoder.h5")
    parser.add_argument('--load_decoder', help='Path to existing discriminator1 weights file',
                        default="Models_AE/decoder.h5")
    parser.add_argument('--output_directory', help="Directoy to save weights and images to.",
                        default="Output")
    parser.add_argument('--train', type=str2bool, help="Start the training process.",
                        default=False)
    args = parser.parse_args()

    # Load the model
    CAAE = CAAE(args.load_encoder, args.load_decoder, args.output_directory)

    # Check if the folder with the encoder is empty and the flag training is on
    if not(os.path.exists(args.load_encoder)) and args.train:
        print('Training the model')
        CAAE.train(epochs=10, batch_size_AE = 1000, batch_size_MI = 1000, k = 5, n = 2)
        CAAE.train_MI(epochs_MI=100, batch_size_MI=1000, k = 5, n = 2)
        CAAE.test(batch_size_AE = 4000, k = 5, n = 2)
    # Just testing
    elif os.path.exists(args.load_encoder):
        print('Testing using the loaded models in your folder. If you want to train the model, delete the models')
        CAAE.test(batch_size_AE = 4000, k = 5, n = 2)

    else:
        print('Error in the models loading, please check the correct path')
