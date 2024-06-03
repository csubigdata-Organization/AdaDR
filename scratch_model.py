import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Dense
from utils import random_uniform_init
from clr import cyclic_learning_rate

from inter_conv import inter_gcn, inter_gat, inter_graph, inter_general, inter_le, inter_mf, inter_sage
from intra_conv import intra_gcn, intra_gat, intra_graph, intra_general, intra_le, intra_mf, intra_sage
from inter_intra_source import inter, intra, inter_plus_intra


class Model(object):

    def __init__(self, config):
        """
        :param config:
        """
        self.config = config
        self.architecture = config['scratch_architecture']
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.disease_dim = config['disease_dim']
        self.drug_dim = config['drug_dim']
        self.disease_size = config['disease_size']
        self.drug_size = config['drug_size']
        self.hidden_dim = config['hidden_dim']
        self.mlp_layer_num = 1

        self.global_step = tf.Variable(0, trainable=False)

        # input
        self.disease_drug_Adj = tf.placeholder(dtype=tf.float32,
                                      shape=[self.disease_size, self.drug_size])
        self.disease_disease_Adj = tf.placeholder(dtype=tf.float32,
                                      shape=[self.disease_size, self.disease_size])
        self.drug_drug_Adj = tf.placeholder(dtype=tf.float32,
                                      shape=[self.drug_size, self.drug_size])

        self.input_disease = tf.placeholder(dtype=tf.int32, shape=[None])
        self.input_drug = tf.placeholder(dtype=tf.int32, shape=[None])
        self.label = tf.placeholder(dtype=tf.float32, shape=[None])

        self.disease_embedding = random_uniform_init(name="disease_embedding_matrix",
                                                    shape=[self.disease_size, self.disease_dim])
        self.drug_embedding = random_uniform_init(name="drug_embedding_matrix",
                                                      shape=[self.drug_size, self.drug_dim])


        with tf.variable_scope("model_disease", reuse=tf.AUTO_REUSE):
            if self.architecture[2] == 'inter_plus_intra':
                inter_func = eval(self.architecture[0])
                disease_inter_output = inter_func(self.disease_drug_Adj, self.drug_embedding, self.drug_dim, self.disease_embedding, self.hidden_dim)

                intra_func = eval(self.architecture[1])
                disease_intra_output = intra_func(self.disease_disease_Adj, self.disease_embedding, self.disease_dim, self.hidden_dim)

                self.disease_aggregation = tf.nn.selu(tf.add(disease_inter_output, disease_intra_output))
            elif self.architecture[2] == 'intra':
                intra_func = eval(self.architecture[1])
                disease_intra_output = intra_func(self.disease_disease_Adj, self.disease_embedding, self.disease_dim, self.hidden_dim)
                self.disease_aggregation = tf.nn.selu(disease_intra_output)
            elif self.architecture[2] == 'inter':
                inter_func = eval(self.architecture[0])
                disease_inter_output = inter_func(self.disease_drug_Adj, self.drug_embedding, self.drug_dim, self.disease_embedding, self.hidden_dim)
                self.disease_aggregation = tf.nn.selu(disease_inter_output)


        with tf.variable_scope("model_drug", reuse=tf.AUTO_REUSE):
            if self.architecture[2] == 'inter_plus_intra':
                inter_func = eval(self.architecture[0])
                drug_inter_ouput = inter_func(tf.transpose(self.disease_drug_Adj), self.disease_embedding, self.disease_dim, self.drug_embedding, self.hidden_dim)

                intra_func = eval(self.architecture[1])
                drug_intra_output = intra_func(self.drug_drug_Adj, self.drug_embedding, self.drug_dim, self.hidden_dim)

                self.drug_aggregation = tf.nn.selu(tf.add(drug_inter_ouput, drug_intra_output))
            elif self.architecture[2] == 'intra':
                intra_func = eval(self.architecture[1])
                drug_intra_output = intra_func(self.drug_drug_Adj, self.drug_embedding, self.drug_dim, self.hidden_dim)
                self.drug_aggregation = tf.nn.selu(drug_intra_output)
            elif self.architecture[2] == 'inter':
                inter_func = eval(self.architecture[0])
                drug_inter_ouput = inter_func(tf.transpose(self.disease_drug_Adj), self.disease_embedding, self.disease_dim, self.drug_embedding, self.hidden_dim)
                self.drug_aggregation = tf.nn.selu(drug_inter_ouput)

        with tf.variable_scope("model_fusion", reuse=tf.AUTO_REUSE):
            disease_aggregation_batch = tf.nn.embedding_lookup(self.disease_aggregation, self.input_disease)  # batch_size * disease_hidden_dim
            drug_aggregation_batch = tf.nn.embedding_lookup(self.drug_aggregation, self.input_drug)  # batch_size * drug_hidden_dim
            input_temp = tf.multiply(disease_aggregation_batch, drug_aggregation_batch)
            for l_num in range(self.mlp_layer_num):
                input_temp = Dense(self.disease_dim, activation='selu', kernel_initializer='lecun_uniform')(input_temp)  # MLP hidden layer
            z = Dense(1, kernel_initializer='lecun_uniform', name='prediction')(input_temp)
            z = tf.squeeze(z)

        self.label = tf.squeeze(self.label)
        self.loss = tf.losses.sigmoid_cross_entropy(self.label, z)
        self.z = tf.sigmoid(z)

        # train
        with tf.variable_scope("optimizer"):
            self.model_opt = tf.train.AdamOptimizer(learning_rate=cyclic_learning_rate(global_step=self.global_step,
                                                                                 learning_rate=self.lr*0.1,
                                                                                 max_lr=self.lr,
                                                                                 mode='exp_range',
                                                                                 gamma=.999))



            variables = tf.trainable_variables()

            model_parameters = [v for v in variables if 'architecture_component' not in v.name]

            # apply grad clip to avoid gradient explosion
            self.model_grads_vars = self.model_opt.compute_gradients(self.loss, var_list=model_parameters)

            model_capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                       for g, v in self.model_grads_vars]


            self.model_train_op = self.model_opt.apply_gradients(model_capped_grads_vars, self.global_step)

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)


    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        disease_drug_Adj, disease_disease_Adj, drug_drug_Adj, input_disease, input_drug, label = batch
        feed_dict = {
            self.disease_drug_Adj: np.asarray(disease_drug_Adj),
            self.disease_disease_Adj: np.asarray(disease_disease_Adj),
            self.drug_drug_Adj: np.asarray(drug_drug_Adj),
            self.input_disease: np.asarray(input_disease),
            self.input_drug: np.asarray(input_drug),
            self.label: np.asarray(label)
        }

        if is_train:
            global_step, loss, z, _, _ = sess.run(
                [self.global_step, self.loss, self.z, self.model_grads_vars, self.model_train_op], feed_dict)
            return global_step, loss, z
        else:
            z, labels = sess.run([self.z, self.label], feed_dict)
            return z, labels
