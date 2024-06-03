import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Dense
from utils import random_uniform_init

from supernet_mix import mix_aggregation_inter, mix_aggregation_intra, mix_source


class SuperNet(object):

    def __init__(self, config):
        """
        :param config:
        """
        self.config = config
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



        self.search_space = [['inter_gcn', 'inter_gat', 'inter_graph', 'inter_general', 'inter_le', 'inter_mf', 'inter_sage'],
                             ['intra_gcn', 'intra_gat', 'intra_graph', 'intra_general', 'intra_le', 'intra_mf', 'intra_sage'],
                             ['inter', 'intra', 'inter_plus_intra']]




        self.arch_lr = config['arch_lr']

        self.architecture_alphas = []
        for i, component in enumerate(self.search_space):
            component_alpha = tf.Variable(1e-3 * tf.random.normal([len(component)]), name=f'architecture_component_{i}')
            self.architecture_alphas.append(component_alpha)


        with tf.variable_scope("model_disease", reuse=tf.AUTO_REUSE):

            alphas_0 = tf.nn.softmax(self.architecture_alphas[0])

            disease_inter_output = mix_aggregation_inter(self.disease_drug_Adj, self.drug_embedding, self.drug_dim, self.disease_embedding, self.hidden_dim,
                                                         self.search_space[0], alphas_0)
            alphas_1 = tf.nn.softmax(self.architecture_alphas[1])
            disease_intra_output = mix_aggregation_intra(self.disease_disease_Adj, self.disease_embedding, self.disease_dim, self.hidden_dim,
                                                         self.search_space[1], alphas_1)
            alphas_2 = tf.nn.softmax(self.architecture_alphas[2])
            self.disease_aggregation = tf.nn.selu(mix_source(disease_inter_output, disease_intra_output,
                                                             self.search_space[2], alphas_2))

        with tf.variable_scope("model_drug", reuse=tf.AUTO_REUSE):
            alphas_0 = tf.nn.softmax(self.architecture_alphas[0])
            drug_inter_ouput = mix_aggregation_inter(tf.transpose(self.disease_drug_Adj), self.disease_embedding, self.disease_dim, self.drug_embedding, self.hidden_dim,
                                                     self.search_space[0], alphas_0)
            alphas_1 = tf.nn.softmax(self.architecture_alphas[1])
            drug_intra_output = mix_aggregation_intra(self.drug_drug_Adj, self.drug_embedding, self.drug_dim, self.hidden_dim,
                                                      self.search_space[1], alphas_1)
            alphas_2 = tf.nn.softmax(self.architecture_alphas[2])
            self.drug_aggregation = tf.nn.selu(mix_source(drug_inter_ouput, drug_intra_output,
                                                          self.search_space[2], alphas_2))

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
            self.model_opt = tf.train.AdamOptimizer(learning_rate=self.lr)



            self.arch_opt = tf.train.AdamOptimizer(learning_rate=self.arch_lr)



            variables = tf.trainable_variables()

            model_parameters = [v for v in variables if 'architecture_component' not in v.name]
            arch_parameters = [v for v in variables if 'architecture_component' in v.name]

            # apply grad clip to avoid gradient explosion
            self.model_grads_vars = self.model_opt.compute_gradients(self.loss, var_list=model_parameters)
            self.arch_grads_vars = self.arch_opt.compute_gradients(self.loss, var_list=arch_parameters)

            model_capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in self.model_grads_vars]
            arch_capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                       for g, v in self.arch_grads_vars]

            self.model_train_op = self.model_opt.apply_gradients(model_capped_grads_vars, self.global_step)
            self.arch_train_op = self.arch_opt.apply_gradients(arch_capped_grads_vars)

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)


    def run_step(self, sess, data_type, alpha_mode, batch):
        """
        :param sess: session to run the batch
        :param data_type: current data type, train/valid/test
        :param alpha_mode: the mode for updating the architecture alpha
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

        if data_type == 'train':
            if alpha_mode == 'train':
                global_step, loss, z, _, _, _, _ = sess.run(
                    [self.global_step, self.loss, self.z, self.model_grads_vars, self.model_train_op,
                     self.arch_grads_vars, self.arch_train_op], feed_dict)
                return global_step, loss, z
            elif alpha_mode == 'valid':
                global_step, loss, z, _, _, = sess.run(
                    [self.global_step, self.loss, self.z, self.model_grads_vars, self.model_train_op], feed_dict)
                return global_step, loss, z
        elif data_type == 'valid':
            if alpha_mode == 'valid':
                z, labels, _, _, _ = sess.run(
                    [self.z, self.label, self.loss, self.arch_grads_vars, self.arch_train_op], feed_dict)
                return z, labels
            elif alpha_mode == 'train':
                z, labels = sess.run([self.z, self.label], feed_dict)
                return z, labels
        elif data_type == 'test':
            z, labels = sess.run([self.z, self.label], feed_dict)
            return z, labels


    def get_best_architecture_alpha(self, sess):
        architecture = []

        for i, alpha in enumerate(self.architecture_alphas):
            alpha = tf.nn.softmax(alpha)
            armax_indice = tf.argmax(alpha)
            armax_indice = sess.run(armax_indice)
            argmax_component = self.search_space[i][armax_indice]
            architecture.append(argmax_component)

        return '||'.join(architecture)
