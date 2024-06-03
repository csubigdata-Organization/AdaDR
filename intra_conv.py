import tensorflow as tf
from keras.regularizers import l2
from tensorflow.python.keras.layers import Dense
from utils import random_uniform_init
from clr import cyclic_learning_rate


def intra_gcn(adj, ner_inputs, feature_dim, hidden_dim):
    """
    Aggregate information from neighbor nodes
    :param adj: Adjacency matrix
    :param ner_inputs: ner embedding
    :param hidden_dim: output dimension
    :return:
    """
    # aggregate intra information
    edges = tf.reduce_sum(adj, 1)
    edges = tf.tile(tf.expand_dims(edges, 1), [1, feature_dim])
    edges = tf.divide(tf.matmul(adj, ner_inputs), edges)

    weight = tf.get_variable('intra_gcn_weight', shape=[feature_dim, hidden_dim],
                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
    bias = tf.get_variable('intra_gcn_bias', shape=[hidden_dim],
                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
    embedding = tf.nn.xw_plus_b(edges, weight, bias)





    return embedding

def intra_gat(adj, ner_inputs, feature_dim, hidden_dim):
    """
    Aggregate information from neighbor nodes
    :param adj: Adjacency matrix
    :param ner_inputs: ner embedding
    :param hidden_dim: output dimension
    :return:
    """
    # aggregate intra information


    alpha = tf.get_variable('intra_gat_alpha', shape=[adj.shape[0], feature_dim],
                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
    alpha = tf.multiply(ner_inputs, alpha)

    edges = tf.matmul(adj, alpha)

    weight = tf.get_variable('intra_gat_weight', shape=[feature_dim, hidden_dim],
                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
    bias = tf.get_variable('intra_gat_bias', shape=[hidden_dim],
                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))

    embedding = tf.nn.xw_plus_b(edges, weight, bias)

    return embedding

def intra_graph(adj, ner_inputs, feature_dim, hidden_dim):
    """
    Aggregate information from neighbor nodes
    :param adj: Adjacency matrix
    :param ner_inputs: ner embedding
    :param hidden_dim: output dimension
    :return:
    """
    # aggregate intra information

    edges = tf.matmul(adj, ner_inputs)

    weight = tf.get_variable('intra_graph_weight', shape=[feature_dim, hidden_dim],
                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
    bias = tf.get_variable('intra_graph_bias', shape=[hidden_dim],
                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))


    embedding = tf.nn.xw_plus_b(edges, weight, bias)
    return embedding


def intra_general(adj, ner_inputs, feature_dim, hidden_dim):
    """
    Aggregate information from neighbor nodes
    :param adj: Adjacency matrix
    :param ner_inputs: ner embedding
    :param hidden_dim: output dimension
    :return:
    """
    # aggregate intra information

    edge_index = tf.where(tf.not_equal(adj, 0))
    num_nodes = tf.shape(adj)[0]

    h_j = tf.gather(ner_inputs, edge_index[:, 1])
    aggr_out = tf.math.unsorted_segment_sum(h_j, edge_index[:, 0], num_segments=num_nodes)
    weight = tf.get_variable('intra_general_weight', shape=[feature_dim, hidden_dim],
                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
    bias = tf.get_variable('intra_general_bias', shape=[hidden_dim],
                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
    embedding = tf.nn.xw_plus_b(aggr_out, weight, bias)

    h_i = tf.gather(ner_inputs, edge_index[:, 0])
    aggr_out_i = tf.math.unsorted_segment_sum(h_i, edge_index[:, 0], num_segments=num_nodes)
    weight_i = tf.get_variable('intra_general_weight_i', shape=[feature_dim, hidden_dim],
                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
    bias_i = tf.get_variable('intra_general_bias_i', shape=[hidden_dim],
                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
    embedding_i = tf.nn.xw_plus_b(aggr_out_i, weight_i, bias_i)

    embedding = tf.add(embedding, embedding_i)

    weight_self = tf.get_variable('intra_general_weight_self', shape=[feature_dim, hidden_dim],
                               initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
    bias_self = tf.get_variable('intra_general_bias_self', shape=[hidden_dim],
                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))

    embedding = embedding + tf.nn.xw_plus_b(ner_inputs, weight_self, bias_self)

    embedding = tf.nn.l2_normalize(embedding, axis=-1)

    return embedding





def intra_sage(adj, ner_inputs, feature_dim, hidden_dim):
    """
    Aggregate information from neighbor nodes
    :param adj: Adjacency matrix
    :param ner_inputs: ner embedding
    :param hidden_dim: output dimension
    :return:
    """
    # aggregate intra information


    edge_index = tf.where(tf.not_equal(adj, 0))
    num_nodes = tf.shape(adj)[0]

    h_i = tf.gather(ner_inputs, edge_index[:, 1])
    mean_feature = tf.math.unsorted_segment_mean(h_i, edge_index[:, 0], num_segments=num_nodes)

    weight = tf.get_variable('intra_sage_weight', shape=[feature_dim, hidden_dim],
                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
    bias = tf.get_variable('intra_sage_bias', shape=[hidden_dim],
                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))

    embedding = tf.nn.xw_plus_b(mean_feature, weight, bias)

    return embedding



def intra_mf(adj, ner_inputs, feature_dim, hidden_dim, max_degree=10):
    """
    Aggregate information from neighbor nodes
    :param adj: Adjacency matrix
    :param ner_inputs: ner embedding
    :param hidden_dim: output dimension
    :return:
    """
    # aggregate intra information


    edge_index = tf.where(tf.not_equal(adj, 0))
    num_nodes = tf.shape(adj)[0]

    edge_weight = tf.gather_nd(adj, edge_index)

    deg = tf.math.unsorted_segment_sum(edge_weight, edge_index[:, 0], num_segments=num_nodes)

    deg = tf.clip_by_value(deg, clip_value_min=0, clip_value_max=max_degree)
    deg = tf.cast(deg, tf.int32)

    h = tf.matmul(adj, ner_inputs)

    out_shape = tf.concat([tf.shape(h)[:-1], [hidden_dim]], axis=0)
    out = tf.zeros(out_shape, dtype=h.dtype)

    for i in range(max_degree + 1):
        idx = tf.where(tf.equal(deg, i))
        idx = tf.reshape(idx, [-1])
        index_select = tf.gather(h, idx)

        weight = tf.get_variable(f'intra_mf_weight_deg{i}', shape=[feature_dim, hidden_dim],
                                 initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
        bias = tf.get_variable(f'intra_mf_bias_deg{i}', shape=[hidden_dim],
                               initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
        r = tf.nn.xw_plus_b(index_select, weight, bias)

        scatter_indices = tf.expand_dims(idx, axis=-1)
        scatter_shape = tf.shape(out, out_type=scatter_indices.dtype)
        updates = tf.scatter_nd(scatter_indices, r, scatter_shape)
        out = out + updates

    return out

def intra_le(adj, ner_inputs, feature_dim, hidden_dim):
    """
    Aggregate information from neighbor nodes
    :param adj: Adjacency matrix
    :param ner_inputs: ner embedding
    :param hidden_dim: output dimension
    :return:
    """
    # aggregate intra information


    edge_index = tf.where(tf.not_equal(adj, 0))
    num_nodes = tf.shape(adj)[0]

    weight = tf.get_variable('intra_le_weight', shape=[feature_dim, hidden_dim],
                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
    bias = tf.get_variable('intra_le_bias', shape=[hidden_dim],
                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
    h = tf.nn.xw_plus_b(ner_inputs, weight, bias)

    edge_weight = tf.gather_nd(adj, edge_index)

    deg = tf.math.unsorted_segment_sum(edge_weight, edge_index[:, 0], num_segments=num_nodes)

    h_j = tf.gather(h, edge_index[:, 1])
    h_j = tf.reshape(edge_weight, [-1, 1]) * h_j
    aggr_out = tf.math.unsorted_segment_sum(h_j, edge_index[:, 0], num_segments=num_nodes)

    weight_lin1 = tf.get_variable('intra_le_weight_lin1', shape=[feature_dim, hidden_dim],
                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
    bias_lin1 = tf.get_variable('intra_le_bias_lin1', shape=[hidden_dim],
                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))

    weight_lin2 = tf.get_variable('intra_le_weight_lin2', shape=[feature_dim, hidden_dim],
                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
    bias_lin2 = tf.get_variable('intra_le_bias_lin2', shape=[hidden_dim],
                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))

    out = (tf.reshape(deg, [-1, 1]) * tf.nn.xw_plus_b(ner_inputs, weight_lin1, bias_lin1) + aggr_out) \
          + tf.nn.xw_plus_b(ner_inputs, weight_lin2, bias_lin2)

    return out
