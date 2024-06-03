import tensorflow as tf
from keras.regularizers import l2
from tensorflow.python.keras.layers import Dense
from utils import random_uniform_init
from clr import cyclic_learning_rate


def inter_gat(adj, ner_inputs, ner_dim, target_inputs, hidden_dim):
    """
    Aggregate information from neighbor nodes
    :param adj: Adjacency matrix
    :param ner_inputs: disease or drug embedding
    :param ner_dim: ner_inputs dimension
    :param target_inputs: target embedding
    :param hidden_dim: output dimension
    :return:
    """
    # aggregate inter information
    query = tf.tile(tf.reshape(target_inputs, (target_inputs.shape[0], 1, target_inputs.shape[1])),
                    [1, ner_inputs.shape[0], 1])
    key = tf.tile(tf.reshape(ner_inputs, (1, ner_inputs.shape[0], ner_inputs.shape[1])),
                  [target_inputs.shape[0], 1, 1])
    key_query = tf.reshape(tf.concat([key, query], -1), [ner_inputs.shape[0] * target_inputs.shape[0], -1])
    alpha = Dense(hidden_dim, activation='relu', use_bias=True, kernel_regularizer=l2(1.0))(key_query)
    alpha = Dense(1, activation='relu', use_bias=True, kernel_regularizer=l2(1.0))(alpha)
    alpha = tf.reshape(alpha, [target_inputs.shape[0], ner_inputs.shape[0]])
    alpha = tf.multiply(alpha, adj)
    alpha_exps = tf.nn.softmax(alpha, 1)
    weight = tf.get_variable('inter_gat_weight', shape=[ner_dim, hidden_dim],
                         initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
    bias = tf.get_variable('inter_gat_bias', shape=[hidden_dim],
                         initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
    alpha_exps = tf.tile(tf.expand_dims(alpha_exps, -1), [1, 1, ner_inputs.shape[1]])
    e_r = tf.nn.xw_plus_b(tf.reduce_sum(tf.multiply(alpha_exps, key), 1), weight, bias)




    return e_r


def inter_gcn(adj, ner_inputs, ner_dim, target_inputs, hidden_dim):
    """
    Aggregate information from neighbor nodes
    :param adj: Adjacency matrix
    :param ner_inputs: disease or drug embedding
    :param ner_dim: ner_inputs dimension
    :param target_inputs: target embedding
    :param hidden_dim: output dimension
    :return:
    """
    # aggregate inter information
    ner_edges = tf.reduce_sum(adj, 1)
    ner_edges = 1 - tf.nn.softmax(ner_edges, 0)
    ner_edges = tf.tile(tf.expand_dims(ner_edges, 1), [1, ner_dim])
    edges = tf.multiply(tf.matmul(adj, ner_inputs), ner_edges)

    weight = tf.get_variable('inter_gcn_weight', shape=[ner_dim, hidden_dim],
                         initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
    bias = tf.get_variable('inter_gcn_bias', shape=[hidden_dim],
                         initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
    e_r = tf.nn.xw_plus_b(edges, weight, bias)


    # add target_inputs
    weight_target = tf.get_variable('inter_gcn_weight_target', shape=[ner_dim, hidden_dim],
                             initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
    e_r_target = tf.matmul(target_inputs, weight_target)

    e_r_final = tf.add(e_r, e_r_target)
    return e_r_final

def inter_graph(adj, ner_inputs, ner_dim, target_inputs, hidden_dim):
    """
    Aggregate information from neighbor nodes
    :param adj: Adjacency matrix
    :param ner_inputs: disease or drug embedding
    :param ner_dim: ner_inputs dimension
    :param target_inputs: target embedding
    :param hidden_dim: output dimension
    :return:
    """
    # aggregate inter information
    edges = tf.matmul(adj, ner_inputs)

    weight = tf.get_variable('inter_graph_weight', shape=[ner_dim, hidden_dim],
                         initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
    bias = tf.get_variable('inter_graph_bias', shape=[hidden_dim],
                         initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
    e_r = tf.nn.xw_plus_b(edges, weight, bias)


    # add target_inputs
    weight_target = tf.get_variable('inter_graph_weight_target', shape=[ner_dim, hidden_dim],
                             initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
    e_r_target = tf.matmul(target_inputs, weight_target)

    e_r_final = tf.add(e_r, e_r_target)



    return e_r_final

def inter_general(adj, ner_inputs, ner_dim, target_inputs, hidden_dim):
    """
    Aggregate information from neighbor nodes
    :param adj: Adjacency matrix
    :param ner_inputs: disease or drug embedding
    :param ner_dim: ner_inputs dimension
    :param target_inputs: target embedding
    :param hidden_dim: output dimension
    :return:
    """

    edge_index = tf.where(tf.not_equal(adj, 0))
    num_nodes = tf.shape(adj)[0]

    h_j = tf.gather(ner_inputs, edge_index[:, 1])
    aggr_out = tf.math.unsorted_segment_sum(h_j, edge_index[:, 0], num_segments=num_nodes)
    weight = tf.get_variable('inter_general_weight', shape=[ner_dim, hidden_dim],
                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
    bias = tf.get_variable('inter_general_bias', shape=[hidden_dim],
                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
    e_r = tf.nn.xw_plus_b(aggr_out, weight, bias)

    h_i = tf.gather(target_inputs, edge_index[:, 0])
    aggr_out_i = tf.math.unsorted_segment_sum(h_i, edge_index[:, 0], num_segments=num_nodes)
    weight_i = tf.get_variable('inter_general_weight_i', shape=[ner_dim, hidden_dim],
                               initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
    bias_i = tf.get_variable('inter_general_bias_i', shape=[hidden_dim],
                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
    e_r_i = tf.nn.xw_plus_b(aggr_out_i, weight_i, bias_i)

    e_r = tf.add(e_r, e_r_i)

    weight_self = tf.get_variable('inter_general_weight_self', shape=[ner_dim, hidden_dim],
                                  initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
    bias_self = tf.get_variable('inter_general_bias_self', shape=[hidden_dim],
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))

    e_r_final = e_r + tf.nn.xw_plus_b(target_inputs, weight_self, bias_self)

    e_r_final = tf.nn.l2_normalize(e_r_final, axis=-1)
    return e_r_final


def inter_sage(adj, ner_inputs, ner_dim, target_inputs, hidden_dim):
    """
    Aggregate information from neighbor nodes
    :param adj: Adjacency matrix
    :param ner_inputs: disease or drug embedding
    :param ner_dim: ner_inputs dimension
    :param target_inputs: target embedding
    :param hidden_dim: output dimension
    :return:
    """
    edge_index = tf.where(tf.not_equal(adj, 0))
    num_nodes = tf.shape(adj)[0]

    h_i = tf.gather(ner_inputs, edge_index[:, 1])
    mean_feature = tf.math.unsorted_segment_mean(h_i, edge_index[:, 0], num_segments=num_nodes)


    weight = tf.get_variable('inter_sage_weight', shape=[ner_dim, hidden_dim],
                             initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
    bias = tf.get_variable('inter_sage_bias', shape=[hidden_dim],
                           initializer=tf.truncated_normal_initializer(mean=0, stddev=1))

    e_r = tf.nn.xw_plus_b(mean_feature, weight, bias)

    # add target_inputs
    h_target = tf.gather(target_inputs, edge_index[:, 0])
    mean_feature_target = tf.math.unsorted_segment_mean(h_target, edge_index[:, 0], num_segments=num_nodes)

    weight_target = tf.get_variable('inter_sage_weight_target', shape=[ner_dim, hidden_dim],
                                    initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
    e_r_target = tf.matmul(mean_feature_target, weight_target)


    e_r_final = tf.add(e_r, e_r_target)

    return e_r_final


def inter_mf(adj, ner_inputs, ner_dim, target_inputs, hidden_dim, max_degree=10):
    """
    Aggregate information from neighbor nodes
    :param adj: Adjacency matrix
    :param ner_inputs: disease or drug embedding
    :param ner_dim: ner_inputs dimension
    :param target_inputs: target embedding
    :param hidden_dim: output dimension
    :return:
    """
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

        weight = tf.get_variable(f'inter_mf_weight_deg{i}', shape=[ner_dim, hidden_dim],
                                 initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
        bias = tf.get_variable(f'inter_mf_bias_deg{i}', shape=[hidden_dim],
                               initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
        r = tf.nn.xw_plus_b(index_select, weight, bias)

        weight_target = tf.get_variable(f'inter_mf_weight_target_deg{i}', shape=[ner_dim, hidden_dim],
                                 initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
        bias_target = tf.get_variable(f'inter_mf_bias_target_deg{i}', shape=[hidden_dim],
                               initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))

        index_select_target = tf.gather(target_inputs, idx)

        r_target = tf.nn.xw_plus_b(index_select_target, weight_target, bias_target)

        r = tf.add(r, r_target)


        scatter_indices = tf.expand_dims(idx, axis=-1)
        scatter_shape = tf.shape(out, out_type=scatter_indices.dtype)
        updates = tf.scatter_nd(scatter_indices, r, scatter_shape)
        out = out + updates

    return out

def inter_le(adj, ner_inputs, ner_dim, target_inputs, hidden_dim):
    """
    Aggregate information from neighbor nodes
    :param adj: Adjacency matrix
    :param ner_inputs: disease or drug embedding
    :param ner_dim: ner_inputs dimension
    :param target_inputs: target embedding
    :param hidden_dim: output dimension
    :return:
    """
    edge_index = tf.where(tf.not_equal(adj, 0))
    num_nodes = tf.shape(adj)[0]

    weight = tf.get_variable('inter_le_weight', shape=[ner_dim, hidden_dim],
                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
    bias = tf.get_variable('inter_le_bias', shape=[hidden_dim],
                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
    h = tf.nn.xw_plus_b(ner_inputs, weight, bias)

    edge_weight = tf.gather_nd(adj, edge_index)

    deg = tf.math.unsorted_segment_sum(edge_weight, edge_index[:, 0], num_segments=num_nodes)

    h_j = tf.gather(h, edge_index[:, 1])
    h_j = tf.reshape(edge_weight, [-1, 1]) * h_j
    aggr_out = tf.math.unsorted_segment_sum(h_j, edge_index[:, 0], num_segments=num_nodes)

    weight_lin1 = tf.get_variable('inter_le_weight_lin1', shape=[ner_dim, hidden_dim],
                                  initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
    bias_lin1 = tf.get_variable('inter_le_bias_lin1', shape=[hidden_dim],
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))

    weight_lin2 = tf.get_variable('inter_le_weight_lin2', shape=[ner_dim, hidden_dim],
                                  initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
    bias_lin2 = tf.get_variable('inter_le_bias_lin2', shape=[hidden_dim],
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))

    e_r = (tf.reshape(deg, [-1, 1]) * tf.nn.xw_plus_b(target_inputs, weight_lin1, bias_lin1) + aggr_out) \
          + tf.nn.xw_plus_b(target_inputs, weight_lin2, bias_lin2)
    return e_r
