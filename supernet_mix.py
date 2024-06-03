
from inter_conv import inter_gat, inter_gcn, inter_graph, inter_general, inter_le, inter_mf, inter_sage
from intra_conv import intra_gcn, intra_gat, intra_graph, intra_general, intra_le, intra_mf, intra_sage
from inter_intra_source import inter, intra, inter_plus_intra


def mix_aggregation_inter(adj, ner_inputs, ner_dim, target_inputs, hidden_dim, action_list, arch_weight):
    """
    Aggregate information from neighbor nodes
    :param adj: Adjacency matrix
    :param ner_inputs: disease or drug embedding
    :param ner_dim: ner_inputs dimension
    :param target_inputs: target embedding
    :param hidden_dim: output dimension
    :param action_list: search component
    :return:
    """
    fin = []
    for i, action in enumerate(action_list):
        action = eval(action)
        weight = arch_weight[i]
        action_output = weight * action(adj, ner_inputs, ner_dim, target_inputs, hidden_dim)
        fin.append(action_output)
    return sum(fin)

def mix_aggregation_intra(adj, target_inputs, feature_dim, hidden_dim, action_list, arch_weight):
    """
    Aggregate information from neighbor nodes
    :param adj: Adjacency matrix
    :param target_inputs: target embedding
    :param hidden_dim: output dimension
    :param action_list: search component
    :return:
    """
    fin = []
    for i, action in enumerate(action_list):
        action = eval(action)
        weight = arch_weight[i]
        action_output = weight * action(adj, target_inputs, feature_dim, hidden_dim)
        fin.append(action_output)
    return sum(fin)

def mix_source(inter_embedding, intra_embedding, action_list, arch_weight):
    """
    Mix the source of aggregated information
    :param inter_embedding: the embedding from inter aggregation
    :param intra_embedding: the embedding from intra aggregation
    :return:
    """
    fin = []
    for i, action in enumerate(action_list):
        action = eval(action)
        weight = arch_weight[i]
        action_output = weight * action(inter_embedding, intra_embedding)
        fin.append(action_output)
    return sum(fin)
