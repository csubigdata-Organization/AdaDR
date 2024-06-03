import tensorflow as tf
from keras.regularizers import l2
from tensorflow.python.keras.layers import Dense
from utils import random_uniform_init
from clr import cyclic_learning_rate


def intra(inter_embedding, intra_embedding):
    return intra_embedding

def inter(inter_embedding, intra_embedding):
    return inter_embedding

def inter_plus_intra(inter_embedding, intra_embedding):
    return tf.add(inter_embedding, intra_embedding)



