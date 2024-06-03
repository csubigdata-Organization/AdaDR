import tensorflow as tf
import math



def create_model(session, Model_class, path, config):
    # create model, reuse parameters if exists
    model = Model_class(config)

    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


# initialize variables
def random_uniform_init(shape, name, dtype=tf.float32):
    with tf.name_scope('uniform_normal'):
        std = 1.0 / math.sqrt(shape[1])
        embeddings = tf.get_variable(name, shape=shape, dtype=dtype,
                                     initializer=tf.initializers.random_normal(stddev=std))
    return tf.nn.l2_normalize(embeddings, 1)