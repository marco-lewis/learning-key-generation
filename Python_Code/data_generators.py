import numpy as np
import tensorflow as tf

# Same definition in Abadi/Andersen's code
def make_weights(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def gen_binary(instances, length):
    int_arr = np.random.randint(0,2, size=(instances, length))
    float_arr = int_arr.astype(np.float32)
    return float_arr

def gen_zeros(instances, length):
    return np.zeros((instances, length))