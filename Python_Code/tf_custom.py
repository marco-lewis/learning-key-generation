import tensorflow as tf
import numpy as np
import angle
import bit_measure as bm
from tensorflow.python.framework import ops


# See https://github.com/tensorflow/tensorflow/issues/1095#issuecomment-224845561
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)
    
    
def tf_angle(tensor_in, name=None):
    with ops.name_scope(name, "angle", tensor_in) as name:
        tensor = py_func(angle.angle_func,
                         tensor_in,
                         tf.float32,
                         name=name,
                         grad=tf_angle_grad
                        )
    return tensor

    
def tf_inv_angle(tensor_in, name=None):
    with ops.name_scope(name, "inv_angle", tensor_in) as name:
        tensor = py_func(angle.inv_angle_func,
                         tensor_in,
                         tf.float32,
                         name=name,
                         grad=tf_inv_angle_grad
                        )
    return tensor

def tf_angle_grad(op, grad):
    diff_tensor = tf.py_func(angle.angle_func_grad, [op.inputs[0]], tf.float32)
    out = tf.multiply(diff_tensor, grad)
    return out

def tf_inv_angle_grad(op, grad):
    diff_tensor = tf.py_func(angle.inv_angle_gradient, [op.inputs[0]], tf.float32)
    out = tf.multiply(diff_tensor, grad)
    return out


def tf_bit_measure(tensor_in, name=None):
    with ops.name_scope(name, "entropy", tensor_in) as name:
        tensor = py_func(bm.bit_measure,
                         tensor_in,
                         tf.float32,
                         name=name,
                         grad=tf_bit_measure_grad
                        )
    return tensor

def tf_bit_measure_grad(op, grad):
    diff_tensor = tf.py_func(bm.bit_measure_grad, [op.inputs[0]], tf.float32)
    out = tf.multiply(diff_tensor, grad)
    return out
    