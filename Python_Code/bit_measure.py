import numpy as np
import tensorflow as tf

def bit_measure(x):
    return x * (1-x)

def bit_measure_grad(x):
    return 1- 2*x

def bit_measure_alt(x):
    scale = 8
    out = 1 / np.cosh(scale * (x-0.5))
    return out

def bit_measure_grad_alt(x):
    scale = 8
    f = scale * (x - 0.5)
    out = (-5) * np.tanh(f) / np.cosh(f)
    return out