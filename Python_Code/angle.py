import numpy as np

def angle_func(bit):
    return np.arccos(1 - (2 * bit))

def angle_func_grad(bit):
    eps = 0.0001
    bit = bit
    return (1 / (eps + np.sqrt((1 - bit) * bit)))
    
def inv_angle_func(angle):
    return (1 - np.cos(angle)) / 2

def inv_angle_gradient(angle):
    return np.sin(angle)/2