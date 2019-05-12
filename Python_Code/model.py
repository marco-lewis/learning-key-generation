class Model(object):
    def __init__(self, sess, const_len, key_len, batchsize, lr):
        self.sess = sess
        self.const_len = const_len
        self.key_len = key_len
        self.batchsize = batchsize
        self.lr = lr
    
    def error_func(ins, out):
        raise Exception("No error function defined")
      
    def build_model():
        raise Exception("No build function defined")
    
    def train_model():
        raise Exception("No train model function defined")
    
    def train():
        raise Exception("No training function defined")
        
    def plot():
        raise Exception("No plot function defined")
        