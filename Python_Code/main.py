import tensorflow as tf
import numpy as np
import angle
import data_generators
from key_gen_model import Key_Gen_Model
from fixed import Fixed_Model
from timeit import default_timer as timer
from result_collector import directory_results
import os

def main():
    get_results = True
    
    if get_results:
        rootdir = os.path.dirname(os.path.abspath(__file__)) + str("\Experiment_Results")
        results, key_ratios, ex_direcs = directory_results(rootdir)
        key_successes = []
        for result in results:
            exp_successes = all(result)
            key_successes.append(exp_successes)
            print(result)
        print(key_successes)
        print(key_ratios)
        print(ex_direcs)
    else:
        run_model()
    
    
def run_model():
    bs = 256
    lr = 0.0001
    fixed_perm = True
    run_example = False
    
    if run_example:
        with tf.Session() as sess:
            key_len = 4
            const_len = 4
            example(sess, const_len, key_len, bs, lr, None)
            arr = np.array([[0,1,0,0],[0,0,0,1],[0,0,1,0],[1,0,0,0]], dtype=np.float32)
            example(sess, const_len, key_len, bs, lr, arr)
    else:
        experiments(0, 8, bs, lr, fixed_perm)
        experiments(0, 16, bs, lr, fixed_perm)
        experiments(0, 32, bs, lr, fixed_perm)
        experiments(4, 4, bs, lr, fixed_perm)
        experiments(8, 8, bs, lr, fixed_perm)
        experiments(16, 16, bs, lr, fixed_perm)

def experiments(const_len, key_len, bs, lr, fixed_perm):
        for i in range(0, 20):
            p_weights = np.random.permutation(np.identity(key_len, dtype=np.float32))
            run_experiment(const_len, key_len, bs, lr, fixed_perm, p_weights)
                
                
def run_experiment(const_len, key_len, bs, lr, fixed_perm, p_weights):
    g = tf.Graph()
    with g.as_default():
        with tf.Session() as sess:
            model = Key_Gen_Model(sess, const_len, key_len, bs, lr, fixed_perm, p_weights)
            model.build_model()
            epochs = 1500
            iterations = 25
            model.generate_directory()
            start = timer()
            model.train_model(epochs, iterations)
            end = timer()
            process_time = end-start
            print("Tests")
            model.test(1024)
            model.plot(epochs)
            model.save_data(process_time)

            
def example(sess, const_len, key_len, bs, lr, perm):
    fixed_model = Fixed_Model(sess, const_len, key_len, bs, lr, key_id=perm)
    fixed_model.run_example(10)
    
    
            
if __name__ == '__main__':
    main()