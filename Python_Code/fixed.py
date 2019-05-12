import tensorflow as tf
import numpy as np
import data_generators as dg
from angle import angle_func, inv_angle_func
from model import Model
import tf_custom as tfc

class Fixed_Model(Model):
    def __init__(self, sess, const_len, key_len, batchsize, lr, key_id=None):
        super().__init__(sess, const_len, key_len, batchsize, lr)
        if key_id is None:
            self.key_id = np.identity(self.key_len, dtype=np.float32)
        else:
            self.key_id = key_id
        print("Variables fixed")
        self.build_model()
        print("Finished model building")
        
    def build_model(self):
        # Inputs
        self.const_in = tf.placeholder("float", shape=[None, self.const_len], name="const_in")
        self.key_a_in = tf.placeholder("float", shape=[None, self.key_len], name="key_a_in")
        self.key_b_in = tf.placeholder("float", shape=[None, self.key_len], name="key_b_in")
        
        self.build_alice_bob()
        
        
    def build_alice_bob(self):
        # Weights
        comms_input_length = self.const_len + self.key_len
        
        self.key_inv = np.linalg.inv(self.key_id)
        
        self.a = np.vstack((dg.gen_binary(self.const_len, self.key_len), self.key_id))
        self.b = np.vstack((dg.gen_binary(self.const_len, self.key_len), self.key_inv))
       
        self.a_weights = tf.constant(self.a, name="Alice_weight")
        self.b_weights = tf.constant(self.b, name="Bob_weight")

        # AB generating partial keys
        with tf.name_scope('Alice_PK') as scope:
            self.a_in = tf.concat([self.const_in, self.key_a_in], 1, "a_in")
            self.a_angles = tf.py_func(angle_func, [self.a_in], tf.float32, name="a_angles")
            self.a_out_angles = tf.nn.relu(tf.matmul(self.a_angles, self.a_weights), name="a_op")
            self.a_out = tfc.tf_inv_angle([self.a_out_angles], name="pk_a")
        
        with tf.name_scope('Bob_PK') as scope:
            self.b_in = tf.concat([self.const_in, self.key_b_in], 1, "b_in")
            self.b_angles = tf.py_func(angle_func, [self.b_in], tf.float32, name="b_angles")
            self.b_out_angles = tf.nn.relu(tf.matmul(self.b_angles, self.b_weights), name="b_op")
            self.b_out = tfc.tf_inv_angle([self.b_out_angles], name="pk_b")
        
        # Partial keys
        self.a_pk = self.a_out
        self.b_pk = self.b_out
        
        # Send keys to other network and compute secret, use same weights
        with tf.name_scope('Alice_Secret') as scope:
            self.a2_in = tf.concat([self.const_in, self.b_pk], 1, name="a2_in")
            self.a2_angles = tf.py_func(angle_func, [self.a2_in], tf.float32, name="a2_angles")
            self.a2_out_angles = tf.nn.relu(tf.matmul(self.a2_angles, self.a_weights), name="a2_op")
            self.a2_out = tf.squeeze(tfc.tf_inv_angle([self.a2_out_angles]), name="a_secret")
        
        with tf.name_scope('Bob_Secret') as scope:
            self.b2_in = tf.concat([self.const_in, self.a_pk], 1, name="b2_in")
            self.b2_angles = tf.py_func(angle_func, [self.b2_in], tf.float32, name="b2_angles")
            self.b2_out_angles = tf.nn.relu(tf.matmul(self.b2_angles, self.b_weights), name="b2_op")
            self.b2_out = tfc.tf_inv_angle([self.b2_out_angles])
        
        # Permutation network (either fixed or learned) set up
        self.p_weights = tf.constant(self.key_id, name="perm_weights")
                
        self.p_input = self.b2_out
        self.p_out = tf.nn.relu(tf.matmul(self.p_input, self.p_weights))
        
        # Secrets of A and B
        self.a_secret = self.a2_out
        self.b_secret = tf.squeeze(self.p_out, name="b_secret")
        
        
    def run_example(self, batches):
        const_ins = dg.gen_binary(batches, self.const_len)
        key_a_ins = dg.gen_zeros(batches, self.key_len)
        key_b_ins = dg.gen_zeros(batches, self.key_len)
        
        print("Constants\n", const_ins)
#         print("Key A\n", key_a_ins)
#         print("Key B\n", key_b_ins)
        feed = {self.const_in: const_ins, self.key_a_in: key_a_ins, self.key_b_in: key_b_ins}
        secrets = self.sess.run([self.a_secret, self.b_secret], feed_dict=feed)
        print("Secret A\n", secrets[0])
        print("Secret B\n", secrets[1])
        print("Secrets Same? " + str((secrets[0] == secrets[1]).all()))
        print("A Network\n", self.a)
        print("B Network\n", self.b)
        print("\n")
