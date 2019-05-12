from model import Model
import tensorflow as tf
from angle import angle_func, inv_angle_func
from data_generators import gen_binary, make_weights, gen_zeros
import bit_measure as bm
import tf_custom as tfc
import matplotlib.pyplot as plt
import numpy as np
import shutil
import uuid
import datetime
import json
import os

class Key_Gen_Model(Model):
    def __init__(self, sess, const_len, key_len, batchsize, lr, fixed_perm, p_weights):
        super().__init__(sess, const_len, key_len, batchsize, lr)
        self.fixed_perm = fixed_perm
        self.p_weights = p_weights
        print("Variables fixed")
    
    
    def generate_directory(self):
        dir_id = uuid.uuid4()
        data_folder = os.path.dirname(os.path.abspath(__file__)) + str("\Data")
        self.path = data_folder + "\\Key-Model-Exp-" + str(self.const_len) + "-" + str(self.key_len)+ "-" + str(dir_id)
        os.mkdir(self.path)
        
    def delete_directory(self):
        shutil.rmtree(self.path)
    
    def build_model(self):
        self.build_inputs()
        self.build_alice_bob()
        print("Built Alice and Bob")
        self.build_eve()
        print("Built Eve")
        self.build_data()

        
    def build_inputs(self):
        self.const_in = tf.placeholder("float", shape=[None, self.const_len], name="const_in")
        self.key_a_in = tf.placeholder("float", shape=[None, self.key_len], name="key_a_in")
        self.key_b_in = tf.placeholder("float", shape=[None, self.key_len], name="key_b_in")
        self.initial = tf.placeholder("float", shape=[None, self.key_len], name="Initial_Vector")
        
    def build_eve(self):
        adversary_input_length = 2 * self.key_len + self.const_len
        self.e_weights = make_weights("eve_weights", [adversary_input_length, self.key_len])
        
        with tf.name_scope('Eve') as scope:
            self.e_in = tf.concat([self.const_in, self.a_pk, self.b_pk], 1, name="e_in")
            self.e_angles = tfc.tf_angle([self.e_in], name="e_angles")
        
            # E fully connected layer input -> output
            self.e_out_angles = (tf.matmul(self.e_in, self.e_weights, name="e_out_angles"))        
            self.e_ang_inv = tfc.tf_inv_angle([self.e_out_angles], "e_secret")
    
            self.e_secret = self.e_ang_inv
                
        
    def build_alice_bob(self):
        comms_input_length = 2 * self.key_len + self.const_len
        self.a_weights = make_weights("alice_weights", [comms_input_length, self.key_len])
        self.b_weights = make_weights("bob_weights", [comms_input_length, self.key_len])
        

        # AB generating public keys
        with tf.name_scope('Alice_PK') as scope:
            self.a_in = tf.concat([self.const_in, self.key_a_in, self.initial], 1, "a_in")
            self.a_angles = tfc.tf_angle([self.a_in], name="a_angles")
            self.a_out_angles = (tf.matmul(self.a_angles, self.a_weights, name="a_op"))
            self.a_out = tfc.tf_inv_angle([self.a_out_angles], name="pk_a")
        
        with tf.name_scope('Bob_PK') as scope:
            self.b_in = tf.concat([self.const_in, self.key_b_in, self.initial], 1, "b_in")
            self.b_angles = tfc.tf_angle([self.b_in], name="b_angles")
            self.b_out_angles = (tf.matmul(self.b_angles, self.b_weights, name="b_op"))
            self.b_out = tfc.tf_inv_angle([self.b_out_angles], name="pk_b")
        
        # Public keys
        self.a_pk = self.a_out
        self.b_pk = self.b_out
        
        # Send keys to other network and compute secret key, use same weights
        with tf.name_scope('Alice_Secret') as scope:
            self.a2_in = tf.concat([self.const_in, self.key_a_in, self.b_pk], 1, name="a2_in")
            self.a2_angles = tfc.tf_angle([self.a2_in], name="a2_angles")
            self.a2_out_angles = (tf.matmul(self.a2_angles, self.a_weights, name="a2_op"))
            self.a2_out = tf.squeeze(tfc.tf_inv_angle([self.a2_out_angles]), name="a_secret")
        
        with tf.name_scope('Bob_Secret') as scope:
            self.b2_in = tf.concat([self.const_in, self.key_b_in, self.a_pk], 1, name="b2_in")
            self.b2_angles = tfc.tf_angle([self.b2_in], name="b2_angles")
            self.b2_out_angles = (tf.matmul(self.b2_angles, self.b_weights, name="b2_op"))
            self.b2_out = tfc.tf_inv_angle([self.b2_out_angles], name="b_unordered_secret")
        
        # Permutation network (either fixed or learned) set up
        with tf.name_scope('Permutation') as scope:
            if (not self.fixed_perm):
                self.p_weights = make_weights("perm_weights", [self.key_len, self.key_len])
                self.fixed_perm = False
                print("Perm Made")
            else:
                self.p_weights = tf.Variable(self.p_weights, name="perm_weights")
                print("Perm Set")
                
            
            self.p_input = self.b2_out
            self.p_out = tf.matmul(self.p_input, self.p_weights)
        
        # Secrets of A and B
        self.a_secret = self.a2_out
        self.b_secret = tf.squeeze(self.p_out, name="b_secret")
        
        
    # Collects various data, some not used in Loss Functions
    def build_data(self):
        with tf.name_scope('Bit_Distance') as scope:
            self.a_e_dist = tf.abs(self.a_secret - self.e_secret, name="Alice_Eve")
            self.b_e_dist = tf.abs(self.b_secret - self.e_secret, name="Bob_Eve")
            self.a_b_dist = tf.abs(self.a_secret - self.b_secret, name="Alice_Bob")
        
        with tf.name_scope('Avg_Bit_Distances') as scope:
            self.a_e_abd = tf.reduce_mean(self.a_e_dist, name="Alice_Eve")
            self.b_e_abd = tf.reduce_mean(self.b_e_dist, name="Bob_Eve")
            self.a_b_abd = tf.reduce_mean(self.a_b_dist, name="Alice_Bob")
            
        with tf.name_scope('Max_Bit_Distance') as scope:
            self.a_b_max = tf.reduce_max(self.a_b_dist, name ="Alice_Bob")
            
        with tf.name_scope('Bit_Measure') as scope:
            self.a_bm = tfc.tf_bit_measure([self.a_secret], name="A")
            self.b_bm = tfc.tf_bit_measure([self.b_secret], name="B")
            self.a_ed = tf.reduce_mean(self.a_bm, name="Alice")
            self.b_ed = tf.reduce_mean(self.b_bm, name="Bob")
            
        with tf.name_scope('Max_Measure') as scope:
            self.a_max_bm = tf.reduce_max(self.a_bm, name="Alice")
            self.b_max_bm = tf.reduce_mean(self.b_bm, name="Bob")
            
    def train_model(self, epochs, iterations):
        print("Training Initialization")        
        self.e_loss = self.a_e_abd * self.b_e_abd
        
        self.ab_loss = (self.a_b_abd) - (tf.minimum(self.e_loss, 0.25))
        
        self.a_loss = self.ab_loss + 0.1 * self.a_ed
        self.b_loss = self.ab_loss + 0.1 * self.b_ed
        print("Loss functions defined")
        
        self.b_vars = [self.b_weights]
        if not (self.fixed_perm):
            self.b_vars.append(self.p_weights)
        else:
            print("Set permutation")
            
        with tf.name_scope('Optimizers') as scope:
            self.a_optimizer = tf.train.AdamOptimizer(self.lr, name="Alice_Opt").minimize(loss=self.a_loss, var_list=[self.a_weights])
            self.b_optimizer = tf.train.AdamOptimizer(self.lr, name="Bob_Opt").minimize(loss=self.b_loss, var_list=self.b_vars)
            self.e_optimizer = tf.train.AdamOptimizer(self.lr, name="Eve_Opt").minimize(loss=self.e_loss, var_list=[self.e_weights])
        print("Optimizers defined")
        
        ab_errors, ae_errors, be_errors = [], [], []
        tf.global_variables_initializer().run()
        
        tf.summary.scalar("Alice_Eve_ABDistance", self.a_e_abd)
        tf.summary.scalar("Bob_Eve_ABDistance", self.b_e_abd)
        tf.summary.scalar("Alice_Bob_ABDistance", self.a_b_abd)
        tf.summary.scalar("Alice_Loss", self.a_loss)
        tf.summary.scalar("Bob_Loss", self.b_loss)
        tf.summary.scalar("Eve_Loss", self.e_loss)
        tf.summary.scalar("Alice_Bob_Loss", self.ab_loss)
        self.merged = tf.summary.merge_all()
        
        self.eve_writer = tf.summary.FileWriter(self.path + '/eve', self.sess.graph)
        self.ab_writer = tf.summary.FileWriter(self.path + '/ab', self.sess.graph)

        print("Training Start")
        for i in range(epochs):
            print("Training Alice and Bob, Epoch: ", i + 1)
            ab,_,_ = self.train("comms", iterations, i)
            ab_errors.append(float(ab))
            
            print("Training Eve, Epoch: ", i + 1)
            _,ae,be = self.train("adv", iterations, i)
            ae_errors.append(float(ae))
            be_errors.append(float(be))
        
        self.errors = [ab_errors, ae_errors, be_errors]
        print("Trained")
        self.ab_writer.close()
        self.eve_writer.close()
        
            
    def train(self, who_training, iterations, current_epoch):
        # Set initial errors
        ab_secret_error, ae_secret_error, be_secret_error = np.inf, np.inf, np.inf
        e_loss = np.inf
        
        train_bs = self.batchsize
        if (who_training == "adv"):
            train_bs = 2 * train_bs
        
        iv = gen_zeros(train_bs, self.key_len)
        # Run for minibatch:
        for i in range(iterations):
            # Generate inputs (constants, Alice key, Bob key)
            const_ins = gen_binary(train_bs, self.const_len)
            key_a_ins = gen_binary(train_bs, self.key_len)
            key_b_ins = gen_binary(train_bs, self.key_len)
            feed = {self.const_in: const_ins, self.key_a_in: key_a_ins, self.key_b_in: key_b_ins, self.initial: iv}
            
            # Run for Alice/Bob:
            if who_training == "comms":
                comms_fetches = [self.a_optimizer, self.b_optimizer, self.a_b_abd, self.merged]
                _, _, ab_t_error, summary= self.sess.run(comms_fetches, feed_dict=feed)
                
                if ab_t_error < ab_secret_error:
                    final_summary = summary
                    ab_secret_error = ab_t_error
                
                if i == (iterations - 1):
                    self.ab_writer.add_summary(final_summary, current_epoch)
                    
            # OR Run for Eve:
            elif who_training == "adv":
                adv_fetches = [self.e_optimizer, self.e_loss, self.a_e_abd, self.b_e_abd, self.merged]
                _, e_t_loss, ae_t_error, be_t_error, summary = self.sess.run(adv_fetches, feed_dict=feed)
                
                if e_loss > e_t_loss:
                    final_summary = summary
                    e_loss = e_t_loss
                    ae_secret_error = ae_t_error
                    be_secret_error = be_t_error
                
                if i == (iterations - 1):
                    self.eve_writer.add_summary(final_summary, current_epoch)
            
        # Return secret errors
        return ab_secret_error, ae_secret_error, be_secret_error
    
    def test(self, batches):
        const_ins = gen_binary(batches, self.const_len)
        key_a_ins = gen_binary(batches, self.key_len)
        key_b_ins = gen_binary(batches, self.key_len)
        iv_in = gen_zeros(batches, self.key_len)
        
        a_round, a_old = self.round_weights(self.a_weights)
        b_round, b_old = self.round_weights(self.b_weights)

        feed = {self.const_in: const_ins, self.key_a_in: key_a_ins, self.key_b_in: key_b_ins, self.initial: iv_in}
        a_out, b_out = self.sess.run([self.a_secret, self.b_secret], feed)
        
        print("Tests")
        print("Alice (Round):\n", a_round)
        print("Bob (Round):\n", b_round)
        print("Alice Secret:\n", a_out)
        print("Bob Secret:\n", b_out)
        
        self.result = np.all(a_out == b_out)
        
        self.sess.run(tf.assign(self.a_weights, a_old))
        self.sess.run(tf.assign(self.b_weights, b_old))
        
    def round_weights(self, weight_to_round):
        old_weights = self.sess.run(weight_to_round)
        new_weights = np.rint(old_weights)
        self.sess.run(tf.assign(weight_to_round, new_weights))
        return new_weights, old_weights
        
    
    def save_data(self, process_time):
        path = self.path
        # Create a directory w/ unique name/key
        print(path)
        
        # Time file
        time_file = open(path + "\\time.txt", "w")
        time_file.write("Run Date\n")
        time_file.write(str(datetime.datetime.now()))
        time_file.write("\nProcess Time\n")
        time_file.write(str(process_time))
        time_file.close()
        
        # Save errors into a txt file, backup
        error_file = open(path + "\\errors.txt", "w")
        json.dump(self.errors, error_file)
        error_file.close()
        
        # Weight File
        weight_file = open(path + "\\weights.txt", "w")
        weight_file.write("Alice\n")
        weight_file.write(str(self.sess.run(self.a_weights)))
        weight_file.write("\nBob\n")
        weight_file.write(str(self.sess.run(self.b_weights)))
        weight_file.write("\nEve\n")
        weight_file.write(str(self.sess.run(self.e_weights)))
        weight_file.close()
        
        result_file = open(path + "\\result.txt", "w")
        result_file.write(str(self.result))
        result_file.close()
        
        # Save model
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, path + "\\model.ckpt")
        
    def plot(self, epoch):
        xs = range(epoch)
        i = 1
        details = "(Const: " + str(self.const_len) + ", Key: " + str(self.key_len) + ")"
        for elist in self.errors:
            plt.plot(xs, elist)
            
            if i == 1:
                plt.title("Alice and Bob Average Bit Error " + details)
            if i == 2:
                plt.title("Alice and Eve Average Bit Error " + details)
            if i == 3:
                plt.title("Bob and Eve Average Bit Error " + details)
                
            plt.savefig(self.path + "\\" + str(i) + '.png')
            i += 1
            plt.clf()