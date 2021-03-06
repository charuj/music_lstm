import scipy
from scipy.io.wavfile import read
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn
import matplotlib.pyplot as plt

def normalize(data):
    mean = data.mean(axis=0)
    std = data.max(axis=0)
    data = (data-mean)/std
    data = data
    data = data
    return data


bach_set= read('cell-8b-8khz.wav')
bach_array= np.array(bach_set[1]) # made this int, should it be float instead

bach_array=normalize(bach_array)
# DONE: batch training instead of full dataset all at once
# DONE: multiple layers
# TODO: dropout
# TODO: graph training and validation error
# TODO: output file as wav
# TODO: increase validation set size
# TODO: add stereo channel to extend dataset
# TODO: try overlapping batches

#Defining some hyper-params
input_size = 1      #num_units and input_size will be the same
num_units = input_size*100    #this is the parameter for input_size in the basic LSTM cell


seq_len = 80
num_epochs=100
number_of_layers = 2
def data_process(bach_array):



    total_seq= bach_array.shape[0]
    num_seq= total_seq/seq_len
    bach_array= bach_array[:num_seq*seq_len]
    bach_array= bach_array.reshape(num_seq, seq_len)
    bach_array=np.transpose(bach_array)


    bach_lists=[]
    for i in range(seq_len-1):
        row=bach_array[i,:]
        row= np.expand_dims(row, axis=1)
        bach_lists.append(row)

    target_list= bach_array[-1,:]
    target_list=target_list.tolist()

    target_list=  np.expand_dims(target_list, axis=1)



    return target_list, bach_lists, num_seq

num_batches = 300
training_set_size = (bach_array.shape[0])/num_batches

#
# batch_size = bach_array
# bach_array= bach_array[:total_seq*seq_len]
# bach_array = bach_array.reshape([batch_size,-1])
#
# bach_lists=[]
# target_list =[]
# for i in range(total_seq/batch_size):
#     array_to_append= (bach_array[:,i*seq_len: (i+1)*seq_len-1])
#     target_list.append(bach_array[:,(i+1)*seq_len-1])
#     #print (len(target_list ))
#     bach_lists.append(array_to_append)

#training_input= bach_lists[0:(total_seq*3)/4]
# print (len(training_input))
# print (training_input[0].shape)
#training_targets= np.asarray(target_list[0:(total_seq*3)/4])
validation_start = int(0.2*num_batches*training_set_size)
valid_target, valid_input, num_seq= data_process(bach_array[-validation_start:])


# MODEL
cell = rnn_cell.BasicLSTMCell(num_units)
cell = rnn_cell.MultiRNNCell([cell] * number_of_layers)


inputs = [tf.placeholder(tf.float32,shape=[None,input_size]) for _ in range(seq_len-1)]
result = tf.placeholder(tf.float32, shape=[None,input_size])



outputs, states = rnn.rnn(cell, inputs, dtype=tf.float32)

outputs2 = outputs[-1]

W_o = tf.Variable(tf.random_normal([num_units,input_size], stddev=0.01))
b_o = tf.Variable(tf.random_normal([input_size], stddev=0.01))

outputs3 = tf.nn.tanh(tf.matmul(tf.nn.tanh(outputs2),W_o) + b_o)

cost = tf.reduce_mean(tf.abs(outputs3-result))

train_op = tf.train.AdamOptimizer(0.005).minimize(cost)

saver = tf.train.Saver()

with tf.Session() as sess:

    tf.initialize_all_variables().run()     #initialize all variables in the model
    #temp_dict= {inputs:training_input,result:training_targets}

    val_dict = {inputs[i]:valid_input[i] for i in range(seq_len-1)}
    val_dict.update({ result: valid_target})
    train_error=[]
    valid_error=[]
    for k in range(num_epochs):
        training_targets, training_input, num_seq = data_process(bach_array[(k*training_set_size)%(bach_array.shape[0]-validation_start):((k+1)*training_set_size)%(bach_array.shape[0]-validation_start)])
        temp_dict = {inputs[i]:training_input[i] for i in range(seq_len-1)}
        temp_dict.update({result: training_targets})
        sess.run(train_op,feed_dict=temp_dict)
        train_cost = sess.run(cost, feed_dict = temp_dict )
        train_error.append(train_cost)
        valid_cost= sess.run(cost, feed_dict = val_dict)
        valid_error.append(valid_cost)
        y1 = outputs3.eval(feed_dict=val_dict)
        y2 = val_dict[result]
        print "tc: {} Validation cost: {}, on Epoch {}".format(train_cost, valid_cost, k)
    save_path = saver.save(sess, "model_lstm4")
    print("Model saved in file: %s" % save_path)

    epochs=np.arange(num_epochs)
    plt.plot(epochs, train_error, 'r--', epochs, valid_error, 'b--')
    plt.show()