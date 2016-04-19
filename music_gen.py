import scipy
from scipy.io.wavfile import read
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn



bach_set= read('cell-8b-8khz.wav')
bach_array= np.array(bach_set[1]) # made this int, should it be float instead



#Defining some hyper-params
input_size = 1      #num_units and input_size will be the same
num_units = input_size*2       #this is the parameter for input_size in the basic LSTM cell


seq_len = 80
num_epochs=100
rounding= len(bach_array)/seq_len
batch_size = rounding
bach_array= bach_array[:rounding*seq_len]
bach_array = bach_array.reshape([batch_size,-1])

bach_lists=[]
target_list =[]
for i in range(rounding/batch_size):
    array_to_append= (bach_array[:,i*seq_len: (i+1)*seq_len-1])
    target_list.append(bach_array[:,(i+1)*seq_len-1])
    #print (len(target_list ))
    bach_lists.append(array_to_append)

training_input= bach_lists[0:(rounding*3)/4]
print (len(training_input))
print (training_input[0].shape)
training_targets= np.asarray(target_list[0:(rounding*3)/4])
valid_input= bach_lists[(rounding*3)/4:]
valid_target= np.asarray(target_list[(rounding*3)/4:])

# MODEL
cell = rnn_cell.BasicLSTMCell(num_units)

inputs = [tf.placeholder(tf.float32,shape=[batch_size,input_size]) for _ in range(seq_len-1)]
result = tf.placeholder(tf.float32, shape=[batch_size,input_size])

outputs, states = rnn.rnn(cell, inputs, dtype=tf.float32)

outputs2 = outputs[-1]

W_o = tf.Variable(tf.random_normal([num_units,input_size], stddev=0.01))
b_o = tf.Variable(tf.random_normal([input_size], stddev=0.01))

outputs3 = tf.tanh(tf.matmul(tf.tanh(outputs2),W_o) + b_o)

cost = tf.reduce_mean(tf.abs(outputs3-result))

train_op = tf.train.AdamOptimizer(0.05).minimize(cost)

with tf.Session() as sess:

    tf.initialize_all_variables().run()     #initialize all variables in the model
    #temp_dict= {inputs:training_input,result:training_targets}
    temp_dict = {inputs[i]:training_input[i] for i in range(seq_len-1)}
    temp_dict.update({result: training_targets})
    val_dict = {inputs[i]:valid_input[i] for i in range(seq_len-1)}
    val_dict.update({ result: valid_target})
    for k in range(num_epochs):
        sess.run(train_op,feed_dict=temp_dict)
        train_cost = sess.run(cost, feed_dict = temp_dict )
        valid_cost= sess.run(cost, feed_dict = val_dict)
        y1 = outputs3.eval(feed_dict=val_dict)
        y2 = val_dict[result]
        print "tc: {} Validation cost: {}, on Epoch {}".format(train_cost, valid_cost, k)


