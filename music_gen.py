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

batch_size = 1000
seq_len = 8000
num_epochs=100
rounding= len(bach_array)/seq_len
bach_array= bach_array[:rounding*seq_len]

bach_lists=[]
target_list =[]
for i in range(rounding):
    array_to_append= (bach_array[i*seq_len: (i+1)*seq_len-1])
    target_list.append(bach_array[(i+1)*seq_len-1])
    print len(target_list )
    bach_lists.append(array_to_append)

training_input= bach_lists[0:(rounding*3)/4]
training_targets= target_list[0:(rounding*3)/4]
valid_input= bach_lists[(rounding*3)/4:]
valid_target= target_list[(rounding*3)/4:]

print 'blag'
