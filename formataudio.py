import scipy
from scipy.io.wavfile import read
import numpy as np

def normalize(data):
    mean = data.mean(axis=0)
    std = data.max(axis=0)
    data = (data-mean)/std
    data = data * 255
    data = data + 127
    data= data.astype(int)
    return data


bach_set= read('cell-8b-8khz.wav')
bach_array= np.array(bach_set[1])
bach_array=normalize(bach_array)
N=len(bach_array)
bach_array= bach_array[:N/8]

zero_matrix=np.zeros((len(bach_array),256))
zero_matrix[np.arange(len(bach_array)), bach_array] = 1

print bach_array

