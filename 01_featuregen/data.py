from numpy import genfromtxt
import numpy as n
import matplotlib.pyplot as p
import collections as c

dataset = genfromtxt('data.csv', delimiter=',')

colors = ['b', 'g', 'c', 'm', 'y', 'k']

def iirfilter(dataset, alpha):
    filtered = n.zeros(dataset.shape[0]);
    for i, val in enumerate(dataset):
        filtered[i] = val*alpha + (1-alpha)*filtered[i-1]
    return filtered

def sd(values):
    mean = n.mean(values)
    return n.sqrt(n.mean(n.power(values - mean, 2)))

class Feature_SD:
    buffer_size = 0
    no_value = 0

    def __init__(self, size=10, no_value=0):
        self.buffer_size = size
        self.no_value = no_value
        self.val_buffer = c.deque([], self.buffer_size)

    def update(self, val):
        self.val_buffer.append(val)
        if (len(self.val_buffer) == self.buffer_size):
            return sd(n.asarray(self.val_buffer))
        else:
            return self.no_value

def create_sd_feature(values, buffer_size, no_value):
    f_buffer = Feature_SD(buffer_size, no_value)
    new_feature = n.zeros(values.shape[0])

    for i, val in enumerate(values):
        new_feature[i] = f_buffer.update(values[i])

    return new_feature
            
if __name__ == "__main__":
    time = n.round((dataset[:,0]-dataset[0,0])/10000)
    firstC = dataset[:,1]
    filtered = []
    for a in n.arange(0,1.1,0.1):
        filtered.append(iirfilter(firstC, a))
    
    for i, a in enumerate(filtered):
        p.plot(time, a, colors[i % len(colors)], label="alpha = {0}".format(i * 0.1))
    p.plot(time, firstC, 'r', label="original data")
    p.legend()
    p.show()

    stat = dataset[:, 3]
    p.plot(time, stat, 'r', label="original data")
    p.plot(time, create_sd_feature(stat, 50, 0), 'b', label="buffer 50")
    p.plot(time, create_sd_feature(stat, 300, 0), 'g', label="buffer 300")
    p.legend()
    p.show()
    
    sampling_rate = 10.0
    time_interval = 1.0/sampling_rate
    length = len(time)
    k = n.arange(length)
    T = length/sampling_rate
    frequency = k/T

    fourier_1 = n.fft.fft(firstC)/len(time)
    fourier_2 = n.fft.fft(dataset[:,2])/len(time)
    fourier_3 = n.fft.fft(stat)/len(time)
    p.plot(frequency, abs(fourier_1), 'b', label="second column")
    p.legend()
    p.show()

    p.plot(frequency, abs(fourier_2), 'b', label="third column")
    p.legend()
    p.show()

    p.plot(frequency, abs(fourier_3), 'b', label="fourth column")
    p.legend()
    p.show()

