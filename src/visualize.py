"""
对不同超参数下的loss变化进行可视化
"""

import numpy as np
import matplotlib.pyplot as plt 

train_accuracys_01 = np.load('val_accuracys_01.npy', allow_pickle=True) 
train_accuracys_005 = np.load('val_accuracys_005.npy', allow_pickle=True) 
train_accuracys_001 = np.load('val_accuracys_001.npy', allow_pickle=True) 
# train_losses = np.load('train_losses.npy') 
train_numbers = np.load('train_numbers.npy') 

# print(train_accuracys.shape, train_losses.shape, train_numbers.shape)
# print(train_numbers)
# train_numbers = train_numbers/2100
train_numbers = np.array(range(1,61))
plt.title("Val accuracy of different Learn Rates")
plt.plot(train_numbers, train_accuracys_01, color='orange', label='0.1')
plt.plot(train_numbers, train_accuracys_005, 'b', label='0.05')
plt.plot(train_numbers, train_accuracys_001, 'r', label='0.01')
plt.legend()
plt.show()
