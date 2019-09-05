from keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import cifar100

import numpy.random as rndm
import matplotlib.pyplot as plt

def plot_data(x,y,figname):
	num_samples=x.shape[0]
	titles=y[:,0]
	print('size of dataset:',num_samples)
	r, c = 8,8
	fig, axs = plt.subplots(r, c)
	fig.set_size_inches(13,11)
	count = 0
	rndm.seed(20160903)
	index=rndm.randint(0,num_samples,size=(r*c))
	for i in range(r): 
		for j in range(c):
			print(count,i,j,index[count])
			axs[i,j].imshow(x[index[count]],interpolation='lanczos') 
			axs[i,j].axis('off')
			axs[i,j].set_title(titles[index[count]])
			count += 1
	fig.tight_layout()
	#plt.show()
	fig.savefig(figname+'.png')
	#plt.close()
	

'''
x_train, x_test: uint8 array of grayscale image data with shape (num_samples, 28, 28).
y_train, y_test: uint8 array of digit labels (integers in range 0-9) with shape (num_samples,).
'''
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('mnist:',x_train.shape,'x',y_train.shape)

'''
x_train, x_test: uint8 array of RGB image data with shape (num_samples, 3, 32, 32) or (num_samples, 32, 32, 3) based on the image_data_format backend setting of either channels_first or channels_last respectively.
y_train, y_test: uint8 array of category labels (integers in range 0-9) with shape (num_samples,).
'''

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('cifar:',x_train.shape,'x',y_train.shape)
plot_data(x_train,y_train,'testcifar10')
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
print('cifar:',x_train.shape,'x',y_train.shape)
plot_data(x_train,y_train,'testcifar100')

