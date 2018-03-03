import numpy as np
import copy
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Dense, Reshape, Flatten, Activation, Input, Lambda, merge
from keras.optimizers import Adam, RMSprop
from keras.layers.convolutional import Convolution2D
from keras.applications.vgg16 import VGG16
from RL_network import PGAgent 
from keras import regularizers
from keras import backend as K
from keras.utils.vis_utils import plot_model
import tensorflow as tf
class USNet:
	def __init__(self, dim_feature, batchsize, alpha, beta):
		self.learning_rate = 0.01
		self.dim_feature = dim_feature
		self.batchsize = batchsize
		self.alpha = alpha
		self.beta = beta
		self.model= self._build_model()
		self.relax = 0.5
		self.center_prob = 0.5

	def RELoss(self):
		def reconstruction_loss(y_true, y_pred):
			return K.mean(K.square(y_pred-y_true),axis= -1)
		return reconstruction_loss

	def TOLoss(self,xx,yy):
		def total_loss(y_true, y_pred):
			output_list = []
			temp = []
			num_post = [1]*self.dim_feature
			for i in range(self.dim_feature):
				temp.append(y_pred[:,i*self.dim_feature+i])
			p0 = tf.transpose(tf.stack(temp))
			for j in range(len(yy)):
				temp[yy[j]] += y_pred[:,xx[j]*self.dim_feature+yy[j]]
				num_post[yy[j]] += 1
			for k in range(len(num_post)):
				temp[k] /= num_post[k]

			output_list = temp

			p = tf.transpose(tf.stack(output_list))
			return 100*(K.mean(K.square(K.mean(p, axis = 0)-self.center_prob),axis = -1)-self.alpha*K.mean(K.log(K.abs(p-self.center_prob)+self.relax), axis = -1)+self.beta*K.mean(K.square(p0-p), axis = -1))
		return total_loss
	def make_parallel(self, model, gpu_count):
		def get_slice(data, idx, parts):
		    shape = tf.shape(data)
		    size = tf.concat([ shape[:1] // parts, shape[1:] ],axis=0)
		    stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ],axis=0)
		    start = stride * idx
		    return tf.slice(data, start, size)

		outputs_all = []
		for i in range(len(model.outputs)):
		    outputs_all.append([])

		#Place a copy of the model on each GPU, each getting a slice of the batch
		for i in range(gpu_count):
		    with tf.device('/gpu:%d' % i):
		        with tf.name_scope('tower_%d' % i) as scope:

		            inputs = []
		            #Slice each input into a piece for processing on this GPU
		            for x in model.inputs:
		                input_shape = tuple(x.get_shape().as_list())[1:]
		                slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
		                inputs.append(slice_n)                

		            outputs = model(inputs)
		            
		            if not isinstance(outputs, list):
		                outputs = [outputs]
		            
		            #Save all the outputs for merging back together later
		            for l in range(len(outputs)):
		                outputs_all[l].append(outputs[l])

		# merge outputs on CPU
		with tf.device('/cpu:0'):
		    merged = []
		    for outputs in outputs_all:
		        merged.append(merge(outputs, mode='concat', concat_axis=0))
            
        	return Model(input=model.inputs, output=merged)
	def _build_model(self):

		x = Input(shape=(4096,))
		#model = Sequential()
		#model = VGG16(weights = 'imagenet', include_top = 'true')
		#model.layers.pop()	
		#for layer in model.layers:
			#layer.trainable = False
		#model.layers.pop()
		#model.summary()
		#plot_model(model, to_file='model1.png',show_shapes=True)
		#x = model.output

		#x = model(x)
		u1 = Dense(4096, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01), name = 'fc00')(x)
		u2 = Dense(4096, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01), name = 'fc01')(u1)
		u3 = Dense(4096, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01),)(u2)
		#u4 = Dense(4096)(u3)
		#x1 = Dense(512)(u2)
		w1 = Dense(self.dim_feature*self.dim_feature, name = 'w1')(u3)

		model_x = Model(inputs= x,outputs=w1)
		#model_x = self.make_parallel(model_x, 2)
		return model_x

	def predict(self,X):
		w,p = self.model.predict_on_batch(X)
		return w,p

	def sigmoid(self,x):
		return 1.0/(1+np.exp(-x))

	def return_loss(self,X,xx,yy):
		p = np.zeros((X.shape[0], self.dim_feature))
		#print('X',X)
		num_post = [1]*self.dim_feature
		w1 = self.model.predict(X)
		for j in range(self.dim_feature):
			p[:,j] = w1[:,j*self.dim_feature+j]
		p0 = copy.deepcopy(p)
		print('X',p0[0,:],p[0,:])
		print('loss',np.mean(np.square(np.mean(p, axis = 0)-self.center_prob),axis = -1),-self.alpha*np.mean(np.mean(np.log(np.abs(p-self.center_prob)+self.relax))), self.beta*np.mean(np.mean(np.square(p0-p))))
		for j in range(self.dim_feature):
			p[:,j] = w1[:,j*self.dim_feature+j]
			for i in range(len(xx)):
				#print('prob_af1',prob_af[n,yy[i]])
				if j == yy[i]:
					p[:,yy[i]] += w1[:,xx[i]*self.dim_feature+yy[i]]
					num_post[j] += 1
			p[:,j] /= num_post[j]
		print('XX',p[0,:], np.mean(p, axis = 0))
		print('XXX',p[:,0])
		calculate_loss_af = 100*(np.mean(np.square(np.mean(p, axis = 0)-self.center_prob),axis = -1)-self.alpha*np.mean(np.mean(np.log(np.abs(p-self.center_prob)+self.relax)))+self.beta*np.mean(np.mean(np.square(p0-p))))
		#calculate_loss_af = np.mean(np.mean(np.log(np.abs(p-0.5)+2)))
		print('loss',np.mean(np.square(np.mean(p, axis = 0)-self.center_prob),axis = -1),-self.alpha*np.mean(np.mean(np.log(np.abs(p-self.center_prob)+self.relax))), self.beta*np.mean(np.mean(np.square(p0-p))))
		return calculate_loss_af

	def train(self,X,xx,yy):
		opt=keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
		self.model.compile(loss=self.TOLoss(xx,yy),optimizer=opt)
		self.model.train_on_batch(X, X)

		'''
		#self.model.compile(loss={'fc1':self.RELoss(), 'w1':self.TOLoss(xx,yy,self.x1)},optimizer='adadelta',loss_weights={'fc1':1,'w1':1})
		#get_last3_layer_output = K.function([self.model.layers[0].input],[self.model.layers[-2].output])
		#layer_output3 = get_last3_layer_output([X])[0]	
		#prob_bf = layer_output3
		#print('X',X.shape, prob_bf.shape)
		#self.model.train_on_batch(X, {'fc1':X,'w1':X})
		else:
			#self.model.compile(loss={'fc2':self.TLoss()},optimizer='adadelta',loss_weights={'fc2':1})
			#self.model.train_on_batch(X, {'fc2':X})
			self.model.compile(loss={'fc1':self.TLoss(), 'w1':self.Loss()},optimizer='adadelta',loss_weights={'fc1':1,'w1':1})
			get_last3_layer_output = K.function([self.model.layers[0].input],[self.model.layers[-2].output])
			layer_output3 = get_last3_layer_output([X])[0]	
			prob_bf = layer_output3

			#print('X',X, prob_bf)
			self.model.train_on_batch(X, {'fc1':X,'w1':prob_bf})'''

	def fit(self,X,xx,yy,epochs):
		opt=keras.optimizers.RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
		self.model.compile(loss=self.TOLoss(xx,yy,X),optimizer=opt)
		#print('shape',X.shape)
		self.model.fit(X, X, batch_size=self.batchsize, epochs=epochs)

	def load(self, name):
		self.model.load_weights(name)

	def save(self, name):
		self.model.save_weights(name)

