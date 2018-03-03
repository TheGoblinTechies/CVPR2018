import numpy as np
import copy
#import US_network as us
from keras.models import Model, Sequential
from keras.layers import Dense, Reshape, Flatten, Activation
from keras.optimizers import Adam, RMSprop
from keras.layers.convolutional import Convolution2D
from keras import backend as K
import random
class PGAgent:
    def __init__(self, state_size, action_size, dim_feature, batchsize):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = 0.00002
        self.n_filters = 32
        self.connect_thr = 0.00001
        self.remove_thr = 0.001
        self.batchsize = batchsize
        self.dim_feature = dim_feature
        self.states = []
        self.gradients1 = []
        self.rewards1 = []
        self.gradients2 = []
        self.rewards2 = []
        self.probs = []
        self.maxprob = 0
        self.minprob = 0
        self.epsilon = 0.05
        self.model = self._build_model()
        self.model.summary()

    def _build_model(self):
		model = Sequential()
		model.add(Reshape((self.dim_feature, self.dim_feature, 1), input_shape=(self.state_size,)))
		model.add(Convolution2D(self.n_filters, 5, 5, border_mode='same',
                                activation='relu', init='he_uniform'))
		model.add(Convolution2D(self.n_filters, 3, 3, border_mode='same',
                                activation='relu', init='he_uniform'))
		model.add(Convolution2D(1, 1, 1, border_mode='same',
                                 init='he_uniform'))
		model.add(Flatten())
		model.add(Activation(('softmax')))
		model.add(Reshape((self.dim_feature, self.dim_feature)))
		opt = RMSprop(lr=self.learning_rate)
		model.compile(loss='categorical_crossentropy', optimizer=opt)
		return model

    def remember(self, state, action1, action2, prob, reward1, reward2):
        if action1>-1:
        	y1 = np.zeros([self.dim_feature, self.dim_feature])
        	y1[int(action1/self.dim_feature),action1%self.dim_feature] = 1
        	self.gradients1.append(np.array(y1).astype('float32') - prob)
        	self.rewards1.append(reward1)
        else:
			self.gradients1.append(np.array(np.zeros([1, self.dim_feature, self.dim_feature])).astype('float32'))
			self.rewards1.append([0])
        if action2>-1:
         	y2 = np.zeros([self.dim_feature, self.dim_feature])
        	y2[int(action2/self.dim_feature),action2%self.dim_feature] = 1
        	self.gradients2.append(-(np.array(y2).astype('float32') - np.log(1-prob))/(1-prob))
        	self.rewards2.append(reward2)
        else:
			self.gradients2.append(np.array(np.zeros([1, self.dim_feature, self.dim_feature])).astype('float32'))
			self.rewards2.append([0])
        #print('g1',self.gradients1)
        #print('g2',self.gradients2)
        self.states.append(state)


    def act(self, state, act_times, cur_x):
		act = 0
		action1 = -1
		action2 = -1
		action3 = -1
		
		aprob = self.model.predict(state, batch_size=1)

		#print('probsum',np.sum(aprob, axis = -1))

		bprob = copy.deepcopy(-np.log(aprob))


		if np.sum(np.sum(aprob)) != 0:
			prob = aprob/np.sum(np.sum(aprob))
		else:
			prob = np.ones((1,self.dim_feature,self.dim_feature))/(self.dim_feature*self.dim_feature)
		#print('prob',prob)

		self.probs.append(prob)
		prob_trans = prob.reshape((self.dim_feature*self.dim_feature))

		if np.sum(np.sum(bprob))==0:
			antiprob = np.ones((1,self.dim_feature,self.dim_feature))/(self.dim_feature*self.dim_feature)
		else:
			antiprob = bprob/np.sum(np.sum(bprob))
		antiprob_trans = antiprob.reshape((self.dim_feature*self.dim_feature))
		self.maxprob = np.max(np.max(prob))
		self.minprob = np.min(np.min(prob))
		print('thr',self.connect_thr,self.remove_thr, np.max(np.max(prob)), np.min(np.min(prob)))
		if max(prob_trans)>self.connect_thr:
			random_num = random.random()
			if random_num>self.epsilon:
				action1 = np.random.choice(self.action_size, 1, p=prob_trans)[0]
			else:
				action1 = np.random.choice(self.action_size, 1)[0]
			act = 1
		if min(prob_trans)<self.remove_thr and np.sum(np.sum(bprob))!=0:
			random_num = random.random()
			if random_num>self.epsilon:
				action2 = np.random.choice(self.action_size, 1, p=antiprob_trans)[0]
			else:
				action2 = np.random.choice(self.action_size, 1)[0]
			act = 1
		if act == 0 or act_times>0:
			action3 = 1
		return action1, action2, action3, prob

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in range(0, rewards.size):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self):
        if len(self.gradients1) > 0:
		    gradients1 = np.vstack(self.gradients1)

		    rewards1 = np.vstack(self.rewards1)
		    #print('reward1',rewards1.shape)
		    rewards1 = self.discount_rewards(rewards1)

		    if len(rewards1) > 1 and np.mean(rewards1)!=0:
		    	rewards1 = rewards1 / np.std(rewards1 - np.mean(rewards1))
		    	#print('r1',rewards1)
		    #print('g1shape',gradients1.shape)
		    for i in range(gradients1.shape[0]):
		    	gradients1[i,:,:] = np.squeeze(rewards1[i])*gradients1[i,:,:]
		    #print('gg1',gradients1)
		    
        else:
			gradients1 = np.zeros((1,self.dim_feature, self.dim_feature))
        if len(self.gradients2) > 0:
		    gradients2 = np.vstack(self.gradients2)
		    rewards2 = np.vstack(self.rewards2)
		    rewards2 = self.discount_rewards(rewards2)
		    #print('r2',rewards2)
		    if len(rewards2) > 1 and np.mean(rewards2)!=0:
		    	rewards2 = rewards2 / np.std(rewards2 - np.mean(rewards2))
		    for i in range(gradients2.shape[0]):
		    	gradients2[i,:,:] = np.squeeze(rewards2[i])*gradients2[i,:,:]
		    #print('gg2',gradients2.shape)
		    #print('g2shape',gradients2.shape)
        else:
			gradients2 = np.zeros((1,self.dim_feature, self.dim_feature))
        
 
        self.probs = np.squeeze(self.probs)
        X = np.squeeze(np.vstack([self.states]))
        if np.squeeze(np.vstack([gradients1])).shape[0] != self.probs.shape[0]:
			Y = self.probs[:-1,:,:] + self.learning_rate * (np.squeeze(np.vstack([gradients1]))+np.squeeze(np.vstack([gradients2])))
        else:
        	Y = self.probs + self.learning_rate * (np.squeeze(np.vstack([gradients1]))+np.squeeze(np.vstack([gradients2])))

        if X.shape[0] == self.dim_feature*self.dim_feature:
			X = np.reshape(X,(1,self.dim_feature*self.dim_feature))
			Y = np.reshape(Y,(1,self.dim_feature,self.dim_feature))
        #print('X',X,Y)
        self.model.train_on_batch(X, Y)
        self.states, self.probs, self.gradients1, self.gradients2, self.rewards1, self.rewards2 = [], [], [], [], [], []

    #def load(self, name):
        #self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

