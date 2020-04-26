#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist= input_data.read_data_sets("/tmp/data/",one_hot=True)

#parameters 

learning_rate=0.001
training_epochs=20
batch_size=1000
display_step=1

#Network parameters

n_hidden_1=256 # 1st layer of neurons
n_hidden_2=256 # 2nd layer neurons
n_input=784 # MNIST data input(img shape:28*28)
n_classes=10 #MNIST total classes(0-9 digits)

#tf Graph inputs
X=tf.placeholder("float",[None,n_input])
Y=tf.placeholder("float",[None,n_classes])


#Store ayers weight and bias
weights={
  'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
  'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
   'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
}
biases={
 'b1':tf.Variable(tf.random_normal([n_hidden_1])),
 'b2':tf.Variable(tf.random_normal([n_hidden_2])),
 'out':tf.Variable(tf.random_normal([n_classes]))
}

#Create a model
def multilayer_perceptron(x):
	#Hidden fully connected layer with 256 neurons
 	layer_1=tf.add(tf.matmul(x,weights['h1']),biases['b1'])

	#Hidden fully connected layer with 256 neurons
        layer_2=tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])

	#Output fully connected layer a neauron for each class

	out_layer=tf.matmul(layer_2,weights['out'])+biases['out']
	return out_layer

#Construct model

logits=multilayer_perceptron(X)


#Define loss and optimizer

loss_op=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op=optimizer.minimize(loss_op)


# Initializing the variables

init=tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	#Training cycle
	for epoch in range(training_epochs):
		avg_cost=0
		x,y=mnist.train.next_batch(batch_size)
		#Run optimization op and cost op
		_,c=sess.run([train_op,loss_op],feed_dict={X:x,Y:y})
		#Compute average loss
		avg_cost=c
		#Display logs per epoch step
		if epoch %display_step==0:
			print("Epoch:",(epoch+1))
			print("cost={:9f}".format(c))
	print("Optimization Finished")
	#test the model
	pred=tf.nn.softmax(logits)# Apply softmax to logits
	correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
	
	#Calculate accuracy
	accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
	writer=tf.summary.FileWriter('./tfb1',sess.graph)
	writer.close()
	print("Accuracy:",accuracy.eval({X:mnist.test.images,y:mnist.test.labels}))
	









