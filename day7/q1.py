#!/usr/bin env python
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist =input_data.read_data_sets("/tmp/data/",one_hot=True)

Xtrain,ytrain=mnist.train.next_batch(5000)
Xtest,ytest=mnist.test.next_batch(10)

print(Xtrain.shape)

xtrain=tf.placeholder('float',[None,784])
xtest=tf.placeholder('float',[784])

distance=tf.reduce_sum(tf.abs(tf.add(xtrain,tf.negative(xtest))),reduction_indices=1)
prediction=tf.arg_min(distance,0)
print(prediction)

accuracy=0

init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)
for i in range(len(Xtest)):
	nn_index=sess.run(prediction,feed_dict={xtrain:Xtrain,xtest:Xtest[i,:]})
	writer=tf.summary.FileWriter('g1',sess.graph)
	writer.close()
	print(nn_index)
	print("Prediction:",np.argmax(ytrain[nn_index]))
	print("True Class:",np.argmax(ytest[i]))

