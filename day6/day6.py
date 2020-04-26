#!/usr/bin/env python
import tensorflow as tf

x=tf.zeros([2,3],tf.int32)
sess=tf.Session()
print(sess.run(x))

#q2
x1=tf.constant([[1,2,3],[4,5,6]],tf.int32)
print(sess.run(x1))

y=tf.zeros_like([[1,2,3],[4,5,6]],tf.int32)
print(sess.run(y))

#q3-q4

y=tf.ones_like([[1,2,3],[4,5,6]],tf.int32)
print(sess.run(y))
#q5
y=tf.zeros([2,3],tf.int32)
print ('new',sess.run(y))
z=y+5

#method2
z=tf.Variable(y+5,tf.int32)
model=tf.global_variables_initializer()
sess.run(model)

print(sess.run(z))
#q6
t=tf.constant([[1,3,5],[4,6,8]],tf.float32)
print(sess.run(t))

#q7
y=tf.zeros([2,3],tf.int32)
print ('new',sess.run(y))
z=y+4

#method2
z=tf.Variable(y+4,tf.int32)
model=tf.global_variables_initializer()
sess.run(model)

print(sess.run(z))
#q8
x=tf.linspace(5.0,10.0,50)
print('linspace:',sess.run(x))
#q9
x=tf.random_normal([3,2],mean=0,stddev=2)
print('normal:',sess.run(x))

#q10
x=tf.random_uniform([3,2],minval=0,maxval=2,dtype=tf.dtypes.float32)
print('Random:',sess.run(x))

#q11
x=tf.random_shuffle([[1,2],[3,4],[5,6]])
print ('Shuffle:',sess.run(x))
#q12
y=tf.random_normal([10,10,3])
print('Normal:',sess.run(y))
x=tf.random_crop(y,[5,5,3])
print('Crop:',sess.run(x))
#q13
x=tf.constant([[-1,-2,-3],[0,1,2]],tf.int32)
print(sess.run(x))
y=tf.zeros([2,3],tf.int32)
print(sess.run(y))
result=tf.not_equal(x,y)
print('Check',sess.run(result))
#q14method 1
z=tf.math.add(x=8,y=5)
print('addition',sess.run(z))
y=tf.math.subtract(x=9,y=6)
print('subtraction'.sess.run(z))
w=tf.math.multiply(x=7,y=6)
print('multiply',sess.run(w))
#method2q14-q19
w=tf.constant([[1,2,3],[4,5,6]])
y=tf.constant([2,4,6])
result=w-y
print('subtract',sess.run(result))
#
w=tf.constant([[1,2,3],[4,5,6]])
y=tf.constant([2,4,6])
result=w+y
print('add',sess.run(result))
#
w=tf.constant([[1,2,3],[4,5,6]])
y=tf.constant([2,4,6])
result=w*y
print('multipy'.sess.run(result))
#
w=tf.constant([[1,2,3],[4,5,6]])
y=tf.constant([5])
result=w*y
print('single no multiply',sess.run(result))
#
x=tf.constant([[1,2,3],[4,5,6]])
y=tf.constant([2,4,6])
z=tf.constant([[4,6,8],[4,5,6]])
result=x+y+z
print('x+y+z',sess.run(result))

#q20
r=tf.Variable(1.0,name='Weight')
model=tf.global_variables_initializer()
sess.run(model)
print(sess.run(r))

#q21

x=tf.Variable(17,name='x')
y=tf.Variable(26,name='y')
z=tf.Variable(35,name='z')
tot=x+y+z
model=tf.global_variables_initializer()
sess.run(model)
print('x+y+z:',sess.run(tot))

#q22
x=tf.placeholder(tf.float32)
sess=tf.Session()
print('placeholder',sess.run(x,feed_dict={x:10}))
#q23
x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
z=tf.placeholder(tf.float32)
total=tf.add_n([x,y,z])
print(total)
sess=tf.Session()
print('feeddict',sess.run(total,feed_dict={x:10,y:20,z:30}))


