"""Tinyflow example code.

This code is adapted from Tensorflow's MNIST Tutorial with minimum code changes.
"""
from __future__ import print_function

import tinyflow as tf
from tinyflow.datasets import get_mnist

from nnvm import symbol, graph

batch_size = 100
# Create the model
x = tf.placeholder(tf.float32, [batch_size, 784, 1, 1], name='images')
W = tf.Variable(tf.zeros([784, 10]), name='fc/weight')
W2 = tf.Variable(tf.zeros([3, 3, 1, 10]), name='conv/weight')

with tf.attr_scope(parallelism='data'):
    net = tf.nn.conv2d(x, W2, padding='SAME', name='conv/conv2d')

net = tf.reshape(net, [batch_size, 7840], name='fc/reshape')
with tf.attr_scope(parallelism='model'):
    net = tf.matmul(net, W, name='fc/matmul')

y = tf.nn.softmax(net, name='y')

y_ = tf.placeholder(tf.float32, [batch_size, 10], name='labels')

# Apply transform
g = graph.create(y)
print('Before')
g = g.apply('DumpGraph')

print('\n\nParallelizing')
g = g.apply('Parallelize')

print('\n\nAfter')
g = g.apply('DumpGraph')
g.apply('DotGraph')

## Define loss
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]), name='cross_entropy')

## Define accuracy
#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(correct_prediction, name='accuracy')


print('\n\nVariable Initialization')
sess = tf.Session()
sess.run(tf.initialize_all_variables())

## get the mnist dataset
#mnist = get_mnist(flatten=True, onehot=True)
#batch_xs, batch_ys = mnist.train.next_batch(100)
#sess.run([y, cross_entropy, accuracy], feed_dict={x: batch_xs, y_:batch_ys})

## Optimizer to train
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)



#for i in range(1000):
#    batch_xs, batch_ys = mnist.train.next_batch(100)
#    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

#print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
