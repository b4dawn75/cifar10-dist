'''
Basic Operations example using TensorFlow library.
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf

# Basic constant operations
# The value returned by the constructor represents the output
# of the Constant op.
a = tf.constant(2)
b = tf.constant(3)

# Launch the default graph.
with tf.Session() as sess:
    print("a=2, b=3")
    print("Addition with constants: %i" % sess.run(a + b))
    print("Multiplication with constants: %i" % sess.run(a * b))

# Basic Operations with variable as graph input
# The value returned by the constructor represents the output
# of the Variable op. (define as input when running session)
# tf Graph input
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# Define some operations
add = tf.add(a, b)
mul = tf.mul(a, b)

# Launch the default graph.
with tf.Session() as sess:
    # Run every operation with variable input
    print("Addition with variables: %i" % sess.run(add,
                                                   feed_dict={a: 2, b: 3}))
    print("Multiplication with variables: %i" %
          sess.run(mul, feed_dict={a: 2, b: 3}))


# ----------------
# More in details:
# Matrix Multiplication from TensorFlow official tutorial

# Create a Constant op that produces a 1x2 matrix.  The op is
# added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
matrix1 = tf.constant([[3., 3.]])

# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.], [2.]])

# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.
product = tf.matmul(matrix1, matrix2)

# To run the matmul op we call the session 'run()' method, passing 'product'
# which represents the output of the matmul op.  This indicates to the call
# that we want to get the output of the matmul op back.
#
# All inputs needed by the op are run automatically by the session.  They
# typically are run in parallel.
#
# The call 'run(product)' thus causes the execution of threes ops in the
# graph: the two constants and matmul.
#
# The output of the op is returned in 'result' as a numpy `ndarray` object.
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
    # ==> [[ 12.]]


def mysum(a, b, name=None):
    with tf.op_scope([a, b], name, "mysum") as scope:
        v = tf.get_variable("v", 1)
        v2 = tf.Variable([0], name="v2")
        assert v.name == "v:0", v.name
        assert v2.name == "mysum/v2:0", v2.name
        return tf.add(a, b)


def mysum2(a, b, name=None):
    with tf.variable_op_scope([a,b],name,"mysum2") as scope:
        v = tf.get_variable("v", 1)
        v2 = tf.Variable([0], name="v2")
        assert v.name == "mysum2/v:0", v.name
        assert v2.name == "mysum2/v2:0", v2.name
        return tf.add(a, b)

with tf.Graph().as_default():
    op = mysum(tf.Variable(1), tf.Variable(2))
    op2 = mysum2(tf.Variable(1), tf.Variable(2))
    assert op.name == 'mysum/Add:0', op.name
    assert op2.name == 'mysum2/Add:0', op2.name


with tf.Graph().as_default():
    with tf.name_scope("name_scope") as scope:
        v = tf.get_variable("v", [1])
        op = tf.add(v, v)
        v2 = tf.Variable([0], name="v2")
        assert v.name == "v:0", v.name
        assert op.name == "name_scope/Add:0", op.name
        assert v2.name == "name_scope/v2:0", v2.name

with tf.Graph().as_default():
    with tf.variable_scope("name_scope") as scope:
        v = tf.get_variable("v", [1])
        op = tf.add(v, v)
        v2 = tf.Variable([0], name="v2")
        assert v.name == "name_scope/v:0", v.name
        assert op.name == "name_scope/Add:0", op.name
        assert v2.name == "name_scope/v2:0", v2.name
