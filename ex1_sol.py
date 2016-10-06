# Copyright 2016 Joongsoo Lee @ ETRI. All Rights Reserved.

"""An exercise program.

Make a program that returns average of fist elements of inner list.

For example,
when given grad_and_vars = [[1, 5], [3, 7]]

The result should be 2.
"""

import tensorflow as tf

grad_and_vars = [[1.1, 2.0], [0.7, 0.9], [0.6, 0.4], [1.2, 5.4]]

sess = tf.Session()

grads = []

for g, _ in grad_and_vars:
    expanded_g = tf.expand_dims(g, 0)
#    print(sess.run(expanded_g))
    grads.append(expanded_g)
#    grads.append(g)


#print(sess.run(grads))
grad = tf.concat(0, grads)
#print(sess.run(grads))
#grad = tf.reduce_mean(grads, 0)
grad = tf.reduce_mean(grad, 0)

#sess = tf.Session()
print(sess.run(grad))
