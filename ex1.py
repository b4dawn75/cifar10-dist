# Copyright 2016 Joongsoo Lee @ ETRI. All Rights Reserved.

"""An exercise program.

Make a program that returns average of fist elements of inner list.

For example,
when given grad_and_vars = [[1, 5], [3, 7]]

The result should be 2.

tf.expand_dims()
tf.concat()
tf.reduce_mean()

"""

import tensorflow as tf

grad_and_vars = [[1.1, 2.0], [0.7, 0.9], [0.6, 0.4], [1.2, 5.4]]
