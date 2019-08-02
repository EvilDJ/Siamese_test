import tensorflow as tf
with tf.name_scope('keep_prob') as scope:
    keep_prob = tf.placeholder(tf.float32)
    
with tf.variable_scope('input_x1') as scope:
    x1 = tf.placeholder(tf.float32, shape=[None, 784])


print(x1)
#print(keep_prob)