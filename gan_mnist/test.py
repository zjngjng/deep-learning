import pickle as pkl
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, [None, real_dim], name='inputs_real')
    inputs_z = tf.placeholder(tf.float32, [None, z_dim], name='inputs_z')
    return inputs_real, inputs_z

def generator(z, out_dim, n_units=128, reuse=False, alpha=0.01):
    with tf.VariableScope('generator', reuse=reuse):
        h1 = tf.layers.dense(z, n_units, activation=None)
        h1 = tf.maximum(alpha*h1, h1)
        logits = tf.layers.dense(h1, out_dim, activation=None)
        out = tf.tanh(logits)
        return out

def discriminator(x, n_units=128, reuse=False, alpha=0.01):
    with tf.VariableScope('discriminator', reuse=reuse):
        h1 = tf.layers.dense(x, n_units, activation=None)
        h1 = tf.maximum(h1*alpha, h1)
        logits = tf.layers.dense(h1, 1, activation=None)
        out = tf.sigmoid(logits)
        return logits, out


mnist = input_data.read_data_sets('mnist_data')
input_size = 784
z_size = 100

g_hidden_size = 128
d_hidden_size = 128

alpha = 0.01
smooth = 0.1

lr = 0.02
tf.reset_default_graph()
input_real, input_z = model_inputs(input_size, z_size)
g_out = generator(input_z, g_hidden_size)

d_logits_real, d_out_real = discriminator(input_real)
d_logits_fake, d_out_fake = discriminator(g_out, reuse=True)

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_logits_real)*(1-smooth)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_logits_real)))
d_loss = d_loss_real + d_loss_fake

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)))

t_vars = tf.trainable_variables()
g_vars = [var for var in t_vars if var.name.startswith('generator')]
d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

g_train_op = tf.train.AdamOptimizer(lr).minimize(g_loss, var_list=g_vars)
d_train_op = tf.train.AdamOptimizer(lr).minimize(d_loss, var_list=d_vars)

batch_size = 100
epochs = 100
samples = []
losses = []
saver = tf.train.Saver(var_list=g_vars)
with tf.session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for ii in range(mnist.train.num_examples/batch_size):
            batch = mnist.train.next_batch(batch_size)

            batch_images = batch[0].reshape(batch_size, 784)
            batch_images = batch_images*2-1

            batch_z = np.random.uniform(-1, 1, size=(batch_size,z_size))
            _ = sess.run(d_train_op, feed_dict={input_real:batch_images, input_z:batch_z})
            _ = sess.run(g_train_op, feed_dict={input_z:batch_z})
        train_loss_d = sess.run(d_loss, feed_dict={input_real:batch_images, input_z:batch_z})
        train_loss_g =  sess.run(g_loss, feed_dict={input_z:batch_z})
        print("Epoch {}/{}...".format(e+1, epochs),
              "Discriminator Loss: {:.4f}...".format(train_loss_d),
              "Generator Loss: {:.4f}".format(train_loss_g))
        # Save losses to view after training
        losses.append((train_loss_d, train_loss_g))

        sample_z = np.random.uniform(-1, 1, size=(16, z_size))
        gen_samples = sess.run(generator(input_z, input_size, reuse=True), feed_dict={input_z: sample_z})
        samples.append(gen_samples)
        saver.save(sess, './checkpoints/generator.ckpt')

# Save training generator samples
with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)