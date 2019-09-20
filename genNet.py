import tensorflow as tf
import numpy as np
from scipy.stats import norm
# Set the seed for reproducible results
from mpl_toolkits.mplot3d import Axes3D

import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.backend import binary_crossentropy
import os


# Data loading
def get_filenames(path):
    sample_paths = []
    for root, dirs, files in os.walk(path):
        # print(root)

        if len(files) == 0:
            continue
        # print(files[0])
        sample_paths.append(os.path.join(root, files[0]))
    length = len(sample_paths)
    return sorted(sample_paths), length


def load_samples(paths):
    item = np.load(paths.decode())
    voxel = item["voxel"]
    # print("Test")
    return voxel.astype(np.float32)[np.newaxis].T


def create_iterator(batch_size, path):
    '''
    Generate iterator for given dataset path.
    https://stackoverflow.com/questions/48889482/feeding-npy-numpy-files-into-tensorflow-data-pipeline
    :param batch_size:
    :param path:
    :return: iterator, length of dataset
    '''
    filenames, length = get_filenames(path)
    print(length)
    dx = tf.data.Dataset.from_tensor_slices(filenames).map(
        lambda item: tf.py_func(load_samples, [item], [tf.float32]))
    dx = dx.batch(batch_size).shuffle(buffer_size=11630)

    iterator = dx.make_initializable_iterator()
    return iterator, length


# Define layers
def conv(inputs, filters, kernel_size, strides, activation=tf.nn.relu, padding="SAME"):
    return tf.layers.conv3d(inputs, filters, kernel_size, strides=(strides, strides, strides),
                            padding=padding, activation=activation, )


def deconv(inputs, filters, kernel_size, strides, activation=tf.nn.relu, padding='SAME'):
    return tf.layers.conv3d_transpose(inputs, filters, kernel_size, strides=(strides, strides, strides),
                                      padding='SAME', activation=activation)


def sample(mean, std):
    with tf.variable_scope('random_sampling'):
        shape = tf.shape(mean)
        random_sample = tf.random_normal(shape)
        return mean + tf.exp(std * .5) * random_sample

def batch_norm(input, training, axis=-1):
    return tf.layers.batch_normalization(input, training=training, axis=axis)


# Loss functions
def kl_loss_n(mean, log_var):
    '''kl_loss
    Calculates D_KL(N(mean, sigma)||N(0, 1))
    mean: is the mean value
    log_var: is the logarithm of variance (log(sigma^2))
    '''
    # log_var = tf.log(var)
    return - .5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))


def vis_voxel_reconstruction(res, val, iteration):
    vis = 2
    fig, axes = plt.subplots(vis, 2, subplot_kw=dict(projection='3d'), figsize=(25,25))
    for i in range(vis):
        voxel_rec = res[1]
        label = val[0][i].T[0]
        print(label.shape)
        pred = voxel_rec[i].T[0]
        # pred[pred < 0.5] = 0
        # pred[pred >= 0.5] = 1
        # colors = np.empty(spatial_axes + [4], dtype=np.float32)
        alpha = .4
        # colors[0] = [1, 0, 0, alpha]
        # colors[1] = [0, 1, 0, alpha]
        # colors[2] = [0, 0, 1, alpha]
        # colors[3] = [1, 1, 0, alpha]
        # colors[4] = [0, 1, 1, alpha]

        axes[i,0].voxels(label, edgecolors='k', )
        axes[i,1].voxels(pred, edgecolors='k')
    plt.savefig("out/"+str(iteration)+".png")

# (5,30,30,30,8)
def main():
    # Variables
    batch_size = 150

    # supress tensorflow Warnings
    tf.logging.set_verbosity(tf.logging.ERROR)

    path = "data/dataset/qube/train"

    # load dataset
    iterator, length = create_iterator(batch_size, path)
    next_element = iterator.get_next()

    # graph
    with tf.variable_scope('Image-Input'):
        img_input = tf.placeholder(tf.float32, [None, 32, 32, 32, 1])
        training = tf.placeholder_with_default(True, shape=())
        print(img_input.get_shape())

    with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE):
        encoding = batch_norm(img_input, training=training)
        encoding = conv(encoding, 8, 3, 1, padding="valid")
        # (5, 30,30,30,8)
        encoding = batch_norm(encoding, training=training)
        print(encoding.get_shape())
        encoding = conv(encoding, 16, 3, 2)
        encoding = batch_norm(encoding, training=training)
        print(encoding.get_shape())
        encoding = conv(encoding, 32, 3, 1,padding="valid")
        encoding = batch_norm(encoding, training=training)
        print(encoding.get_shape())
        encoding = conv(encoding, 64, 3, 2)
        encoding = batch_norm(encoding, training=training)
        print(encoding.get_shape())
        flatten = tf.layers.flatten(encoding)
        flatten = tf.layers.dense(flatten, 512, activation=tf.nn.relu)
        flatten = batch_norm(flatten, training=training, axis=1)
        # print(flatten.get_shape())
        mean = tf.layers.dense(flatten, 512)
        mean = batch_norm(mean, training=training, axis=1)
        std = tf.layers.dense(flatten, 512)
        std = batch_norm(std, training=training, axis=1)

    with tf.variable_scope("Sample"):
        samples = sample(mean, std)
        # print(samples.get_shape())

    with tf.variable_scope("Decoder", reuse=tf.AUTO_REUSE):
        decode = tf.layers.dense(samples, 512, activation=tf.nn.relu)
        # decode = batch_norm(decode, training=training, axis=1)
        # print(decode.get_shape())
        decode = tf.reshape(decode, [-1, 8, 8, 8, 1])
        # print(decode.get_shape())
        decode = deconv(decode, 64, 3, 1, activation=tf.nn.relu, )
        # decode = batch_norm(decode, training=training)
        print(decode.get_shape())
        decode = deconv(decode, 32, 5, 2, activation=tf.nn.relu)
        # decode = batch_norm(decode, training=training)
        print(decode.get_shape())
        decode = deconv(decode, 16, 3, 1, activation=tf.nn.relu)
        # decode = batch_norm(decode, training=training)
        print(decode.get_shape())
        decode = deconv(decode, 8, 3, 2, activation=tf.nn.relu)
        # decode = batch_norm(decode, training=training)
        # print()
        print(decode.get_shape())
        decode = deconv(decode, 1, 3, 1, activation=tf.nn.sigmoid, padding="valid")
        print(decode.get_shape())

    with tf.variable_scope("Losses", reuse=tf.AUTO_REUSE):
        # print(kl_loss_n(mean, std).get_shape())
        kl_divergence = tf.reduce_sum(kl_loss_n(mean, std))
        # print(kl_divergence.get_shape())
        rec_loss = tf.reduce_sum(binary_crossentropy(img_input, decode))
        # print(rec_loss)
        loss = kl_divergence + rec_loss
        # print(loss.get_shape())

    with tf.variable_scope('2D_Train', reuse=tf.AUTO_REUSE):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train = tf.train.AdamOptimizer(0.0001).minimize(loss)

    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    #     train_op = optimizer.minimize(loss)
    train_sess = tf.Session()
    train_sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    # try:
    #     saver.restore(train_sess, 'model_batch_norm.ckpt')
    # except ValueError:
    #     pass
    # dataloop
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(0, 50001):
            val = sess.run(next_element)
            # val = (tf.convert_to_tensor(val[0]))
            # print(len(val))
            # print(val[0][0].T[0].shape)
            res = train_sess.run([loss,decode, train], feed_dict={img_input: val[0]})

            print("Epoch:", i, "loss:", res[0])

            # break
            # if (i % 100) == 0:
            #     saver.save(train_sess, 'model_batch_norm.ckpt')
            #     print("Model saved!")
            # if (i % 100) == 0:
            #     vis_voxel_reconstruction(res, val, i)
            if (i + 1) % (length // batch_size) == 0 and i > 0:
                print("reinitialized")
                sess.run(iterator.initializer)


if __name__ == '__main__':
    main()
    # _vae_tools()
