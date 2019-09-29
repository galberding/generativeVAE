import tensorflow as tf
import numpy as np
from scipy.stats import norm
# Set the seed for reproducible results
from mpl_toolkits.mplot3d import Axes3D

import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.backend import binary_crossentropy
import os
import io
from tensorboardX import SummaryWriter


# from StringIO import StringIO

def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


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
    dx = dx.batch(batch_size).shuffle(buffer_size=60)

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
def kl_loss(mean1, mean2, log_var1, log_var2):
    '''KL-divergence between two Gaussian'''
    return - .5 * (1 + log_var1 - log_var2 - ((tf.exp(log_var1) + tf.square(mean1 - mean2)) / tf.exp(log_var2)))


def kl_loss_n(mean, log_var):
    '''KL-divergence between an abitrary Gaussian and the normal distribution'''
    return kl_loss(mean, 0., log_var, 0.)


def vis_voxel_reconstruction(writer, reconstruction, val, iteration):
    vis = 2
    fig, axes = plt.subplots(vis, 2, subplot_kw=dict(projection='3d'), figsize=(25, 25))
    # print(type(reconstruction))
    # print(reconstruction)
    # if not os.path.exists(path):
    #     os.makedirs(path)
    for i in range(vis):
        voxel_rec = reconstruction
        label = val[0][i].T[0]
        # print(label.shape)
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
        # print(label.shape)
        axes[i, 0].voxels(label, edgecolors='k', )
        axes[i, 1].voxels(pred, edgecolors='k')


    # plt.savefig(os.path.join(path,str(iteration) + ".png"))
    writer.add_figure("Reconstruction", fig, global_step=iteration)
    plt.close(fig)


# (5,30,30,30,8)
def main():
    # Variables
    batch_size = 2
    OUT_FILE = "out/train_2"
    MODEL_PATH = os.path.join(OUT_FILE, "model.ckpt")
    CONFIG_PATH = os.path.join(OUT_FILE, "config.npz")

    if not os.path.exists(OUT_FILE):
        os.makedirs(OUT_FILE)
    # supress tensorflow Warnings
    tf.logging.set_verbosity(tf.logging.ERROR)

    train_path = "data/dataset/qube/train"
    test_path = "data/dataset/qube/test"

    # load dataset
    iterator, length = create_iterator(batch_size, train_path)
    next_element = iterator.get_next()

    test_iterator, test_length = create_iterator(batch_size, test_path)
    test_next_element = test_iterator.get_next()

    # graph
    with tf.variable_scope('Image-Input'):
        img_input = tf.placeholder(tf.float32, [None, 32, 32, 32, 1])
        training = tf.placeholder_with_default(True, shape=())
        print(img_input.get_shape())

    with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE):
        # encoding = batch_norm(img_input, training=training)
        encoding = conv(img_input, 8, 3, 1, padding="valid")
        # (5, 30,30,30,8)
        # encoding = batch_norm(encoding, training=training)
        print(encoding.get_shape())
        encoding = conv(encoding, 16, 3, 2)
        # encoding = batch_norm(encoding, training=training)
        print(encoding.get_shape())
        encoding = conv(encoding, 32, 3, 1, padding="valid")
        # encoding = batch_norm(encoding, training=training)
        print(encoding.get_shape())
        encoding = conv(encoding, 64, 3, 2)
        # encoding = batch_norm(encoding, training=training)
        print(encoding.get_shape())
        flatten = tf.layers.flatten(encoding)
        flatten = tf.layers.dense(flatten, 512, activation=tf.nn.relu)
        # flatten = batch_norm(flatten, training=training, axis=1)
        # print(flatten.get_shape())
        mean = tf.layers.dense(flatten, 128)
        # mean = batch_norm(mean, training=training, axis=1)
        std = tf.layers.dense(flatten, 128)
        # std = batch_norm(std, training=training, axis=1)

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

    # with tf.variable_scope("Create_Reconstruction"):
    #     reconstruction = decode
    #     figure = vis_voxel_reconstruction(reconstruction, img_input, 0)
    #     rec_sum = tf.summary.image(
    #         "Training data", plot_to_image(figure))

    with tf.variable_scope("Losses", reuse=tf.AUTO_REUSE):
        # print(kl_loss_n(mean, std).get_shape())
        kl_divergence = tf.reduce_sum(kl_loss_n(mean, std))
        # print(kl_divergence.get_shape())
        rec_loss = tf.reduce_sum(binary_crossentropy(img_input, decode))
        # print(rec_loss)
        loss = kl_divergence + rec_loss
        train_sum = tf.summary.scalar("loss", loss)
        # print(loss.get_shape())

    with tf.variable_scope("Metrics"):
        threshold_mask = tf.greater(decode, 0.5)
        binarize =  tf.cast(threshold_mask, dtype=tf.int32)
        img_binary = tf.cast(img_input, dtype=tf.int32)
        labels = (img_binary)
        pred =  (binarize)

        # area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
        # area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)
        #
        # iou = (area_intersect / area_union)

        area_union = tf.bitwise.bitwise_or(labels, pred)
        area_intersection = tf.bitwise.bitwise_and(labels, pred)

        iou = tf.reduce_sum(area_intersection) / tf.reduce_sum(area_union)
        iou_sum = tf.summary.scalar("IoU", iou)
        # iou = tf.metrics.mean_iou(labels, labels,2)

    with tf.variable_scope('Optimize', reuse=tf.AUTO_REUSE):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train = tf.train.AdamOptimizer(0.0001).minimize(loss)

    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    #     train_op = optimizer.minimize(loss)
    train_sess = tf.Session()
    train_sess.run(tf.global_variables_initializer())
    train_sess.run(tf.local_variables_initializer())

    # merged_train = tf.summary.merge([train_sum, iou_sum])
    # merged_rec = tf.summary.merge([rec_sum])
    train_writer = SummaryWriter(os.path.join(OUT_FILE, "train"))
    test_writer = SummaryWriter(os.path.join(OUT_FILE,"test"))
    # test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')
    # tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    try:
        saver.restore(train_sess, MODEL_PATH)
        epo_iter = np.load(CONFIG_PATH)["state"]
        epoch = epo_iter[0]
        i = epo_iter[1]
    except Exception:
        print("Could not restore model.")
        i = 0
        epoch = 0
        # np.savez(OUT)

    # dataloop
    # epoch = 0
    # i = 0
    with tf.Session() as sess:
        sess.run([iterator.initializer, test_iterator.initializer])
        # writer  = tf.summary.FileWriter("out/logs", sess.graph)
        # initialise the variables
        # for i in range(0, 50001):
        while True:
            train_val, test_val = sess.run([next_element, test_next_element])

            # print(train_val[0].shape)
            # print(test_val[0].shape)
            # val = (tf.convert_to_tensor(val[0]))
            # print(len(val))
            # print(val[0][0].T[0].shape)
            loss_, reconstruction, _ , iou_, kl_divergence_, rec_loss_ = train_sess.run(
                [loss, decode, train, iou, kl_divergence, rec_loss], feed_dict={img_input: train_val[0]})

            test_loss_, test_reconstruction, test_iou_out, test_kl_divergence, test_rec_loss = train_sess.run(
                [loss, decode, iou, kl_divergence, rec_loss], feed_dict={img_input: test_val[0]})

            print("Epoch:", epoch, "Iteration:", i, "train_loss:", loss_, "validation loss:", test_loss_, "IoU train:", iou_, "IoU test:", test_iou_out)
            # train_writer.add_summary(summary, i)
            # test_writer.add_summary(test_summary, i)
            train_writer.add_scalar("Loss:", loss_, global_step=i)
            train_writer.add_scalar("Reconstrruction Loss", rec_loss_, global_step=i)
            train_writer.add_scalar("Kl Divergence", kl_divergence_, global_step=i)
            train_writer.add_scalar("IoU", iou_, global_step=i)

            test_writer.add_scalar("Loss:", test_loss_, global_step=i)
            test_writer.add_scalar("Reconstrruction Loss", test_rec_loss, global_step=i)
            test_writer.add_scalar("Kl Divergence", test_kl_divergence, global_step=i)
            test_writer.add_scalar("IoU", test_iou_out, global_step=i)

            # print(iou_out)
            # print(test_iou_out)
            # break
            # break

            if (i % 1) == 0:
                saver.save(train_sess, MODEL_PATH)
                np.savez(CONFIG_PATH, state=np.array([epoch, i]))
                print("Model saved!")
            if (i % 1000) == 0:
                print("Save images")
                vis_voxel_reconstruction(train_writer,reconstruction, train_val, i,)
                vis_voxel_reconstruction(test_writer,reconstruction, test_val, i)

            if (i + 1) % (length // batch_size) == 0 and i > 0:
                print("reinitialized")
                epoch += 1
                sess.run(iterator.initializer)
            if (i + 1) % (test_length // batch_size) == 0 and i > 0:
                print("test reinitialized")
                epoch += 1
                sess.run(test_iterator.initializer)
            i += 1



if __name__ == '__main__':
    main()
    # _vae_tools()
