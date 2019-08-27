import vae_tools# Always import first to define if keras or tf.kreas should be used
import vae_tools.sanity
import vae_tools.viz
import vae_tools.callbacks
from vae_tools.mmvae import MmVae, ReconstructionLoss
import tensorflow as tf
vae_tools.sanity.check()
import numpy as np
import os

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
    print("Test")
    return voxel.astype(np.float32)

def main():

    path = "data/dataset/qube/train"
    batch_size = 40

    iterator, length = create_iterator(batch_size, path)

    next_element = iterator.get_next()

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(10000):
            val = sess.run(next_element)
            # print(val)
            if (i + 1) % (length // batch_size) == 0 and i > 0:
                print("reinitialized")
                sess.run(iterator.initializer)



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
        lambda item: tuple(tf.py_func(load_samples, [item], [tf.float32])))
    dx = dx.batch(batch_size)
    iterator = dx.make_initializable_iterator()
    return iterator, length


if __name__ == '__main__':
    main()