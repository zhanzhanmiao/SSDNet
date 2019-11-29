from datasets import dataset_factory
import tensorflow as tf
from preprocessing import preprocessing_factory
slim = tf.contrib.slim

def main():
    dataset_dir = 'E:/DeepLearning/SSD-tensorflow-master/datasets/train/'
    dataset_split_name = 'train'
    dataset_name = 'pascalvoc_2012'
    dataset = dataset_factory.get_dataset(dataset_name, dataset_split_name, dataset_dir)

    batch_size = 1
    with tf.device(deploy_config.inputs_device()):
        with tf.name_scope(dataset_name + '_data_provider'):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=64,
                common_queue_capacity=20 * batch_size,
                common_queue_min=10 * batch_size,
                shuffle=True)
        # Get for SSD network: image, labels, bboxes.
        [image, shape, glabels, gbboxes] = provider.get(['image', 'shape',
                                                         'object/label',
                                                         'object/bbox'])
        # Pre-processing image, labels and bboxes.
        image, glabels, gbboxes = \
            image_preprocessing_fn(image, glabels, gbboxes,
                                   out_shape=ssd_shape,
                                   data_format=DATA_FORMAT)
        # Encode groundtruth labels and bboxes.
        gclasses, glocalisations, gscores = \
            ssd_net.bboxes_encode(glabels, gbboxes, ssd_anchors)
        batch_shape = [1] + [len(ssd_anchors)] * 3

        # Training batches and queue.
        r = tf.train.batch(
            tf_utils.reshape_list([image, gclasses, glocalisations, gscores]),
            batch_size=batch_size,
            num_threads=1,
            capacity=5 * batch_size)
        b_image, b_gclasses, b_glocalisations, b_gscores = \
            tf_utils.reshape_list(r, batch_shape)

        # Intermediate queueing: unique batch computation pipeline for all
        # GPUs running the training.
        batch_queue = slim.prefetch_queue.prefetch_queue(
            tf_utils.reshape_list([b_image, b_gclasses, b_glocalisations, b_gscores]),
            capacity=2 * deploy_config.num_clones)