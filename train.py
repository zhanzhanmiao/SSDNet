from datasets import dataset_factory
import tensorflow as tf
import Config
from preprocessing import preprocessing_factory
from nets import nets_factory
import tf_utils

slim = tf.contrib.slim
GLOBAL_STEP = slim.create_global_step()


MODEL_NAME = Config.model_name
PREPROCESSING_NAME = Config.preprocessing_name
DATASET_DIR = Config.dataset_dir
DATASET_NAME = Config.dataset_name
DATASET_SPLIT_NAME = Config.dataset_split_name

BATCH_SIZE = Config.batch_size
DATA_FORMAT = Config.data_format
NUM_CLASSES = Config.num_classes


def main():
    # 为将要被记录的的东西（日志）设置开始入口
    tf.logging.set_verbosity(tf.logging.DEBUG)

    dataset_dir = DATASET_DIR
    dataset_split_name = DATASET_SPLIT_NAME
    dataset_name = DATASET_NAME
    dataset = dataset_factory.get_dataset(dataset_name, dataset_split_name, dataset_dir)

    # Get the SSD network and its anchors.
    model_name = MODEL_NAME
    ssd_class = nets_factory.get_network(model_name)
    num_classes = NUM_CLASSES
    ssd_params = ssd_class.default_params._replace(num_classes=num_classes)
    ssd_net = ssd_class(ssd_params)
    ssd_shape = ssd_net.params.img_shape
    ssd_anchors = ssd_net.anchors(ssd_shape)

    # Select the preprocessing function.
    preprocessing_name = PREPROCESSING_NAME
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name, is_training=True)

    # 指定运行设备
    # with tf.device():
    with tf.name_scope(dataset_name + '_data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=64,
            common_queue_capacity=20 * BATCH_SIZE,
            common_queue_min=10 * BATCH_SIZE,
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
        batch_size=BATCH_SIZE,
        num_threads=1,
        capacity=5 * BATCH_SIZE)
    b_image, b_gclasses, b_glocalisations, b_gscores = \
        tf_utils.reshape_list(r, batch_shape)

    # Intermediate queueing: unique batch computation pipeline for all
    # GPUs running the training.
    batch_queue = slim.prefetch_queue.prefetch_queue(
        tf_utils.reshape_list([b_image, b_gclasses, b_glocalisations, b_gscores]),
        capacity=2 * deploy_config.num_clones)