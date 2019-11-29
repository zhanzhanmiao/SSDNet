import math
from collections import namedtuple
import tensorflow as tf
import Config
slim = tf.contrib.slim

SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])
class SSDNet(object):
    def __init__(self):
        return

    default_params = SSDParams(

    )

    def BaseNet(self, input):

        return