import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.draw
import skimage.io

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Root directory of the project
ROOT_DIR = os.path.abspath("../..")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.crowd import crowd


class SketchSegment:
    def __init__(self, model_dir, data_dir):
        self.name = 'seg'
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.model = None
        self.name_map = crowd.name_map

    def load_model(self):
        config = crowd.BalloonConfig()

        class InferenceConfig(config.__class__):
            # Run detection on one image at a time
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            NUM_CLASSES = 1 + len(self.name_map)  # Background + area + path + people + add geometry later
            BACKBONE = "resnet101"

            IMAGE_RESIZE_MODE = "none"
            IMAGE_MIN_DIM = 512
            IMAGE_MAX_DIM = 512
            DETECTION_MIN_CONFIDENCE = 0.9

        config = InferenceConfig()
        config.display()

        self.model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=self.model_dir)

        # Find last trained weights
        weights_path = self.model.find_last()

        self.model.load_weights(weights_path, by_name=True)
        self.model.keras_model._make_predict_function()

        dataset = crowd.BalloonDataset()
        dataset.load_balloon(self.data_dir, "test")

        # Must call before using the dataset
        dataset.prepare()
        self.class_names = dataset.class_names
        #self.class_ids = dataset.class_ids

    def segment_imgV2(self, image):
        r = self.model.detect([image], verbose=1)[0]
        masks = r['masks']
        class_ids = r['class_ids']
        # scores = r['scores']
        # rois = r['rois']
        N = class_ids.shape[0]

        #labels = [ for class_id in class_ids]
        mask_dict = {}
        for i in range(N):
            label = self.class_names[class_ids[i]]
            mask_1_channel = masks[:, :, i]
            if label not in mask_dict:
                mask_dict[label] = []
            mask_dict[label].append(mask_1_channel.astype(np.uint8).tolist())
        return mask_dict

    def segment_img(self, image):
        r = self.model.detect([image], verbose=1)[0]
        masks = r['masks']
        class_ids = r['class_ids']
        scores = r['scores']
        rois = r['rois']
        N = class_ids.shape[0]
        white_img = np.ones(image.shape) * 255
        results = {}
        for i in range(N):
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = self.class_names[class_id]
            mask_1_channel = masks[:, :, i]
            mask_3_channel = np.zeros_like(white_img)
            for ch in range(3):
                mask_3_channel[:, :, ch] = mask_1_channel
            extracted_img = np.where(mask_3_channel, image, white_img).astype(np.uint8)
            skimage.io.imsave('test' + str(i) + label + '.png', extracted_img)
            if label not in results:
                results[label] = []
            results[label].append((extracted_img, rois[i]))
        return results




