import os
from typing import Optional

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.models import load_model


class SegmentModel:
    def __init__(self,
                 model_path: str,
                 custom_objects: dict,
                 input_shape: tuple = (256, 256, 3),
                 device: Optional[str] = None):
        self.model_path = model_path
        self.custom_objects = custom_objects
        self.input_shape = input_shape
        self.device = device if device is not None else "/cpu:0"

        assert os.path.isfile(self.model_path)

        self.model = None

        # loading the model
        self.load_model()

    @staticmethod
    def get_swish():
        def swish(x):
            return K.tf.nn.swish(x)

        return swish

    @staticmethod
    def get_dropout():
        class FixedDropout(keras.layers.Dropout):
            def _get_noise_shape(self, inputs):
                if self.noise_shape is None:
                    return self.noise_shape

                symbolic_shape = K.shape(inputs)
                noise_shape = [symbolic_shape[axis] if shape is None else shape
                               for axis, shape in enumerate(self.noise_shape)]
                return tuple(noise_shape)

        return FixedDropout

    def load_model(self, use_compile: bool = False):
        with tf.device(self.device):
            self.model = load_model(self.model_path,
                                    custom_objects={
                                        "swish": self.get_swish(),
                                        "FixedDropout": self.get_dropout(),
                                    },
                                    compile=use_compile)

    def pred_to_mask(self, image: np.ndarray, use_hair_segment: bool = False):
        # 0 : bg, 1 : hair, 2 : face
        masking = np.zeros(self.input_shape, dtype=np.uint8)
        sparse_mask_image = np.argmax(image, axis=-1).squeeze()

        v: int = 1 if use_hair_segment else 2
        for i in range(self.input_shape[0]):
            for j in range(self.input_shape[1]):
                if sparse_mask_image[i, j] == v:
                    masking[i, j] = [255, 255, 255]
        return masking

    def inference(self, img: np.ndarray):
        assert self.input_shape == img.shape

        img = np.expand_dims(img, axis=0)
        img = (img / 127.5) - 1.  # [-1, 1]

        segment_img = self.model.predict(img)[0]
        return segment_img

    def get_mask(self, img: np.ndarray, get_hair: bool = True):
        x = self.inference(img)
        x = self.pred_to_mask(x, use_hair=get_hair)
        return x
