import time

import numpy as np
import tensorflow as tf
from skimage.segmentation import mark_boundaries

from utils.logger import debug, timing, debug_mem, info


class LIME():
    def __init__(self, model_path: str):
        super().__init__()
        self.model = tf.keras.models.load_model(model_path)

    def explain(self, image_array: np.ndarray,  GLOBAL_SEED=0) -> list:
        explainer = self.model.LimeImageExplainer(random_state=GLOBAL_SEED)
        explanation = explainer.explain_instance(
            image_array.astype(np.double),
            self.model.predict,
            top_labels=3, hide_color=0, num_samples=50
        )

        im, mask = explanation.get_image_and_mask(
            self.model.predict(
                image_array[None, :, :, :]
            ).argmax(axis=1)[0],
            positive_only=False, hide_rest=False, num_features=50)

        im = mark_boundaries(im, mask)
        return im

    def reconstruct(self, series: np.ndarray) -> list:
        raise Exception('LIME doesn\'t support reconstruct yet.')

    def get_name(self):
        return type(self).__name__

    def get_type(self):
        return self.type

    # def set_type(self, method_type: ImageExplainerType):
    #     if method_type == ImageExplainerType.approximate:
    #         raise Exception('Signal2vec does not support only approximation. The series has to be transformed firstly')
    #     self.type = method_type

    def map_into_vectors(self, sequence):
        start_time = time.time()
        sequence_of_vectors = [self.embedding[str(i)] for i in sequence]
        timing('Appending vectors to list : {}'.format(round(time.time() - start_time, 2)))
        return sequence_of_vectors


class SaliencyMap():
    def __init__(self, model_path: str):
        super().__init__()
        self.model = tf.keras.models.load_model(model_path)

    def explain(self, image_array: np.ndarray):
        img = np.expand_dims(image_array, axis=0)
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        result = self.model(img)
        max_idx = tf.argmax(result, axis=1)

        with tf.GradientTape() as tape:
            tape.watch(img)
            result = self.model(img)
            max_score = result[0, max_idx[0]]
        grads = tape.gradient(max_score, img)

        return grads[0]

    def normalize_image(self, image_array: np.ndarray,):
        grads_norm = image_array[:, :, 0] + image_array[:, :, 1] + image_array[:, :, 2]
        grads_norm = (grads_norm - tf.reduce_min(grads_norm)) / (tf.reduce_max(grads_norm) - tf.reduce_min(grads_norm))
        return grads_norm

