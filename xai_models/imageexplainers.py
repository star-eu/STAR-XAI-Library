import time

import numpy as np
import tensorflow as tf
from skimage.segmentation import mark_boundaries

from utils.logger import debug, timing, debug_mem, info

class LIME():
    def __init__(self, model_path: str):
        super().__init__()
        self.model = tf.keras.models.load_model(model_path)

    def explain(self, image_array: np.ndarray, window: int = 1, should_fit: bool = True, GLOBAL_SEED=0) -> list:
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

    # def discretize(self, data):
    #     debug('Length of data {}'.format(len(data)))
    #     start_time = time.time()
    #
    #     pred = self.clf.predict(data.reshape(-1, 1))
    #
    #     timing('clf.predict: {}'.format(round(time.time() - start_time, 2)))
    #     debug('Length of predicted sequence {}'.format(len(pred)))
    #     debug('Type of discrete sequence {}'.format(type(pred)))
    #
    #     return pred

    def map_into_vectors(self, sequence):
        start_time = time.time()
        sequence_of_vectors = [self.embedding[str(i)] for i in sequence]
        timing('Appending vectors to list : {}'.format(round(time.time() - start_time, 2)))
        return sequence_of_vectors


