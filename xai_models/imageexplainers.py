import time

import numpy as np
import tensorflow as tf
from utils.logger import debug, timing, debug_mem, info

class LIME():
    def __init__(self, model_path: str,):
        super().__init__()
        self.model = tf.keras.models.load_model(model_path)
        # embedding = pd.read_csv(embedding_path)
        # self.embedding = embedding.reset_index().to_dict('list')
        # self.type = TransformerType.transform_and_approximate
        # self.num_of_representative_vectors = num_of_representative_vectors

    def __repr__(self):
        return f"Signal2Vec num_of_representative_vectors: {self.num_of_representative_vectors}"

    # def transform(self, series: np.ndarray, sample_period: int = 6) -> np.ndarray:
    #     discrete_series = self.discretize_in_chunks(series, sample_period)
    #     debug_mem('Time series {} MB', series)
    #     debug_mem('Discrete series {} MB', discrete_series)
    #
    #     vector_representation = self.map_into_vectors(discrete_series)
    #     debug_mem('Sequence of vectors : {} MB', vector_representation)
    #
    #     return np.array(vector_representation)
    #
    # def approximate(self, data_in_batches: np.ndarray, window: int = 1, should_fit: bool = True) -> list:
    #     # TODO: Window is used only by signal2vec, move it to constructor or extract it as len(segment).
    #     if self.num_of_representative_vectors > 1:
    #         window = int(window / self.num_of_representative_vectors)
    #         data_in_batches = np.reshape(data_in_batches,
    #                                      (len(data_in_batches), window, 300 * self.num_of_representative_vectors))
    #
    #     squeezed_seq = np.sum(data_in_batches, axis=1)
    #     vf = np.vectorize(lambda x: x / window)
    #     squeezed_seq = vf(squeezed_seq)
    #     return squeezed_seq

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


