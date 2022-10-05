import time

import cv2
import numpy as np
import tensorflow as tf
from skimage.segmentation import mark_boundaries

from utils.helper_functions import sigmoid
from utils.logger import debug, timing, debug_mem, info
from tensorflow.keras.models import Model


class LIME():
    def __init__(self, model_path: str):
        super().__init__()
        self.model = tf.keras.models.load_model(model_path)

    def explain(self, image_array: np.ndarray, GLOBAL_SEED=0) -> list:
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

    def normalize_image(self, image_array: np.ndarray, ):
        grads_norm = image_array[:, :, 0] + image_array[:, :, 1] + image_array[:, :, 2]
        grads_norm = (grads_norm - tf.reduce_min(grads_norm)) / (tf.reduce_max(grads_norm) - tf.reduce_min(grads_norm))
        return grads_norm


class GradCam():

    def __init__(self, model_path: str):
        super().__init__()
        self.model = tf.keras.models.load_model(model_path)

    def superimpose(self, img_bgr, cam, thresh, emphasize=False):

        '''
        Superimposes a grad-cam heatmap onto an image for model interpretation and visualization.


        Args:
          image: (img_width x img_height x 3) numpy array
          grad-cam heatmap: (img_width x img_width) numpy array
          threshold: float
          emphasize: boolean

        Returns
          uint8 numpy array with shape (img_height, img_width, 3)

        '''
        heatmap = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
        if emphasize:
            heatmap = sigmoid(heatmap, 50, thresh, 1)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        hif = .7
        superimposed_img = heatmap * hif + img_bgr
        superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)  # scale 0 to 255
        superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

        return superimposed_img_rgb

    def explain(self, model, img_array, layer_name, eps=1e-8):
        '''
         Creates a grad-cam heatmap given a model and a layer name contained with that model


         Args:
           model: tf model
           img_array: (img_width x img_width) numpy array
           layer_name: str


         Returns
           uint8 numpy array with shape (img_height, img_width)

         '''
        gradModel = Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(layer_name).get_output_at(0), model.output])

        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(img_array, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, 0]

        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (img_array.shape[2], img_array.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        # heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap


