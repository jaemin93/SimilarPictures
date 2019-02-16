# -*- coding: utf-8 -*-
"""
MobileNet V2 클래스 입니다. Tensorflow Hub에서 다운받은 Pre-train 모델을 사용합니다.
"""
import tensorflow as tf
import tensorflow_hub as hub

slim = tf.contrib.slim

def get_encoded_image(image_path):
    encoded_image = tf.gfile.FastGFile(image_path, 'rb').read()
    return encoded_image

    def _get_init_fn(checkpoint_path):
        """Returns a function run by the chief worker to warm-start the training.

        Note that the init_fn is only run when initializing the model during the very
        first global step.

        Returns:
          An init function run by the supervisor.
        """
        if checkpoint_path is None:
            return None

        # Warn the user if a checkpoint exists in the train_dir. Then we'll be
        # ignoring the checkpoint anyway.
        if tf.train.latest_checkpoint(checkpoint_path):
            tf.logging.info(
                'Ignoring --checkpoint_path because a checkpoint already exists in %s'
                % FLAGS.train_dir)
            return None

        exclusions = []
        if FLAGS.checkpoint_exclude_scopes:
            exclusions = [scope.strip()
                          for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

        # TODO(sguada) variables.filter_variables()
        variables_to_restore = []
        for var in slim.get_model_variables():
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    break
            else:
                variables_to_restore.append(var)

        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path

        tf.logging.info('Fine-tuning from %s' % checkpoint_path)

        return slim.assign_from_checkpoint_fn(
            checkpoint_path,
            variables_to_restore,
            ignore_missing_vars=FLAGS.ignore_missing_vars)

class Inception_resnet_v2:
    def __init__(self):
        self.module_url = "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1"
        self.filename = tf.placeholder(tf.string, shape=[None], name='filename')
        self.encoded_images = tf.placeholder(tf.string, shape=[None], name='encoded_images')
        self.features = self.build_model()
        self.output_size = 1536

    def build_model(self):
        # build mobilenet v2 model using tensorflow hub
        image_module = hub.Module(self.module_url)
        image_size = hub.get_expected_image_size(image_module)

        def _decode_and_resize_image(encoded: tf.Tensor) -> tf.Tensor:
            decoded = tf.image.decode_jpeg(encoded, channels=3)
            decoded = tf.image.convert_image_dtype(decoded, tf.float32)
            return tf.image.resize_images(decoded, image_size)

        decoded_images = tf.map_fn(_decode_and_resize_image, self.encoded_images, tf.float32)  # type: tf.Tensor
        return image_module(decoded_images)



model = Inception_resnet_v2()