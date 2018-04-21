import os
import numpy as np
from scipy.io import wavfile
import subprocess
import tempfile
import tensorflow as tf
import traceback
import csv

from audio_recog_v1.utils import vggish, youtube8m
import audio_recog_v1.aiy.audio  as audio # noqa

RECORD_DURATION_SECONDS = 3
AIY_PROJECTS_DIR = os.path.dirname(os.path.dirname(__file__))

YOUTUBE_CHECKPOINT_FILE = 'models/youtube_model.ckpt'
VGGISH_MODEL='models/vggish_model.ckpt'

SAMPLE_RATE = 16000
MAX_FRAMES = 300

def make_spectrogram():
    """record a few secs of audio with the mic, convert to spectrogram"""
    temp_file, temp_path = tempfile.mkstemp(suffix='.wav')
    os.close(temp_file)

    # load in our ML model
    try:
        # TODO: eventually, we want this to continuously run
        input("Press enter to record " + str(RECORD_DURATION_SECONDS) + " seconds of audio, and convert to spectrogram")
        audio.record_to_wave(temp_path, RECORD_DURATION_SECONDS)
        sr, data = wavfile.read(temp_path)


# init everything

        _vggish_sess = None
        _youtube_sess = None
        _class_map = {}

        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session()
            vggish.model.define_vggish_slim(training=False)
            vggish.model.load_vggish_slim_checkpoint(sess, VGGISH_MODEL)
            print(_vggish_sess)
            _vggish_sess = sess

        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session()
            youtube8m.model.load_model(sess, YOUTUBE_CHECKPOINT_FILE)
            print(_youtube_sess)
            _youtube_sess = sess


        with open('models/class_labels_indices.csv') as f:
            next(f)  # skip header
            reader = csv.reader(f)
            for row in reader:
                _class_map[int(row[0])] = row[2]

# get predictions
        samples = data / 32768.0  # Convert to [-1.0, +1.0]
        examples_batch = vggish.input.waveform_to_examples(samples, sr)

        # get features
        sess =_vggish_sess
        features_tensor = sess.graph.get_tensor_by_name(
            'vggish/input_features:0')
        embedding_tensor = sess.graph.get_tensor_by_name(
            'vggish/embedding:0')

        [embedding_batch] = sess.run(
            [embedding_tensor],
            feed_dict={features_tensor: examples_batch}
        )

        pca_params = np.load('models/vggish_pca_params.npz')
        pca_matrix = pca_params['pca_eigen_vectors']
        pca_means = pca_params['pca_means']

        postprocessed_batch = np.dot(
            pca_matrix, (embedding_batch.T - pca_means)
        ).T

        features = postprocessed_batch

        # process features
        sess = _youtube_sess
        num_frames = np.minimum(features.shape[0], MAX_FRAMES)
        data = youtube8m.input.resize(features, 0, MAX_FRAMES)
        data = np.expand_dims(data, 0)
        num_frames = np.expand_dims(num_frames, 0)

        input_tensor = sess.graph.get_collection("input_batch_raw")[0]
        num_frames_tensor = sess.graph.get_collection("num_frames")[0]
        predictions_tensor = sess.graph.get_collection("predictions")[0]

        predictions_val, = sess.run(
            [predictions_tensor],
            feed_dict={
                input_tensor: data,
                num_frames_tensor: num_frames
            })

        predictions = predictions_val

        # filter predictions
        count = 20
        hit = 0.1

        top_indices = np.argpartition(predictions[0], -count)[-count:]
        line = ((_class_map[i], float(predictions[0][i])) for
                i in top_indices if predictions[0][i] > hit)
        predictions =  sorted(line, key=lambda p: -p[1])

        ################
        print(predictions)





            # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

            # sess.run(
            #     set_up_init_ops(tf.get_collection_ref(tf.GraphKeys.LOCAL_VARIABLES))
            # )
            # feed our data into our ML model
            # input_tensor = sess.graph.get_collection("input_batch_raw")[0]
            # num_frames_tensor = sess.graph.get_collection("num_frames")[0]
            # predictions_tensor = sess.graph.get_collection("predictions")[0]
            #

            # with tf.device('/cpu:0'):
            #     predictions = sess.run(
            #         [predictions_tensor],
            #         feed_dict={
            #             input_tensor: data,
            #             num_frames_tensor: num_frames_tensor
            #         })


        # TODO: take ML model output and write to kinesis
        print(predictions) # for now, just print them

    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass

# Workaround for num_epochs issue.
def set_up_init_ops(variables):
    init_op_list = []
    for variable in list(variables):
        if "train_input" in variable.name:
            init_op_list.append(tf.assign(variable, 1))
            variables.remove(variable)
    init_op_list.append(tf.variables_initializer(variables))
    return init_op_list


def enable_audio_driver():
    print("Enabling audio driver for VoiceKit.")
    configure_driver = os.path.join(AIY_PROJECTS_DIR, 'scripts', 'configure-driver.sh')
    subprocess.check_call(['sudo', configure_driver])


def main():
    enable_audio_driver()
    make_spectrogram()


if __name__ == '__main__':
    try:
        main()
        input('Press Enter to close...')
    except Exception:  # pylint: disable=W0703
        traceback.print_exc()
        input('Press Enter to close...')
