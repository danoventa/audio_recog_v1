import os
import numpy
from scipy.io import wavfile
import subprocess
import tempfile
import tensorflow as tf
import traceback

import aiy.audio  # noqa
import vggish_input as vin


RECORD_DURATION_SECONDS = 3
AIY_PROJECTS_DIR = os.path.dirname(os.path.dirname(__file__))


tf_model_path = 'safesense/tensorflow/yt8m'

g_sess = tf.Session()

def make_spectrogram():
    """record a few secs of audio with the mic, convert to spectrogram"""
    temp_file, temp_path = tempfile.mkstemp(suffix='.wav')
    os.close(temp_file)



    # load in our ML model
    with tf.Graph().as_default():
        sess = tf.Session()
        print(sess)
        checkpoint_path = tf_model_path + '/youtube_model.ckpt'
        meta_graph_location = checkpoint_path + '.meta'

        print(meta_graph_location)

        saver = tf.train.import_meta_graph(
            meta_graph_location, clear_devices=True, import_scope='m2'
        )
        print('Saver init')

        saver.restore(sess, checkpoint_path)

        print('saver restpre')

        sess.run(
            set_up_init_ops(tf.get_collection_ref(tf.GraphKeys.LOCAL_VARIABLES))
        )
        print(sess)
        g_sess = sess
        print(g_sess)

    print(g_sess)
    try:
        # TODO: eventually, we want this to continuously run
        input("Press enter to record " + str(RECORD_DURATION_SECONDS) + "seconds of audio, and convert to spectrogram")
        aiy.audio.record_to_wave(temp_path, RECORD_DURATION_SECONDS)
        sr, data = wavfile.read(temp_path)

        # convert to our model input with VGGish
        samples = data / 32768.0  # Convert to [-1.0, +1.0]
        examples_batch = vin.waveform_to_examples(samples, sr)

        # feed our data into our ML model
        input_tensor = g_sess.graph.get_collection("input_batch_raw")[0]
        num_frames_tensor = g_sess.graph.get_collection("num_frames")[0]
        predictions_tensor = g_sess.graph.get_collection("predictions")[0]

        predictions, = g_sess.run(
            [predictions_tensor],
            feed_dict={
                input_tensor: data,
                num_frames_tensor: num_frames_tensor
            })

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
