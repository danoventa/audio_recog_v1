import os
import numpy
from scipy.io import wavfile
import subprocess
import tempfile
import tensorflow as tf
import traceback

import aiy.audio  # noqa
from utils import vggish

RECORD_DURATION_SECONDS = 3
AIY_PROJECTS_DIR = os.path.dirname(os.path.dirname(__file__))


tf_model_path = 'safesense/tensorflow/yt8m'

def make_spectrogram():
    """record a few secs of audio with the mic, convert to spectrogram"""
    temp_file, temp_path = tempfile.mkstemp(suffix='.wav')
    os.close(temp_file)

    # load in our ML model
    saver = tf.train.Saver()

    try:
        # TODO: eventually, we want this to continuously run
        input("Press enter to record " + RECORD_DURATION_SECONDS + "seconds of audio, and convert to spectrogram")
        aiy.audio.record_to_wave(temp_path, RECORD_DURATION_SECONDS)
        sr, data = wavfile.read(temp_path)

        # convert to our model input with VGGish
        samples = data / 32768.0  # Convert to [-1.0, +1.0]
        examples_batch = vggish.vggish_input.waveform_to_examples(samples, sample_rate)

        # feed our data into our ML model
        with tf.Session() as sess:
            saver.restore(sess, tf_model_path + '/model.ckpt')
            input_tensor = sess.graph.get_collection("input_batch_raw")[0]
            num_frames_tensor = sess.graph.get_collection("num_frames")[0]
            predictions_tensor = sess.graph.get_collection("predictions")[0]

            predictions, = sess.run(
                [predictions_tensor],
                feed_dict={
                    input_tensor: data,
                    num_frames_tensor: num_frames
                })

        # TODO: take ML model output and write to kinesis
        print(predictions) # for now, just print them

    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass


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
