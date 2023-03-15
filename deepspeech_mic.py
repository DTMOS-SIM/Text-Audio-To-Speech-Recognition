 # adapted from https://github.com/mozilla/DeepSpeech/blob/master/native_client/python/client.py
# which has a Mozzila Public License:
# https://github.com/mozilla/DeepSpeech/blob/master/LICENSE

from deepspeech import Model, version
import librosa as lr
import numpy as np
import sounddevice as sd
import os


## set up the model
scorer_english = "english/deepspeech-0.9.3-models.scorer"
model_english = "english/deepspeech-0.9.3-models.pbmm"

scorer_italian = "italian/kenlm_it.scorer"
model_italian = "italian/output_graph_it.pbmm"

scorer_spanish = "spanish/kenlm_es.scorer"
model_spanis = "spanish/output_graph_es.pbmm"

assert os.path.exists(scorer_english), "You need to download a scroere  from the deepspeech release page: https://github.com/mozilla/DeepSpeech/releases"
assert os.path.exists(model_english), "You need to download a  model from the deepspeech release page: https://github.com/mozilla/DeepSpeech/releases"
assert os.path.exists(scorer_italian), "You need to download a scroere  from the deepspeech release page: https://github.com/mozilla/DeepSpeech/releases"
assert os.path.exists(model_italian), "You need to download a  model from the deepspeech release page: https://github.com/mozilla/DeepSpeech/releases"
assert os.path.exists(scorer_spanish), "You need to download a scroere  from the deepspeech release page: https://github.com/mozilla/DeepSpeech/releases"
assert os.path.exists(model_spanis), "You need to download a  model from the deepspeech release page: https://github.com/mozilla/DeepSpeech/releases"

ds = Model(model_english)
ds.enableExternalScorer(scorer_english)
desired_sample_rate = ds.sampleRate()

## set up the sound device
device_info = sd.query_devices(0, 'input')
samplerate = int(device_info['default_samplerate'])
#q = queue.Queue()
audio_buffer = []
state = {"quiet_seen":0, "quiet_want":2, "trigger_stt": False}

threshold = 32768 / 50

def callback(indata, frames, time, status):
    """audio callback function ."""
    if status:
        print(status, file=sys.stderr)

    data = indata.reshape(1, len(indata))[0]
    audio_buffer.extend(data)

    avg = np.mean(np.abs(data))
    if avg < threshold:
        state["quiet_seen"] = state["quiet_seen"] + 1
        if state["quiet_seen"] > state["quiet_want"]:
            # trigget reocog
            state["quiet_seen"] = 0
            state["trigger_stt"] = True
    else:
        state["quiet_seen"] = 0

# fire up the audio device
with sd.InputStream(samplerate=desired_sample_rate, 
                     blocksize = 8000, 
                     dtype='int16',
                     channels=1, callback=callback):
    while True:
        if state["trigger_stt"]:
            audio = audio_buffer
            print("received audio", len(audio), "running neural analysis :) ")
            res = ds.stt(audio)
            print("result: "+res)
            state["trigger_stt"] = False
            audio_buffer.clear()
            
