## minimal vosk mic input example
## is a stipped out version of this:
## ## https://github.com/alphacep/vosk-api/blob/master/python/example/test_microphone.py
## which has an Apache 2.0 license
## https://github.com/alphacep/vosk-api/blob/master/COPYING

# import argparse
import os
import queue
from vosk import Model, KaldiRecognizer
import sys
import json
import wave
import sounddevice as sd

def callback(indata, frames, time, status):
    """audio callback function ."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))            


def loadAssistant(model, audio_file):
    try:
        with wave.open(audio_file) as wf:
            assert wf.getnchannels() == 1, "must be a mono wav"
            assert wf.getsampwidth() == 2, "must be a 16bit wav"
            assert wf.getcomptype() == "NONE", "must be PCM data"

            rec = KaldiRecognizer(model, wf.getframerate())
            text = ""
            while True:
                data = wf.readframes(1000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    jres = json.loads(rec.Result())
                    text = text + " " + jres["text"]
            jres = json.loads(rec.FinalResult())
            text = text + " " + jres["text"]
            return text
    except Exception as e:
        print(e.args)


def menu():
    print("Welcome to your virtual airport assistant")
    print("Please select your preferred language: ")
    print("[1] Load English Language Personal Audio")
    print("[2] Load English Language Coursera Sample")
    print("[3] Load Italian Language Coursera Sample")
    print("[4] Load Spanish Language Coursera Sample")
    print("[5] Load Microphone Input")
    print("[0] Exit")


def wer(ref, hyp, debug=True):
    r = ref.split()
    h = hyp.split()

    # costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]

    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3
    DEL_PENALTY = 1
    INS_PENALTY = 1
    SUB_PENALTY = 1

    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r) + 1):
        costs[i][0] = DEL_PENALTY * i
        backtrace[i][0] = OP_DEL

    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                costs[i][j] = costs[i - 1][j - 1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i - 1][j - 1] + SUB_PENALTY  # penalty is always 1
                insertionCost = costs[i][j - 1] + INS_PENALTY  # penalty is always 1
                deletionCost = costs[i - 1][j] + DEL_PENALTY  # penalty is always 1

                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL

    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        print("OP\tREF\tHYP")
        lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("OK\t" + r[i] + "\t" + h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("SUB\t" + r[i] + "\t" + h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j -= 1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i -= 1
            if debug:
                lines.append("DEL\t" + r[i] + "\t" + "****")
    if debug:
        lines = reversed(lines)
        for line in lines:
            print(line)
        print("#cor " + str(numCor))
        print("#sub " + str(numSub))
        print("#del " + str(numDel))
        print("#ins " + str(numIns))
    # return (numSub + numDel + numIns) / (float) (len(r))
    wer_result = round((numSub + numDel + numIns) / (float)(len(r)), 3)
    return {'WER': wer_result, 'numCor': numCor, 'numSub': numSub, 'numIns': numIns, 'numDel': numDel,
            "numCount": len(r)}

# Model Declaration
english_model_dir = "english/small-en-us-0.15"
italian_model_dir = "italian/small-it-0.22"
spanish_model_dir = "spanish/small-es-0.42"

#Audio input device declaration
device_info = sd.query_devices(0, 'input')
samplerate = int(device_info['default_samplerate'])
q = queue.Queue()

"""
Test File Initialisation
------------------------

Total Sample:

- English = 5
- Italian = 5
- Spanish = 5

------------------------ 

"""
# SELF RECORDINGS (CLEAN)
english_clean_reference = "this is my very first short sentence for the demostration of my system and it will be a sentence of gibberrish"
english_audio_file_clean = "english/Voice/english_audio_eval_1.wav"

# SELF RECORDINGS (NOISY)
english_noisy_reference = "alright this is my short sentence number two where i will be demostrating with background noises such as rain and youtube videos"
english_audio_file_noisy = "english/Voice/english_audio_eval_2.wav"

# ENGLISH TEST FILES
english_list = [["where is the check in desk", "english/Voice/checkin.wav"],
                ["i have lost my parents", "english/Voice/parents.wav"],
                ["please i have lost my suitcase", "english/Voice/suitcase.wav"],
                ["what time is my plane", "english/Voice/what_time.wav"],
                ["where are the restuarants and shops", "english/Voice/where.wav"]]

# ITALIAN TEST FILES
italian_list = [["dove è il pancone", "italian/Voice/checkin_it.wav"],
                ["ho perso i miei genitori", "italian/Voice/parents_it.wav"],
                ["per favore. Ho perso la mia", "italian/Voice/suitcase_it.wav"],
                ["a che ora e’ il mio aereo", "italian/Voice/what_time_it.wav"],
                ["dove sono i ristoranti e i negozi", "italian/Voice/where_it.wav"]]

# SPANISH TEST FILES
spanish_list = [["dónde están los mostrador", "spanish/Voice/checkin_es.wav"],
                ["he perdido a mis padres", "spanish/Voice/parents_es.wav"],
                ["por favor he perdido mi maleta", "spanish/Voice/suitcase_es.wav"],
                ["a qué hora es mi avión", "spanish/Voice/what_time_es.wav"],
                ["dónde están los restaurantes y las tiendas", "spanish/Voice/where_es.wav"]]

#Assertion
assert os.path.exists(
    english_model_dir), "Please download a model for your language from https://alphacephei.com/vosk/models"
assert os.path.exists(
    italian_model_dir), "Please download a model for your language from https://alphacephei.com/vosk/models"
assert os.path.exists(
    spanish_model_dir), "Please download a model for your language from https://alphacephei.com/vosk/models"

english_model = Model(english_model_dir)
italian_model = Model(italian_model_dir)
spanish_model = Model(spanish_model_dir)

flag = True

while flag:
    menu()
    option = int(input("Your selection is: "))

    if option == 1:
        print("English Language Personal Test Mode Selected!")
        processed_node_1 = loadAssistant(english_model, english_audio_file_clean)
        processed_node_2 = loadAssistant(english_model, english_audio_file_noisy)
        percentage_1 = wer(english_noisy_reference, processed_node_1)
        percentage_2 = wer(english_noisy_reference, processed_node_2)
        print(percentage_1, "\n", percentage_2)

    elif option == 2:
        print("English Language Coursera Sample Files Selected!")
        for item in english_list:
            processed_node = loadAssistant(english_model, item[1])
            print(processed_node)
            percentage = wer(item[0], processed_node)
            print(percentage)
            print("\n")
        flag = False

    elif option == 3:
        print("Italian Language Coursera Sample Files Selected!")
        for item in italian_list:
            processed_node = loadAssistant(italian_model, item[1])
            print(processed_node)
            percentage = wer(item[0], processed_node)
            print(percentage)
            print("\n")
        flag = False

    elif option == 4:
        print("Spanish Language Coursera Sample Files Selected!")
        for item in spanish_list:
            processed_node = loadAssistant(spanish_model, item[1])
            print(processed_node)
            percentage = wer(item[0], processed_node)
            print(percentage)
            print("\n")
        flag = False

    elif option == 5:
        print("Select Language Support:")
        print("[0] English")
        print("[1] Italian")
        print("[2] Spanish")
        language_support = int(input("Your selection is: "))
        if language_support == 1:
            with sd.RawInputStream(samplerate=samplerate, 
                blocksize = 8000, 
                dtype='int16',
                channels=1, callback=callback):
                rec = KaldiRecognizer(english_model, samplerate)
                while True:
                    data = q.get()
                    if rec.AcceptWaveform(data):
                        res = json.loads(rec.Result())
                        print(res["text"])
        
        elif language_support == 2:
            with sd.RawInputStream(samplerate=samplerate, 
                blocksize = 8000, 
                dtype='int16',
                channels=1, callback=callback):
                rec = KaldiRecognizer(italian_model, samplerate)
                while True:
                    data = q.get()
                    if rec.AcceptWaveform(data):
                        res = json.loads(rec.Result())
                        print(res["text"])
        
        elif language_support == 3:
            with sd.RawInputStream(samplerate=samplerate, 
                blocksize = 8000, 
                dtype='int16',
                channels=1, callback=callback):
                rec = KaldiRecognizer(spanish_model, samplerate)
                while True:
                    data = q.get()
                    if rec.AcceptWaveform(data):
                        res = json.loads(rec.Result())
                        print(res["text"])

    elif option == 0:
        print("We'll see you again! Good Day!")
        flag = False

    else:
        print("Invalid option, please choose an appropriate number")
        option = int(input("Your selection is: "))
