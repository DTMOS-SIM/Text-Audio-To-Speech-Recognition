from deepspeech import Model
import noisereduce as nr
import librosa as lr
import numpy as np
import os 

def loadAssistant(model, scorer, audio_file, noise_sample_start, noise_sample_end):
    try:
        #Model Declaration
        ds = Model(model)
        ds.enableExternalScorer(scorer)
        
        #Sample Rate
        desired_sample_rate = ds.sampleRate()
        
        #Librosa load audio
        raw, sample_rate = lr.load(audio_file, sr=desired_sample_rate)
        noisy_part = raw[noise_sample_start:noise_sample_end]
        # perform noise reduction
        audio = nr.reduce_noise(y=raw, y_noise=noisy_part, sr=desired_sample_rate)
        
        audio = (audio * 32767).astype(np.int16) # scale from -1 to 1 to +/-32767
        res = ds.stt(audio)
        
        return res
    except Exception as e:
        print(e.args)

def menu():
    print("Welcome to your virtual airport assistant")
    print("Please select your preferred language: ")
    print("[1] Load English Language Personal Audio")
    print("[2] Load English Language Coursera Sample")
    print("[3] Load Italian Language Coursera Sample")
    print("[4] Load Spanish Language Coursera Sample")
    print("[0] Exit")

def framework_selection():
    print("Please select your preferred framwork: ")
    print("[1] Load Mozilla Deepspeech")
    print("[2] Load Quartznet")
    print("[0] Exit")

# Code adapted from https://web.archive.org/web/20171215025927/http://progfruits.blogspot.com/2014/02/word-error-rate-wer-and-word.html
def wer(ref, hyp ,debug=True):
    
    r = ref.split()
    h = hyp.split()
    
    #costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
 
    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3
    DEL_PENALTY = 1
    INS_PENALTY = 1
    SUB_PENALTY = 1
    
    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r)+1):
        costs[i][0] = DEL_PENALTY*i
        backtrace[i][0] = OP_DEL
    
    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS
    
    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                costs[i][j] = costs[i-1][j-1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i-1][j-1] + SUB_PENALTY # penalty is always 1
                insertionCost    = costs[i][j-1] + INS_PENALTY   # penalty is always 1
                deletionCost     = costs[i-1][j] + DEL_PENALTY   # penalty is always 1
                 
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
            i-=1
            j-=1
            if debug:
                lines.append("OK\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub +=1
            i-=1
            j-=1
            if debug:
                lines.append("SUB\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j-=1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i-=1
            if debug:
                lines.append("DEL\t" + r[i]+"\t"+"****")
    if debug:
        lines = reversed(lines)
        for line in lines:
            print(line)
        print("#cor " + str(numCor))
        print("#sub " + str(numSub))
        print("#del " + str(numDel))
        print("#ins " + str(numIns))
    # return (numSub + numDel + numIns) / (float) (len(r))
    wer_result = round( (numSub + numDel + numIns) / (float) (len(r)), 3)
    return {'WER':wer_result, 'numCor':numCor, 'numSub':numSub, 'numIns':numIns, 'numDel':numDel, "numCount": len(r)}

#Deepspeech Model Initialisation
scorer_english_default = "english/deepspeech-0.9.3-models.scorer"
model_english_default = "english/deepspeech-0.9.3-models.pbmm"
scorer_italian_default = "italian/kenlm_it.scorer"
model_italian_default = "italian/output_graph_it.pbmm"
scorer_spanish_default = "spanish/kenlm_es.scorer"
model_spanish_default = "spanish/output_graph_es.pbmm"

#SELF RECORDINGS (CLEAN)
english_clean_reference = "this is my very first short sentence for the demostration of my system and it will be a sentence of gibberrish"
english_audio_file_clean = "english/Voice/english_audio_eval_1.wav"

#SELF RECORDINGS (NOISY)
english_noisy_reference = "alright this is my short sentence number two where i will be demostrating with background noises such as rain and youtube videos"
english_audio_file_noisy = "english/Voice/english_audio_eval_2.wav"


"""
Test File Initialisation
------------------------

Total Sample:

- English = 5
- Italian = 5
- Spanish = 5

------------------------ 

"""
#ENGLISH TEST FILES
english_list = []
english_list.append(["where is the check in desk","english/Voice/checkin.wav"])
english_list.append(["i have lost my parents","english/Voice/parents.wav"])
english_list.append(["please i have lost my suitcase","english/Voice/suitcase.wav"])
english_list.append(["what time is my plane","english/Voice/what_time.wav"])
english_list.append(["where are the restuarants and shops","english/Voice/where.wav"])

#ITALIAN TEST FILES
italian_list = []
italian_list.append(["dove è il pancone","italian/Voice/checkin_it.wav"])
italian_list.append(["ho perso i miei genitori","italian/Voice/parents_it.wav"])
italian_list.append(["per favore. Ho perso la mia","italian/Voice/suitcase_it.wav"])
italian_list.append(["a che ora e’ il mio aereo","italian/Voice/what_time_it.wav"])
italian_list.append(["dove sono i ristoranti e i negozi","italian/Voice/where_it.wav"])

#SPANISH TEST FILES
spanish_list = []
spanish_list.append(["dónde están los mostrador","spanish/Voice/checkin_es.wav"])
spanish_list.append(["he perdido a mis padres","spanish/Voice/parents_es.wav"])
spanish_list.append(["por favor he perdido mi maleta","spanish/Voice/suitcase_es.wav"])
spanish_list.append(["a qué hora es mi avión","spanish/Voice/what_time_es.wav"])
spanish_list.append(["dónde están los restaurantes y las tiendas","spanish/Voice/where_es.wav"])


#Assertion implementation to check for file existence     
assert os.path.exists(scorer_english_default), "English Scorer not in directory"
assert os.path.exists(model_english_default), "English Model not in directory"
assert os.path.exists(scorer_italian_default), "Italian Scorer not in directory"
assert os.path.exists(model_italian_default), "Italian Model not in directory"
assert os.path.exists(scorer_spanish_default), "Spanish Scorer not in directory"
assert os.path.exists(model_spanish_default), "Spanish Model not in directory"

flag = True

while(flag):
    menu()
    option = int(input("Your selection is: "))
    
    if(option == 1):
        print("English Language Personal Test Mode Selected!")
        processed_node_1 = loadAssistant(model_english_default, scorer_english_default,english_audio_file_clean)
        processed_node_2 = loadAssistant(model_english_default, scorer_english_default,english_audio_file_noisy)
        percentage_1 = wer(english_noisy_reference,processed_node_1)
        percentage_2 = wer(english_noisy_reference,processed_node_2)                
        print(percentage_1, "\n", percentage_2)
    
    elif(option == 2):
        print("English Language Coursera Sample Files Selected!")
        for item in english_list:
            if(item[1] == "english/Voice/checkin.wav" or item[1] == "english/Voice/parents.wav"):
                processed_node = loadAssistant(model_english_default,scorer_english_default,item[1],0,122)
                print(processed_node)
                percentage = wer(item[0],processed_node)
                print(percentage)
                print("\n")
            elif(item[1] == "english/Voice/where.wav"):
                processed_node = loadAssistant(model_english_default,scorer_english_default,item[1],1721,1852)
                print(processed_node)
                percentage = wer(item[0],processed_node)
                print(percentage)
                print("\n")
            elif(item[1] == "english/Voice/what_time.wav"):
                processed_node = loadAssistant(model_english_default,scorer_english_default,item[1],0,0)
                print(processed_node)
                percentage = wer(item[0],processed_node)
                print(percentage)
                print("\n")
            elif(item[1] == "english/Voice/suitcase.wav"):
                processed_node = loadAssistant(model_english_default,scorer_english_default,item[1],713,964)
                print(processed_node)
                percentage = wer(item[0],processed_node)
                print(percentage)
                print("\n")
        flag = False                 
    
    elif(option == 3):
        print("Italian Language Coursera Sample Files Selected!")
        for item in italian_list:
            if(item[1] == "italian/Voice/checkin_it.wav" or item[1] == "italian/Voice/parents_it.wav"):
                processed_node = loadAssistant(model_italian_default,scorer_italian_default,item[1],0,500)
                print(processed_node)
                percentage = wer(item[0],processed_node)
                print(percentage)
                print("\n")
            elif(item[1] == "italian/Voice/where_it.wav"):
                processed_node = loadAssistant(model_italian_default,scorer_italian_default,item[1],2347,2647)
                print(processed_node)
                percentage = wer(item[0],processed_node)
                print(percentage)
                print("\n")
            elif(item[1] == "italian/Voice/what_time_it.wav"):
                processed_node = loadAssistant(model_italian_default,scorer_italian_default,item[1],0,215)
                print(processed_node)
                percentage = wer(item[0],processed_node)
                print(percentage)
                print("\n")
            elif(item[1] == "italian/Voice/suitcase_it.wav"):
                processed_node = loadAssistant(model_italian_default,scorer_italian_default,item[1],0,199)
                print(processed_node)
                percentage = wer(item[0],processed_node)
                print(percentage)
                print("\n")
        flag = False 

    elif(option == 4):
        print("Spanish Language Coursera Sample Files Selected!")
        for item in spanish_list:
            if(item[1] == "spanish/Voice/checkin_es.wav" or item[1] == "spanish/Voice/parents_es.wav"):
                processed_node = loadAssistant(model_spanish_default,scorer_spanish_default,item[1],0,500)
                print(processed_node)
                percentage = wer(item[0],processed_node)
                print(percentage)
                print("\n")
            elif(item[1] == "spanish/Voice/where_es.wav"):
                processed_node = loadAssistant(model_spanish_default,scorer_spanish_default,item[1],2347,2647)
                print(processed_node)
                percentage = wer(item[0],processed_node)
                print(percentage)
                print("\n")
            elif(item[1] == "spanish/Voice/what_time_es.wav"):
                processed_node = loadAssistant(model_spanish_default,scorer_spanish_default,item[1],0,215)
                print(processed_node)
                percentage = wer(item[0],processed_node)
                print(percentage)
                print("\n")
            elif(item[1] == "spanish/Voice/suitcase_es.wav"):
                processed_node = loadAssistant(model_spanish_default,scorer_spanish_default,item[1],0,199)
                print(processed_node)
                percentage = wer(item[0],processed_node)
                print(percentage)
                print("\n")
        flag = False   
    
    elif(option == 0):
        print("We'll see you again! Good Day!")
        flag = False
    
    else:
        print("Invalid option, please choose an appropriate number")
        option = int(input("Your selection is: "))


   
        
