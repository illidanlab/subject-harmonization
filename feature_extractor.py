#import library
import liwc
import pandas as pd
import pickle
import glob
import os
import copy 
import numpy as np
from lexical_diversity import lex_div as ld

#get parse and categories from LIWC
parse, category_names = liwc.load_token_parser("LIWC2007_English.dic")
category_position = {}

for i in range(len(category_names)):
    category_position[category_names[i]] = i

#get pandas
df_syn = pd.read_csv("rawdata/syntactic_complexity_measures.csv")
df_labels = pd.read_csv("rawdata/Baseline_label.csv")
id2feature = {}

#get all paths to text file
video_chat_paths = sorted(glob.glob("Transcriptions/*.csv"))
paths = video_chat_paths 

def respone_feature(df):
    df.dropna(subset = ['asr'], inplace=True)

    data = []
    current_role = str(df["role"].iloc[0])
    current_text = ""

    for i in range(df.shape[0]):
        if len(str(df["asr"].iloc[i]).split()) == 0:
            continue

        if str(df["role"].iloc[i]) != current_role:
            tokens = current_text.split()
            current_text = ""
            if len(tokens) != 0:
                if current_role == "Participant":
                    data.append(len(tokens))

        current_role = df["role"].iloc[i]
        current_text += str(df["asr"].iloc[i]).strip()
        current_text += " "

    tokens = current_text.split()
    if len(tokens) != 0:
        if current_role == "Participant":
            data.append(len(tokens))
    
    return [np.mean(data), np.var(data)]

def lexical(text):
    flt = ld.flemmatize(text)
    return [ld.ttr(flt), ld.root_ttr(flt), ld.log_ttr(flt), ld.maas_ttr(flt), ld.msttr(flt), ld.mattr(flt), ld.hdd(flt), ld.mtld(flt), ld.mtld_ma_wrap(flt), ld.mtld_ma_bid(flt)]

for fileName in paths:

    subject = os.path.splitext(os.path.basename(fileName))[0].split("_")[0]
    baseName = os.path.basename(fileName)[:-3]+'txt'
        
    #check if subject has label or not
    if subject not in df_labels["ts_sub_id"].values:
        continue
    
    #check if subject has syntactic feature or not
    if baseName not in df_syn["Filename"].values:
        continue
   
    #get Participant text from text file
    df = pd.read_csv(fileName)
    df_save = copy.deepcopy(df)

    df = df[df["role"] == "Participant"]
    df.dropna(subset = ['asr'], inplace=True) #can replace

    texts = list(df["asr"])
    for i in range(len(texts)):
        texts[i] = str(texts[i]).strip().lower()
    texts = " ".join(texts)
    tokens = texts.split()

    if(len(tokens) == 0):
        continue

    if subject not in id2feature:
        id2feature[subject] = []

    #get LIWC categories from each tokens
    feature = [0 for i in range(64)]
    for token in tokens:
        for category in parse(token):
            feature[category_position[category]] +=1
    
    #add Syntactic feature
    feature.extend(df_syn[df_syn["Filename"] == baseName].iloc[0, 1:].tolist())

    #add Lexical feature
    feature.extend(lexical(" ".join(tokens)))

    #add responses feature
    feature.extend(respone_feature(df_save))

    id2feature[subject].append(feature)

with open("rawdata/id2feature.p", "wb") as f:
    pickle.dump(id2feature, f)