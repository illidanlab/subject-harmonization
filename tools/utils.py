import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn.functional as F


def load_raw_data():

    with open("rawdata/id2feature.p", "rb") as f:
        dic_id2feature = pickle.load(f)

    for id in dic_id2feature:
        dic_id2feature[id] = np.array(dic_id2feature[id])
        
    df_labels = pd.read_csv("rawdata/Baseline_label.csv")

    nl_subject = df_labels["ts_sub_id"][df_labels["nac_normcog"] == 1].to_list()
    mci_subject = df_labels["ts_sub_id"][df_labels["nac_normcog"] == 0].to_list()
    nl_subject = [subject for subject in nl_subject if subject in dic_id2feature]
    mci_subject = [subject for subject in mci_subject if subject in dic_id2feature]

    return dic_id2feature, df_labels, nl_subject, mci_subject
       
def get_feature_from_id(train, test, dic_id2feature, df_labels):

    x_train, y_train, g_train, x_test, y_test, g_test  = [], [], [], [], [], []

    for id in train:
        label = 1-int(df_labels[df_labels["ts_sub_id"] == id]['nac_normcog'].values[0])
        df = df_labels[df_labels["ts_sub_id"] == id]
        var_confounder = [int(df['nac_a1_age'].values[0]), int(df['nac_sex'].values[0]), int(df['nac_educ'].values[0])]
        [age, gender, educ] = [int(df['nac_a1_age'].values[0]), int(df['nac_sex'].values[0]), int(df['nac_educ'].values[0])]
        for feature in dic_id2feature[id]:
            feature = np.concatenate([feature, var_confounder])
            x_train.append(feature)
            y_train.append(label*1.0)
            g_train.append([id, age, gender, educ])

    for id in test:
        label = 1-int(df_labels[df_labels["ts_sub_id"] == id]['nac_normcog'].values[0])
        df = df_labels[df_labels["ts_sub_id"] == id]
        var_confounder = [int(df['nac_a1_age'].values[0]), int(df['nac_sex'].values[0]), int(df['nac_educ'].values[0])]
        [age, gender, educ] = [int(df['nac_a1_age'].values[0]), int(df['nac_sex'].values[0]), int(df['nac_educ'].values[0])]
        for feature in dic_id2feature[id]:
            feature = np.concatenate([feature, var_confounder])
            x_test.append(feature)
            y_test.append(label)
            g_test.append([id, age, gender, educ])

    return np.array(x_train), np.array(y_train), g_train, np.array(x_test), np.array(y_test), g_test
