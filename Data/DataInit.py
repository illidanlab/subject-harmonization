from sklearn.model_selection import train_test_split
from tools.utils import get_feature_from_id
import numpy as np

def data_init(cfg_proj, mci_subject, nl_subject, dic_id2feature, df_labels, seed):

    mci_train, mci_test = train_test_split(mci_subject, test_size=0.2, random_state = seed)
    nl_train, nl_test = train_test_split(nl_subject, test_size=0.2, random_state = seed)

    id_train = mci_train + nl_train
    id_test = mci_test + nl_test
    
    x_train, y_train, g_train, x_test, y_test, g_test = get_feature_from_id(id_train, id_test, dic_id2feature, df_labels)
    
    return x_train, y_train, g_train, x_test, y_test, g_test