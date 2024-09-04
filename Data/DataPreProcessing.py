from sklearn.preprocessing import StandardScaler
import numpy as np

def data_pre_processing(cfg_proj, cfg_m, x_train_raw, y_train, g_train, x_test_raw, y_test, g_test): 

    if cfg_proj.solver not in ["Baseline_confounder_solver"]:
        x_train_raw = x_train_raw[:, :-3]
        x_test_raw = x_test_raw[:, :-3]
    
    if cfg_proj.solver not in ["confounder_harmonization_solver"]:
        g_train, g_test = [g[0] for g in g_train], [g[0] for g in g_test]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw)
    x_test = scaler.transform(x_test_raw)

    return x_train, y_train, g_train, x_test, y_test, g_test