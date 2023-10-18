import os
from configs.default_configs import get_default_configs

def init_cfg(cfg_proj):
    n_solver = cfg_proj.solver
    config = get_default_configs()
    config.Note = None

    if n_solver in ["Standard_solver", "whiting_confounder_solver" , "whiting_confounderS_solver", "whiting_solver", "Baseline_confounder_solver"]:
        config.data.dim_out = 2
        config.training.epochs = 100
        config.training.batch_size = 512
        config.training.lr_init = 1.0e-3
        config.training.tol = 1e-4

        config.l2_lambda = None #or None
        config.l1_lambda = None

        config.training.epochs_whiting = 60

    if n_solver == "whiting_confounder_solver":
        config.training.confounder_var = "educ" #["sbj", "age", "gender", "educ"]
    return config
