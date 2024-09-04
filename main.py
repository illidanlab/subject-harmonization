import os
import argparse
from time import localtime, strftime
from tools.utils import load_raw_data
from Data.DataInit import data_init
from Data.DataPreProcessing import data_pre_processing
from configs.cfg import init_cfg


def main(cfg_proj, cfg_m):
    from Solvers.Solver_loader import solver_loader
    solver = solver_loader(cfg_proj, cfg_m)

    # Load raw data
    dic_id2feature, df_labels, nl_subject, mci_subject = load_raw_data()
    
    solver.setLabels(df_labels)

    for step in range(cfg_proj.num_total_runs):
        seed = step if cfg_proj.seed is None else cfg_proj.seed
        solver.set_random_seed(seed)
        
        # Split to train and test
        x_train, y_train, g_train, x_test, y_test, g_test = data_init(cfg_proj, mci_subject, nl_subject, dic_id2feature, df_labels, seed)
        
        # Data preprocessing
        x_train, y_train, g_train, x_test, y_test, g_test = data_pre_processing(cfg_proj, cfg_m, x_train, y_train, g_train, x_test, y_test, g_test)

        # Run the experiment
        auc, f1, sens, spec, auc_sbj, f1_sbj, sens_sbj, spec_sbj = solver.run(x_train, y_train, g_train, x_test, y_test, g_test, seed)

        print("step-%d, auc=%.3f,f1=%.3f,sens=%.3f,spec=%.3f, sbj:auc=%.3f,f1=%.3f,sens=%.3f,spec=%.3f"%(step, auc, f1, sens, spec, auc_sbj, f1_sbj, sens_sbj, spec_sbj))
    
    # print results
    solver.save_results()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="template")
    parser.add_argument("--gpu", type=str, default="3", required=False)
    parser.add_argument("--seed", type=int, default = None, required=False) 
    parser.add_argument("--num_total_runs", type=int, default = 100, required=False) 
    parser.add_argument("--flag_generatePredictions", default = ["Sex", "Edu", "Age"])
    parser.add_argument("--number_of_feature", type=int, default = 99, required=False)  
    parser.add_argument("--vote_threshold", type=int, default = 0.5, required=False) 

    #Standard_solver, Baseline_confounder_solver, subject_harmonization_solver, confounder_harmonization_solver
    parser.add_argument("--solver", type=str, default="subject_harmonization_solver", required=False)  
    parser.add_argument("--classifier", type=str, default="MLP", required=False)  #LR, MLP
    parser.add_argument("--flag_log", type=str, default = True, required=False) 
    parser.add_argument("--save_whitening", type=bool, default = False, required=False) 
    parser.add_argument("--flag_time", type=str, default = strftime("%Y-%m-%d_%H-%M-%S", localtime()), required=False)
    parser.add_argument("--flag_load", type=str, default = None, required=False)    #if is not None, then the file of loaded para need to contain the str
    cfg_proj = parser.parse_args()

    cfg_m = init_cfg(cfg_proj)
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%(cfg_proj.gpu)
    if cfg_proj.save_whitening:
        cfg_proj.num_total_runs = 1
    main(cfg_proj, cfg_m)