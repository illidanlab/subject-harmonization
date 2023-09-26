from tqdm import tqdm
import random
import torch
import numpy as np
import logging
import torch.nn.functional as F
from sklearn.metrics import (
    precision_recall_fscore_support as prf,
    accuracy_score,
    roc_auc_score,
)
import os
import torch.nn as nn
import time
import ml_collections
from sklearn import metrics
import pandas as pd
import math


class Solver_Base:
    
    def __init__(self, cfg_proj, cfg_m, name):
        self.name = name
        self.cfg_proj = cfg_proj
        self.cfg_m = cfg_m
        self.loss_func = torch.nn.functional.cross_entropy
        self.init_env(name)
        self.performance_main = {"AUC":[], "F1":[], "Sens":[], "Spec":[]}
        self.performance = {}

    def setLabels(self, df_labels):
        self.df_labels = df_labels
        
    def init_env(self, name, log_folder = "checkpoints"):
        #init device
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.cfg_m.device = self.device

        #init log sys
        self.log_sub_folder = "%s/log_%s_%s"%(log_folder, name, self.cfg_proj.flag_time)
        self.log_id = "log_%s_%s"%(name, self.cfg_proj.flag_time)
        self.logger = logging.getLogger(name)
        if self.cfg_proj.flag_log:
            os.makedirs(self.log_sub_folder, exist_ok=True)
            logging.basicConfig(
                format =' [%(asctime)s] - %(message)s',
                datefmt = '%Y/%m/%d %H:%M:%S',
                level = logging.INFO,
                filename = '%s/%s.log'%(self.log_sub_folder, self.log_id))
        setting_log = "----Setting----"
        for n in self.cfg_m:
            if isinstance(self.cfg_m[n], ml_collections.ConfigDict):
                for n_sub in self.cfg_m[n]:
                    setting_log = setting_log + "\n" + "%s - %s - %s"%(n, n_sub, self.cfg_m[n][n_sub])
            else:
                setting_log = setting_log + "\n" + "%s - %s"%(n, self.cfg_m[n])
        for p in vars(self.cfg_proj):
            setting_log = setting_log + "\n" + "%s - %s"%(p, getattr(self.cfg_proj, p))
        setting_log = setting_log + "\n" + "----log----"
        self.logger.info(setting_log)

    def eval_func(self, model, x_test, y_test, g_test):
        model.eval()
        
        #conversation-wise
        pred = self.predict(model, x_test)
        pred_proba = self.predict_proba(model, x_test)[:, 1]

        # Evaluation metric
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, pred).ravel()
        auc = metrics.roc_auc_score(y_test, pred_proba)
        f1 = metrics.f1_score(y_test, pred)
        sens = tp/(tp+fn)
        spec = tn/(fp+tn)

        #subject-wise
        pred_sbj = []
        pred_proba_sbj = []
        y_test_subject = []
        id2pred = {}
        id2pred_proba = {}

        g_test_unique = list(set(g_test))
        for g in g_test_unique:
            index = [i for i in range(len(g_test)) if g_test[i] == g]
            pred_single_sbj = pred_proba[index]
            
            # get most confidence conversations
            pred_single_sbj = list(pred_single_sbj)
            pred_single_sbj.sort(key = lambda x: -abs(x-0.5))
            
            id2pred[g] = pred[index]
            id2pred_proba[g] = pred_proba[index]
            pred_proba_single_sbj =np.sum(pred_single_sbj)/len(pred_single_sbj)
            pred_single_sbj = 1 if pred_proba_single_sbj >= self.cfg_proj.vote_threshold else 0
            pred_sbj.append(pred_single_sbj)
            pred_proba_sbj.append(pred_proba_single_sbj)
            assert np.all(y_test[index] == y_test[index[0]])
            y_test_subject.append(y_test[index[0]])
        
        self.generatePredictions(id2pred, id2pred_proba)

        # Evaluation metric
        tn_sbj, fp_sbj, fn_sbj, tp_sbj = metrics.confusion_matrix(y_test_subject, pred_sbj).ravel()
        auc_sbj = metrics.roc_auc_score(y_test_subject, pred_proba_sbj)
        f1_sbj = metrics.f1_score(y_test_subject, pred_sbj)
        sens_sbj = tp_sbj/(tp_sbj+fn_sbj)
        spec_sbj = tn_sbj/(fp_sbj+tn_sbj)
        
        self.save_results_each_run(auc, f1, sens, spec, auc_sbj, f1_sbj, sens_sbj, spec_sbj)
        return auc, f1, sens, spec, auc_sbj, f1_sbj, sens_sbj, spec_sbj

    def load_ckp(self, model, optimizer, lr_scheduler, seed, contain_t):

        def getFilesInPath(flag_load, folder, suffix, contain_t):
            name_list = []
            f_list = sorted(os.listdir(folder))
            try:
                folder_sub = [f_n for f_n in f_list if flag_load in f_n][0] #time stamp is unique, for sure
            except:
                return name_list
            folder_sub = os.path.join(folder, folder_sub)
            f_sub_list = sorted(os.listdir(folder_sub))

            for f_n in f_sub_list:
                if contain_t is not None:
                    if suffix in os.path.splitext(f_n)[1] and "seed_%d"%(seed) in os.path.splitext(f_n)[0] and contain_t in os.path.splitext(f_n)[0]:
                        pathName = os.path.join(folder_sub, f_n)
                        name_list.append(pathName)
                else:
                    if suffix in os.path.splitext(f_n)[1]:
                        pathName = os.path.join(folder_sub, f_n)
                        name_list.append(pathName)

            return name_list
        flag_load = self.cfg_proj.flag_load if self.cfg_proj.flag_load is not None else self.cfg_proj.flag_time
        file_ns = getFilesInPath(flag_load, folder = "checkpoints", suffix = "pt", contain_t = contain_t)

        if len(file_ns) == 1:
            checkpoint = torch.load(file_ns[0])
            try:
                model.load_state_dict(checkpoint['net'])    #, strict=False)  #ignore the unmatched key
            except RuntimeError:
                try:
                    model.load_state_dict(checkpoint['net'], strict=False)  #ignore the unmatched key
                    print("unmatched keys in paras are loaded to model at stage: %s" % (contain_t))
                except:
                    model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['net'].items()})
            if optimizer is not None: optimizer.load_state_dict(checkpoint['optimizer'])
            if lr_scheduler is not None: lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            epoch_start = checkpoint['epoch']
            str_record = "load ckp - %s, epoch_start = %d at stage: %s" % (file_ns[-1], epoch_start, contain_t)
            print(str_record)
            self.logger.info(str_record)
        else:
            epoch_start = 0
            print("Warning - no paras are loaded to model at stage: %s" % (contain_t))

        return model, optimizer, lr_scheduler, epoch_start

    def save_ckp(self, model, optimizer, lr_scheduler, seed, epoch, stage):
        if os.path.exists(self.log_sub_folder):
            state_ckp = {'net':model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), 'optimizer':optimizer.state_dict() if optimizer is not None else None, \
                        'lr_scheduler':lr_scheduler.state_dict() if lr_scheduler is not None else None, 'epoch':epoch}
            torch.save(state_ckp, '%s/%s_%s_%s_seed_%d_epoch_%04d.pt'%(self.log_sub_folder, self.cfg_proj.backbone, stage, self.cfg_proj.flag_time, seed, epoch))
            time.sleep(1)
        else:
            print("no-saving-ckp, %s doesn't exist!" % (self.log_sub_folder))

    def to_parallel_model(self, model):
        if torch.cuda.device_count() > 1:
            # print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
            model = model.to(self.device)
            return model, model.module
        else:
            model = model.to(self.device)
            return model, model
        
    def set_random_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available(): 
            torch.cuda.manual_seed(seed)
            if torch.cuda.device_count() > 1: torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        self.seed_current = seed


    def predict(self, model, X, flag_prob = False):
        X = torch.from_numpy(X)

        model.eval()
        with torch.no_grad():
            X = X.float().to(self.device)
            pred = model(X)
            pred = pred if torch.is_tensor(pred) else pred[1]
        
        pred = torch.argmax(pred, 1)
        return pred.detach().cpu().numpy() 

    def predict_proba(self, model, X, flag_prob = True):
        X = torch.from_numpy(X)

        model.eval()
        with torch.no_grad():
            X = X.float().to(self.device)
            pred = model(X)
            pred = pred if torch.is_tensor(pred) else pred[1]
            pred = torch.nn.functional.softmax(pred, dim = 1)
        
        return pred.detach().cpu().numpy() 
    
    def cross_entropy_regs(self, model, Yhat, Y, l2_lambda, l1_lambda):    #pred, train_Y
        Y_t = torch.zeros((Y.shape[0], 2)).to(Y)
        Y_t[:, 1] = Y.data
        Y_t[:, 0] = 1 - Y.data
        loss_mean = F.cross_entropy(Yhat, Y_t)
       
        if l2_lambda is not None:
            l2_reg = torch.tensor(0., requires_grad=True)
            for name, param in model.named_parameters():
                if 'weight' in name:
                    l2_reg = l2_reg + torch.norm(param, 2)
            l2_reg = l2_lambda * l2_reg
            loss_mean += l2_reg
        if l1_lambda is not None:
            l1_reg = torch.tensor(0., requires_grad=True)
            for name, param in model.named_parameters():
                if 'weight' in name:
                    l1_reg = l1_reg + torch.norm(param, 1)
            l1_reg = l1_lambda * l1_reg
            loss_mean += l1_reg
        return loss_mean
    
    def basic_train(self, model, dataloader_train, criterion, optimizer, lr_scheduler):
        loss_train_trace = []
        for epoch in range(self.cfg_m.training.epochs):
            model.train()
            loss_epoch = []
            for train_X, train_Y, _ , idx in dataloader_train:   

                train_X, train_Y = train_X.float().to(self.device), train_Y.to(self.device)
                Y_hat = model(train_X)
                Y_hat = Y_hat if torch.is_tensor(Y_hat) else Y_hat[1]
                loss = criterion(model, Y_hat, train_Y, l2_lambda = self.cfg_m.l2_lambda, l1_lambda = self.cfg_m.l1_lambda)
                loss_epoch.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                loss_train_trace.append(np.mean(loss_epoch))
        return model, loss_train_trace
    
    def freeze_grad(self, model, except_full_names = [None], except_str = [None]):
        for n, para in model.named_parameters():
            para.requires_grad = False
        for n, para in model.named_parameters():
            for f_n in except_full_names: 
                if f_n == n: para.requires_grad = True
            for s in except_str:
                if s is not None:
                    if s in n: para.requires_grad = True
        return model

    def save_results_each_run(self, auc, f1, sens, spec, auc_sbj, f1_sbj, sens_sbj, spec_sbj):
        self.performance_main["AUC"].append([auc, auc_sbj])
        self.performance_main["F1"].append([f1, f1_sbj])
        self.performance_main["Sens"].append([sens, sens_sbj])
        self.performance_main["Spec"].append([spec, spec_sbj])

    def save_results(self):
        AUC, F1, Sens, Spec = np.array(self.performance_main["AUC"]), np.array(self.performance_main["F1"]), np.array(self.performance_main["Sens"]), np.array(self.performance_main["Spec"])
        info_ = "step-%d, auc=%.3f\u00B1%.3f,f1=%.3f\u00B1%.3f,sens=%.3f\u00B1%.3f,spec=%.3f\u00B1%.3f, sbj:auc=%.3f\u00B1%.3f,f1=%.3f\u00B1%.3f,sens=%.3f\u00B1%.3f,spec=%.3f\u00B1%.3f"%(-1,
               np.mean(AUC[:, 0]), np.std(AUC[:, 0]), np.mean(F1[:, 0]), np.std(F1[:, 0]), np.mean(Sens[:, 0]), np.std(Sens[:, 0]), 
               np.mean(Spec[:, 0]), np.std(Spec[:, 0]), np.mean(AUC[:, 1]), np.std(AUC[:, 1]), np.mean(F1[:, 1]), np.std(F1[:, 1]), 
               np.mean(Sens[:, 1]), np.std(Sens[:, 1]), np.mean(Spec[:, 1]), np.std(Spec[:, 1]))
        print(info_)
        self.logger.info(info_)
        self.logger.info("AUC  = %.3f\u00B1%.3f, AUC_sbj  = %.3f\u00B1%.3f"%(np.mean(AUC[:, 0]), np.std(AUC[:, 0]), np.mean(AUC[:, 1]), np.std(AUC[:, 1])))
        self.logger.info("F1   = %.3f\u00B1%.3f, f1_sbj   = %.3f\u00B1%.3f"%(np.mean(F1[:, 0]), np.std(F1[:, 0]), np.mean(F1[:, 1]), np.std(F1[:, 1])))
        self.logger.info("Sens = %.3f\u00B1%.3f, Sens_sbj = %.3f\u00B1%.3f"%(np.mean(Sens[:, 0]), np.std(Sens[:, 0]), np.mean(Sens[:, 1]), np.std(Sens[:, 1])))
        self.logger.info("Spec = %.3f\u00B1%.3f, Spec_sbj = %.3f\u00B1%.3f"%(np.mean(Spec[:, 0]), np.std(Spec[:, 0]), np.mean(Spec[:, 1]), np.std(Spec[:, 1])))

        if len(self.cfg_proj.flag_generatePredictions) != 0:
            for category in self.cfg_proj.flag_generatePredictions:
                for group in self.performance[category]:
                    AUC, F1, Sens, Spec = np.array(self.performance[category][group]["AUC"]), np.array(self.performance[category][group]["F1"]), np.array(self.performance[category][group]["Sens"]), np.array(self.performance[category][group]["Spec"])
                    if len(AUC) == 0:
                        AUC, F1, Sens, Spec = np.ones([1, 2])*-1, np.ones([1, 2])*-1, np.ones([1, 2])*-1, np.ones([1, 2])*-1
                    self.logger.info(category + " " + str(group))
                    self.logger.info("AUC  = %.3f\u00B1%.3f, AUC_sbj  = %.3f\u00B1%.3f"%(np.mean(AUC[:, 0]), np.std(AUC[:, 0]), np.mean(AUC[:, 1]), np.std(AUC[:, 1])))
                    self.logger.info("F1   = %.3f\u00B1%.3f, f1_sbj   = %.3f\u00B1%.3f"%(np.mean(F1[:, 0]), np.std(F1[:, 0]), np.mean(F1[:, 1]), np.std(F1[:, 1])))
                    self.logger.info("Sens = %.3f\u00B1%.3f, Sens_sbj = %.3f\u00B1%.3f"%(np.mean(Sens[:, 0]), np.std(Sens[:, 0]), np.mean(Sens[:, 1]), np.std(Sens[:, 1])))
                    self.logger.info("Spec = %.3f\u00B1%.3f, Spec_sbj = %.3f\u00B1%.3f"%(np.mean(Spec[:, 0]), np.std(Spec[:, 0]), np.mean(Spec[:, 1]), np.std(Spec[:, 1])))

        self.logger.info("Conversation-wise " + str(np.mean(np.array(self.performance_main["AUC"])[:, 0])))
        self.logger.info("Subject-wise " + str(np.mean(np.array(self.performance_main["AUC"])[:, 1])))

    def generatePredictions(self, id2pred, id2pred_proba):
        dic_pd_index = {"Age":'nac_a1_age', "Edu":'nac_educ', "Sex":'nac_sex'}
        dic_attr_index = {"Age":["75-80", "81-87", "88-94"], "Edu":["12-15", "16-18", "19-21"], "Sex":[1, 2]}
        for cat_sub in self.cfg_proj.flag_generatePredictions:
            if cat_sub not in self.performance:
                self.performance[cat_sub] = {} 
            for attr in dic_attr_index[cat_sub]:
                if attr not in self.performance[cat_sub]:
                    self.performance[cat_sub][attr] = {"AUC":[], "F1":[], "Spec":[] , "Sens":[]}
                [lower, upper] = [int(attr[:2]), int(attr[3:])] if cat_sub != "Sex" else [attr, attr]
                auc, f1, sens, spec, auc_sbj, f1_sbj, sens_sbj, spec_sbj = self.getClinicalPerformance(id2pred, id2pred_proba, lower, upper, dic_pd_index[cat_sub])
                if auc != -1:
                    self.performance[cat_sub][attr]["AUC"].append([auc, auc_sbj])
                    self.performance[cat_sub][attr]["Sens"].append([sens, sens_sbj])
                    self.performance[cat_sub][attr]["F1"].append([f1, f1_sbj])
                    self.performance[cat_sub][attr]["Spec"].append([spec, spec_sbj])
                
    def getClinicalPerformance(self, id2pred, id2pred_proba, lower, upper, clinical):
        pred_proba = []
        pred = []
        y_test = []
        for id in id2pred:
            if int(self.df_labels[self.df_labels["ts_sub_id"] == id][clinical].values[0]) <= upper and int(self.df_labels[self.df_labels["ts_sub_id"] == id][clinical].values[0]) >= lower:
                pred_proba.extend(list(id2pred_proba[id]))
                pred.extend(list(id2pred[id]))
                y_test.extend([1-int(self.df_labels[self.df_labels["ts_sub_id"] == id]['nac_normcog'].values[0]) for i in range(len(id2pred_proba[id]))])

        if len(set(y_test)) < 2:
            return -1, -1, -1, -1, -1, -1, -1, -1

        # Evaluation metric
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, pred).ravel()
        auc = metrics.roc_auc_score(y_test, pred_proba)
        f1 = metrics.f1_score(y_test, pred)
        sens = tp/(tp+fn)
        spec = tn/(fp+tn)
        
        #subject-wise
        pred_sbj = []
        pred_proba_sbj = []
        y_test_subject = []

        for id in id2pred_proba:
            if int(self.df_labels[self.df_labels["ts_sub_id"] == id][clinical].values[0]) <= upper and int(self.df_labels[self.df_labels["ts_sub_id"] == id][clinical].values[0]) >= lower:
                pred_single_sbj = id2pred_proba[id]
                
                # get most confidence conversations
                pred_single_sbj = list(pred_single_sbj)
                pred_single_sbj.sort(key = lambda x: -abs(x-0.5))
                
                pred_proba_single_sbj = np.sum(pred_single_sbj)/len(pred_single_sbj)
                pred_single_sbj = 1 if pred_proba_single_sbj >= self.cfg_proj.vote_threshold else 0
                pred_sbj.append(pred_single_sbj)
                pred_proba_sbj.append(pred_proba_single_sbj)
                y_test_subject.append(1-int(self.df_labels[self.df_labels["ts_sub_id"] == id]['nac_normcog'].values[0]))
        
        # Evaluation metric
        tn_sbj, fp_sbj, fn_sbj, tp_sbj = metrics.confusion_matrix(y_test_subject, pred_sbj).ravel()
        auc_sbj = metrics.roc_auc_score(y_test_subject, pred_proba_sbj)
        f1_sbj = metrics.f1_score(y_test_subject, pred_sbj)
        sens_sbj = tp_sbj/(tp_sbj+fn_sbj)
        spec_sbj = tn_sbj/(fp_sbj+tn_sbj)
        
        return auc, f1, sens, spec, auc_sbj, f1_sbj, sens_sbj, spec_sbj