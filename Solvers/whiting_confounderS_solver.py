from tkinter import E
import numpy as np
from Solvers.Solver_Base import Solver_Base
import torch
from torch.utils.data import DataLoader
from Moldes.model import MLP_pytorch, MLP_whiting, CustomDataset_coufounder, MLP_whiting_confounders
import torch.nn.functional as F
import pickle
from sklearn import metrics
import math


def generalization(values, values_f, num_class):
    v_mean = np.mean(values)
    v_std = np.std(values)
    values_norm = [v for v in values if v >= v_mean-3*v_std and v <= v_mean+3*v_std]
    v_min = min(values_norm)
    v_max = max(values_norm)
    class_interval = (v_max-v_min)/num_class
    class_abnorm = int(num_class/2) #-1, int(num_class/2), num_class
    values_g = [max(min(int((v-v_min)/class_interval), num_class-1), 0)  for v in values]
    values_abnorm_id = [i for i, v in enumerate(values) if v < v_mean-3*v_std or v > v_mean+3*v_std]
    values_g = [v if i not in values_abnorm_id else class_abnorm for i,v in enumerate(values_g)]

    values_f_g = [max(min(int((v-v_min)/class_interval), num_class-1), 0)  for v in values_f]
    values_f_abnorm_id = [i for i, v in enumerate(values_f) if v < v_mean-3*v_std or v > v_mean+3*v_std]
    values_f_g = [v if i not in values_f_abnorm_id else class_abnorm for i,v in enumerate(values_f_g)]

    return values_g, values_f_g


class whiting_confounderS_solver(Solver_Base):
    
    def __init__(self, cfg_proj, cfg_m, name = "white_c_s"):
        Solver_Base.__init__(self, cfg_proj, cfg_m, name)
        
    def run(self, x_train, y_train, g_train, x_test, y_test, g_test, seed):

        g_train_sbj, g_test_sbj = [g[0] for g in g_train], [g[0] for g in g_test]
        g_train_age, g_test_age = [g[1] for g in g_train], [g[1] for g in g_test]
        g_train_age, g_test_age = generalization(g_train_age, g_test_age, 5)
        g_train_gender, g_test_gender = [g[2] for g in g_train], [g[2] for g in g_test]
        g_train_educ, g_test_educ = [g[3] for g in g_train], [g[3] for g in g_test]
        g_train_educ, g_test_educ = generalization(g_train_educ, g_test_educ, 3)


        dataloader_train = DataLoader(CustomDataset_coufounder(x_train, y_train, [g_train_sbj, g_train_age, g_train_gender, g_train_educ]), 
                                      batch_size = self.cfg_m.training.batch_size, drop_last=True, shuffle = True, pin_memory=True, worker_init_fn = np.random.seed(seed))
        
        model = MLP_whiting_confounders(input_dim = len(x_train[0]), confounders_dim = [len(list(set(g_train_age))), len(list(set(g_train_gender))), len(list(set(g_train_educ)))], task_in_dim = len(x_train[0]), task_out_dim = self.cfg_m.data.dim_out) 
        model = model.to(self.device)
        
        #whiting features
        self.freeze_grad(model, except_str = ["feature_mapping", "out_age", "out_gender", "out_educ"])
        optimizer_sbj = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad == True], lr = self.cfg_m.training.lr_init)
        lr_scheduler_sbj = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_sbj, int(self.cfg_m.training.epochs_whiting*len(dataloader_train)))   #very useful
        model, loss_train_trace = self.sbj_train(model, dataloader_train, None, optimizer_sbj, lr_scheduler_sbj)
        
        #feature_preprocessing - outlier detection and feature selection
        dataloader_train_ = DataLoader(CustomDataset_coufounder(x_train, y_train, [g_train_sbj, g_train_age, g_train_gender, g_train_educ]), 
                                       batch_size = self.cfg_m.training.batch_size, drop_last=False, shuffle = False, pin_memory=True, worker_init_fn = np.random.seed(seed))
        
        #feature_preprocessing - outlier detection and feature selection
        dataloader_test_ = DataLoader(CustomDataset_coufounder(x_test, y_test, [g_test_sbj, g_test_age, g_test_gender, g_test_educ]), 
                                       batch_size = self.cfg_m.training.batch_size, drop_last=False, shuffle = False, pin_memory=True, worker_init_fn = np.random.seed(seed))

        self.feature_preprocessing(model, dataloader_train_, dataloader_test_)
            
        #Task training
        self.freeze_grad(model, except_str = ["out_task"])
        criterion = self.cross_entropy_regs
        optimizer_task = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad == True], lr = self.cfg_m.training.lr_init)
        lr_scheduler_task = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_task, int(self.cfg_m.training.epochs*len(dataloader_train)))   #very useful
        
        model, loss_train_trace = self.basic_train(model, dataloader_train, criterion, optimizer_task, lr_scheduler_task)

        # Evaluation
        auc, f1, sens, spec, auc_sbj, f1_sbj, sens_sbj, spec_sbj = self.eval_func(model, x_test, y_test, g_test, g_test_sbj)

        return auc, f1, sens, spec, auc_sbj, f1_sbj, sens_sbj, spec_sbj
    

    def sbj_train(self, model, dataloader_train, criterion, optimizer, lr_scheduler):
        loss_train_trace = []
        loss_mse = torch.nn.MSELoss()
        loss_cross_ent= torch.nn.CrossEntropyLoss()
        for epoch in range(self.cfg_m.training.epochs_whiting):
            model.train()
            loss_epoch = []
            for train_X, train_Y, _, train_G_age, train_G_gender, train_G_educ, idx in dataloader_train:   
                train_X, train_Y = train_X.float().to(self.device), train_Y.to(self.device)
                train_G_age, train_G_gender, train_G_educ = train_G_age.to(self.device), train_G_gender.to(self.device), train_G_educ.to(self.device)
                [features, logits_age, logits_gender, logits_educ] = model(train_X, id = "0,1") #
                loss_sbj_mse = loss_mse(train_X, features)
                loss_age = 15 - loss_cross_ent(logits_age, train_G_age)
                loss_gender = 15 - loss_cross_ent(logits_gender, train_G_gender)
                loss_educ = 15 - loss_cross_ent(logits_educ, train_G_educ)
                loss = loss_age*0.2 + loss_gender*0.2 + loss_educ*0.2 + loss_sbj_mse*0.3
                loss_epoch.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                loss_train_trace.append(np.mean(loss_epoch))
        return model, loss_train_trace
    

    def cross_entropy_regs(self, model, Yhat, Y, l2_lambda, l1_lambda):    #pred, train_Y
        Y_t = torch.zeros((Y.shape[0], 2)).to(Y)
        Y_t[:, 1] = Y.data
        Y_t[:, 0] = 1 - Y.data
        loss_mean = F.cross_entropy(Yhat, Y_t)
      
        l2_lambda = 0.5
        if l2_lambda is not None:
            l2_reg = torch.tensor(0., requires_grad=True)
            for name, param in model.named_parameters():
                if 'weight' in name and "out_task" in name:
                    l2_reg = l2_reg + torch.norm(param, 2)
            l2_reg = l2_lambda * l2_reg
            loss_mean += l2_reg
        l1_lambda = 0.0005
        if l1_lambda is not None:
            l1_reg = torch.tensor(0., requires_grad=True)
            for name, param in model.named_parameters():
                if 'weight' in name and "out_task" in name:
                    l1_reg = l1_reg + torch.norm(param, 1)
            l1_reg = l1_lambda * l1_reg
            loss_mean += l1_reg
        return loss_mean
    

    def feature_preprocessing(self, model, dataloader_train, dataloader_test):
        model.eval()
        features_np = np.zeros(dataloader_train.dataset.X.shape)
        id2feature = {}

        with torch.no_grad():
            for train_X, train_Y, train_G, _, _, _, idx in dataloader_train:   
                train_X, train_Y, train_G = train_X.float().to(self.device), train_Y.to(self.device), train_G.to(self.device)
                features = model(train_X, id = "0") #
                features_np[idx] = features.data.detach().cpu().numpy()
                features = features.data.detach().cpu().numpy().tolist()
                train_G = train_G.data.detach().cpu().numpy().tolist()
                for i in range(len(features)):
                    if dataloader_train.dataset.subject_id[train_G[i]] not in id2feature:
                        id2feature[dataloader_train.dataset.subject_id[train_G[i]]] = []
                    id2feature[dataloader_train.dataset.subject_id[train_G[i]]].append(features[i]) 

        if self.cfg_proj.save_whitening:
            with torch.no_grad():
                for test_X, test_Y, test_G, _, _, _, idx in dataloader_test:   
                    test_X, test_Y, test_G = test_X.float().to(self.device), test_Y.to(self.device), test_G.to(self.device)
                    features = model(test_X, id = "0") #
                    features = features.data.detach().cpu().tolist()
                    test_G = test_G.data.detach().cpu().tolist()
                    for i in range(len(features)):
                        if dataloader_test.dataset.subject_id[test_G[i]] not in id2feature:
                            id2feature[dataloader_test.dataset.subject_id[test_G[i]]] = []
                        id2feature[dataloader_test.dataset.subject_id[test_G[i]]].append(features[i]) 

            with open("rawdata/id2feature_whitening.p", "wb") as output_file:
                pickle.dump(id2feature, output_file) 

        dataset_train_updated = dataloader_train.dataset

    #for the id
    def basic_train(self, model, dataloader_train, criterion, optimizer, lr_scheduler):
        loss_train_trace = []
        for epoch in range(self.cfg_m.training.epochs):
            model.train()
            loss_epoch = []
            for train_X, train_Y, _ , _, _, _, idx in dataloader_train:   

                train_X, train_Y = train_X.float().to(self.device), train_Y.to(self.device)
                Y_hat = model(train_X, id = "2")
                Y_hat = Y_hat if torch.is_tensor(Y_hat) else Y_hat[1]
                loss = criterion(model, Y_hat, train_Y, l2_lambda = self.cfg_m.l2_lambda, l1_lambda = self.cfg_m.l1_lambda)
                loss_epoch.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                loss_train_trace.append(np.mean(loss_epoch))
        return model, loss_train_trace
    

    def predict(self, model, X, flag_prob = False):
        X = torch.from_numpy(X)

        model.eval()
        with torch.no_grad():
            X = X.float().to(self.device)
            pred = model(X, id = "2")
            pred = pred if torch.is_tensor(pred) else pred[1]
        
        pred = torch.argmax(pred, 1)
        return pred.detach().cpu().numpy() 

    def predict_proba(self, model, X, flag_prob = True):
        X = torch.from_numpy(X)

        model.eval()
        with torch.no_grad():
            X = X.float().to(self.device)
            pred = model(X, id = "2")
            pred = pred if torch.is_tensor(pred) else pred[1]
            pred = torch.nn.functional.softmax(pred, dim = 1)
        
        return pred.detach().cpu().numpy() 

    def eval_func(self, model, x_test, y_test, g_test, g_test_sbj):
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

        g_test_sbj_unique = list(set(g_test_sbj))
        for g in g_test_sbj_unique:
            index = [i for i in range(len(g_test_sbj)) if g_test_sbj[i] == g]
            pred_single_sbj = pred_proba[index]
            
            # get most confidence conversations
            pred_single_sbj = list(pred_single_sbj)
            pred_single_sbj.sort(key = lambda x: -abs(x-0.5))
            pred_single_sbj = pred_single_sbj[:math.ceil(self.cfg_proj.most_confidence*len(pred_single_sbj))]
            
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