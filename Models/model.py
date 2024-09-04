import torch
from torch.utils.data import Dataset
import numpy as np

# Define model
class MLP_pytorch(torch.nn.Module):
    def __init__(self, input_dim, output_dim, classifier = "MLP"):
        super(MLP_pytorch, self).__init__()

        self.classifier = classifier

        if self.classifier == "LR":
            # LR 
            self.linear1 = torch.nn.Linear(input_dim, output_dim)
        else:
            # MLP
            self.linear1 = torch.nn.Linear(input_dim, 32)
            self.relu1 = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(32, output_dim)
    
    def forward(self, x):
        if self.classifier == "LR":
            # LR
            return self.linear1(x)

        # MLP
        outputs = self.linear1(x)
        outputs = self.relu1(outputs)
        outputs = self.linear2(outputs)
        return outputs
    
class LR_pytorch(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LR_pytorch, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.linear1(x) 

class MSE_pytorch(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MSE_pytorch, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.linear1(x) 
    

class MLP_whiting(torch.nn.Module):
    feature_idx = None
    def __init__(self, input_dim, sbj_dim, task_in_dim, task_out_dim, classifier = "MLP"):
        super(MLP_whiting, self).__init__()
        self.feature_mapping = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, input_dim),
        )
        self.out_sbj = torch.nn.Linear(input_dim, sbj_dim)
        self.classifier = classifier
        if self.classifier == "LR":
            self.out_task = torch.nn.Sequential(
                # LR
                torch.nn.Linear(task_in_dim, task_out_dim),
            )
        else:
            self.out_task = torch.nn.Sequential(
                # MLP
                torch.nn.Linear(task_in_dim, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, task_out_dim)
            )
    
    def forward(self, x, id):
        feature = self.feature_mapping(x)
        if id == "0":
            return feature
        elif id == "1":
            return self.out_sbj(feature)
        elif id == "0,1":
            return [feature, self.out_sbj(feature)]
        elif id == "2":
            if self.feature_idx is not None:
                return self.out_task(feature[:, self.feature_idx])
            else:
                return self.out_task(feature)
            

class MLP_whiting_confounders(torch.nn.Module):
    feature_idx = None
    def __init__(self, input_dim, confounders_dim, task_in_dim, task_out_dim, classifier = "MLP"):
        super(MLP_whiting_confounders, self).__init__()
        self.feature_mapping = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, input_dim),
        )
        # self.out_sbj = torch.nn.Linear(input_dim, confounders_dim[0])
        self.out_age = torch.nn.Linear(input_dim, confounders_dim[0])
        self.out_gender = torch.nn.Linear(input_dim, confounders_dim[1])
        self.out_educ = torch.nn.Linear(input_dim, confounders_dim[2])
        self.classifier = classifier
        if self.classifier == "LR":
            self.out_task = torch.nn.Sequential(
                # LR
                torch.nn.Linear(task_in_dim, task_out_dim),
            )
        else:
            self.out_task = torch.nn.Sequential(
                # MLP
                torch.nn.Linear(task_in_dim, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, task_out_dim)
            )
    
    def forward(self, x, id):
        feature = self.feature_mapping(x)
        if id == "0":
            return feature
        elif id == "1":
            return self.out_sbj(feature)
        elif id == "0,1":
            return [feature, self.out_age(feature), self.out_gender(feature), self.out_educ(feature)]
        elif id == "2":
            if self.feature_idx is not None:
                return self.out_task(feature[:, self.feature_idx])
            else:
                return self.out_task(feature)

class CustomDataset(Dataset):
    def __init__(self, X, Y, G):
        self.X = X
        self.Y = np.array(Y)
        self.G = np.zeros(len(G), dtype=np.int64)
        self.subject_id = {}
        g_unique = list(sorted(set(G)))
        for i, g in enumerate(g_unique):
            index = [i for i in range(len(G)) if G[i] == g]
            self.G[index] = i
            self.subject_id[i] = g
 
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.G[idx], idx
    
    def kept(self, idx_kept):
        self.X = self.X[idx_kept]
        self.Y = self.Y[idx_kept]
        self.G = self.G[idx_kept]
        return self
    
class CustomDataset_coufounder(Dataset):
    def __init__(self, X, Y, G):
        self.X = X
        self.Y = np.array(Y)
        self.G = np.zeros(len(G[0]), dtype=np.int64)
        self.G_age = np.zeros(len(G[1]), dtype=np.int64)
        self.G_gender = np.zeros(len(G[2]), dtype=np.int64)
        self.G_educ = np.zeros(len(G[3]), dtype=np.int64)
        self.subject_id = {}
        g_unique = list(sorted(set(G[0])))
        for i, g in enumerate(g_unique):
            index = [i for i in range(len(G[0])) if G[0][i] == g]
            self.G[index] = i
            self.subject_id[i] = g
        g_unique = list(sorted(set(G[1])))
        for i, g in enumerate(g_unique):
            index = [i for i in range(len(G[1])) if G[1][i] == g]
            self.G_age[index] = i
        g_unique = list(sorted(set(G[2])))
        for i, g in enumerate(g_unique):
            index = [i for i in range(len(G[2])) if G[2][i] == g]
            self.G_gender[index] = i
        g_unique = list(sorted(set(G[3])))
        for i, g in enumerate(g_unique):
            index = [i for i in range(len(G[3])) if G[3][i] == g]
            self.G_educ[index] = i
 
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.G[idx], self.G_age[idx], self.G_gender[idx], self.G_educ[idx], idx
    
    def kept(self, idx_kept):
        self.X = self.X[idx_kept]
        self.Y = self.Y[idx_kept]
        self.G = self.G[idx_kept]
        self.G_age = self.G_age[idx_kept]
        self.G_gender = self.G_gender[idx_kept]
        self.G_educ = self.G_educ[idx_kept]
        return self

class CustomDatasetGroup(Dataset):
    def __init__(self, X, Y, G):
        self.X = X
        self.Y = np.array(Y)
        self.G = np.array(G)
 
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.G[idx], idx
    
    def kept(self, idx_kept):
        self.X = self.X[idx_kept]
        self.Y = self.Y[idx_kept]
        return self