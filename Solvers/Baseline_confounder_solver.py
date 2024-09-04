import numpy as np
from Solvers.Solver_Base import Solver_Base
import torch
from torch.utils.data import DataLoader
from Models.model import MSE_pytorch, CustomDataset, LR_pytorch

class Baseline_confounder_solver(Solver_Base):
    
    def __init__(self, cfg_proj, cfg_m, name = "baseline"):
        Solver_Base.__init__(self, cfg_proj, cfg_m, name)
        
    def run(self, x_train, y_train, g_train, x_test, y_test, g_test, seed):
        # Set seed
        self.set_random_seed(seed)
        
        # train for confounder classifier
        epochs = 50
        X_confounder, X_test_confounder = x_train[:, -3:], x_test[:, -3:]
        
        for i in range(x_train.shape[-1] - 3):
            Y = x_train[:, i]
            dataloader_train_c = DataLoader(CustomDataset(X_confounder, Y, g_train), batch_size = self.cfg_m.training.batch_size, drop_last=True, shuffle = True)
            model_confounder = MSE_pytorch(input_dim = len(X_confounder[0]), output_dim = 1)
            optimizer_c = torch.optim.AdamW(model_confounder.parameters(), lr = self.cfg_m.training.lr_init)
            lr_scheduler_c = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_c, int(epochs*len(dataloader_train_c)))   #very useful
            criterion_c = torch.nn.MSELoss()
            model_confounder, _ = self.to_parallel_model(model_confounder)
            model_confounder, _ = self.basic_train_confounder(model_confounder, dataloader_train_c, criterion_c, optimizer_c, lr_scheduler_c, epochs)

            model_confounder.eval()
            with torch.no_grad():
                x_train_task_pred = model_confounder(torch.from_numpy(X_confounder).float().to(self.device))
                x_train_task_pred = x_train_task_pred.data.detach().cpu().numpy().flatten()
            x_train[:, i] = x_train[:, i] - x_train_task_pred
            with torch.no_grad():
                x_test_task_pred = model_confounder(torch.from_numpy(X_test_confounder).float().to(self.device))
                x_test_task_pred = x_test_task_pred.data.detach().cpu().numpy().flatten()
            x_test[:, i] = x_test[:, i] - x_test_task_pred

        # train for task classifier
        x_train, x_test = x_train[:, :-3], x_test[:, :-3]
        dataloader_train = DataLoader(CustomDataset(x_train, y_train, g_train), batch_size = self.cfg_m.training.batch_size, drop_last=True, shuffle = True)
        model_task = LR_pytorch(input_dim = len(x_train[0]), output_dim = self.cfg_m.data.dim_out)
        optimizer = torch.optim.AdamW(model_task.parameters(), lr = self.cfg_m.training.lr_init)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(self.cfg_m.training.epochs*len(dataloader_train)))   #very useful
        criterion = self.cross_entropy_regs
        model_task, _ = self.to_parallel_model(model_task)
        model_task, loss_train_trace = self.basic_train(model_task, dataloader_train, criterion, optimizer, lr_scheduler)

        # Evaluation
        auc, f1, sens, spec, auc_sbj, f1_sbj, sens_sbj, spec_sbj = self.eval_func(model_task, x_test, y_test, g_test)
        
        return auc, f1, sens, spec, auc_sbj, f1_sbj, sens_sbj, spec_sbj
    
    def basic_train_confounder(self, model, dataloader_train, criterion, optimizer, lr_scheduler, epochs):
        loss_train_trace = []
        for epoch in range(epochs):
            model.train()
            loss_epoch = []
            for train_X, train_Y, _ , idx in dataloader_train:   

                train_X, train_Y = train_X.float().to(self.device), train_Y.float().to(self.device)
                Y_hat = model(train_X)
                loss = criterion(train_Y, Y_hat.flatten())
                loss_epoch.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                loss_train_trace.append(np.mean(loss_epoch))
        return model, loss_train_trace