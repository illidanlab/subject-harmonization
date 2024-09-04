from Solvers.Solver_Base import Solver_Base
import torch
from torch.utils.data import DataLoader
from Models.model import MLP_pytorch, CustomDataset

class Standard_solver(Solver_Base):
    
    def __init__(self, cfg_proj, cfg_m, name = "Std"):
        Solver_Base.__init__(self, cfg_proj, cfg_m, name)
        
    def run(self, x_train, y_train, g_train, x_test, y_test, g_test, seed):
        # Set seed
        self.set_random_seed(seed)
        
        # Initialize
        dataloader_train = DataLoader(CustomDataset(x_train, y_train, g_train), batch_size = self.cfg_m.training.batch_size, drop_last=True, shuffle = True)
        model = MLP_pytorch(input_dim = len(x_train[0]), output_dim = self.cfg_m.data.dim_out, classifier = self.cfg_proj.classifier) 
        optimizer = torch.optim.AdamW(model.parameters(), lr = self.cfg_m.training.lr_init)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(self.cfg_m.training.epochs*len(dataloader_train)))   #very useful
        criterion = self.cross_entropy_regs
        model, _ = self.to_parallel_model(model)
        
        # Train classifier
        model, loss_train_trace = self.basic_train(model, dataloader_train, criterion, optimizer, lr_scheduler)

        # Evaluation
        auc, f1, sens, spec, auc_sbj, f1_sbj, sens_sbj, spec_sbj = self.eval_func(model, x_test, y_test, g_test)
        
        return auc, f1, sens, spec, auc_sbj, f1_sbj, sens_sbj, spec_sbj