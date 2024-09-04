import sys

#solver load
from Solvers.Standard_solver import Standard_solver
from Solvers.subject_harmonization_solver import subject_harmonization_solver
from Solvers.confounder_harmonization_solver import confounder_harmonization_solver
from Solvers.Baseline_confounder_solver import Baseline_confounder_solver

#solver_loader = lambda cfg_proj, cfg_m : getattr(sys.modules[__name__], cfg_proj.solver)(cfg_proj, cfg_m)

def solver_loader(cfg_proj, cfg_m):
    if cfg_proj.solver == "Standard_solver":
        s = Standard_solver(cfg_proj, cfg_m)
    elif cfg_proj.solver == "subject_harmonization_solver":
        s = subject_harmonization_solver(cfg_proj, cfg_m)
    elif cfg_proj.solver == "Baseline_confounder_solver":
        s = Baseline_confounder_solver(cfg_proj, cfg_m)
    elif cfg_proj.solver == "confounder_harmonization_solver":
        s = confounder_harmonization_solver(cfg_proj, cfg_m)
    return s