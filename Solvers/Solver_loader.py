import sys

#solver load
from Solvers.Standard_solver import Standard_solver
from Solvers.whiting_solver import whiting_solver
from Solvers.whiting_confounder_solver import whiting_confounder_solver
from Solvers.whiting_confounderS_solver import whiting_confounderS_solver
from Solvers.Baseline_confounder_solver import Baseline_confounder_solver

#solver_loader = lambda cfg_proj, cfg_m : getattr(sys.modules[__name__], cfg_proj.solver)(cfg_proj, cfg_m)

def solver_loader(cfg_proj, cfg_m):
    if cfg_proj.solver == "Standard_solver":
        s = Standard_solver(cfg_proj, cfg_m)
    elif cfg_proj.solver == "whiting_solver":
        s = whiting_solver(cfg_proj, cfg_m)
    elif cfg_proj.solver == "Baseline_confounder_solver":
        s = Baseline_confounder_solver(cfg_proj, cfg_m)
    elif cfg_proj.solver == "whiting_confounder_solver":
        s = whiting_confounder_solver(cfg_proj, cfg_m)
    elif cfg_proj.solver == "whiting_confounderS_solver":
        s = whiting_confounderS_solver(cfg_proj, cfg_m)
    return s

