import numpy as np
from plot_functions import *
from measures import *
from ising import IsingSystem
from dynamics import GlauberDynamics
from coarsegraining import CoarseGraining
from utils import store_system_information 

N_spin = 6     # number of spins
T = 0.001         #temperature
beta = 1/T      # inverse temperature
block_sizes= [1,3,1,1]
N_macro_spin = len(block_sizes)
patterns = np.array([[1,1,1,1,1,1],
                     [1,1,1,-1,-1,-1]])
                     
#patterns = np.array([[1,1,1],[1,1,-1],[1,-1,1]])

system_info = store_system_information(N_spin, T, N_macro_spin, block_sizes)

show_J = True
show_micro_TPM = True

micro_system = IsingSystem(N_spin, patterns, beta)

micro_dynamics = GlauberDynamics(micro_system, flips=6)

if show_J == True:
    J_im = heatmap(micro_system.J, title='J for micro system', cbar_label = 'Interaction strength')
    annotate_heatmap(J_im) 
    plt.show()

if show_micro_TPM == True:
    if N_spin > 8: 
        pass
    else: 
        micro_TPM_im = heatmap(micro_dynamics.TPM, micro_system.configs, title = 'TPM for micro system', cbar_label ='Probability')
        annotate_heatmap(micro_TPM_im)
        plt.show()


coarsed_system = CoarseGraining(micro_system, micro_dynamics, patterns, block_sizes)  #questo passaggio non mi piace. non salva robe utili che system o dynamics non abbia se non i macro patterns

macro_system, macro_dynamics = coarsed_system.build_macro_model()

TPM_macro_im = heatmap(macro_dynamics.TPM, macro_system.configs, title = 'TPM for macro system', cbar_label ='Probability')
annotate_heatmap(TPM_macro_im)
plt.show()

#%%
micro_measures = compute_all_measures(micro_dynamics.TPM, micro_system.n_configs)
macro_measures = compute_all_measures(macro_dynamics.TPM, macro_system.n_configs)

CE_data = causal_emergence(micro_measures, macro_measures)

results_im = plot_measures_subplots(micro_measures, macro_measures, CE_data, system_info)
