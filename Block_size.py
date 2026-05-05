import numpy as np
from plot_functions import *
from measures import *
from ising import IsingSystem
from dynamics import GlauberDynamics
from coarsegraining import CoarseGraining
from utils import store_system_information 
from collections import defaultdict

N_spin = 6     # number of spins
T = 0.001         #temperature
beta = 1/T      # inverse temperature
n_flips = 1      # number of flips at time 

list_block_sizes= ([3,1,1,1],[1,3,1,1],[1,1,3,1],[1,1,1,3])
group_labels = ['1','2','3','4']
patterns = np.array([[1,1,1,1,1,1],
                     [1,1,1,-1,-1,-1]])

# create micro system and get its TPM
micro_system = IsingSystem(N_spin, patterns, beta)
    
micro_dynamics = GlauberDynamics(micro_system, flips = n_flips)
 
# create dictionary where to store the results w.r.t. different groupings 
macro_measures_vs_groupping = defaultdict(list)

for block_sizes in list_block_sizes:
    
    N_macro_spin = len(block_sizes)
    system_info = store_system_information(N_spin, T, N_macro_spin, block_sizes)
    
    show_J = False
    show_micro_TPM = False
   
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
    
    micro_measures, _ = compute_all_measures(micro_dynamics.TPM, micro_system.n_configs)
    macro_measures, _ = compute_all_measures(macro_dynamics.TPM, macro_system.n_configs)
    
    #CE_data = causal_emergence(micro_measures, macro_measures)
    
    for key, value in macro_measures.items():
        macro_measures_vs_groupping[key].append(value)
        
        
results_im = plot_measures_subplots(micro_measures, macro_measures_vs_groupping, system_info = system_info, scale_labels = group_labels)
