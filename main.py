import numpy as np
import pandas as pd
from plot_functions import *
from measures import *
from ising import IsingSystem
from dynamics import GlauberDynamics
from coarsegraining import CoarseGraining
from utils import store_system_information, concatenate_data, doble_weel_patterns, random_patterns
N_spin = 9   # number of spins (MULTIPLE OF THREE)
M = 4        # memorized patterns 
T = 3         #temperature (not reduce below e^-2 to avoid 'overflow' in the glauber exponent)
beta = 1/T      # inverse temperature
mapping_list = [[3,3,3],[5,3,1],[7,1,1]]
#[[5,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,5,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,5]]
np.random.seed(seed)

patterns = random_patterns(N_spin, M)

sm = IsingSystem(N_spin, patterns, beta)

dm = GlauberDynamics(sm)

micro_measures = CausalMeasure(dm.TPM, sm.n_configs).compute_all_measures(label = 'micro')

measures_by_scale = micro_measures
for i, mapping in enumerate(mapping_list): 
    
    N_macro_spin = len(mapping)
    system_info = store_system_information(N_spin, T, N_macro_spin, groups = 'multiple groups')
    
    cs = CoarseGraining(sm, mapping, dm)  #questo passaggio non mi piace. non salva robe utili che system o dynamics non abbia se non i macro patterns

    sM, dM = cs.build_macro_model()
        
    group_label = ','.join(str(x) for x in mapping)
    macro_measures  = CausalMeasure(dM.TPM, sM.n_configs).compute_all_measures(label = group_label)
    
    J_im = heatmap(dM.TPM, title='M@TPM for macro system', cbar_label = 'Interaction strength')
    annotate_heatmap(J_im) 
    plt.show()

    
    measures_by_scale = pd.concat([measures_by_scale, macro_measures])
    
show_J = False
show_micro_TPM = False
show_macro_TPM = False

if show_J == True:
    J_im = heatmap(sm.J, title='J for micro system', cbar_label = 'Interaction strength')
    annotate_heatmap(J_im) 
    plt.show()

if show_micro_TPM == True:
    if N_spin > 8: 
        pass
    else: 
        micro_TPM_im = heatmap(dm.TPM, sm.configs, title = 'TPM for micro system', cbar_label ='Probability')
        annotate_heatmap(micro_TPM_im)
        plt.show()

if show_macro_TPM == True:
    if N_macro_spin > 8: 
        pass
    else: 
        TPM_macro_im = heatmap(dM.TPM, sM.configs, title = 'TPM for macro system', cbar_label ='Probability')
        annotate_heatmap(TPM_macro_im)
        plt.show()
#%%       
CE_data_v = causal_emergence_vectorized(measures_by_scale)

results_im = plot_measures_subplots(measures_by_scale, CE_data_v,
                                    system_info,
                                    title = 'Causal Measures Across Scales')







#%% dynamics
# =============================================================================
# steps = N_spin
# N_configs = 2**N_spin
# np.random.seed(seed)
# initial_state = np.random.choice([-1,1], N_spin)
# 
# #uniform_micro_distribution = 1/N_configs * np.ones(N_configs)
# #evolved_micro_distribution = dm.evolution(uniform_micro_distribution, steps=3)
# 
# trajectory = dm.run_with_monitoring(initial_state, steps)
# trajectory_array = np.asarray(trajectory)
# overlap = compute_overlap_matrix(patterns, trajectory_array)
# overlap_im = plot_state_sequence_and_overlap(trajectory_array, overlap)
# plt.show()
# plt.plot(evolved_micro_distribution, linewidth = 0.3)
# #
# =============================================================================
    