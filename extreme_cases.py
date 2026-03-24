import numpy as np
from plot_functions import *
from measures import causal_emergence, compute_all_measures
import matplotlib.pyplot as plt
from utils import store_system_information


n_states = 4 # number of states

degenerated_matrix = (np.zeros((n_states,n_states)).T + np.array([1,0,0,0])) # first state degeneretion

noisy_matrix = np.ones((n_states ,n_states)) / n_states

causal_matrix = np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]])

measures = ['determinism', 'degeneracy', 'effectiveness', 'EI']



results_degenerated = compute_all_measures(degenerated_matrix, n_states)

results_noisy =  compute_all_measures(noisy_matrix, n_states)

results_perfect = compute_all_measures(causal_matrix, n_states)

#%%
def map_micro_to_macro(block_sizes): 
    '''
    This is the mapping from the micro phace space to the macro phace space. 
    
    Returns:
    -------
      macro_idx: array_like
        This array associates to each micro state
        (identified by its position along 'macro_idx') the index referring 
        to the macro state it maps into. 
    '''
    macro_idx = np.concatenate([
        np.full(size, i) for i, size in enumerate(block_sizes)
    ])

    return macro_idx

def build_projection_matrix(n_micro_states, n_macro_states, block_sizes):

    M = np.zeros((n_micro_states, n_macro_states))

    # set 1 in correspondece of the mapping from a micro configuration 
    # to a macro configuration
    macro_idx = map_micro_to_macro(block_sizes)
    M[np.arange(n_micro_states), macro_idx] = 1                              

    return M

def coarse_graining_tpm(TPM, block_sizes):
    
    n_micro_states = np.shape(TPM)[0]
    n_macro_states = len(block_sizes)
    
    M = build_projection_matrix(n_micro_states, n_macro_states, block_sizes)

    TPM_macro = M.T @ TPM @ M                                
    
    # counts the number of multiple micro realizations of each macro configuration
    macro_idx = map_micro_to_macro(block_sizes)
    counts = np.bincount(macro_idx)
    
    TPM_macro /= counts[:, None]                                        
    
    return TPM_macro  

#%%  Example of positive CE

n_micro_states = 9
T = 0.5         #temperature
block_sizes= [3,3,3]
n_macro_states = len(block_sizes)

system_info = store_system_information(n_micro_states, T, n_macro_states, block_sizes)


b1 = np.array([[0.5, 0, 0.5],[0.3, 0.3, 0.3],[0.8, 0.1, 0.1]])
TPM_block_micro = np.block([[b1/2,            np.zeros((3,3)),np.zeros((3,3))],
                            [np.zeros((3,3)), b1,             np.zeros((3,3))],
                            [np.zeros((3,3)), np.zeros((3,3)),b1]])

TPM_micro_im = heatmap(TPM_block_micro, title = 'TPM for micro system', cbar_label ='Probability')
annotate_heatmap(TPM_micro_im)
plt.show()

TPM_block_macro = coarse_graining_tpm(TPM_block_micro, block_sizes)

TPM_macro_im = heatmap(TPM_block_macro, title = 'TPM for macro system', cbar_label ='Probability')
annotate_heatmap(TPM_macro_im)
plt.show()

measures_micro_block = compute_all_measures(TPM_block_micro, n_micro_states)
measures_macro_block = compute_all_measures(TPM_block_macro, n_macro_states)

CE_data = causal_emergence(measures_micro_block, measures_macro_block)
print(CE_data)

plot_measures_subplots(measures_micro_block, measures_macro_block, CE_data, system_info)

