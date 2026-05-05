

"""
Phase diagram creation parallelizing the parameters
"""
# Importing useful modules/functions
import numpy as np
from plot_functions import heatmap_diagram_phase, plot_diagram_phase
from measures import function
from utils import doble_weel_patterns, store_system_information, compute_all_alphas, PhaseParameters
import time
from concurrent.futures import ProcessPoolExecutor



if __name__ == "__main__":
    
    seed = 139390
    np.random.seed(seed)
    
    trials = 4
    N_spin = 12   # number of spins (MULTIPLE OF THREE)
    mappings = [[3,3]]#[[3,3,3,3,3],[5,5,5]]  
    micro = True
    phase = 'effectiveness'
    
    T_min = 0.01; T_max = 3.01; T_points = 5
    temps = np.linspace(T_min, T_max, T_points)       #temperature (not reduce below e^-2 to avoid 'overflow' in the glauber exponent)
    
    # n_memories (array of even numbers)
    M_min = 2; M_max = 10; M_add = 2
    n_memories = np.arange(M_min, M_max, M_add)
     
    n_maps = len(mappings) if mappings is not None else 0
    #phase_data = np.zeros((trials, len(temps), len(n_memories), n_maps + 1))

    patterns_list = [doble_weel_patterns(N_spin, M_max) for _ in range(trials)]
    parameters = [(T, M) for T in temps for M in n_memories]
    parallell_tasks = len(parameters)
    t_0 = time.perf_counter()
    
    with ProcessPoolExecutor() as executor:
       results = list(executor.map(
            function,
            parameters, 
            patterns_list*parallell_tasks,
            [phase]*parallell_tasks,
            [mappings] * parallell_tasks, 
            [micro] * parallell_tasks))

    phase_data = np.stack(results)
    running_time = time.perf_counter()-t_0

    print("Time:", running_time)
    # average over different nominal patterns 
    phase_data_averaged = np.mean(phase_data, axis=0)

#%%    
# =============================================================================
#     
# # average over different nominal patterns 
# phase_data_averaged = np.mean(phase_data, axis=0)
# 
#  
# 
# info_system = store_system_information(N_spin, T = 'range', N_macro = N_spin, groups = 'None')
# 
# all_alphas = compute_all_alphas(N_spin, n_memories, mappings) # size (len(n_memories), len(mappings))
# alpha_max = all_alphas.max()
#         
# fig, ax = heatmap_diagram_phase(
#     all_alphas[:,0],
#     temps,
#     phase_data_averaged[:,:,0],
#     info_system,
#     title="Phase Diagram",
#     x_label="alpha",
#     y_label="T",
#     cbar_label= ' '.join(['mean', phase])
# )
# 
# for k, mapp in enumerate(mappings):
#     N_macro = len(mapp)
#     
#     info_system = store_system_information(N_spin, T = 'range', N_macro = N_macro, groups = str(mapp))
#     fig, ax = heatmap_diagram_phase(
#         all_alphas[:,0],
#         temps,
#         phase_data_averaged[:,:, k+1],
#         info_system,
#         title="Phase Diagram",
#         x_label="alpha",
#         y_label="T",
#         cbar_label= ' '.join(['mean', phase]) 
#         )      
#     
#     
# # plot showing eff(alphas) for different T and for all mappings
# fig, ax = plot_diagram_phase(
#      temps,
#      all_alphas,
#      phase_data_averaged,
#      mappings,
#      title="Phase Diagram of effectiveness for different mappings system",
#      x_label="alpha",
#      y_label=' '.join(['mean', phase]),
#      cbar_label= 'Temperatures'
# )   
# 
# =============================================================================

   
