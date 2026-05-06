"""
phase diagram Hopfield model 
"""

import numpy as np
from plot_functions import heatmap_diagram_phase, plot_diagram_phase
from measures import compute_overlap_matrix, compute_mean_overlap
from ising import IsingSystem
from coearsegraining import CoarseGraining
from utils import doble_weel_patterns, PhaseParameters, store_system_information
import time

trials = 20
N_spin = 15   # number of spins (MULTIPLE OF THREE)
mappings = [[3,3,3,3,3],[5,5,5],[13,1,1]]
micro = True 
phase = 'overlap'
T_min = 0.01; T_max = 3.01; T_points = 5
temps = np.linspace(T_min, T_max, T_points) 

# n_memories (array of even numbers)
M_min = 2; M_max = 20; M_add = 2
n_memories = np.arange(M_min, M_max, M_add)
alphas = n_memories/N_spin

# the number of patterns that have the first and last spins up
first_group_members = 10

n_maps = len(mappings) if mappings is not None else 0

global_overlap = np.zeros((trials, len(temps), len(n_memories), n_maps + 1))
patterns_list = [doble_weel_patterns(N_spin, M_max, first_group_members) for _ in range(trials)]

t_0 = time.time()

for p, patterns in enumerate(patterns_list):
    
    for i, T in enumerate(temps):
        beta = 1/T
        
        for j, M in enumerate(n_memories):
            sm = IsingSystem(N_spin, patterns[:M], beta)
            
            if micro == True:
                boltzmann_pdf = sm.canonical_distribution()
                half_configs = sm.configs[:int(sm.n_configs/2)]
                overlap_matrix = compute_overlap_matrix(patterns[:M], half_configs)
                _, global_overlap[p, i, j, 0] = compute_mean_overlap(overlap_matrix, 
                                                                        boltzmann_pdf)
            else:  global_overlap[p, i, j, 0] = np.nan
            
            if mappings is not None:
                for k, mapp in enumerate(mappings):
                    cs = CoarseGraining(sm, mapp)
                    sM = cs.build_macro_system()
                    boltzmann_pdf = sM.canonical_distribution()
                    half_configs = sM.configs[:int(sM.n_configs/2)]
                    overlap_matrix = compute_overlap_matrix(patterns[:M], half_configs)
                    _, global_overlap[p, i, j, k + 1]  = compute_mean_overlap(overlap_matrix, 
                                                                              boltzmann_pdf)
    print(f'check point: trial number {k} executed')

                                                                              
running_time = time.time() - t_0
print("Time:", running_time)
# average over different nominal patterns 
phase_data_averaged = np.mean(global_overlap, axis=0)

                   

info_system = store_system_information(N_spin, T = 'range', N_macro = N_spin, groups = 'None')

fig, ax = heatmap_diagram_phase(
    alphas,
    temps,
    phase_data_averaged[:,:,0],
    info_system,
    title="Phase Diagram",
    x_label="alpha",
    y_label="T",
    cbar_label= ' '.join(['mean', phase])
)

for k, mapp in enumerate(mappings):
    N_macro = len(mapp)
    
    info_system = store_system_information(N_spin, T = 'range', N_macro = N_macro, groups = str(mapp))
    fig, ax = heatmap_diagram_phase(
        alphas,
        temps,
        phase_data_averaged[:,:, k+1],
        info_system,
        title="Phase Diagram",
        x_label="alpha",
        y_label="T",
        cbar_label= ' '.join(['mean', phase]) 
        )      
