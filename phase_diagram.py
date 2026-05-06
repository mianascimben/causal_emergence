"""
Phase diagram creation 
"""
# Importing useful modules/functions
import numpy as np
from plot_functions import heatmap_diagram_phase, plot_diagram_phase
from measures import compute_phase_diagram
from utils import doble_weel_patterns, store_system_information, compute_all_alphas, PhaseParameters
import time

seed = 139390
np.random.seed(seed)

trials = 20

N_spin = 15   # number of spins (MULTIPLE OF THREE)
mappings = [[3,3,3,3,3],[5,5,5],[13,1,1]]
micro = True

n_maps = len(mappings) if mappings is not None else 0

phase = 'effectiveness'

T_min = 0.01; T_max = 3.01; T_points = 5
temps = np.linspace(T_min, T_max, T_points) 


# n_memories (array of even numbers)
M_min = 2; M_max = 20; M_add = 2
n_memories = np.arange(M_min, M_max, M_add)
alphas = n_memories/N_spin

# the number of patterns that have the first and last spins up
first_group_members = 10

phase_data = np.zeros((trials, len(temps), len(n_memories), n_maps + 1))

phase_parameters = PhaseParameters(
    temps = temps,
    n_memories = n_memories,
    causal_measure = phase,
    mappings = mappings, 
    micro = micro)

patterns_list = [doble_weel_patterns(N_spin, M_max, first_group_members) for _ in range(trials)]

t_0 = time.time()

for k in range(trials): 
    phase_data[k] = compute_phase_diagram(phase_parameters, patterns_list[k])#, mapping = mappings
    print(f'check point: trial number {k} executed')

running_time = time.time() - t_0
print("Time:", running_time)

# average over different nominal patterns 
phase_data_averaged = np.mean(phase_data, axis=0)
    
 
#%% This part is to print the phase diagram as a Heatmap

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


#%% show single trials

k = 1 # which mapping to plot
for i in range(10):
     fig, ax = heatmap_diagram_phase(
         alphas,
         temps,
         phase_data[i,:,:,k],
         title="Phase Diagram macro system trial 0",
         x_label="alpha",
         y_label="T",
         cbar_label= phase
    )    
   
