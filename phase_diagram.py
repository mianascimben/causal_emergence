"""
Phase diagram creation 
"""
# Importing useful modules/functions
import cupy as np
from plot_functions import heatmap_diagram_phase, plot_diagram_phase
from measures import compute_phase_diagram
from utils import doble_weel_patterns, store_system_information, compute_all_alphas, PhaseParameters
import time



seed = 139390
np.random.seed(seed)

trials = 2

N_spin = 15   # number of spins (MULTIPLE OF THREE)
mappings = [[5,5,5]]#[[3,3,3,3,3],[5,5,5]]

n_maps = len(mappings) if mappings is not None else 0

phase = 'effectiveness'

T_min = 0.01; T_max = 3.01; T_points = 2
temps = np.linspace(T_min, T_max, T_points)       #temperature (not reduce below e^-2 to avoid 'overflow' in the glauber exponent)

# n_memories (array of even numbers)
M_min = 2; M_max = 5; M_add = 2
n_memories = np.arange(M_min, M_max, M_add)
phase_data = np.zeros((trials, len(temps), len(n_memories), n_maps + 1))

phase_parameters = PhaseParameters(
    temps = temps,
    n_memories = n_memories,
    causal_measure = phase,
    mappings = mappings,
    micro = True
    )

patterns_list = [doble_weel_patterns(N_spin, M_max) for _ in range(trials)]

t_0 = time.time()

for k in range(trials): 
    phase_data[k] = compute_phase_diagram(phase_parameters, patterns_list[k])#, mapping = mappings


running_time = time.time() - t_0
print("Time:", running_time)

# average over different nominal patterns 
phase_data_averaged = np.mean(phase_data, axis=0)

#%%

info_system = store_system_information(N_spin, T = 'range', N_macro = N_spin, groups = 'None')

all_alphas = compute_all_alphas(N_spin, n_memories, mappings) # size (len(n_memories), len(mappings))
alpha_max = all_alphas.max()
        
fig, ax = heatmap_diagram_phase(
    all_alphas[:,0],
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
        all_alphas[:,0],
        temps,
        phase_data_averaged[:,:, k+1],
        info_system,
        title="Phase Diagram",
        x_label="alpha",
        y_label="T",
        cbar_label= ' '.join(['mean', phase]) 
        )      
    
    
# plot showing eff(alphas) for different T and for all mappings
fig, ax = plot_diagram_phase(
     temps,
     all_alphas,
     phase_data_averaged,
     mappings,
     title="Phase Diagram of effectiveness for different mappings system",
     x_label="alpha",
     y_label=' '.join(['mean', phase]),
     cbar_label= 'Temperatures'
)   

#%% show single trials

k = 2
for i in range(trials):
     fig, ax = heatmap_diagram_phase(
         all_alphas[:, k],
         temps,
         phase_data[i, :, :, k],
         title="Phase Diagram macro system",
         x_label="alpha",
         y_label="T",
         cbar_label= phase
    )    
   
