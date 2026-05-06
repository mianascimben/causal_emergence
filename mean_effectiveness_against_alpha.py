# -*- coding: utf-8 -*-
"""
Effectiveness v/s alpha at minimum T=0.01
"""

# Importing useful modules/functions
import numpy as np
from plot_functions import heatmap_diagram_phase, plot_diagram_phase
from measures import compute_phase_diagram
from utils import doble_weel_patterns, PhaseParameters
import time

seed = 139390
np.random.seed(seed)

trials = 20

N_spin = 15   # number of spins (MULTIPLE OF THREE)
mappings = [[3,3,3,3,3],[5,5,5],[13,1,1]]
micro = False

n_maps = len(mappings) if mappings is not None else 0

phase = 'effectiveness'

#T_min = 0.01; T_max = 3.01; T_points = 5
#temps = np.linspace(T_min, T_max, T_points) 
T= 0.01      
temps = np.array([T])

# n_memories (array of even numbers)
M_min = 2; M_max = 60; M_add = 4
n_memories = np.arange(M_min, M_max, M_add)
alphas = n_memories/N_spin # parameter to plot in the x axis

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

    
# plots eff v/s alphas for all mappings
fig, ax = plot_diagram_phase(
     temps,
     alphas,
     phase_data_averaged,
     mappings,
     title="Effectiveness v/s alpha for different mapping methods",
     x_label="alpha",
     y_label=' '.join(['mean', phase]),
     cbar_label= 'Temperatures'
)   
