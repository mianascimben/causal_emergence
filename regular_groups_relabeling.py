# -*- coding: utf-8 -*-
"""
Analyze the relabeling influence on the effectiveness
"""

import numpy as np
import pandas as pd
from plot_functions import *
from measures import *
from ising import IsingSystem
from dynamics import GlauberDynamics
from coarsegraining import CoarseGraining
from utils import store_system_information, doble_weel_patterns, shuffle_spin_order, relabeling

seed = 139390
np.random.seed(seed)

N_spin = 15   # number of spins (MULTIPLE OF THREE)
M = 20        # memorized patterns 
T = 0.01         #temperature (not reduce below e^-2 to avoid 'overflow' in the glauber exponent)
beta = 1/T      # inverse temperature

#creation of the nominal patterns
patterns = doble_weel_patterns(N_spin, M)

mapping = [3,3,3,3,3]

N_macro_spin = len(mapping)
system_info = store_system_information(N_spin, T, N_macro_spin, groups = mapping)

# number of relabels that will be analyzed
n_relabelings = 5
# cration of the list of spin relabelings

relabeling_list = relabeling(N_spin, n_relabelings, seed)

# micro system
sm = IsingSystem(N_spin, patterns, beta)
dm = GlauberDynamics(sm)
micro_measures = CausalMeasure(dm.TPM, sm.n_configs).compute_all_measures(label = 'micro')

measures_by_relabaling = micro_measures

# macro system without relabeling
coarsed_system = CoarseGraining(sm, dm, patterns, mapping)  #questo passaggio non mi piace. non salva robe utili che system o dynamics non abbia se non i macro patterns

sM, dM = coarsed_system.build_macro_model()
 
group_label = 'None'
macro_measures  = CausalMeasure(dM.TPM, sM.n_configs).compute_all_measures(label = group_label)

measures_by_relabaling = pd.concat([measures_by_relabaling, macro_measures])

r = 0 # variable to number the different relabeling in the plot
for relabel in relabeling_list: 
    

    coarsed_system = CoarseGraining(sm, dm, patterns, mapping, relabel)  #questo passaggio non mi piace. non salva robe utili che system o dynamics non abbia se non i macro patterns
    
    sM, dM = coarsed_system.build_macro_model()
     
    group_label = ''.join(['r ', str(r)])
    macro_measures  = CausalMeasure(dM.TPM, sM.n_configs).compute_all_measures(label = group_label)
    
    measures_by_relabaling = pd.concat([measures_by_relabaling, macro_measures])
    r += 1

CE_data_v = causal_emergence_vectorized(measures_by_relabaling)

results_im = plot_measures_subplots(measures_by_relabaling, CE_data_v, system_info, title = 'Causal measures for different groupings')


