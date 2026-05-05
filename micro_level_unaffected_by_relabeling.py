# -*- coding: utf-8 -*-
"""
check that micro system ordered and micro system shuffled are the same system
"""

import numpy as np
import pandas as pd
from measures import *
from ising import IsingSystem
from dynamics import GlauberDynamics
from utils import  doble_weel_patterns, relabeling, shuffle_patterns
seed = 139390
N_spin = 15   # number of spins (MULTIPLE OF THREE)
M = 20        # memorized patterns 
T = 0.01         #temperature (not reduce below e^-2 to avoid 'overflow' in the glauber exponent)
beta = 1/T      # inverse temperature
np.random.seed(seed)

# consider a single random relabeling
relabel = relabeling(N_spin, 10)[8]

patterns = doble_weel_patterns(N_spin, M)

# ordered micro system
sm = IsingSystem(N_spin, patterns, beta)

dm = GlauberDynamics(sm)

micro_measures = CausalMeasure(dm.TPM, sm.n_configs).compute_all_measures(label = 'micro ordered')


# shuffled micro system
patterns_shuffled = shuffle_patterns(patterns, relabel)

sm_s = IsingSystem(N_spin, patterns_shuffled, beta)

dm_s = GlauberDynamics(sm_s)

micro_measures_s = CausalMeasure(dm_s.TPM, sm_s.n_configs).compute_all_measures(label = 'micro shuffle')

measures_by_scale = pd.concat([micro_measures, micro_measures_s])

# compare the causal measures obtained by each micro system and check if the results are equal
if np.allclose(micro_measures, micro_measures_s):
    print ("The spin relabeling procedure does not affect the micro system")