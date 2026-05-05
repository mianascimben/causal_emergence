# -*- coding: utf-8 -*-
"""
Compute the hamming distance

WORK IN PROGRESS
"""
import numpy as np
from measures import hamming_distance
from ising import IsingSystem
from dynamics import GlauberDynamics
from coarsegraining import CoarseGraining
from utils import store_system_information, concatenate_data, doble_weel_patterns, random_patterns

N_spin = 9   # number of spins (MULTIPLE OF THREE)
M = 4        # memorized patterns 
T = 1         #temperature (not reduce below e^-2 to avoid 'overflow' in the glauber exponent)
beta = 1/T      # inverse temperature
mapping_list = [[3,3,3],[5,3,1],[7,1,1]]

patterns = random_patterns(N_spin, M)

sm = IsingSystem(N_spin, patterns, beta)
dm = GlauberDynamics(sm)

cs = CoarseGraining(sm, dm, patterns, mapping)  #questo passaggio non mi piace. non salva robe utili che system o dynamics non abbia se non i macro patterns

micro_realizations = {}
# group micro configs depending on the macro state they map into 
for idx in len(mapping):
    micro_realizations[idx] = list(cs.micro_configs[cs.macro_idx==idx])

n_macro_configs = cs.macro_idx.max()

distance = 0
for I in range(n_macro_configs):
    for J in range(I, n_macro_configs):
        for i in len(micro_realizations[I]):
            for j in len(micro_realizations[J]):
                distance += hamming_distance(micro_realizations[I][i], micro_realizations[J][j])