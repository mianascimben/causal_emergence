import numpy as np 
import pandas as pd
import networkx as nx

class PhaseParameters:
    '''
    Store the informations necessary for the computation of the phase diagram.
    An instance of this class should be passed into 'compute_phase_diagram' 
    function of the 'measure' module.
    '''
    
    def __init__(self, temps, n_memories,  
                 causal_measure = None, mappings = None,
                 micro = True):
        '''
        Parameters:
        -----------
            temps: array-like of float
                range of temperatures 
            n_memories: array-like of int
                range of the number of memories to store in the system
            causal_measure: function 
                Choose between 'degeneracy', 'determinism', 'effectivness' or 'EI'
            mappings: list of lists, optional 
                the mappings for the coarse-graining. Default None
            micro: bool, optional 
                Determins the computation of the phase diagram of the micro system. 
                If just a coearsenphase diagram is need, then setting micro = False
                avoids heavy and useless computations.
        '''
        self.temps = temps
        self.n_memories = n_memories
        
        # compute or not the phase diagram for the micro system
        self.micro = micro 

        if mappings is not None:
            self.mappings = mappings 
        if causal_measure is not None:
            self.causal_measure = causal_measure

def store_system_information(N_micro, T, N_macro, groups = ''):
    "Create a dictionary for storing the system information."
    return {'N_m': N_micro,
            'Temperature': T,
            'N_M': N_macro,
            'mapping': groups}
            
def indexing_configs(configs):
    bits = (configs + 1) // 2
    powers = 2**np.arange(configs.shape[1]-1, -1, -1)

    return bits @ powers

def concatenate_data(data):
        
    return pd.concat(data)

def create_graph(J):
    return nx.from_numpy_array(J)

def doble_weel_patterns(N_spin, M, first_group_members):
    '''
    It create 'M' patterns divided in 3 parts of equal length: 
        first part is all +
        second part is always random
        third part is + for the first 'M/2' patterns and - for the other half. 
    The number of spins 'N_spin' must be a multiple of three and M must be even. 

    Parameters
    ----------
    N_spin : int
        Number of spins.
    M : int
        Number of patterns to store.
    first_group_members: int 
        The number of patterns that have the first and last spins up


    Returns
    -------
    patterns: array-like (M x N_spin)
    

    '''
    members_first_groups = int(first_group_members)
    members_second_groups = M - members_first_groups
    pattern_first_group = np.concat( ((np.ones((int(members_first_groups),int(N_spin/3)))), 
                                      np.random.choice([-1,1], (int(members_first_groups), int(N_spin/3))),
                                      np.ones((int(members_first_groups),int(N_spin/3)))),
                                      axis = 1)
    pattern_second_group = np.concat(( np.ones((int(members_second_groups),int(N_spin/3))) , 
                                      np.random.choice([-1,1], (int(members_second_groups), int(N_spin/3))),
                                      -np.ones((int(members_second_groups),int(N_spin/3)))),
                                      axis = 1)
    patterns = np.concat((pattern_first_group, pattern_second_group), axis=0) 
    return patterns


def random_patterns(N, M):
    return np.random.choice([-1,1], (M, N))

def shuffle_interaction_matrix(J): # not used up to now
    # number of spins
    N = J.shape[0]
    # permute randomly the spin indexing
    permutation = np.random.shuffle(np.arange(N))
    
    J_row = J[[permutation]]
    return J_row[:,[permutation]]

def relabeling(N, n, seed):
    rng = np.random.default_rng(seed)
    perms = []
    for i in range(n):
        perm = rng.permutation(N)
        perms.append(perm)
    return perms

def shuffle_spin_order(configs, permutation):
    '''
    Shuffle the order of the spins

    Parameters
    ----------
    configs : ndarray
        Configurations of the system.
    
    permutation: array-like
        The permutation for the reordering of the spin states.

    Returns
    -------
    ndaaray
        The input 'configs' with the columns randomly exchanged.

    '''
    return configs[:, permutation]


    
