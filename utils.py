import cupy as np 
import pandas as pd
import networkx as nx

class PhaseParameters:
    def __init__(self, temps, n_memories,  
                 causal_measure = None, mappings = None,
                 micro = True):
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
    return nx.from_cupy_array(J)

def doble_weel_patterns(N_spin, M):
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


    Returns
    -------
    patterns: array-like (M x N_spin)
    

    '''
    pattern_first_group = np.concat( ((np.ones((int(M/2),int(N_spin/3)))), 
                                      np.random.choice([-1,1], (int(M/2), int(N_spin/3))),
                                      np.ones((int(M/2),int(N_spin/3)))),
                                      axis = 1)
    pattern_second_group = np.concat(( np.ones((int(M/2),int(N_spin/3))) , 
                                      np.random.choice([-1,1], (int(M/2), int(N_spin/3))),
                                      -np.ones((int(M/2),int(N_spin/3)))),
                                      axis = 1)
    patterns = np.concat((pattern_first_group, pattern_second_group), axis=0) 
    return patterns

def orthogonal_patterns(N, M): #useless function
    '''
    Creates 'M' patterns with the first half all + and the second half orthogonal 
    to the first: (+...+,-...,-).
    Both 'M' and 'N' must be even.

    Parameters
    ----------
    N : int
        Number of spins.
    M : int
        Number of patterns to store.

    Returns
    -------
    array like

    '''
    plus = np.ones((int(M/2), int(N/2)))
    pattern_up = np.concat((plus, plus), axis=1)
    pattern_up_down = np.concat((plus, -plus), axis=1)
    return np.concat((pattern_up,pattern_up_down), axis =0)

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

def compute_all_alphas(N_spin, n_memories, mappings):
    """
    Compute the load parameter alpha = M / N for both micro and macro systems.
    M is the number of stored patterns and N is the number of spins.

    Parameters
    ----------
    N_spin : int
        Number of spins in the microscopic (original) system.

    n_memories : np.ndarray
        1D array containing the number of stored patterns (M).
        Typically an increasing sequence.

    mappings : list of list of int
        List of coarse-graining mappings. Each mapping is a list of integers
        specifying the sizes of groups that define macro-spins.

        Example:
            [[3,3,3], [5,1,1,1,1], [7,1,1]]

        Each mapping reduces the system to a number of macro-spins equal to
        len(mapp).

    Returns
    -------
    alphas_all : dict of {int: np.ndarray}
        Dictionary mapping each system to its corresponding array of alpha values.

        Keys are integer value indicating the system size:
            - 'N_spin' → micro system
            - 'N_macro' → macro systems
    
        Each element has the same shape as `memories`.

    """
    
    # creation of the array to store the alphas 
    alphas = np.zeros((len(n_memories), len(mappings)+1))
    # --- Micro system ---
    alphas[:,0] = n_memories / N_spin
    
    # --- Macro systems ---
    for k, mapp in enumerate(mappings):
        N_macro = len(mapp)
        
        if N_macro == 0:
            raise ValueError("Mapping with zero macro-spins encountered.")
        
        alphas[:,k+1] = n_memories / N_macro    
    return alphas
    
