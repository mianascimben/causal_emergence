''' 
This script contains functions usefull for the system simulation
'''
import numpy as np
from itertools import product, combinations
from functools import reduce

# I have to group the micro spins into macro spins

# function to be written 


#%% I have to compute all the configs of each spin group


# I combine them to create the micro configurations 
    
def group_configs(spin_groups_cardinality):
    
    group_configs = []
    for N in spin_groups_cardinality: 
        group_configs.append(create_configs(N))  
    return group_configs

    
def conf_prod(N):
    return np.array(list(product([-1, 1], repeat=N)))



    
def majority_rule_classifier(configs):
    '''
    This function takes in input the micro configurations associated to the spins 
    grouped into the same macro spin and classifies them 
    into two groups depending on the macro state +1 or -1 into which they map. 
    
    The map transforming micro configurations into {-1,+1} is the majority rule
    

    Parameters
    ----------
    configs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    sums = configs.sum(axis=1)

    configs_up = configs[sums > 0]
    configs_down = configs[sums <= 0]
    
    return {1: configs_up, -1: configs_down}


def combination_confs(conf_group1, conf_group2):
    '''
    Compute all combinations between the configurations stored into 'conf_group1' and 'conf_group2'.
    
    Parameters: 
    ----------
    conf_group1: np.ndarray
        2D array where each row is a configuration of 'conf_group1.shape[1]' spins. 
    conf_group2: np.ndarray
        2D array where each row is a configuration of 'conf_group2.shape[1]' spins.
  
    Return: 
    -------
    conf_12: np.ndarray
        2D array where each row is a combination of the configurations stored into 'conf_group1' and 'conf_group2'.
        The size of the 'conf_12' is (n1 x n2, N1 + N2), where n1 and n2 are the 
        number of configurations of 'conf_group1' and 'conf_group2' respectively, while N1 and N2 are the 
        number of spins of each group.
    '''
    # define number of configurations within each group (n1 and n2)
    # and the number of spins (N1 and N2)
    n1, N1 = conf_group1.shape
    n2, N2 = conf_group2.shape
  
    # use broadcasting to create combinations 
    conf_group1_broad = np.broadcast_to(conf_group1[:,None,:],(n1,n2,N1))
    conf_group2_broad = np.broadcast_to(conf_group2[None,:,:],(n1,n2,N2))
    conf_12_broad = np.concatenate((conf_group1_broad, conf_group2_broad), axis=2) 
    
    # from 3D to 2D array with size (n1*n2,N1+N2)
    conf_group12 = conf_12_broad.reshape(-1, N1 + N2)
    return conf_group12

def mapping_micro_confs_into_macro_confs(group_configs):
    '''
    
    '''
    return reduce(combination_confs,  group_configs)
  
def majority_rule_dict(configs):
  sums = configs.sum(axis=1)

  configs_up = configs[sums > 0]
  configs_down = configs[sums <= 0]
    
  return {1: configs_up, -1: configs_down}

def majority_rule(configs):
  '''
  This function maps the configurations into the macro states {1,-1}.
  It computes the majority rule of a set of configurations.
  It sums all the elements in the 'configs' rows and assigns:
  1 if the sum is positive and -1 otherwise.

  Return:
  ------
    nd.array
      1D array of 1 and -1, representing the macro state of each configuration.
  '''
  sums = configs.sum(axis=1)
  macro_state = configs[np.where(sums > 0, [1,-1])]
  return macro_state

#def mapping_micro_confs_into_macro_confs(configs):
#  starter = np.zeros((configs.shape[0], 1))
#  for i, j in idx_spin_grouping.items():
#    group_configs = configs[:,j]
#    macro_spin_state = majority_rule(group_configs)
#    macro_configs = np.concatenate((starter, macro_spin_state), axis = 1)

#  macro_configs.delete(0, 1) #delete the first column
#  Config_to_idx = {tuple(Config): i 
#                             for i, config in enumerate(configs)}
#  return macro_configs, Config_to_idx




  
#def build_micro_macro_map_idx(N, block_sizes):

  # create the configurations from N spins 
  #  micro_configs = create_configs(N)

  # map micro configurations into macro configurations
  #  macro_configs = micro_to_macro(micro_configs, block_sizes)

  # index the macro and micro configurations
  #  macro_idx = indexing_configs(macro_configs)
  #  micro_idx = indexing_configs(micro_configs)

  # create a dictionary which connects the index referring to the micro configurations
  # to the macro configurations they map into
  #  idx_micro_macro = {i: I for i, I in zip(micro_idx, macro_idx)}

  #  return idx_micro_macro, macro_idx, micro_idx


    
