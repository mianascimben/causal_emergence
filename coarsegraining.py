import numpy as np
from utils import indexing_configs
from ising import IsingSystem
from dynamics import GlauberDynamics

class CoarseGraining:
    '''
    This class containes is a useful pipeline for creating the coarsen 
    system and dynamics.
    '''
    
    def __init__(self, system, dynamics, patterns, block_sizes):
        self.system = system
        self.micro_configs = system.configs
        self.TPM_micro = dynamics.TPM
        self.block_sizes = block_sizes # qui poi con un if calcolerò i vari blocchi nel caso non siano passati a mano

        self.macro_patterns, self.macro_patterns_idx = self.map_micro_to_macro(patterns) # se vuoi poi calcella macro_patterns_idx
        self.macro_configs, self.macro_idx = self.map_micro_to_macro(self.micro_configs)
        self.macro_system, self.macro_dynamics  = self.build_macro_model()
        
    def map_micro_to_macro(self, configs): 
      '''
      This is the mapping from the micro phace space to the macro phace space. 
      
      Apply the majority rule on all the micro spin blocks that compose each micro 
      configuration to give the sequence of macro spin states (alias the 
      macro configuration). 
 
      Returns:
      -------
        macro_configs: nd.array
          2D array where the row "i" is the image of a micro configuration "i"
          after the mapping. Then 'macro' has 
          macro.shape()=(number of micro configurations, number of macro spins)
      '''
      splits = np.cumsum(self.block_sizes)[:-1]
      
      # divide the spins into blocks, so the global configurations are splitted 
      # to form the configurations of the blocks
      blocks = np.split(configs, splits, axis=1)
      
      # calculate the macro states for the single block configurations 
      # to form the global macro configurations (majority rule)
      macro_configs = np.array([np.sign(b.sum(axis=1)) for b in blocks]).T
      
      # index the macro and micro configurations
      macro_idx = indexing_configs(macro_configs) 
      
      return macro_configs, macro_idx
      
    def build_projection_matrix(self):
    
        n_micro = len(self.micro_configs) # number of micro configurations
        n_macro = self.macro_idx.max() + 1 # number of macro configurations
    
        M = np.zeros((n_micro, n_macro))
    
        # set 1 in correspondece of the mapping from a micro configuration 
        # to a macro configuration
        M[np.arange(n_micro), self.macro_idx] = 1                              
    
        return M
    
    def coarse_graining_tpm(self):

        M = self.build_projection_matrix()

        TPM_macro = M.T @ self.TPM_micro @ M                                
        
        # counts the number of multiple micro realizations of each macro configuration
        counts = np.bincount(self.macro_idx)

        TPM_macro /= counts[:, None]                                        
        
        return TPM_macro  
    
    def build_macro_model(self):
        
        # macro configurations realized by the micro configurations 
        macro_configs = np.unique(self.macro_configs, axis=0)
        
        # macro TPM
        macro_TPM = self.coarse_graining_tpm()
        
        # number of macro spins
        N_macro = macro_configs.shape[1]
        
        macro_system = IsingSystem(N_macro, self.macro_patterns, self.system.beta, macro_configs)
        macro_dynamics = GlauberDynamics(macro_system, macro_TPM)
        
        return macro_system, macro_dynamics