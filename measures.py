import numpy as np
import pandas as pd
from scipy.stats import entropy
from ising import IsingSystem
from dynamics import GlauberDynamics
from coarsegraining import CoarseGraining
from utils import doble_weel_patterns
'''
qui la classe ci starebbe solo perche abbiamo molte quantità simili che si ripetono
salverei il normalization, la evolved_eq_dist, la uniform_dist 
Tuttavia come includo CE in questa classe? 

d altra parte non dovrei passare system e dynamics ma solo equilibrium_dist e TPM 
per rendere queste funzioni usabili anche per casi inventati ad hoc

'''

class CausalMeasure: 
    def __init__(self, TPM, n_configs):
        self.TPM = TPM
        self.n_configs = n_configs
        self.base = 2 # the Kullback-Leibler div. is measured in bits
        # computed quantities
        self.normalization = self.normalization_factor()
        self.deg = self.degeneracy()
        self.det = self.determinism()
        self.eff = self.effectiveness()
        self.ei = self.EI()

    def normalization_factor(self):
        '''
        Parameters
        ----------
        n_configs : int
            Number of configurations.
    
        Returns
        -------
        float
        
        '''
        return 1/np.log2(self.n_configs)
    
    def degeneracy(self):
        '''
        Compute the degree of deterministic convergence by 
        evaluating the Kullback-Leibler divergence between the effect unconstrained 
        distribution and the uniform distribution. 
    
        Returns
        -------
        TYPE
            DESCRIPTION.
    
        '''
       
        uniform_distribution = 1/self.n_configs * np.ones(self.n_configs)
     
        distribution_effects = uniform_distribution @ self.TPM
        
        return self.normalization * entropy(distribution_effects, 
                                                         uniform_distribution, 
                                                         self.base)
            
    def determinism(self):
         '''
         For each configuration compute the Kullback-Leibler divergence 
         between its associated constrained effect distribution and the 
         uniform distribution, then it averages the results over all 
         configurations with equal weights.
             
         Returns
         -------
         TYPE
             DESCRIPTION.
        
         '''
           
         # We put axis = 1 to calculate the entropy of each row of the TPM 
         # we use a formula that is equivalent to the Dkl to save memory
         det = np.log2(self.n_configs) - entropy(self.TPM, base=self.base, axis=1)
              
         return self.normalization * np.mean(det) 

    def EI(self):
        '''
        For each configuration, compute the Kullback-Leibler divergence 
        between its associated constrained effect distribution and the 
        unconstrained effect distribution, then it averages the
        results over all configurations according to the uniform distribution.
        
        
        uniform_distribution = 1/n_configs * np.ones(n_configs)
        
        distribution_effects = uniform_distribution @ TPM
    
        effect_information = entropy(TPM, distribution_effects[None,:], base, axis = 1) 
        return  np.mean(effect_information) 
        '''
        return self.eff / self.normalization
        
    def effectiveness(self):
        '''
        eff = normalization_factor(n_configs) * EI(TPM, n_configs)
        
        det = determinism(TPM, n_configs)
        deg = degeneracy(TPM, n_configs)
        
        check = np.isclose(eff, det - deg)
        print(f"eff=det-deg: {check}")
        '''
        
        return self.det - self.deg
        

    def compute_all_measures(self, label = ''):                                # I want to implement it for multiple scales
        '''
        Computes different causal measures 
    
        Parameters
        ----------
        equilibrium_distribution : np.ndarray
        TPM : np.ndarray
            Transition probability matrix.
    
        Returns
        -------
        DataFrame
            DESCRIPTION.
    
        
        det = determinism(TPM, n_configs)
        deg = degeneracy(TPM, n_configs)
        eff = effectiveness(TPM, n_configs)
        ei = EI(TPM, n_configs)
        '''
        
        return pd.DataFrame({
            "determinism": [self.det],
            "degeneracy": [self.deg],
            "effectiveness": [self.eff],
            "EI": [self.ei],
            "number of states": [self.n_configs]  # this measures is used for computing the
                                           # effectivenesss and size contribution in CE
                                           }, index=[label])

            
# =============================================================================
def normalization_factor(n_configs):
#     '''
#     Parameters
#     ----------
#     n_configs : int
#         Number of configurations.
# 
#     Returns
#     -------
#     float
#     
#     '''
    return 1/np.log2(n_configs)
# 
# def degeneracy(TPM, n_configs):
#     '''
#     Compute the degree of deterministic convergence by 
#     evaluating the Kullback-Leibler divergence between the effect unconstrained 
#     distribution and the uniform distribution. 
# 
#     Returns
#     -------
#     TYPE
#         DESCRIPTION.
# 
#     '''
#     base = 2
#     uniform_distribution = 1/n_configs * np.ones(n_configs)
#  
#     distribution_effects = uniform_distribution @ TPM
#     
#     return normalization_factor(n_configs) * entropy(distribution_effects, 
#                                                      uniform_distribution, 
#                                                      base)
#         
def determinism(TPM, n_configs):
      '''
      For each configuration compute the Kullback-Leibler divergence 
      between its associated constrained effect distribution and the 
      uniform distribution, then it averages the results over all 
      configurations with equal weights.
          
      Returns
      -------
      TYPE
          DESCRIPTION.
     
      '''
      uniform_distribution = 1/n_configs * np.ones(n_configs)

        
      # We put axis = 1 to calculate the entropy of each row of the TPM 
      # we use a formula that is equivalent to the Dkl to save memory
      det = entropy(TPM, uniform_distribution[None,:], base=2, axis=1)
           
      return normalization_factor(n_configs) * np.mean(det) 
# 
# def EI(TPM, n_configs):
#     '''
#     For each configuration, compute the Kullback-Leibler divergence 
#     between its associated constrained effect distribution and the 
#     unconstrained effect distribution, then it averages the
#     results over all configurations according to the uniform distribution.
#     '''
#     
#     uniform_distribution = 1/n_configs * np.ones(n_configs)
#     
#     distribution_effects = uniform_distribution @ TPM
# 
#     effect_information = entropy(TPM, distribution_effects[None,:], base=2, axis = 1) 
#     return  np.mean(effect_information) 
#     
#     
# def effectiveness(TPM, n_configs):
#     
#     eff = normalization_factor(n_configs) * EI(TPM, n_configs)
#     
#     det = determinism(TPM, n_configs)
#     deg = degeneracy(TPM, n_configs)
#     
#     if eff - (det-deg)<0.0001:  print("eff = det-deg: True") # poi cancella se vuoi
#     else:  print("eff = det-deg: False")
#     
#     
#     return eff
#     
# 
# def compute_all_measures(TPM, n_configs, label = ''):                                # I want to implement it for multiple scales
#     '''
#     Computes different causal measures 
# 
#     Parameters
#     ----------
#     equilibrium_distribution : np.ndarray
#     TPM : np.ndarray
#         Transition probability matrix.
# 
#     Returns
#     -------
#     DataFrame
#         DESCRIPTION.
#     '''
#     
#     det = determinism(TPM, n_configs)
#     deg = degeneracy(TPM, n_configs)
#     eff = effectiveness(TPM, n_configs)
#     ei = EI(TPM, n_configs)
#     
#     
#     return {"determinism": det,
#     "degeneracy": deg,
#     "effectiveness": eff,
#     "EI": ei,
#     "number of states": n_configs}, pd.DataFrame({
#         "determinism": [det],
#         "degeneracy": [deg],
#         "effectiveness": [eff],
#         "EI": [ei],
#         "number of states": [n_configs]  # this measures is used for computing the
#                                        # effectivenesss and size contribution in CE
#                                        }, index=[label])
# =============================================================================
# =============================================================================
# 
# def causal_emergence(measures_micro, measures_macro):
#     '''
#     Computes the causal emergence (CE) between two scales as the difference 
#     in effective information (EI). Furthermore, it returns the 
#     contribution due both to the variation of effectiveness and the size reduction
# 
#     Parameters
#     ----------
#     measures_micro : dataframe
#         storing the values of EI, effectiveness and number of states
#         for the micro scale.
#     measures_macro : dataframe 
#         The same as above, but for the macro scale.
# 
#     Returns
#     -------
#     DataFrame
#     It containes the values of : 
#                                 CE: float
#                                 effectiveness_variation : float
#                                 size_reduction : float 
# 
#     '''
#     
#     # calculate the difference of between any macro measure and the micro one
#     EI_m = measures_micro['EI']
#     EI_M= measures_macro['EI']
#     
#     CE = EI_M - EI_m
#     
#     eff_micro = measures_micro['effectiveness']
#     eff_macro = measures_macro['effectiveness']
#     
#     size_micro = np.log2(measures_micro['number of states'])
#     size_macro = np.log2(measures_macro['number of states'])
#     
#     size_reduction = eff_micro * (size_macro - size_micro)
#     
#     effectiveness_variation = size_macro * (eff_macro-eff_micro)
#     
#     if CE - (effectiveness_variation + size_reduction) <0.0001: print("Deff + DI = CE: True") # poi cancella se vuoi
#     else:  print("Deff + Dsize = CE: False")
#     
#     return {'CE': CE,
#             'D_{eff}': effectiveness_variation, 
#             'D_{size}': size_reduction
#             }
#                              
# =============================================================================
def causal_emergence_vectorized(data):

    data = data.astype(float)

    log_n = np.log2(data['number of states'])
    diff = data - data.loc['micro']

    CE_series = diff['EI']
    D_eff_series = log_n * diff['effectiveness']
    D_size_series = data.loc['micro', 'effectiveness'] * (log_n - log_n.loc['micro'])

    CE_df = pd.DataFrame({
        'CE': CE_series,
        'D_{eff}': D_eff_series,
        'D_{size}': D_size_series
    })
    # --- check ---
    check_series = np.isclose(CE_series, D_eff_series + D_size_series)
    print(f"For each mapping -- CE = D_eff + D_size : {check_series}")
    
    return CE_df

def compute_overlap_matrix(patterns, configs): 
    """ measures how similar the configurations are to each nominal pattern.
      parameters: 
      ----------
          patterns: array-like
              stores the nominal patterns
          configs: array-like
              2D array where on the row there are the configurations to compare with 
              the 'patterns' 
      returns:
      ----------
          overlap_matrix: 2D array
              stores the overlap value between each couple (configuration, pattern).
              The row stands for the reference patterns, while the columns for the 
              configuration in 'configs'.           
    """
        
    N = np.shape(patterns)[1] # number of spins 
    overlap_matrix = 1/N  * (patterns @ configs.T)  
    return overlap_matrix

def compute_mean_overlap(overlap_matrix, boltzmann_pdf):
    '''
    

    Parameters
    ----------
    overlap_matrix : 2d-array
        stores the overlap value between each couple (configuration, pattern).
        The row stands for the reference patterns, while the columns for the 
        configuration in 'configs'.  
    boltzmann_pdf : array
        boltzmann weights of the configurations of the system

    Returns
    -------
    mean_overlap_per_pattern : array like
        given a pattern, the overlap between each config with that pattern is 
        averaged according to the boltzmann weights. The result is a measure of
        how similar the configs of the system are to each pattern at the equilibrium.
    system_mean_overlap : float
        arithmetic average of the elements in 'mean_overlap_per_pattern'.
        It measures how much the configs at the equilibrium looks like the 
        memorized patterns

    '''
    mean_overlap_per_pattern = np.average(overlap_matrix, axis = 1, weights = boltzmann_pdf)
    system_mean_overlap = np.mean(mean_overlap_per_pattern)
    return mean_overlap_per_pattern, system_mean_overlap
# =============================================================================
# 
# def compute_phase_diagram(temperatures, memories, patterns, f, mapping = None):
#     
#     """
# 
#     #     Evaluate a causal measure over a two-dimensional parameter space to 
#     #     construct a phase diagram at the microscopic or macroscopic level.
#     # 
#     #     For each pair of parameters (T, alpha), an Ising system is instantiated 
#     #     using a subset of the provided patterns and inverse temperature 
#     #     β = 1 / T. The corresponding Transition Probability Matrix (TPM) is 
#     #     computed via Glauber dynamics. A causal measure f is then evaluated 
#     #     either on the microscopic TPM or on a coarse-grained TPM obtained 
#     #     through a specified mapping.
#     # 
#     #     Parameters
#     #     ----------
#     # 
#     #     temperatures : array-like
#     #         Continuous values of the control parameter along the vertical axis 
#     #         (e.g., temperature T). The inverse temperature is defined as β = 1 / T.
#     #     
#     #     alphas : array-like
#     #         Discrete values of the control parameter along the horizontal axis 
#     #         (e.g., fraction of stored patterns alpha = M/N, with N the number of 
#     #          spins). For each value alpha, the first alpha*N rows of `patterns` are selected.
#     #         
#     #     patterns : ndarray of shape (M_max, N)
#     #         Binary pattern matrix defining the stored configurations of the system, 
#     #         where M_max is the maximum number of patterns and N is the number of spins.
#     # 
#     #     f : str
#     #         Name of the method of the `CausalMeasure` class to be evaluated 
#     #         (e.g., "effectiveness", "determinism"). The method must be callable 
#     #         without arguments.
#     # 
#     #     mapping : list of int, optional
#     #         Coarse-graining scheme specifying how microscopic spins are grouped 
#     #         into macroscopic variables (e.g., [3, 3, 3]). If None, the measure is 
#     #         evaluated at the microscopic level. Otherwise, a coarse-grained model 
#     #         is constructed prior to evaluation.
#     # 
#     #     Returns
#     #     -------
#     #     phase : ndarray 
#     #         Matrix of evaluated measure values. Entry (i, j) corresponds to the 
#     #         value of the measure at parameter pair (temperatures[i], alphas[j]).
#     # 
#     #     Notes
#     #     -----
#     #     The computation proceeds as follows:
#     # 
#     #         1. Select the first alpha*N patterns from the full pattern set.
#     #         2. Construct the microscopic Ising system with N spins and inverse 
#     #            temperature β = 1 / T.
#     #         3. Generate the TPM via Glauber dynamics.
#     #         4. If `mapping` is provided, apply coarse-graining to obtain a 
#     #            macroscopic TPM.
#     #         5. Evaluate the selected causal measure on the resulting TPM.
#     # 
#     #     This function performs a full reconstruction of the system for each 
#     #     parameter pair (temperatures[i], alphas[j]), which may become computationally
#     #     expensive for large systems. Caching or parallelization strategies are recommended 
#     #     for large-scale phase diagram evaluations.
#     """
#     n_maps = len(mapping) if mapping is not None else 0
# 
#     # Creation of the matrix where to store the 'f' values (diagram phase)
#     # the rows detect the same system at different resolutions; the '+1' 
#     # guarantees that the micro system is always saved
#     phases = np.zeros((len(temperatures), len(memories), n_maps + 1))
#     
#     N = patterns.shape[1]
#          
#     for i, T in enumerate(temperatures):
#         beta = 1/T
#              
#         for j, M in enumerate(memories):
#             fraction_of_patterns = patterns[:M]
#             sm = IsingSystem(N, fraction_of_patterns, beta)
#             dm = GlauberDynamics(sm)
#                  
#             phases[i, j, 0] = getattr(CausalMeasure(dm.TPM, sm.n_configs),f)()
#                
#             if mapping is not None: 
#                 for k, mapp in enumerate(mapping): 
#                     cs = CoarseGraining(sm, mapp, dm)
#                     sM, dM = cs.build_macro_model()
#                     phases[i, j, k + 1] = getattr(CausalMeasure(dM.TPM, sM.n_configs), f)()
#                     
#     return phases
# =============================================================================

def compute_phase_diagram(phase_class, patterns):
    
    """

    #     Evaluate a causal measure over a two-dimensional parameter space to 
    #     construct a phase diagram at the microscopic or macroscopic level.
    # 
    #     For each pair of parameters (T, alpha), an Ising system is instantiated 
    #     using a subset of the provided patterns and inverse temperature 
    #     β = 1 / T. The corresponding Transition Probability Matrix (TPM) is 
    #     computed via Glauber dynamics. A causal measure f is then evaluated 
    #     either on the microscopic TPM or on a coarse-grained TPM obtained 
    #     through a specified mapping.
    # 
    #     Parameters
    #     ----------
    # 
    #     temperatures : array-like
    #         Continuous values of the control parameter along the vertical axis 
    #         (e.g., temperature T). The inverse temperature is defined as β = 1 / T.
    #     
    #     alphas : array-like
    #         Discrete values of the control parameter along the horizontal axis 
    #         (e.g., fraction of stored patterns alpha = M/N, with N the number of 
    #          spins). For each value alpha, the first alpha*N rows of `patterns` are selected.
    #         
    #     patterns : ndarray of shape (M_max, N)
    #         Binary pattern matrix defining the stored configurations of the system, 
    #         where M_max is the maximum number of patterns and N is the number of spins.
    # 
    #     f : str
    #         Name of the method of the `CausalMeasure` class to be evaluated 
    #         (e.g., "effectiveness", "determinism"). The method must be callable 
    #         without arguments.
    # 
    #     mapping : list of int, optional
    #         Coarse-graining scheme specifying how microscopic spins are grouped 
    #         into macroscopic variables (e.g., [3, 3, 3]). If None, the measure is 
    #         evaluated at the microscopic level. Otherwise, a coarse-grained model 
    #         is constructed prior to evaluation.
    # 
    #     Returns
    #     -------
    #     phase : ndarray 
    #         Matrix of evaluated measure values. Entry (i, j) corresponds to the 
    #         value of the measure at parameter pair (temperatures[i], alphas[j]).
    # 
    #     Notes
    #     -----
    #     The computation proceeds as follows:
    # 
    #         1. Select the first alpha*N patterns from the full pattern set.
    #         2. Construct the microscopic Ising system with N spins and inverse 
    #            temperature β = 1 / T.
    #         3. Generate the TPM via Glauber dynamics.
    #         4. If `mapping` is provided, apply coarse-graining to obtain a 
    #            macroscopic TPM.
    #         5. Evaluate the selected causal measure on the resulting TPM.
    # 
    #     This function performs a full reconstruction of the system for each 
    #     parameter pair (temperatures[i], alphas[j]), which may become computationally
    #     expensive for large systems. Caching or parallelization strategies are recommended 
    #     for large-scale phase diagram evaluations.
    """
    n_maps = len(phase_class.mappings) if phase_class.mappings is not None else 0

    # Creation of the matrix where to store the 'f' values (diagram phase)
    # the rows detect the same system at different resolutions; the '+1' 
    # guarantees that the micro system is always saved
    phases = np.zeros((len(phase_class.temps), len(phase_class.n_memories), n_maps + 1))
    
    N = patterns.shape[1]
         
    for i, T in enumerate(phase_class.temps):
        beta = 1/T
             
        for j, M in enumerate(phase_class.n_memories):
            fraction_of_patterns = patterns[:M]
            sm = IsingSystem(N, fraction_of_patterns, beta)
            dm = GlauberDynamics(sm)
            
            if phase_class.micro == True:
                phases[i, j, 0] = getattr(CausalMeasure(dm.TPM, sm.n_configs),
                                      phase_class.causal_measure)()
            else:   phases[i, j, 0] = np.nan 
            
            if phase_class.mappings is not None: 
                for k, mapp in enumerate(phase_class.mappings): 
                    cs = CoarseGraining(sm, mapp, dm)
                    sM, dM = cs.build_macro_model()
                    phases[i, j, k + 1] = getattr(CausalMeasure(dM.TPM, sM.n_configs), 
                                                  phase_class.causal_measure)()
                    
    return phases

def hamming_distance(array1, array2):
    '''
    Get the Hamming distance between the two binary arrays in input. 
    The Hamming distance is the number of elements for which tro arrays differ.
    '''
    difference = array1 - array2
    return np.sum((difference!=0))

def function(parameters, patterns, causal_measure, mappings=None, micro=True): #to delete if parallelism not used
    n_maps = len(mappings) if mappings is not None else 0

    # Creation of the matrix where to store the 'f' values (diagram phase)
    # the rows detect the same system at different resolutions; the '+1' 
    # guarantees that the micro system is always saved
    
         
    T = parameters[0]; M = parameters[1];
    N = patterns.shape[1]
    beta = 1/T
        
    sm = IsingSystem(N, patterns[:M], beta)
    dm = GlauberDynamics(sm)
    
    phases = np.zeros((n_maps+1))        
    if micro == True:
        phases[0] = getattr(CausalMeasure(dm.TPM, sm.n_configs),
                                      causal_measure)()
    # else:   phases[i, j, 0] = 0 
            
    if mappings is not None: 
        for k, mapp in enumerate(mappings): 
            cs = CoarseGraining(sm, mapp, dm)
            sM, dM = cs.build_macro_model()
            phases[k + 1] = getattr(CausalMeasure(dM.TPM, sM.n_configs), 
                                                  causal_measure)()
                    
    return phases