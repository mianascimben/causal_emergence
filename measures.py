import numpy as np
from scipy.stats import entropy
'''
qui la classe ci starebbe solo perche abbiamo molte quantità simili che si ripetono
salverei il normalization, la evolved_eq_dist, la uniform_dist 
Tuttavia come includo CE in questa classe? 

d altra parte non dovrei passare system e dynamics ma solo equilibrium_dist e TPM 
per rendere queste funzioni usabili anche per casi inventati ad hoc

'''

def evolution(distribution, TPM, step=1):                                       # prova a tenerla come funzione pura
    return distribution @ np.linalg.matrix_power(TPM, step)                         # in questo contesto per ora non serve

def normalization_factor(n_configs):
    '''
    Parameters
    ----------
    n_configs : int
        Number of configurations.

    Returns
    -------
    float
    
    '''
    return 1/np.log2(n_configs)

def degeneracy(TPM, n_configs):
    '''
    Compute the degree of deterministic convergence by 
    evaluating the Kullback-Leibler divergence between the effect unconstrained 
    distribution and the uniform distribution. 

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    base = 2 # the Kullback-Leibler div. is measured in bits
    
    uniform_distribution = 1/n_configs * np.ones(n_configs)
 
    distribution_effects = uniform_distribution @ TPM
    
    return normalization_factor(n_configs) * entropy(distribution_effects, 
                                                     uniform_distribution, 
                                                     base)
        
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
     base = 2 # the Kullback-Leibler div. is measured in bits
     
     uniform_distribution = 1/n_configs * np.ones(n_configs)
     
     # We put axis = 1 to calculate the entropy between each row of the TPM 
     # and the uniform distribution
     det = entropy(TPM, uniform_distribution[None,:], base, axis = 1)                  
     
     return normalization_factor(n_configs) * np.mean(det)                   

def EI(TPM, n_configs):
    '''
    For each configuration, compute the Kullback-Leibler divergence 
    between its associated constrained effect distribution and the 
    unconstrained effect distribution, then it averages the
    results over all configurations according to the uniform distribution.
    '''
    base = 2 # the Kullback-Leibler div. is measured in bits
    
    uniform_distribution = 1/n_configs * np.ones(n_configs)
    
    distribution_effects = uniform_distribution @ TPM

    effect_information = entropy(TPM, distribution_effects[None,:], base, axis = 1) 
    return  np.mean(effect_information) 
        
        
def effectiveness(TPM, n_configs):
    
    eff = normalization_factor(n_configs) * EI(TPM, n_configs)
    
    det = determinism(TPM, n_configs)
    deg = degeneracy(TPM, n_configs)
    
    if eff - (det-deg)<0.0001:  print("eff = det-deg: True") # poi cancella se vuoi
    else:  print("eff = det-deg: False")
    
    return eff
        

def compute_all_measures(TPM, n_configs):                                # I want to implement it for multiple scales
    '''
    Computes different causal measures 

    Parameters
    ----------
    equilibrium_distribution : np.ndarray
    TPM : np.ndarray
        Transition probability matrix.

    Returns
    -------
    dict
        DESCRIPTION.

    '''
    det = determinism(TPM, n_configs)
    deg = degeneracy(TPM, n_configs)
    eff = effectiveness(TPM, n_configs)
    ei = EI(TPM, n_configs)
     
    
    return {
        "determinism": det,
        "degeneracy": deg,
        "effectiveness": eff,
        "EI": ei,
        "number of states": n_configs  # this measures is used for computing the
                                       # effectivenesss and size contribution in CE
    }

def causal_emergence(measures_micro, measures_macro):
    '''
    Computes the causal emergence (CE) between two scales as the difference 
    in effective information (EI). Furthermore, it returns the 
    contribution due both to the variation of effectiveness and the size reduction

    Parameters
    ----------
    measures_micro : dict
        Dictionary storing the values of EI, effectiveness and number of states
        for the micro scale.
    measures_macro : dict 
        The same as above, but for the macro scale.

    Returns
    -------
    dict
    It containes the values of : 
                                CE: float
                                effectiveness_variation : float
                                size_reduction : float 

    '''
    EI_micro = measures_micro['EI']
    EI_macro = measures_macro['EI']
    
    eff_micro = measures_micro['effectiveness']
    eff_macro = measures_macro['effectiveness']
    
    size_micro = np.log2(measures_micro["number of states"])
    size_macro = np.log2(measures_macro["number of states"])

    CE = EI_macro - EI_micro
    effectiveness_variation = size_macro * (eff_macro - eff_micro)
    size_reduction = eff_micro * (size_macro - size_micro)
    
    if CE - (effectiveness_variation + size_reduction) <0.0001: print("Deff + DI = CE: True") # poi cancella se vuoi
    else:  print("Deff + Dsize = CE: False")
    
    return {'CE': CE,
            'D_{eff}': effectiveness_variation, 
            'D_{size}': size_reduction
            }