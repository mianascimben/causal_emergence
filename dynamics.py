import cupy as np
from itertools import combinations

class GlauberDynamics:                                                              #questa potrebbe essere solo la funzione pura che fa TPM, ma per ora va bene così
    
        def __init__(self, system, TPM = None):
            '''
            Parameters
            ----------
            system : class
                Istance of IsingSystem.


            '''
            self.system = system
            self.beta = self.system.beta

            self._update_probability_method = _glauber_probability()
            
            if TPM is None: 
                self.TPM = self.build_TPM()
            else:
                self.TPM =TPM
                       

        def build_TPM(self):
            '''
            Generate the transition probability matrix for a system with N binary neurons 
            interacting through the J matrix. Neuron states can either be -1 or 1.
            The TPM coefficients are the probabilities, given a current state, 
            to select a target configuration and to evolve into it. 
            The probaility for the selection of the target configuration is the 
            uniform distribution, where each target configuration has a chance 1/N_target
            to be selected and N_target is the number of all configuration in which the 
            current state can evolve. 
            The assumption is that the transition can be implemented among 
            configurations differing for onle 1 spin flip, then 
            N_target = number of spins.
            The probability that the proposed transition is 
            accepted follows the Glauber formula. 
            
            
            Parameters                                                          # se teniamo il self questo si elimina
            ----------
            N : int
                Number of neurons composing the system.
            J : np.ndarray
                Interaction matrix.
            beta : float, optional
                It is the inverse of temperature and it measures the degree of
                environmental noise. The default is 1.


            Returns
            -------
            TPM : np.ndarray
                Transition probability matrix .
            all_configs : np.ndarray
                List of all configurations of the system.
            config_to_idx : dict
                All configurations of the system are numerated from 0 to 2^N-1.
            '''
            N = self.system.N
            J = self.system.J
            n_configs = self.system.n_configs                                   # vediamo quante volte lo usiamo n_configs perchè potremmo direttamente calcolarlo
            configs = self.system.configs 
            
            TPM = np.zeros((n_configs, n_configs))
            
            for i in range(n_configs):
                config = configs[i]
                
                fields = J @ config.T
                
                for k in range(N): #flips = 1
                    j = i ^ (1 << k) # use the binary labelling 
                    
                    if j>i:
                        delta_E = 2 * config[k] * fields[k]
                        TPM[i,j] = self._update_probability_method(delta_E, self.beta)
                    else:
                        TPM[i,j] = 1 - TPM[j,i] 
                
                
            # normalization
            TPM /= N
            # add diagonal coefficients
            np.fill_diagonal(TPM, 1 - TPM.sum(axis=1))
            
            # to avoid negative probabilties in the diagonal due to numerical 
            # approximation of the subtraction
            TPM = np.clip(TPM, 0, 1)
            # remormalize 
            TPM /= TPM.sum(axis=1, keepdims=True)
                     
            return TPM
        
        def evolution(self, p, steps = 1):
            '''
            Evolves the array 'p' of a given number of steps
            by multiplying it with the TPM stored in GlauberDynamics

            Parameters
            ----------
            p : 1D array
                
            step : int, optional
                Number of time steps evolution. The default is 1.

            Returns
            -------
            array
                The evolved array.

            '''
            return p @ np.linalg.matrix_power(self.TPM, steps)
        
        def async_run(self, state_0):
            random_idx = np.random.choice(len(state_0))
            state_1 = state_0.copy()
            
            r = np.random.random_sample()
                
            field_idx = np.dot(self.system.J[random_idx,:], state_0)
            delta_E = 2 * state_0[random_idx] * field_idx
            p = self._update_probability_method(delta_E, self.beta)
            
            if r < p: 
                state_1[random_idx] *= -1
                
            return state_1
        
        def run(self, state_0, steps):
            """Runs the dynamics.

            Args:
                nr_steps (float, optional): Timesteps to simulate
            """
            state_update = state_0.copy()
            for t in range(steps):
                # run a step
                state_update = self.async_run(state_update)
            return state_update 
        
        def run_with_monitoring(self, state_0, steps):                      # vedi se si possono usare gli indici 
            """
            Iterates for the fixed steps. Records the network state after every
            iteration

            Args:
                steps:

            Returns:
                a list of 2d network states
            """
            states = list()
            state_update = state_0.copy()
            states.append(state_update)
            for t in range(steps):
                # run a step
                state_update = self.async_run(state_update)
                states.append(state_update.copy())
            return states
                    
def _glauber_probability():
    def compute_probability(delta_E, beta):
        return  1.0 / (1.0 + np.exp(np.clip(beta * delta_E, -100, 100)))
    return compute_probability        
        
