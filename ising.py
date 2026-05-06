import numpy as np 

class IsingSystem: 
    
    def __init__(self, N, patterns, beta, configs = None):
        '''     
        Parameters
        ----------
        N : int
            Number of spins of the system.
        patterns : np.ndarray
            2D array where each row is a memorized pattern.
        beta: float
            Inverse of temperature.
        '''
        self.N = N
        self.patterns = patterns
        self.beta = beta
        
        if configs is None: #used for computing the Hamiltonian 
            self.configs = self.create_configs()
            self.J = self.interaction_matrix()

        else: 
            self.configs = configs # I want to pass them for the macro system to keep track of errors (e.s. block with even spins )

        # quantities often reused
        self.n_configs = np.shape(self.configs)[0]                                      # se tolgo normalization, anche questo non serve

    def create_configs(self):
        '''
        Compute all the configurations of a system with 'self.N' binary elements. 
        The use of the bit representation and the bitwise operatorion speeds up the 
        computation, reducing the running time. 


        Returns
        -------
        configs: np.ndarray
            2D array storing all the 2^'self.N' configurations.

        '''
        configs = np.arange(2 ** self.N)
        
        # create all possible configurations using bitwise operators
        # In particular: 
        # (configs[:,None] >> np.arange(N)) 
        #  |  The numbers from 0 to N-1 are divided for the 
        #  |_ vector [0,2,4,...,2^(N-1)] (through braodcasting)
        #
        # ('....' & 1)
        #  |_ All even results are transformed into 1, while odd results into 0
        bits = ((configs[:,None] >> np.arange(self. N)) & 1)
        
        # The '1' and '0' become '+1' and '-1' respectively
        configs = 2*bits - 1
        return configs
    
    def normalization_factor(self):                                             #qui non viene mai usato 
        return 1/np.log2(self.n_configs)
    
    def interaction_matrix(self):
        '''
        Generates the interaction matrix through the Hebbian rule

        Returns
        -------
        J : np.ndarray
            Interaction matrix

        '''
        J = self.patterns.T @ self.patterns
        np.fill_diagonal(J, 0)
        J = J/self.N
        return J 
    
    def energy_config(self, config):                                            
        '''
        Compute the energy level of a single configuration.

        Parameters
        ----------
        config : nd.array
            1D array representing a configuration of the system.

        Returns
        -------
        float
            energy of 'config'.

        '''
        return -0.5 * np.sum(config * (self.J @ config).T)
    
    def energy_landscape(self):
        '''
        Compute the Hamiltonian for all configurations of the system.

        Returns
        -------
        np.ndarray
            Array storing the energy of each configuration.

        '''        
        # J @ configs.T | N,N @ N,2^N = N,2^N
        local_fields = self.J @ self.configs.T
        
        # local_fields * configs.T | (N,2^N) * (N,2^N) = (N, 2^N)
        # then sum over the rows and get an array (2^N,)
        return  -0.5 * np.sum(local_fields * self.configs.T, axis=0)
    
        
    def canonical_distribution(self):
            """Canonical probability distribution."""
            H = self.energy_landscape()
            
            expH = np.exp(-self.beta * H)
            
            return expH / expH.sum()

        
       
       
 