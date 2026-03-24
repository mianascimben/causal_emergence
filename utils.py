
def store_system_information(N_micro, T, N_macro, groups):
    "Create a dictionary for storing the system information."
    return {'N_m': N_micro,
            'Temperature': T,
            'N_M': N_macro,
            'groups': groups}
            
def indexing_configs(configs):
    bits = (configs + 1) // 2
    powers = 2**np.arange(configs.shape[1]-1, -1, -1)

    return bits @ powers