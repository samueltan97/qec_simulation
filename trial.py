from typing import List
from simulate import Simulation
from typing import List
import numpy as np

if __name__ == "__main__":
    from simulate import Simulation
    import time
    import numpy as np
    from functools import partial
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    import stim
    import pandas as pd
    from itertools import cycle
    from scipy.stats import linregress

    colors = cycle(['tab:blue', 'tab:orange', 'tab:red', 'yellow'])
    num_shots = 1000
    data_dictionary = dict()
    rounds = [30, 50, 70]
    distances = [3, 5, 7]
    noises = np.linspace(0.03, 0.06,5)

    # st = time.time()
    # simulation = Simulation(rounds=rounds, distances=distances, noises=noises, \
    #     circuit_parameters={'code_task': 'surface_code:unrotated_memory_z', 'before_round_data_depolarization':'', 'before_measure_flip_probability':''})
    # simulation_results = simulation.simulate_logical_error_rate(num_shots, 12, False)
    # print('Time taken')
    # print(time.time() - st)
    # print(simulation_results)
    # data_dictionary[0] = simulation_results
    # simulation.simulation_results_to_csv(data_dictionary, 'phenom_new')
    
    
    data_dictionary = pd.read_csv('phenom_new.csv')
    
    simulation = Simulation(rounds=rounds, distances=distances, noises=noises, \
        circuit_parameters={'code_task': 'surface_code:unrotated_memory_z', 'before_round_data_depolarization':'', 'before_measure_flip_probability':''})

    for d_index, distance in enumerate(distances):
        sub_df = data_dictionary[(data_dictionary['Distance'] == distance) & (data_dictionary['Number_of_Rounds'] == rounds[d_index])]
        logical_error_rates = sub_df['Logical_Error_Rate'].to_numpy()
        CI_logical_error_rates = simulation.compute_clopper_pearson_error_bars(logical_error_rates, 0.95, num_shots)
        plt.errorbar(noises, logical_error_rates, yerr=CI_logical_error_rates, fmt='o', capsize=10, c=next(colors), label=str(distance))
        # plt.plot(noises, logical_error_rates, c=next(colors), label=str(distance))
    
    
    plt.ylabel('Logical Error Rate')
    # plt.semilogy()
    plt.xlabel('Physical Error Rate')
    plt.legend()
    plt.savefig('new.png', bbox_inches="tight")
    plt.clf()
