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
    num_shots = 100000
    data_dictionary = dict()
    swapped_data_dictionary = dict()
    rounds = [64, 96, 128, 160, 192]
    distances = [5]
    noises = [0.02]
    burst_error_timesteps = [32, 48, 64, 80, 96]
    burst_error_rates = [0.1]
    for burst_error_rate in burst_error_rates:
        st = time.time()
        simulation = Simulation(rounds=rounds, distances=distances, noises=noises, \
            circuit_parameters={'code_task': 'surface_code:rotated_memory_z', 'before_round_data_depolarization':''})
        simulation_results = simulation.simulate_logical_error_rate(num_shots, 12, True, burst_error_rate, burst_error_timesteps)
        swapped_simulation_results = simulation.simulate_logical_error_rate_with_circuits_swapped(num_shots, 12, True, burst_error_rate, burst_error_timesteps)
        print('Time taken')
        print(time.time() - st)
        print(simulation_results)
        print(swapped_simulation_results)
        data_dictionary[burst_error_rate] = simulation_results
        swapped_data_dictionary[burst_error_rate] = swapped_simulation_results
        simulation.simulation_results_to_csv(data_dictionary, '020_unswapped')
        simulation.simulation_results_to_csv(swapped_data_dictionary, '020_swapped')
    
    
    data_dictionary = pd.read_csv('020_unswapped.csv')
    swapped_data_dictionary = pd.read_csv('020_swapped.csv')
    
    simulation = Simulation(rounds=rounds, distances=distances, noises=noises, \
            circuit_parameters={'code_task': 'surface_code:rotated_memory_z', 'before_round_data_depolarization':''})
    

    logical_error_rates = data_dictionary['Logical_Error_Rate'].to_numpy()
    swapped_logical_error_rates = swapped_data_dictionary['Logical_Error_Rate'].to_numpy()
    CI_logical_error_rates = simulation.compute_clopper_pearson_error_bars(logical_error_rates, 0.95, num_shots)
    swapped_CI_logical_error_rates = simulation.compute_clopper_pearson_error_bars(swapped_logical_error_rates, 0.95, num_shots)
    # norm_dist_error_for_logical_error_rates = simulation.compute_sigma_values_with_wald_interval(logical_error_rates, num_shots)
    # print(CI_logical_error_rates)
    # print(norm_dist_error_for_logical_error_rates)
        
    plt.ylabel('Logical Error Rate')
    plt.semilogy()
    plt.xlabel('Number of Rounds')
    plt.errorbar(rounds, logical_error_rates, yerr=CI_logical_error_rates, fmt='o', capsize=10, c=next(colors))
    plt.errorbar(rounds, swapped_logical_error_rates, yerr=swapped_CI_logical_error_rates, fmt='o', capsize=10, c=next(colors))
    
    # initial_guess_lim_logical_error_rate_per_round = 1 - ((1 - 2*logical_error_rates[int((len(logical_error_rates)/2)) - 1])**(1/rounds[int((len(logical_error_rates)/2)) - 1]))
    # initial_guess_logical_burst_error_rate = 1 - ((1 - 2*logical_error_rates[-1])/((1-initial_guess_lim_logical_error_rate_per_round)**rounds[-1]))
    # popt, pcov = curve_fit(logical_error_rate_function_with_burst_error, rounds, logical_error_rates, [initial_guess_lim_logical_error_rate_per_round, initial_guess_logical_burst_error_rate], norm_dist_error_for_logical_error_rates, absolute_sigma=True)
    # logical_burst_error_rates.append(popt[1])    
    # logical_burst_error_rate_sigmas.append(2*np.sqrt(np.diag(pcov))[1])        
    # x = np.concatenate((np.linspace(1, 300, 300), np.linspace(1, 300, 300)))
    # y = logical_error_rate_function_with_burst_error(x, *popt)
    # plt.plot(x[int(len(x)/2):], y[int(len(x)/2):], c='tab:blue', linestyle='dashed', label='Burst Model Fit')
    # plt.plot(x[:int(len(x)/2)], y[:int(len(x)/2)], c='tab:orange', linestyle='dashed', label='Burstless Model Fit')
    # plt.legend()
    plt.savefig(str(distances[0]) + '_' + str(burst_error_rates[0]) + '_new.png', bbox_inches="tight")
    plt.clf()
