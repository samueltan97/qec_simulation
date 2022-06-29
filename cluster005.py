from typing import List
from simulate import Simulation
from typing import List
import numpy as np

def logical_error_rate_function_with_burst_error(x, lim_logical_error_rate_per_round, logical_burst_error_rate):
    results = []
    for val in x[:int(len(x)/2)]:
        results.append(0.5*(1-((1 - lim_logical_error_rate_per_round)**val)))
    for val in x[int(len(x)/2):]:
        results.append(0.5*(1-((1 - lim_logical_error_rate_per_round)**val)*(1 - logical_burst_error_rate)))
    return np.array(results)

if __name__ == "__main__":
    from simulate import Simulation
    import time
    import numpy as np
    from functools import partial
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    import stim
    import pandas as pd

    data_dictionary = dict()
    rounds = [64, 96, 128, 160, 192, 64, 96, 128, 160, 192]
    distances = [3, 5, 7]
    noises = [0.005]
    burst_error_timesteps = [-1, -1, -1, -1, -1, 32, 48, 64, 80, 96]
    burst_error_rates = np.linspace(0.12, 0.15, 10)
    for burst_error_rate in burst_error_rates:
        st = time.time()
        simulation = Simulation(rounds=rounds, distances=distances, noises=noises, \
            circuit_parameters={'code_task': 'surface_code:rotated_memory_z', 'before_round_data_depolarization':''})
        simulation_results = simulation.simulate_logical_error_rate(100000, 56, True, burst_error_rate, burst_error_timesteps)
        print('Time taken')
        print(time.time() - st)
        print('Burst Error Rate')
        print(burst_error_rate)
        print(simulation_results)
        data_dictionary[burst_error_rate] = simulation_results
        simulation.simulation_results_to_csv(data_dictionary, '005_new_results_expanded_0629')
    
    # data_dictionary = pd.read_csv('015_new_results.csv')
    
    # simulation = Simulation(rounds=rounds, distances=distances, noises=noises, \
    #         circuit_parameters={'code_task': 'surface_code:rotated_memory_z', 'before_round_data_depolarization':''})
    
    # logical_burst_error_rate_dict = dict()

    # for d_index, distance in enumerate(distances):
    #     logical_burst_error_rates = []
    #     logical_burst_error_rate_sigmas = []
    #     for burst_error_rate in burst_error_rates:
    #         sub_df = data_dictionary[(np.isclose(data_dictionary['Burst_Error_Rate'], burst_error_rate)) & (data_dictionary['Distance'] == distance)]
    #         logical_error_rates = sub_df['Logical_Error_Rate'].to_numpy()
    #         CI_logical_error_rates = simulation.compute_clopper_pearson_error_bars(logical_error_rates, 0.95, 10000)
    #         norm_dist_error_for_logical_error_rates = simulation.compute_sigma_values(logical_error_rates, 10000)
    #         plt.ylabel('Logical Error Rate')
    #         plt.semilogy()
    #         plt.xlabel('Number of Rounds')
    #         plt.scatter(rounds[int(len(rounds)/2):], logical_error_rates[int(len(rounds)/2):], label='distance = ' + str(distance) + ', phenomenological noise = 2%, \nburst error rate = ' + str(burst_error_rate * 100) +'%')
    #         plt.scatter(rounds[:int(len(rounds)/2)], logical_error_rates[:int(len(rounds)/2)], label='distance = ' + str(distance) + ', phenomenological noise = 2%')
    #         plt.errorbar(rounds[int(len(rounds)/2):], logical_error_rates[int(len(rounds)/2):], yerr=[x[int(len(rounds)/2):] for x in CI_logical_error_rates], fmt='o', capsize=10)
    #         plt.errorbar(rounds[:int(len(rounds)/2)], logical_error_rates[:int(len(rounds)/2)], yerr=[x[:int(len(rounds)/2)] for x in CI_logical_error_rates], fmt='o', capsize=10)
            
    #         initial_guess_lim_logical_error_rate_per_round = 1 - ((1 - 2*logical_error_rates[int((len(logical_error_rates)/2)) - 1])**(1/rounds[int((len(logical_error_rates)/2)) - 1]))
    #         initial_guess_logical_burst_error_rate = 1 - ((1 - 2*logical_error_rates[-1])/((1-initial_guess_lim_logical_error_rate_per_round)**rounds[-1]))
    #         popt, pcov = curve_fit(logical_error_rate_function_with_burst_error, rounds, logical_error_rates, [initial_guess_lim_logical_error_rate_per_round, initial_guess_logical_burst_error_rate], norm_dist_error_for_logical_error_rates, absolute_sigma=True)
    #         logical_burst_error_rates.append(popt[1])    
    #         logical_burst_error_rate_sigmas.append(2*np.sqrt(np.diag(pcov))[1])        
    #         x = np.concatenate((np.linspace(1, 300, 300), np.linspace(1, 300, 300)))
    #         y = logical_error_rate_function_with_burst_error(x, *popt)
    #         max_y = logical_error_rate_function_with_burst_error(x, *(popt + 2*np.sqrt(np.diag(pcov))))
    #         min_y = logical_error_rate_function_with_burst_error(x, *(popt - 2*np.sqrt(np.diag(pcov))))
    #         plt.plot(x[int(len(x)/2):], y[int(len(x)/2):], 'r--', label='Burst Model Fit')
    #         plt.plot(x[:int(len(x)/2)], y[:int(len(x)/2)], 'g--', label='Burstless Model Fit')
    #         plt.legend()
    #         plt.savefig(str(distance) + '_' + str(burst_error_rate) + '_new.png', bbox_inches="tight")
    #         plt.clf()
    #     logical_burst_error_rate_dict[distance] = (logical_burst_error_rates, logical_burst_error_rate_sigmas)
    
    # for key in list(logical_burst_error_rate_dict.keys()):
    #     print(key, logical_burst_error_rate_dict[key])
    #     plt.plot(burst_error_rates, logical_burst_error_rate_dict[key][0], label='distance = ' + str(key)+ ', phenomenological noise = 2%')
    #     plt.errorbar(burst_error_rates, logical_burst_error_rate_dict[key][0], yerr=logical_burst_error_rate_dict[key][1], fmt='o', capsize=10)

    # plt.legend()
    # plt.ylabel('Logical Burst Error Rate')
    # plt.semilogy()
    # plt.xlabel('Burst Error Rate')
    # plt.savefig('2%_phenomenological_noise_burst_error_threshold_new.png', bbox_inches="tight")
    # plt.clf()
        
        


     
    
    


    

    