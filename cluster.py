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

def compute_error_bars(simulation: Simulation, logical_error_rates: np.ndarray, confidence_interval:float, num_shots: int) -> List:
    CI_upper_bound_logical_error_rates = [simulation.create_clopper_pearson_interval(confidence_interval, x*num_shots, num_shots)[1] for x in logical_error_rates]
    CI_lower_bound_logical_error_rates = [simulation.create_clopper_pearson_interval(confidence_interval, x*num_shots, num_shots)[0] for x in logical_error_rates]
    for i in range(len(logical_error_rates)):
        CI_upper_bound_logical_error_rates[i] = abs(logical_error_rates[i] - CI_upper_bound_logical_error_rates[i])
        CI_lower_bound_logical_error_rates[i] = abs(logical_error_rates[i] - CI_lower_bound_logical_error_rates[i])
    return [CI_lower_bound_logical_error_rates, CI_upper_bound_logical_error_rates]

if __name__ == "__main__":
    from simulate import Simulation
    import time
    import numpy as np
    from functools import partial
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    import stim

    data_dictionary = dict()
    rounds = [64, 96, 128, 160, 192, 64, 96, 128, 160, 192]
    distances = [3, 5, 7]
    noises = [0.01]
    burst_error_timesteps = [-1, -1, -1, -1, -1, 32, 48, 64, 80, 96]
    burst_error_rates = np.linspace(0.1, 0.12, 10)
    for burst_error_rate in burst_error_rates:
        st = time.time()
        simulation = Simulation(rounds=rounds, distances=distances, noises=noises, \
            circuit_parameters={'code_task': 'surface_code:rotated_memory_z', 'before_round_data_depolarization':''})
        simulation_results = simulation.simulate_logical_error_rate(10000, 56, True, burst_error_rate, burst_error_timesteps)
        print('Time taken')
        print(time.time() - st)
        print('Burst Error Rate')
        print(burst_error_rate)
        print(simulation_results)
        data_dictionary[burst_error_rate] = simulation_results
    simulation.simulation_results_to_csv(data_dictionary, '01_results')
    # data_dictionary[0.025] = [[[0.1647, 0.2253, 0.2667, 0.315, 0.3458, 0.1755, 0.2265, 0.2787, 0.317, 0.3465]],\
    #     [[0.0397, 0.0542, 0.0753, 0.093, 0.1069, 0.0443, 0.0626, 0.0846, 0.1005, 0.1129]], [[0.0071,\
    #     0.0112, 0.0186, 0.0204, 0.0211, 0.0136, 0.0176, 0.0203, 0.0225, 0.0285]]]

    # data_dictionary[0.075] = [[[0.1614, 0.2248, 0.2719, 0.3131, 0.3485, 0.2033, 0.2634, 0.3002, 0.3373, 0.3648]],\
    #     [[0.0375, 0.0522, 0.0734, 0.0972, 0.1083, 0.0902, 0.1021, 0.1193, 0.1331, 0.149]], [[0.0091, 0.0115,\
    #     0.0169, 0.0203, 0.0248, 0.0546, 0.0599, 0.0606, 0.0661, 0.0692]]]

    # data_dictionary[0.125] = [[[0.1579, 0.2255, 0.2771, 0.3019, 0.3405, 0.2413, 0.29, 0.3159, 0.3546, 0.3958]],\
    #     [[0.0402, 0.0562, 0.0745, 0.0896, 0.1069, 0.1614, 0.1743, 0.1858, 0.2027, 0.2186]], [[0.0074, 0.0127,\
    #     0.0164, 0.0186, 0.025, 0.1387, 0.1417, 0.1458, 0.1606, 0.1554]]]
    
    simulation = Simulation(rounds=rounds, distances=distances, noises=noises, \
            circuit_parameters={'code_task': 'surface_code:rotated_memory_z', 'before_round_data_depolarization':''})
    
    logical_burst_error_rate_dict = dict()

    for d_index, distance in enumerate(distances):
        logical_burst_error_rate = []
        for burst_error_rate in burst_error_rates:
            logical_error_rates = data_dictionary[burst_error_rate][d_index][0]
            CI_logical_error_rates = simulation.compute_error_bars(simulation, logical_error_rates, 0.95, 10000)
            plt.ylabel('Logical Error Rate')
            plt.semilogy()
            plt.xlabel('Number of Rounds')
            plt.scatter(rounds[int(len(rounds)/2):], logical_error_rates[int(len(rounds)/2):], label='distance = ' + str(distance) + ', phenomenological noise = 2%, \nburst error rate = ' + str(burst_error_rate * 100) +'%')
            plt.scatter(rounds[:int(len(rounds)/2)], logical_error_rates[:int(len(rounds)/2)], label='distance = ' + str(distance) + ', phenomenological noise = 2%')
            plt.errorbar(rounds[int(len(rounds)/2):], logical_error_rates[int(len(rounds)/2):], yerr=[x[int(len(rounds)/2):] for x in CI_logical_error_rates], fmt='o', capsize=10)
            plt.errorbar(rounds[:int(len(rounds)/2)], logical_error_rates[:int(len(rounds)/2)], yerr=[x[:int(len(rounds)/2)] for x in CI_logical_error_rates], fmt='o', capsize=10)
            
            initial_guess_lim_logical_error_rate_per_round = 1 - ((1 - 2*logical_error_rates[int((len(logical_error_rates)/2)) - 1])**(1/rounds[int((len(logical_error_rates)/2)) - 1]))
            initial_guess_logical_burst_error_rate = 1 - ((1 - 2*logical_error_rates[-1])/((1-initial_guess_lim_logical_error_rate_per_round)**rounds[-1]))
            popt, pcov = curve_fit(logical_error_rate_function_with_burst_error, rounds, logical_error_rates, [initial_guess_lim_logical_error_rate_per_round, initial_guess_logical_burst_error_rate])
            print(popt)
            logical_burst_error_rate.append(popt[1])            
            x = np.concatenate((np.linspace(1, 300, 300), np.linspace(1, 300, 300)))
            y = logical_error_rate_function_with_burst_error(x, *popt)
            plt.plot(x[int(len(x)/2):], y[int(len(x)/2):], 'r--', label='Burst Model Fit')
            plt.plot(x[:int(len(x)/2)], y[:int(len(x)/2)], 'g--', label='Burstless Model Fit')
            plt.legend()
            plt.savefig(str(distance) + '_' + str(burst_error_rate) + '_new.png', bbox_inches="tight")
            plt.clf()
        logical_burst_error_rate_dict[distance] = logical_burst_error_rate
    
    for key in list(logical_burst_error_rate_dict.keys()):
        print(key, logical_burst_error_rate_dict[key])
        plt.plot(burst_error_rates, logical_burst_error_rate_dict[key], label='distance = ' + str(key)+ ', phenomenological noise = 2%')
    

    plt.legend()
    plt.ylabel('Logical Burst Error Rate')
    plt.semilogy()
    plt.xlabel('Burst Error Rate')
    # plt.savefig('2%_phenomenological_noise_burst_error_threshold_new.png', bbox_inches="tight")
    plt.clf()
        
        


     
    
    


    

    