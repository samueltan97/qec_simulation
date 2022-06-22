from typing import List
from simulate import Simulation
from typing import List
import numpy as np

def logical_error_rate_function_with_burst_error(x, lim_logical_error_rate_per_round, logical_burst_error_rate):
    return 0.5*(1-((1 - lim_logical_error_rate_per_round)**x)*(1 - logical_burst_error_rate))

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
    rounds = [32, 64, 128, 256]
    distances = [3, 5, 7]
    noises = [0.02]
    burst_error_timesteps = [16, 32, 64, 128]
    burst_error_rates = np.linspace(0.025, 0.1, 5)
    # for burst_error_rate in burst_error_rates:
    #     st = time.time()
    #     simulation = Simulation(rounds=rounds, distances=distances, noises=noises, \
    #         circuit_parameters={'code_task': 'surface_code:rotated_memory_z', 'before_round_data_depolarization':''})
    #     simulation_results = simulation.simulate_logical_error_rate(10000, 12, True, burst_error_rate, burst_error_timesteps)
    #     print('Time taken')
    #     print(time.time() - st)
    #     print('Burst Error Rate')
    #     print(burst_error_rate)
    #     print(simulation_results)
    #     data_dictionary[burst_error_rate] = simulation_results
    data_dictionary[0.025] = [[[0.1046, 0.1774, 0.2838, 0.4395]],\
        [[0.0296, 0.0501, 0.1189, 0.1795]],\
        [[0.0086, 0.0187, 0.0323, 0.0466]]]
    data_dictionary[0.043750000000000004] = [[[0.117, 0.1919, 0.2811, 0.4371]],\
        [[0.0424, 0.0614, 0.1283, 0.1864]],\
        [[0.0177, 0.0332, 0.0422, 0.0566]]]
    data_dictionary[0.0625] = [[[0.1287, 0.2025, 0.297, 0.4467]],\
        [[0.0629, 0.0792, 0.142, 0.2076]],\
        [[0.0333, 0.0493, 0.0611, 0.0708]]]
    data_dictionary[0.08125000000000002] = [[[0.1546, 0.2078, 0.3069, 0.4442]],\
        [[0.0861, 0.0959, 0.1704, 0.2231]],\
        [[0.0605, 0.079,  0.0833, 0.0856]]]
    data_dictionary[0.1] = [[[0.171, 0.2194, 0.3247, 0.4394]],\
        [[0.107,  0.1182, 0.1913, 0.2438]],\
        [[0.093, 0.109,  0.1111, 0.1182]]]   
    simulation = Simulation(rounds=rounds, distances=distances, noises=noises, \
            circuit_parameters={'code_task': 'surface_code:rotated_memory_z', 'before_round_data_depolarization':''})
    
    logical_burst_error_rate_dict = dict()
    for d_index, distance in enumerate(distances):
        logical_burst_error_rate = []
        for burst_error_rate in burst_error_rates:
            logical_error_rates = data_dictionary[burst_error_rate][d_index][0]
            CI_logical_error_rates = compute_error_bars(simulation, logical_error_rates, 0.95, 10000)
            plt.ylabel('Logical Error Rate')
            plt.semilogy()
            plt.xlabel('Number of Rounds')
            plt.scatter(rounds, logical_error_rates, label='distance = ' + str(distance) + ', phenomenological noise = 2%, \nburst error rate = ' + str(burst_error_rate * 100) +'%')
            plt.errorbar(rounds, logical_error_rates, yerr=CI_logical_error_rates, fmt='o', capsize=10)
            
            initial_guess_lim_logical_error_rate_per_round = 1 - ((1 - 2*logical_error_rates[-1])**(1/rounds[-1]))
            # if distance == 7:
            #     print(initial_guess_lim_logical_error_rate_per_round)
            #     initial_guess_lim_logical_error_rate_per_round = 0.00036809
            # initial_guess_logical_burst_error_rate = 1 - ((1 - 2*logical_error_rates[-1])/((1-initial_guess_lim_logical_error_rate_per_round)**rounds[-1]))
            initial_guess_logical_burst_error_rate = 5*initial_guess_lim_logical_error_rate_per_round
            # print(initial_guess_lim_logical_error_rate_per_round)
            # print(initial_guess_logical_burst_error_rate)
            popt, pcov = curve_fit(logical_error_rate_function_with_burst_error, rounds, logical_error_rates, [initial_guess_lim_logical_error_rate_per_round, initial_guess_logical_burst_error_rate])
            logical_burst_error_rate.append(popt[1])            
            x = np.linspace(1, 300, 300)
            y = logical_error_rate_function_with_burst_error(x, *popt)
            plt.plot(x, y, 'g--', label='Model Fit')
            plt.legend()
            plt.savefig(str(distance) + '_' + str(burst_error_rate) + '.png', bbox_inches="tight")
            plt.clf()
        logical_burst_error_rate_dict[distance] = logical_burst_error_rate
    for key in list(logical_burst_error_rate_dict.keys()):
        print(key, logical_burst_error_rate_dict[key])
        plt.plot(burst_error_rates, logical_burst_error_rate_dict[key], label='distance = ' + str(key)+ ', phenomenological noise = 2%')
    plt.legend()
    plt.ylabel('Logical Burst Error Rate')
    plt.semilogy()
    plt.xlabel('Burst Error Rate')
    plt.savefig('2%_phenomenological_noise_burst_error_threshold.png', bbox_inches="tight")
    plt.clf()
        

0.06
[[[0.1594 0.2221 0.2654 0.3211 0.3726 0.4254 0.4304 0.1924 0.255  0.2935
   0.3393 0.3873 0.4202 0.4376]]

 [[0.0395 0.0771 0.1064 0.1063 0.123  0.1613 0.1752 0.0725 0.1113 0.1459
   0.1393 0.1602 0.189  0.2062]]

 [[0.0152 0.0359 0.0253 0.0513 0.0514 0.0398 0.0423 0.0493 0.0716 0.0558
   0.0923 0.088  0.0695 0.0637]]]  


     
    
    


    

    