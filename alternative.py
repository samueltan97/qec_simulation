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
    rounds = [64, 96, 128, 160, 192, 224, 256, 64, 96, 128, 160, 192, 224, 256]
    distances = [3, 5, 7]
    noises = [0.02]
    burst_error_timesteps = [-1, -1, -1, -1, -1, -1, -1, 32, 48, 64, 80, 96, 112, 128]
    burst_error_rates = [0.025, 0.05, 0.075, 0.1]
    for burst_error_rate in burst_error_rates:
        st = time.time()
        simulation = Simulation(rounds=rounds, distances=distances, noises=noises, \
            circuit_parameters={'code_task': 'surface_code:rotated_memory_z', 'before_round_data_depolarization':''})
        simulation_results = simulation.simulate_logical_error_rate(10000, 12, True, burst_error_rate, burst_error_timesteps)
        print('Time taken')
        print(time.time() - st)
        print('Burst Error Rate')
        print(burst_error_rate)
        print(simulation_results)
        data_dictionary[burst_error_rate] = simulation_results

    # data_dictionary[0.06] = [[[0.1594, 0.2221, 0.2654, 0.3211, 0.3726, 0.4254, 0.4304, 0.1924, 0.255, 0.2935,\
    #     0.3393, 0.3873, 0.4202, 0.4376]], [[0.0395, 0.0771, 0.1064, 0.1063, 0.123, 0.1613, 0.1752, 0.0725, \
    #     0.1113, 0.1459, 0.1393, 0.1602, 0.189, 0.2062]], [[0.0152, 0.0359, 0.0253, 0.0513, 0.0514, 0.0398, \
    #     0.0423, 0.0493, 0.0716, 0.0558, 0.0923, 0.088, 0.0695, 0.0637]]] 

    simulation = Simulation(rounds=rounds, distances=distances, noises=noises, \
            circuit_parameters={'code_task': 'surface_code:rotated_memory_z', 'before_round_data_depolarization':''})
        
    for d_index, distance in enumerate(distances):
        logical_burst_error_rate = []
        for burst_error_rate in burst_error_rates:
            logical_error_rates = data_dictionary[burst_error_rate][d_index][0]
            CI_logical_error_rates = compute_error_bars(simulation, logical_error_rates, 0.95, 10000)
            plt.ylabel('Logical Error Rate')
            plt.semilogy()
            plt.xlabel('Number of Rounds')
            plt.scatter(rounds[int(len(rounds)/2):], logical_error_rates[int(len(rounds)/2):], label='distance = ' + str(distance) + ', phenomenological noise = 2%, \nburst error rate = ' + str(burst_error_rate * 100) +'%')
            plt.errorbar(rounds[int(len(rounds)/2):], logical_error_rates[int(len(rounds)/2):], yerr=[x[int(len(rounds)/2):] for x in CI_logical_error_rates], fmt='o', capsize=10)
            
            initial_guess_lim_logical_error_rate_per_round = 1 - ((1 - 2*logical_error_rates[int((len(logical_error_rates)/2)) - 1])**(1/rounds[int((len(logical_error_rates)/2)) - 1]))
            initial_guess_logical_burst_error_rate = 1 - ((1 - 2*logical_error_rates[-1])/((1-initial_guess_lim_logical_error_rate_per_round)**rounds[-1]))
            popt, pcov = curve_fit(logical_error_rate_function_with_burst_error, rounds, logical_error_rates, [initial_guess_lim_logical_error_rate_per_round, initial_guess_logical_burst_error_rate])
            # print(popt)
            logical_burst_error_rate.append(popt[1])            
            x = np.concatenate((np.linspace(1, 300, 300), np.linspace(1, 300, 300)))
            y = logical_error_rate_function_with_burst_error(x, *popt)
            plt.plot(x[int(len(x)/2):], y[int(len(x)/2):], 'g--', label='Model Fit')
            plt.legend()
            plt.savefig(str(distance) + '_' + str(burst_error_rate) + '_new.png', bbox_inches="tight")
            plt.clf()
        # plt.plot(burst_error_rates, logical_burst_error_rate, label='distance = ' + str(distance)+ ', phenomenological noise = 2%')
    plt.legend()
    plt.ylabel('Logical Burst Error Rate')
    plt.semilogy()
    plt.xlabel('Burst Error Rate')
    plt.savefig('2%_phenomenological_noise_burst_error_threshold_new.png', bbox_inches="tight")
    plt.clf()
        
        


     
    
    


    

    