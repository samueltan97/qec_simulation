from distutils.log import error
from typing import List
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

def logical_error_rate_function_with_burst_error(x, lim_logical_error_rate_per_round, logical_burst_error_rate):
    results = []
    for val in x[:int(len(x)/2)]:
        results.append(0.5*(1-((1 - lim_logical_error_rate_per_round)**val)))
    for val in x[int(len(x)/2):]:
        results.append(0.5*(1-((1 - lim_logical_error_rate_per_round)**val)*(1 - logical_burst_error_rate)))
    return np.array(results)

def calc_error_burst_threshold(num_shots:int, rounds: List[int], distances: List[int], \
    noises:List[float], error_burst_rates: np.ndarray, num_cores:int, csv_name:str, \
    process_data:bool, save_intermediate_plots:bool, plot_prefix:str, fix_rounds_to_distance:bool):
    colors = cycle(['tab:blue', 'tab:orange', 'tab:red', 'green'])
    data_dictionary = dict()
    rounds = 2*rounds
    burst_error_timesteps = [-1] * (len(rounds) // 2) + [x//2 for x in rounds[len(rounds) // 2:]] 
    simulation = Simulation(rounds=rounds, distances=distances, noises=noises, \
            circuit_parameters={'code_task': 'surface_code:rotated_memory_z', 'before_round_data_depolarization':'', 'before_measure_flip_probability':''})
    if fix_rounds_to_distance:
            for i in range(len(distances)):
                for j in range(len(noises)):
                    for k in range(len(rounds)):
                        if i != k and i != (k - len(distances)):
                            simulation.circuit_array[i][j][k] = None
    for burst_error_rate in error_burst_rates:
        st = time.time()
        simulation_results = simulation.simulate_logical_error_rate(num_shots, num_cores, True, burst_error_rate, burst_error_timesteps)
        print('Time taken')
        print(time.time() - st)
        print('Burst Error Rate')
        print(burst_error_rate)
        print(simulation_results)
        data_dictionary[burst_error_rate] = simulation_results
        simulation.simulation_results_to_csv(data_dictionary, csv_name)
    
    if process_data:

        data_dictionary = pd.read_csv(csv_name + '.csv')
        
        simulation = Simulation(rounds=rounds, distances=distances, noises=noises, \
                circuit_parameters={'code_task': 'surface_code:rotated_memory_z', 'before_round_data_depolarization':'', 'before_measure_flip_probability':''})
        
        logical_burst_error_rate_dict = dict()

        for d_index, distance in enumerate(distances):
            logical_burst_error_rates = []
            logical_burst_error_rate_sigmas = []
            for burst_error_rate in error_burst_rates:
                sub_df = data_dictionary[(np.isclose(data_dictionary['Burst_Error_Rate'], burst_error_rate)) & (data_dictionary['Distance'] == distance)]
                logical_error_rates = sub_df['Logical_Error_Rate'].to_numpy()
                CI_logical_error_rates = simulation.compute_clopper_pearson_error_bars(logical_error_rates, 0.95, num_shots)
                norm_dist_error_for_logical_error_rates = simulation.compute_sigma_values_with_wald_interval(logical_error_rates, num_shots)        
                
                plt.ylabel('Logical Error Rate')
                plt.semilogy()
                plt.xlabel('Number of Rounds')
                
                
                # For single point fixed rounds to distance plotting
                burst_error_bars = np.zeros(shape=(2,1))
                burstless_error_bars = np.zeros(shape=(2,1))
                for index, bounds in enumerate(CI_logical_error_rates):
                    burst_error_bars[index] = bounds[1]
                    burstless_error_bars[index] = bounds[0]
                plt.scatter([rounds[d_index]], [logical_error_rates[1]], label='distance = ' + str(distance) + ', phenomenological noise = 3%, \nerror burst rate = ' + str(burst_error_rate * 100) +'%')
                plt.scatter([rounds[d_index]], [logical_error_rates[0]], label='distance = ' + str(distance) + ', phenomenological noise = 3%')
                plt.errorbar([rounds[d_index]], [logical_error_rates[1]], yerr=burst_error_bars, fmt='o', capsize=10)
                plt.errorbar([rounds[d_index]], [logical_error_rates[0]], yerr=burstless_error_bars, fmt='o', capsize=10)
                initial_guess_lim_logical_error_rate_per_round = 1 - ((1 - 2*logical_error_rates[0])**(1/rounds[d_index]))
                initial_guess_logical_burst_error_rate = 1 - ((1 - 2*logical_error_rates[-1])/((1-initial_guess_lim_logical_error_rate_per_round)**rounds[d_index]))
                popt, pcov = curve_fit(logical_error_rate_function_with_burst_error, 2*[rounds[d_index]], logical_error_rates, [initial_guess_lim_logical_error_rate_per_round, initial_guess_logical_burst_error_rate], norm_dist_error_for_logical_error_rates, absolute_sigma=True)
                print(popt, pcov)
                logical_burst_error_rates.append(popt[1])    
                logical_burst_error_rate_sigmas.append(2*np.sqrt(np.diag(pcov))[1])        
                x = np.concatenate((np.linspace(1, 300, 300), np.linspace(1, 300, 300)))
                y = logical_error_rate_function_with_burst_error(x, *popt)
                plt.plot(x[int(len(x)/2):], y[int(len(x)/2):], c='tab:blue', linestyle='dashed', label='Burst Model Fit')
                plt.plot(x[:int(len(x)/2)], y[:int(len(x)/2)], c='tab:orange', linestyle='dashed', label='Burstless Model Fit')
                
                
                # plt.scatter(rounds[int(len(rounds)/2):], logical_error_rates[int(len(rounds)/2):], label='distance = ' + str(distance) + ', phenomenological noise = 2.25%, \nerror burst rate = ' + str(burst_error_rate * 100) +'%')
                # plt.scatter(rounds[:int(len(rounds)/2)], logical_error_rates[:int(len(rounds)/2)], label='distance = ' + str(distance) + ', phenomenological noise = 2.25%')
                # plt.errorbar(rounds[int(len(rounds)/2):], logical_error_rates[int(len(rounds)/2):], yerr=[x[int(len(rounds)/2):] for x in CI_logical_error_rates], fmt='o', capsize=10)
                # plt.errorbar(rounds[:int(len(rounds)/2)], logical_error_rates[:int(len(rounds)/2)], yerr=[x[:int(len(rounds)/2)] for x in CI_logical_error_rates], fmt='o', capsize=10)
                
                # initial_guess_lim_logical_error_rate_per_round = 1 - ((1 - 2*logical_error_rates[int((len(logical_error_rates)/2)) - 1])**(1/rounds[int((len(logical_error_rates)/2)) - 1]))
                # initial_guess_logical_burst_error_rate = 1 - ((1 - 2*logical_error_rates[-1])/((1-initial_guess_lim_logical_error_rate_per_round)**rounds[-1]))
                # popt, pcov = curve_fit(logical_error_rate_function_with_burst_error, rounds, logical_error_rates, [initial_guess_lim_logical_error_rate_per_round, initial_guess_logical_burst_error_rate], norm_dist_error_for_logical_error_rates, absolute_sigma=True)
                # logical_burst_error_rates.append(popt[1])    
                # logical_burst_error_rate_sigmas.append(2*np.sqrt(np.diag(pcov))[1])        
                # x = np.concatenate((np.linspace(1, 300, 300), np.linspace(1, 300, 300)))
                # y = logical_error_rate_function_with_burst_error(x, *popt)
                # plt.plot(x[int(len(x)/2):], y[int(len(x)/2):], c='tab:blue', linestyle='dashed', label='Burst Model Fit')
                # plt.plot(x[:int(len(x)/2)], y[:int(len(x)/2)], c='tab:orange', linestyle='dashed', label='Burstless Model Fit')
                plt.legend()
                if save_intermediate_plots:
                    plt.savefig(str(distance) + '_' + str(burst_error_rate) + '.png', bbox_inches="tight")
                plt.clf()
            logical_burst_error_rate_dict[distance] = (logical_burst_error_rates, logical_burst_error_rate_sigmas)
        
        for key in list(logical_burst_error_rate_dict.keys()):
            color = next(colors)
            plt.errorbar(error_burst_rates, logical_burst_error_rate_dict[key][0], yerr=logical_burst_error_rate_dict[key][1], c=color, fmt='o', capsize=10, label='distance = ' + str(key)+ ', phenomenological noise = ' + str(noises[0] * 100) + '%')
            res = linregress(error_burst_rates, logical_burst_error_rate_dict[key][0])
            plt.plot(np.linspace(error_burst_rates[0], error_burst_rates[-1], 300), res.intercept + res.slope*np.linspace(error_burst_rates[0], error_burst_rates[-1], 300), c=color)

        plt.legend()
        plt.ylabel('Logical Error Burst Rate')
        # plt.semilogy()
        plt.xlabel('Error Burst Rate')
        plt.grid(b=True, which='major', linestyle='-')
        plt.grid(b=True, which='minor', linestyle='--')
        plt.minorticks_on()
        plt.savefig(plot_prefix + '_' + str(noises[0] * 100) + '%_phenomenological_noise_burst_error_threshold.png', bbox_inches="tight")
        plt.clf()
        
        
        
        


    

    