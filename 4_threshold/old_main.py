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
    from scipy.stats import linregress
    import stim
    import pandas as pd
    from itertools import cycle

    colors = cycle(['tab:blue', 'tab:orange', 'tab:red', 'yellow'])

    data_dictionary = dict()
    rounds = [64, 96, 128, 160, 192, 64, 96, 128, 160, 192]
    distances = [3, 5, 7]
    noises = [0.02]
    burst_error_timesteps = [-1, -1, -1, -1, -1, 32, 48, 64, 80, 96]
    burst_error_rates = np.linspace(0.1, 0.12, 10)
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
    # simulation.simulation_results_to_csv(data_dictionary, '005_new_results')
    
    data_dictionary = pd.read_csv('020_new_results.csv')
    
    simulation = Simulation(rounds=rounds, distances=distances, noises=noises, \
            circuit_parameters={'code_task': 'surface_code:rotated_memory_z', 'before_round_data_depolarization':''})
    
    logical_burst_error_rate_dict = dict()

    for d_index, distance in enumerate(distances):
        logical_burst_error_rates = []
        logical_burst_error_rate_sigmas = []
        for burst_error_rate in burst_error_rates:
            sub_df = data_dictionary[(np.isclose(data_dictionary['Burst_Error_Rate'], burst_error_rate)) & (data_dictionary['Distance'] == distance)]
            logical_error_rates = sub_df['Logical_Error_Rate'].to_numpy()
            CI_logical_error_rates = simulation.compute_clopper_pearson_error_bars(logical_error_rates, 0.95, 10000)
            norm_dist_error_for_logical_error_rates = simulation.compute_sigma_values_with_wald_interval(logical_error_rates, 10000)
            plt.ylabel('Logical Error Rate')
            plt.semilogy()
            plt.xlabel('Number of Rounds')
            plt.scatter(rounds[int(len(rounds)/2):], logical_error_rates[int(len(rounds)/2):], label='distance = ' + str(distance) + ', phenomenological noise = 1.5%, \nburst error rate = ' + str(burst_error_rate * 100) +'%')
            plt.scatter(rounds[:int(len(rounds)/2)], logical_error_rates[:int(len(rounds)/2)], label='distance = ' + str(distance) + ', phenomenological noise = 1.5%')
            plt.errorbar(rounds[int(len(rounds)/2):], logical_error_rates[int(len(rounds)/2):], yerr=[x[int(len(rounds)/2):] for x in CI_logical_error_rates], fmt='o', capsize=10)
            plt.errorbar(rounds[:int(len(rounds)/2)], logical_error_rates[:int(len(rounds)/2)], yerr=[x[:int(len(rounds)/2)] for x in CI_logical_error_rates], fmt='o', capsize=10)
            
            initial_guess_lim_logical_error_rate_per_round = 1 - ((1 - 2*logical_error_rates[int((len(logical_error_rates)/2)) - 1])**(1/rounds[int((len(logical_error_rates)/2)) - 1]))
            initial_guess_logical_burst_error_rate = 1 - ((1 - 2*logical_error_rates[-1])/((1-initial_guess_lim_logical_error_rate_per_round)**rounds[-1]))
            popt, pcov = curve_fit(logical_error_rate_function_with_burst_error, rounds, logical_error_rates, [initial_guess_lim_logical_error_rate_per_round, initial_guess_logical_burst_error_rate], norm_dist_error_for_logical_error_rates, absolute_sigma=True)
            logical_burst_error_rates.append(popt[1])    
            logical_burst_error_rate_sigmas.append(2*np.sqrt(np.diag(pcov))[1])        
            x = np.concatenate((np.linspace(1, 300, 300), np.linspace(1, 300, 300)))
            y = logical_error_rate_function_with_burst_error(x, *popt)
            if distance == 3 and burst_error_rate == 0.1:
                print(popt)
                print(2*np.sqrt(np.diag(pcov)))
            max_y = logical_error_rate_function_with_burst_error(x, *(popt + 2*np.sqrt(np.diag(pcov))))
            min_y = logical_error_rate_function_with_burst_error(x, *(popt - 2*np.sqrt(np.diag(pcov))))
            plt.plot(x[int(len(x)/2):], y[int(len(x)/2):], c='tab:blue', linestyle='dashed', label='Burst Model Fit')
            plt.plot(x[:int(len(x)/2)], y[:int(len(x)/2)], c='tab:orange', linestyle='dashed', label='Burstless Model Fit')
            plt.legend()
            plt.savefig(str(distance) + '_' + str(burst_error_rate) + '_new.png', bbox_inches="tight")
            plt.clf()
        logical_burst_error_rate_dict[distance] = (logical_burst_error_rates, logical_burst_error_rate_sigmas)
    
    for key in list(logical_burst_error_rate_dict.keys()):
        color = next(colors)
        print(key, logical_burst_error_rate_dict[key])
        # plt.plot(burst_error_rates, logical_burst_error_rate_dict[key][0], c=color, label='distance = ' + str(key)+ ', phenomenological noise = ' + str(noises[0] * 100) + '%')
        plt.errorbar(burst_error_rates, logical_burst_error_rate_dict[key][0], yerr=logical_burst_error_rate_dict[key][1], c=color, fmt='o', capsize=10, label='distance = ' + str(key)+ ', phenomenological noise = ' + str(noises[0] * 100) + '%')
        res = linregress(burst_error_rates, logical_burst_error_rate_dict[key][0])
        plt.plot(np.linspace(0.1, 0.12, 300), res.intercept + res.slope*np.linspace(0.1, 0.12, 300), c=color)

    plt.legend()
    plt.ylabel('Logical Error Burst Rate')
    # plt.semilogy()
    plt.xlabel('Error Burst Rate')
    plt.grid(b=True, which='major', linestyle='-')
    plt.grid(b=True, which='minor', linestyle='--')
    plt.minorticks_on()
    plt.savefig(str(noises[0] * 100) + '%_phenomenological_noise_burst_error_threshold_new.png', bbox_inches="tight")
    plt.clf()

    # iterate through the plot colors
    # colors = itertools.cycle(color_list)
    # 100000 samples
    '''
    For threshold plots, we want double the number of data points. Can try fitting a linear plot.
    Relabel error fitting as wald interval
    Check out section 3.3 in reading as well as numerics section
    Read overlapping recovery method section in topological quantum memory


    For the note to John (target audience: someone who knows toric code but not super familiar with error models)
    Main parts:
    1. Describe what a burst error is and talk about the error models
    2. Describe what exactly I am simulating
    3. Discuss what the results are

    Burst error induced burst error
    Assumptions: 1. We know that a cosmic ray occurred 2. The device is capable of recovering from it (device-side and not information-side)
    Try to infer when the burst error happened from the spike in error rate

    Control error induced burst error (laser power excursions)
    CPU clock can mess up tau to mess up the bit flip gate (overrotation) (refer to first pic) [tau should respect a normal distribution]
    Coupling strength is dependent on the frequency of the laser and its intensity (refer to first picture). Laser could screw up [J should respect a normal distribution. The long tails of the distribution are burst errors]

    Scenario 1: We know how much we have overrotated by : Can just apply the inverse but might not be practical
    Scenario 2: We don't know how much we have overrotated by: Experimentally relevant

    Very hard when we don't know when the burst error took place
    '''
        
        


     
    
    


    

    