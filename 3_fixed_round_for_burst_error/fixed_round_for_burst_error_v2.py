from numpy import insert
import stim

def logical_error_rate_piecewise_function(x, lim_logical_error_rate_per_round, logical_burst_error_rate):
    result = []
    for val in x:
        if val < 70:
            result.append(0.5*(1-((1 - lim_logical_error_rate_per_round)**val)))
        else:
            result.append(0.5*(1-((1 - lim_logical_error_rate_per_round)**val)*(1 - logical_burst_error_rate)))
    return np.array(result)


if __name__ == "__main__":
    from simulate import Simulation
    import time
    import numpy as np
    from functools import partial
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    import stim
    import pandas as pd

    st = time.time()
    # burst error occurs at time step 70
    simulation = Simulation(rounds=[8, 16, 32, 40, 50, 60, 80, 120, 160, 190, 220, 256], distances=[3], noises=[0.015], \
        circuit_parameters={'code_task': 'surface_code:rotated_memory_z', 'before_round_data_depolarization':''})
    simulation_results = simulation.simulate_logical_error_rate(10000, 12, True, 0.1, [-1, -1, -1, -1, -1, -1, 70, 70, 70, 70, 70, 70])
    print(time.time() - st)
    print(simulation_results)
    
    num_rounds = np.array([8, 16, 32, 40, 50, 60, 80, 120, 160, 190, 220, 256])
    logical_error_rates = simulation_results['Logical_Error_Rate'].to_list()
    print(logical_error_rates)
    CI_upper_bound_logical_error_rates = [simulation.create_clopper_pearson_interval(0.95, x*10000, 10000)[1] for x in logical_error_rates]
    CI_lower_bound_logical_error_rates = [simulation.create_clopper_pearson_interval(0.95, x*10000, 10000)[0] for x in logical_error_rates]
    for i in range(len(logical_error_rates)):
        CI_upper_bound_logical_error_rates[i] = abs(logical_error_rates[i] - CI_upper_bound_logical_error_rates[i])
        CI_lower_bound_logical_error_rates[i] = abs(logical_error_rates[i] - CI_lower_bound_logical_error_rates[i])

    CI_logical_error_rates_without_burst = [CI_lower_bound_logical_error_rates[:6], CI_upper_bound_logical_error_rates[:6]]
    CI_logical_error_rates_with_burst = [CI_lower_bound_logical_error_rates[6:], CI_upper_bound_logical_error_rates[6:]]

    plt.ylabel('Logical Error Rate')
    plt.semilogy()
    # plt.semilogx()
    plt.xlabel('Number of Rounds')

    plt.scatter(num_rounds[:6], logical_error_rates[:6], label='distance = 3, phenomenological noise = 1.5%')
    plt.scatter(num_rounds[6:], logical_error_rates[6:], label='distance = 3, phenomenological noise = 1.5%, \nburst error (occurred at round 70) rate = 10%')
    plt.errorbar(num_rounds[:6], logical_error_rates[:6], yerr=CI_logical_error_rates_without_burst, fmt='o', capsize=10)
    plt.errorbar(num_rounds[6:], logical_error_rates[6:], yerr=CI_logical_error_rates_with_burst, fmt='o', capsize=10)
    # logical_burst_error_rate (via non-linear regression's popt) is 0.00920538. Note that logical burst error
    # rate refers to the logical error rate in the single burst error and burst_error_rate refers to the physical error
    # rate due to the burst error.
    initial_guess_lim_logical_error_rate_per_round = 1 - ((1 - 2*logical_error_rates[5])**(1/num_rounds[5]))
    initial_guess_logical_burst_error_rate = 1 - ((1 - 2*logical_error_rates[-1])/((1-initial_guess_lim_logical_error_rate_per_round)**num_rounds[-1]))
    
    popt, pcov = curve_fit(logical_error_rate_piecewise_function, num_rounds, logical_error_rates, [initial_guess_lim_logical_error_rate_per_round, initial_guess_logical_burst_error_rate])
    x = np.linspace(1, 300, 300)
    y = logical_error_rate_piecewise_function(x, *popt)
    plt.plot(x, y, 'r--', label='Model Fit')
    # plt.plot(x, burst_y, 'g--', label='Piecewise Error Model Fit')


    plt.legend()
    plt.savefig('newfig.png', bbox_inches="tight")

    