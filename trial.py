from numpy import insert
import stim

def logical_error_rate_function_with_burst_error(x, logical_burst_error_rate):
    # lim_logical_error_rate_per_round is obtained from non-linear regression obtained from reproduction_of_soft_decoding.py
    lim_logical_error_rate_per_round = 0.00036809
    return 0.5*(1-((1 - lim_logical_error_rate_per_round)**x)*(1 - logical_burst_error_rate))
     
if __name__ == "__main__":
    from simulate import Simulation
    import time
    import numpy as np
    from functools import partial
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    import stim

    # st = time.time()
    simulation = Simulation(rounds=[8, 16, 32, 64, 128, 256], distances=[7], noises=[0.02], \
        circuit_parameters={'code_task': 'surface_code:rotated_memory_z', 'before_round_data_depolarization':''})
    simulation_results = simulation.simulate_logical_error_rate(10000, 12, True, 0.025, [4, 8, 16, 32, 64, 128])
    # print(time.time() - st)
    print(simulation_results)
    # simulation_results =  [[[0.0065, 0.0074, 0.008, 0.0199, 0.0291, 0.0456]]]
    num_rounds = np.array([8, 16, 32, 64, 128, 256])
    logical_error_rates = np.array(simulation_results[0][0])
    logical_error_rates_per_round = []
    CI_upper_bound_logical_error_rates = [simulation.create_clopper_pearson_interval(0.95, x*10000, 10000)[1] for x in logical_error_rates]
    CI_lower_bound_logical_error_rates = [simulation.create_clopper_pearson_interval(0.95, x*10000, 10000)[0] for x in logical_error_rates]
    CI_upper_bound_logical_error_rates_per_round = []
    CI_lower_bound_logical_error_rates_per_round = []
    for i in range(len(logical_error_rates)):
        logical_error_rates_per_round.append(1 - ((1 - 2*logical_error_rates[i])**(1/num_rounds[i])))
        CI_upper_bound_logical_error_rates_per_round.append(abs(logical_error_rates_per_round[i] - (1 - ((1 - 2*CI_upper_bound_logical_error_rates[i])**(1/num_rounds[i])))))
        CI_lower_bound_logical_error_rates_per_round.append(abs(logical_error_rates_per_round[i] - (1 - ((1 - 2*CI_lower_bound_logical_error_rates[i])**(1/num_rounds[i])))))
        CI_upper_bound_logical_error_rates[i] = abs(logical_error_rates[i] - CI_upper_bound_logical_error_rates[i])
        CI_lower_bound_logical_error_rates[i] = abs(logical_error_rates[i] - CI_lower_bound_logical_error_rates[i])

    CI_logical_error_rates = [CI_lower_bound_logical_error_rates, CI_upper_bound_logical_error_rates]
    CI_logical_error_rates_per_round = [CI_lower_bound_logical_error_rates_per_round, CI_upper_bound_logical_error_rates_per_round]

    plt.ylabel('Logical Error Rate')
    plt.semilogy()
    plt.xlabel('Number of Rounds')

    plt.scatter(num_rounds, logical_error_rates, label='distance = 7, phenomenological noise = 2%, \nburst error rate = 2.5%')
    plt.errorbar(num_rounds, logical_error_rates, yerr=CI_logical_error_rates, fmt='o', capsize=10)
    
    # Reasonable initial guess for the logical burst error rate would require us to
    # use the final logical rate and final T as well as the logical error_rate_per_round calculated previously
    lim_logical_error_rate_per_round = 0.00036809
    initial_guess = 1 - ((1 - 2*logical_error_rates[-1])/((1-lim_logical_error_rate_per_round)**num_rounds[-1]))
    
    popt, pcov = curve_fit(logical_error_rate_function_with_burst_error, num_rounds, logical_error_rates, initial_guess)
    # logical_burst_error_rate (via non-linear regression's popt) is 0.00920538. Note that logical burst error
    # rate refers to the logical error rate in the single burst error and burst_error_rate refers to the physical error
    # rate due to the burst error.
    x = np.linspace(1, 300, 300)
    y = logical_error_rate_function_with_burst_error(x, *popt)
    plt.plot(x, y, 'g--', label='Model Fit')


    plt.legend()
    plt.savefig('newfig.png', bbox_inches="tight")

    