def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y

if __name__ == "__main__":
    from simulate import Simulation
    import time
    import numpy as np
    from functools import partial
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    # st = time.time()
    simulation = Simulation(rounds=[8, 16, 32, 64, 128, 256], distances=[7], noises=[0.02], \
        circuit_parameters={'code_task': 'surface_code:rotated_memory_z', 'before_round_data_depolarization':''})
    # simulation_results = simulation.simulate_logical_error_rate(10000, 12)
    # print(time.time() - st)
    # time taken = 1098s
    simulation_results = [[[0.0009, 0.0023, 0.0043, 0.0139, 0.025,0.0436]]]
    # print(simulation_results)
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

    fig, (ax1, ax2) = plt.subplots(2)
    # ax1.set_title('Graph of Logical Error Rate against Number of Rounds')
    ax1.set(ylabel='Logical Error Rate')
    ax1.semilogy()
    # ax2.set_title('Graph of Logical Error Rate per Round against Number of Rounds')
    ax2.set(xlabel='Number of Rounds', ylabel='Logical Error Rate per Round')
    ax2.semilogy()

    ax1_x_scaling = max(num_rounds)
    ax1_y_scaling = max(logical_error_rates)
    ax1.scatter(num_rounds, logical_error_rates, label='distance = 7 and phenomenological noise = 2%')
    ax1.errorbar(num_rounds, logical_error_rates, yerr=CI_logical_error_rates, fmt='o', capsize=10)
    ax1_popt, ax1_pcov = curve_fit(sigmoid, num_rounds/ax1_x_scaling, logical_error_rates/ax1_y_scaling)
    ax1_x = np.linspace(1, 300, 5)
    ax1_x = ax1_x/max(ax1_x)
    ax1_y = sigmoid(ax1_x, *ax1_popt)
    ax1.plot(ax1_x * ax1_x_scaling,ax1_y * ax1_y_scaling, 'g--', label='Sigmoid Fit')


    ax2.scatter(num_rounds, logical_error_rates_per_round)
    ax2.errorbar(num_rounds, logical_error_rates_per_round, yerr=CI_logical_error_rates_per_round, fmt='o', capsize=10)
    ax2_popt, ax2_pcov = curve_fit(sigmoid, num_rounds, logical_error_rates_per_round)

    ax1.legend()
    # ax2.legend()
    # plt.show()
    plt.savefig('newfig.png')