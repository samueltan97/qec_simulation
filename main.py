if __name__ == "__main__":
    from simulate import Simulation
    import time
    import numpy as np
    from functools import partial
    import matplotlib.pyplot as plt

    # st = time.time()
    # simulation = Simulation(num_shots=10000, rounds=[8, 16, 32, 64], distances=[7], noises=[0.005], \
    #     circuit_parameters={'code_task': 'surface_code:rotated_memory_z', 'before_round_data_depolarization':''})
    # simulation_results = simulation.simulate_logical_error_rate(100000, 2)
    # plt.plot([8, 16, 32, 64], simulation_results[0][0])
    # plt.semilogy()
    # plt.semilogx()
    # plt.savefig('newfig')
    # plt.show()
    # print(simulation_results)
    # print(time.time() - st)

    simulation = Simulation(rounds=[8, 16, 32, 64], distances=[3, 5, 7], noises=[0.005], \
        circuit_parameters={'code_task': 'surface_code:rotated_memory_z', 'before_round_data_depolarization':'', \
            'after_clifford_depolarization':'', 'after_reset_flip_probability':'', \
            'before_measure_flip_probability':'', 'before_round_data_depolarization':''})
    simulation_results = simulation.simulate_logical_error_rate(10000, 2)
    simulation.plot_simulation_results(simulation_results, 'round', [8, 16, 32, 64], 'distance' \
        , [3, 5, 7], 'Number of Rounds', 'Logical Error Rate', 'Graph of Logical Error Rate over Number of Rounds')
    print(simulation_results)

