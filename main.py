if __name__ == "__main__":
    from simulate import Simulation
    import time
    import numpy as np
    from functools import partial
    from multiprocessing import Pool

    st = time.time()
    simulation = Simulation(num_shots=10000, rounds=[8, 16, 32, 64], distances=[7], noises=[0.005], \
        circuit_parameters={'code_task': 'surface_code:rotated_memory_z', 'before_round_data_depolarization':''})
    simulation_results = simulation.simulate_logical_error_rate(100000, 2)
    print(simulation_results)
    print(time.time() - st)

