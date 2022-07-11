from typing import List
from simulate import Simulation
from typing import List
import numpy as np

if __name__ == "__main__":
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

    colors = cycle(['tab:blue', 'tab:orange', 'tab:red', 'yellow'])
    num_shots = 10000
    data_dictionary = dict()
    rounds = [30, 50, 70]
    distances = [3, 5, 7]
    noises = np.linspace(0.03, 0.04,5)

    st = time.time()
    simulation = Simulation(rounds=rounds, distances=distances, noises=noises, \
        circuit_parameters={'code_task': 'surface_code:unrotated_memory_z', 'before_round_data_depolarization':'', 'before_measure_flip_probability':''})
    circuit = simulation.circuit_array[0][0][0]
    print(type(circuit))
    n = circuit.num_qubits
    result = stim.Circuit()
    timestep_num = 15
    for instruction in circuit:
        # print(instruction)
        if isinstance(instruction, stim.CircuitRepeatBlock):
            result.append(stim.CircuitRepeatBlock(
                repeat_count=timestep_num - 1,
                body=instruction.body_copy()
            ))

            for single_instruction in instruction.body_copy():
                if single_instruction.name =="DEPOLARIZE1":
                    result.append('DEPOLARIZE1', single_instruction.targets_copy(), 0.1)
                elif single_instruction.name == 'X_ERROR':
                    result.append('X_ERROR', single_instruction.targets_copy(), 2 * 0.1 / 3)
                else:
                    result.append(single_instruction)

            # result.append('DEPOLARIZE1', range(n), burst_error_rate)
            result.append(stim.CircuitRepeatBlock(
                repeat_count=instruction.repeat_count - timestep_num,
                body=instruction.body_copy()
            ))
        else:
            result.append(instruction)
    print(result)