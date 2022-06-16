from numpy import insert
import stim

def insert_burst_error(circuit: stim.Circuit, timestep_num: int, probability: float) -> stim.Circuit:
    n = circuit.num_qubits
    result = stim.Circuit()
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            result.append(stim.CircuitRepeatBlock(
                repeat_count=timestep_num - 2,
                body=instruction.body_copy()
            ))
            result.append('DEPOLARIZE1', range(n), probability)
            result.append(stim.CircuitRepeatBlock(
                repeat_count=instruction.repeat_count - timestep_num + 2,
                body=instruction.body_copy()
            ))
        else:
            result.append(instruction)
    return result

if __name__ == "__main__":
    from simulate import Simulation
    import time
    import numpy as np
    from functools import partial
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    import stim

    phem_noisy_circuit = stim.Circuit.generated(code_task='surface_code:rotated_memory_z', rounds=10, \
        distance=7, before_round_data_depolarization=0.02)
    print(repr(phem_noisy_circuit))

    phem_noisy_circuit_with_burst_error = insert_burst_error(phem_noisy_circuit, 6, 0.1)
    print(repr(phem_noisy_circuit_with_burst_error))

    