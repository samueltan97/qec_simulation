import stim
import decoding
import matplotlib.pyplot as plt
from typing import List, Dict

# TODO Make simulate function take in rounds and x, y agnostic i.e. to be able to plot x and y according to parameters

def simulate_logical_error_rate_wrt_noise(num_shots: int, distances: List[int], noises:List[float], \
    circuit_parameters: Dict[str, str], to_plot: bool = False, save_plots:bool =False, save_plots_title:str = 'plot'):
    
    for d in distances:
        xs = []
        ys = []
        for noise in noises:
            # Adding noise and distance to the circuit parameter dictionary
            compiled_circuit_parameters = circuit_parameters
            compiled_circuit_parameters['distance'] = d
            compiled_circuit_parameters['rounds'] = d * 3
            if compiled_circuit_parameters.get('after_clifford_depolarization') is not None:
                compiled_circuit_parameters['after_clifford_depolarization'] = noise
            if compiled_circuit_parameters.get('after_reset_flip_probability') is not None:
                compiled_circuit_parameters['after_reset_flip_probability'] = noise
            if compiled_circuit_parameters.get('before_measure_flip_probability') is not None:
                compiled_circuit_parameters['before_measure_flip_probability'] = noise
            if compiled_circuit_parameters.get('before_round_data_depolarization') is not None:
                compiled_circuit_parameters['before_round_data_depolarization'] = noise
            circuit = stim.Circuit.generated(**compiled_circuit_parameters)
            xs.append(noise)
            ys.append(decoding.count_logical_errors(circuit, num_shots)/num_shots)
        if to_plot:
            plt.plot(xs, ys, label="d=" + str(d))
    if to_plot:
        plt.semilogy()
        plt.xlabel("physical error rate")
        plt.ylabel("logical error rate")
        plt.legend()
        if save_plots:
            plt.savefig(save_plots_title + '.png')
        plt.show()
    return (xs, ys)