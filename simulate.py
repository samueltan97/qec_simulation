from os import stat
import stim
import decoding
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from multiprocessing import Pool
from functools import partial
import numpy as np
from scipy.stats import beta
import pandas as pd

class Simulation:
    def __init__(self, rounds:List[int], distances: List[int], noises:List[float], \
    circuit_parameters: Dict[str, str]) -> None:
        '''
        :param rounds: List[Int] -> list containing the different number of rounds of measurement and error correction that we want to simulate
        :param distances: List[Int] -> list containing the different surface code distances that we want to simulate
        :param noises: List[Float] -> list containing the different noise parameters that we want to simulate
        :param circuit_paramaters: Dict[Str, Str] -> the keys are the keyword arguments and values are the values corresponding to the keyword arguments. \
            Important keys include "code_task" as well as the types of error we want to simulate. The same noise value would be applied \
                to all noise parameters
        '''
        self.rounds = rounds
        self.distances = distances
        self.noises = noises
        self.circuit_parameters = circuit_parameters
        # Initialize 3D array that will contain the circuit for each simulation corresponding
        # to the code distance, noise parameter, and number of rounds
        self.circuit_array = []
        for d_index, distance in enumerate(distances):
            self.circuit_array.append([])
            for n_index, noise in enumerate(noises):
                self.circuit_array[d_index].append([])
                for round in rounds:
                    # Adding noise and distance and round parameters to the circuit parameter dictionary
                    compiled_circuit_parameters = circuit_parameters
                    compiled_circuit_parameters['distance'] = distance
                    compiled_circuit_parameters['rounds'] = round
                    # Note that the same noise value is given to all sources of noise
                    if compiled_circuit_parameters.get('after_clifford_depolarization') is not None:
                        compiled_circuit_parameters['after_clifford_depolarization'] = noise
                    if compiled_circuit_parameters.get('after_reset_flip_probability') is not None:
                        compiled_circuit_parameters['after_reset_flip_probability'] = noise
                    if compiled_circuit_parameters.get('before_measure_flip_probability') is not None:
                        compiled_circuit_parameters['before_measure_flip_probability'] = noise
                    if compiled_circuit_parameters.get('before_round_data_depolarization') is not None:
                        compiled_circuit_parameters['before_round_data_depolarization'] = noise
                    self.circuit_array[d_index][n_index].append(stim.Circuit.generated(**compiled_circuit_parameters))

    def simulate_logical_error_rate(self, num_shots:int, num_cores:int, with_burst_error:bool=False, burst_error_rate:float=0.0, burst_error_timestep:List[float]=[]) -> pd.DataFrame:
        '''
        :param num_shots: Int -> the number of shots i.e. the number of times we want to repeat the simulation
        :param num_cores: Int -> the number of cores to use for the computation
        :param with_burst_error: Bool -> whether the simulation should include a single timestep burst error
        :param burst_error_rate: Float -> the error rate for the depolarizing burst error
        :param burst_error_timestep: List[Float] -> After which timestep when we ought to insert the burst error; 
        List should have the same length as the list of rounds and -1 means no burst error inserted for that instance
        '''
        completed_simulations = 0
        total_simulations = len(self.rounds) * len(self.distances) * len(self.noises)
        # Initialize 3D array that will contain the logical error rate for each simulation corresponding
        # to the code distance, noise parameter, and number of rounds
        results = []
        for d_index in range(len(self.distances)):
            for n_index in range(len(self.noises)):
                for r_index in range(len(self.rounds)):
                    pool_array = [int(num_shots//num_cores)] * num_cores
                    for i in range(num_shots % num_cores):
                        pool_array[i] += 1
                    with Pool(num_cores) as p:
                        corresponding_circuit = self.circuit_array[d_index][n_index][r_index]
                        if with_burst_error:
                            corresponding_circuit = self.insert_burst_error(corresponding_circuit, burst_error_rate, burst_error_timestep[r_index])
                        logical_error_rate = sum(p.map(partial(decoding.count_logical_errors, circuit=corresponding_circuit), pool_array))/num_shots
                        results.append([burst_error_rate, self.distances[d_index], self.noises[n_index], self.rounds[r_index], logical_error_rate])
                    completed_simulations += 1

                    print('Completed: ' + str(completed_simulations)  + '/' + str(total_simulations) )
        return pd.DataFrame(results, columns=['Burst_Error_Rate', 'Distance','Physical_Error_Rate','Number_of_Rounds', 'Logical_Error_Rate'])
    
    def simulation_results_to_csv(self, simulation_results:dict, csv_name:str):
        '''
        :param simulation_results: Dict -> dictionary with burst error rates as keys and dataframes as value
        :param csv_name: Str -> name of the csv file without .csv at the end
        '''
        df_array = []
        for burst_error_rate in list(simulation_results.keys()):
            df_array.append(simulation_results[burst_error_rate])
        final_df = df_array[0]
        for df in df_array[1:]:
            final_df = pd.concat([final_df, df], ignore_index=True)
        final_df.to_csv('./' + csv_name + '.csv')
    
    

    def plot_simulation_results(self, simulation_results:np.ndarray, is_semilogy:bool, x_axis:str, x_axis_data:List[float], is_semilogx:bool, graph_label:str, graph_label_data:List[float], x_label:Optional[str]='' \
        , y_label:Optional[str]='', plot_title:Optional[str]='', save_fig:bool = False, fig_name: str = 'new_fig'):
        '''
        :param simulation_results: np.ndarray -> simulation_results[distance][noise][# of rounds] = logical error rate
        :param is_semilogy: bool -> whether to plot the y-axis data in a logarithmic axis
        :param x_axis: Str -> one of the following: "distance", "noise", "round"
        :param x_axis_data: List[float] -> list of data for the x_axis
        :param is_semilogx: bool -> whether to plot the x-axis data in a logarithmic axis
        :param graph_label: Str -> one of the following: "distance", "noise", "round". An example would be the plot showing several x-axis vs logical error \
            rate line graphs corresponding to codes with different code distance if "distance" is given as a graph label
        :param graph_label_data: List[float] -> list of data for the different graphs to be plotted on the same plot
        :param x_label: Optional[Str] -> X-axis label
        :param y_label: Optional[Str] -> Y-axis label
        :param plot_title: Optional[str] -> Title for the generated plot
        :param save_fig: bool -> whether you want the plot to be saved
        :param fig_name: Str -> file name for the figure if save_fig is true
        '''
        # Number the different parameters according to which dimension they are in for the simulation_results ndarray
        index_mapping = {'distance': 0, 'noise': 1, 'round': 2}

        # Flatten the 3d array into 2d by naively choosing the first element of the unwanted dimension. Eg. If we want
        # to plot line graphs of logical error rate against number of rounds and each line graph corresponds to different
        # code distance, we will flatten the dimension that corresponds to noise.
        if (graph_label == 'distance' and x_axis == 'round') or (graph_label == 'round' and x_axis == 'distance'):
            filtered_simulation_results = [[j for j in i[0]] for i in simulation_results]
        elif (graph_label == 'distance' and x_axis == 'noise') or (graph_label == 'noise' and x_axis == 'distance'):
            filtered_simulation_results = [[j[0] for j in i] for i in simulation_results]
        elif (graph_label == 'noise' and x_axis == 'round') or (graph_label == 'round' and x_axis == 'noise'):
            filtered_simulation_results = simulation_results[0]
        else:
            raise ValueError('The graph label should not be the same as the x-axis')

        # Transpose the 2D array if the x-axis is now the first dimension of the array and the graph_labels are in the second
        # dimension to allow us to plot each line graph by the different graph_label_values
        if index_mapping[graph_label] > index_mapping[x_axis]:
            filtered_simulation_results = filtered_simulation_results.transpose()
        for index, graph in enumerate(filtered_simulation_results):
            plt.plot(x_axis_data, graph, label=graph_label + " = " + str(graph_label_data[index]))

        if is_semilogx:
            plt.semilogx()
        if is_semilogy:
            plt.semilogy()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if plot_title:
            plt.title(plot_title)
        plt.legend()
        if save_fig:
            plt.savefig(fig_name + '.png')
        plt.show()

    def create_clopper_pearson_interval(self, confidence_interval:float, num_successes: int, num_trials:int) -> tuple:
        '''
        :param confidence_interval: float -> the fraction for the confidence interval of interest i.e. 0.95 for 95% confidence interval
        :param num_successes: int -> the number of successes out of the number of trials i.e. the number of logical errors
        :param num_trials: int -> the number of trials i.e. the number of shots sampled
        '''
        lower_bound = beta.ppf((1 - confidence_interval)/2, num_successes, num_trials - num_successes + 1)
        upper_bound = beta.ppf(1 - ((1 - confidence_interval)/2), num_successes + 1, num_trials - num_successes)
        return (lower_bound, upper_bound)
    
    def insert_burst_error(self, circuit: stim.Circuit, burst_error_rate: float, timestep_num: int,) -> stim.Circuit:
        '''
        :param circuit: stim.Circuit -> the circuit that the burst error should be added to
        :param burst_error_rate: Float -> the error rate for the depolarizing burst error
        :param timestep_num: Int -> at the end of which round should the burst error occur (has to be lesser or equal to the number of rounds)
        '''
        # if timestep_num is -1, burst error will not be inserted into the circuit
        if timestep_num == -1:
            return circuit
        else:
            n = circuit.num_qubits
            result = stim.Circuit()
            for instruction in circuit:
                if isinstance(instruction, stim.CircuitRepeatBlock):
                    result.append(stim.CircuitRepeatBlock(
                        repeat_count=timestep_num - 1,
                        body=instruction.body_copy()
                    ))
                    result.append('DEPOLARIZE1', range(n), burst_error_rate)
                    result.append(stim.CircuitRepeatBlock(
                        repeat_count=instruction.repeat_count - timestep_num + 1,
                        body=instruction.body_copy()
                    ))
                else:
                    result.append(instruction)
            return result

    def compute_clopper_pearson_error_bars(self, logical_error_rates: np.ndarray, confidence_interval:float, num_shots: int) -> List:
        '''
        :param logical_error_rates: np.ndarray -> 1D-array with logical error rate values
        :param confidence_interval: Float -> confidence interval percentage in decimal i.e. 95% CI would be 0.95
        :param num_shots: Int -> the number of shots taken by the sampler
        '''
        CI_upper_bound_logical_error_rates = [self.create_clopper_pearson_interval(confidence_interval, x*num_shots, num_shots)[1] for x in logical_error_rates]
        CI_lower_bound_logical_error_rates = [self.create_clopper_pearson_interval(confidence_interval, x*num_shots, num_shots)[0] for x in logical_error_rates]
        for i in range(len(logical_error_rates)):
            CI_upper_bound_logical_error_rates[i] = abs(logical_error_rates[i] - CI_upper_bound_logical_error_rates[i])
            CI_lower_bound_logical_error_rates[i] = abs(logical_error_rates[i] - CI_lower_bound_logical_error_rates[i])
        return [CI_lower_bound_logical_error_rates, CI_upper_bound_logical_error_rates]

    def compute_sigma_values(self, logical_error_rates: np.ndarray, num_shots: int) -> List:
        return [np.sqrt(x*(1-x)/num_shots) for x in logical_error_rates]