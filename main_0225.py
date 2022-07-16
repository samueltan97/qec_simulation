import routine_simulation
import numpy as np

num_shots = 1000
distances = [5, 7, 9, 11]
rounds = [10 * x for x in distances]
noises = [0.0225]
error_burst_rates = np.linspace(0.09, 0.17, 9)
num_cores = 32
csv_file_name = '0225_results'
process_data = False
save_intermediate_plots = False
plot_prefix = '0715'
fix_round_to_distance = True

routine_simulation.calc_error_burst_threshold(num_shots, rounds, distances, noises, error_burst_rates, num_cores, csv_file_name, process_data, save_intermediate_plots, plot_prefix, fix_round_to_distance)