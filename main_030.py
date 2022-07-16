import routine_simulation
import numpy as np

num_shots = 100000
distances = [5, 7, 9, 11]
rounds = [10 * x for x in distances]
noises = [0.03]
error_burst_rates = np.linspace(0.09, 0.15, 7)
num_cores = 32
csv_file_name = 'x_processed_results'
process_data = True
save_intermediate_plots = True
plot_prefix = '0715'
fix_round_to_distance = True

routine_simulation.calc_error_burst_threshold(num_shots, rounds, distances, noises, error_burst_rates, num_cores, csv_file_name, process_data, save_intermediate_plots, plot_prefix, fix_round_to_distance)