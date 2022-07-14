import routine_simulation
import numpy as np

num_shots = 100
distances = np.array([5, 7, 9, 11, 13])
rounds = 10 * distances
noises = [0.03]
error_burst_rates = [0.16]
num_cores = 32
csv_file_name = 'test_speed'
process_data = False
save_intermediate_plots = True
plot_prefix = 'happy'
fix_round_to_distance = True

routine_simulation.calc_error_burst_threshold(num_shots, rounds, distances, noises, error_burst_rates, num_cores, csv_file_name, process_data, save_intermediate_plots, plot_prefix, fix_round_to_distance)