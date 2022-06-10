import simulate

simulation_results = simulate.simulate_logical_error_rate(1000, [9,15,21], [3,5], [0.1, 0.2, 0.3, 0.4, 0.5], \
    {'code_task': 'surface_code:unrotated_memory_z', 'before_round_data_depolarization':''})
print(simulation_results)
# simulation_results = [[[0.384, 0.468, 0.474], [0.515, 0.505, 0.485], [0.503, 0.48, 0.486], [0.507, 0.517, 0.501], [0.512, 0.511, 0.502]], \
#  [[0.356, 0.401, 0.453], [0.506, 0.462, 0.509], [0.477, 0.511, 0.504], [0.512, 0.506, 0.477], [0.495, 0.499, 0.525]], \
#  [[0.274, 0.356, 0.456], [0.526, 0.482, 0.481], [0.493, 0.513, 0.493], [0.493, 0.481, 0.486], [0.52,  0.482, 0.511]]]
simulate.plot_simulation_results(simulation_results, 'round', [9, 15, 21], 'noise', [0.1, 0.2, 0.3, 0.4, 0.5])