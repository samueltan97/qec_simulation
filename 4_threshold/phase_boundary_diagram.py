from typing import List
from simulate import Simulation
from typing import List
import numpy as np

def logical_error_rate_function_with_burst_error(x, lim_logical_error_rate_per_round, logical_burst_error_rate):
    results = []
    for val in x[:int(len(x)/2)]:
        results.append(0.5*(1-((1 - lim_logical_error_rate_per_round)**val)))
    for val in x[int(len(x)/2):]:
        results.append(0.5*(1-((1 - lim_logical_error_rate_per_round)**val)*(1 - logical_burst_error_rate)))
    return np.array(results)

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
    error_burst_thresholds = [0.135, 0.13, 0.121, 0.118]
    physical_error_rates = [0.005, 0.01, 0.015, 0.02]
    error_burst_threshold_errors_2_sigmas = [0.01, 0.014, 0.018, 0.018]

    # Fitting the data points
    color = next(colors)
    # z = np.polyfit([0.14] + error_burst_thresholds, [0] + physical_error_rates, 2)
    # p = np.poly1d(z)
    # x = np.linspace(0.113, 0.14, 300)
    # y = p(x)
    # plt.plot(np.concatenate([np.linspace(0, 0.113, 300), x]), np.concatenate([np.linspace(0.02, 0.02, 300), y]), c=color, label='Fit')
    # plt.fill_between(np.concatenate([np.linspace(0, 0.14, 300), np.linspace(0.14, 0.14, 300)]), np.concatenate([np.array([0.02]*300), np.linspace(0, 0.02, 300)]) , color=color, alpha=0.3)

    plt.ylabel('Physical Error Rate')
    plt.xlabel('Error Burst Rate')
    plt.errorbar(error_burst_thresholds, physical_error_rates, xerr=error_burst_threshold_errors_2_sigmas,\
        fmt='o', capsize=10, c=color) 
    x = np.concatenate([np.linspace(0, 0.15, 300), np.linspace(0.15, 0.15, 300)])
    y = np.concatenate([np.array([0.02]*300), np.linspace(0, 0.02, 300)]) 
    plt.plot(x, y, c=next(colors), label='Hypothesized Model', linestyle='dashed')

    

    plt.xlim(0, 0.16)
    plt.ylim(0, 0.025)
    plt.grid(b=True, which='major', linestyle='-')
    plt.grid(b=True, which='minor', linestyle='--')
    plt.minorticks_on()
    plt.legend()
    plt.savefig('phase_boundary.png', bbox_inches="tight")
    plt.clf()
    

     
    
    


    

    