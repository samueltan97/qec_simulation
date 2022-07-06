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
    import burst_decoding

    num_shots = 1000
    for d in [3, 5, 7]:
        xs = []
        ys = []
        for noise in np.linspace(0.03, 0.06,5):
            circuit = stim.Circuit.generated(
                "surface_code:unrotated_memory_z",
                rounds=d * 10,
                distance=d,
                # after_clifford_depolarization=noise,
                # after_reset_flip_probability=noise,
                before_measure_flip_probability=noise,
                before_round_data_depolarization=noise)
            xs.append(noise)
            ys.append(burst_decoding.count_logical_errors(circuit, num_shots) / num_shots)
            print(d, noise)
        plt.plot(xs, ys, label="d=" + str(d))
    plt.semilogy()
    plt.xlabel("physical error rate")
    plt.ylabel("logical error rate")
    plt.legend()
    plt.savefig('tutorial.png')