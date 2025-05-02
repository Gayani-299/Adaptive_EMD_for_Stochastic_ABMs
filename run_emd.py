
import sys
import numpy as np
import pandas as pd
import time
sys.path.append('J:\\combine_dist_crowd\\stochastic Stationsim\\AGP_2\\src\\EvolutionaryModelDiscovery')

from EvolutionaryModelDiscovery import EvolutionaryModelDiscovery

# NETLOGO_PATH = "C:\\Program Files\\NetLogo 6.3.0"

model_path = "./StatioSim_IGSS.py"
start_time = time.time()
measurements = ["error"]
ticks = 1000
pop = 50

emd = EvolutionaryModelDiscovery(model_path=model_path, measurement_reporters=measurements, ticks_to_run=ticks, pop_total=pop, start_time=start_time)

emd.set_mutation_rate(0.2)
emd.set_crossover_rate(0.8)
emd.set_generations(50)
emd.set_replications(5)
emd.set_depth(2,6)
emd.set_population_size(50)
emd.set_is_minimize(True)


def simulation_error(results):
    return np.mean(results)

emd.set_objective_function(simulation_error)


if __name__ == '__main__':
    emd.evolve()
    fi = emd.get_factor_importances_calculator("FactorScores.csv")
    GI = fi.get_gini_importances(interactions=True)
    PI = fi.get_permutation_accuracy_importances(interactions=True)
    GI.to_csv('gini_importances.csv', index=False)
    PI.to_csv('permutation_importances.csv', index=False)
    print(GI)
    print(PI)

