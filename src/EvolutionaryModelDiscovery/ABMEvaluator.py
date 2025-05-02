"""EvolutionaryModelDiscovery: Automated agent rule generation and 
importance evaluation for agent-based models with Genetic Programming.
Copyright (C) 2018  Chathika Gunaratne
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>."""

'''- Adapted by Gayani PDP Senanayake, 2025
 - Replaced NetLogo-based simulation logic with a Python ABM (StationSim)
 - Redesigned the fitness evaluation to use RMSE over exit flow metrics
 - Integrated stochastic rule support and multi-run averaging'''

from typing import Dict, List, Callable, Any, Union
import numpy as np
import pandas as pd
from deap import gp
import importlib.util
from .Util import *
from .PythonWriter import PythonWriter


def default_objective(results: pd.DataFrame) -> float:
    """
    Placeholder objective function. User must provide an objective function
    that translates simulation results to fitness for the genetic program.

    :param results: pd.DataFrame of simulation results as returned by NL4Py.
    :return: simulation results translated into fitness for genetic program.
    """
    return 0


OBJECTIVE_FUNCTION = default_objective


def set_objective_function(objective_function: Callable) -> None:
    """
    Sets a custom callable as the objective function for the GP. 

    :param objective_function: Callable to be executed by GP. Must return a fitness value.
    """
    global OBJECTIVE_FUNCTION
    OBJECTIVE_FUNCTION = objective_function


def set_model_factors(model_factors: "EvolutionaryModelDiscovery.ModelFactors",) -> None:
    global MODEL_FACTORS
    MODEL_FACTORS = model_factors


def set_model_init_data(model_init_data: Dict[str, Any]) -> None:
    global MODEL_INIT_DATA
    MODEL_INIT_DATA = model_init_data


def set_python_writer(python_writer: PythonWriter) -> None:
    global PYTHON_WRITER
    PYTHON_WRITER = python_writer


def evaluate(
    individual: Union["gp.creator.IndividualMin", "gp.creator.IndividualMax"]
) -> pd.Series:
    """
    Genetic program's evaluation function. 

    Simplifies and scores factor/factor-interaction presence.
    Compiles gp tree representation into flattened str format.
    Writes rule to new NetLogo model.
    Simulates new NetLogo model and records fitness.
    Cleans up auto-generated NetLogo model.

    :param individual: Union['gp.creator.IndividualMin', 'gp.creator.IndividualMax'] gp individual
    :return: pd.Series containing presence scores, fitness, and compiled rule of executed gp individual
    """

    ind_record = score_factor_presence(individual, MODEL_FACTORS)
    complexity = sum(value for value in ind_record.values())

    newRule = str(
        gp.compile(individual, MODEL_FACTORS.get_DEAP_primitive_set())
    )

    newRule = newRule.replace(" ", "")
    newRule = newRule.replace("((", "(")
    newRule = newRule.replace("))", ")")
    newRule = newRule.replace("(", "", 1)
    newRule = "gate_out=self." + newRule
    newRule = newRule.replace("(all_potential_exits_locations)", "(model,")
    newRule = newRule.replace(")(", ",")
    newRule = newRule.replace("combine", "self.combine")
    newRule = newRule.replace("subtract", "self.subtract")
    newRule = newRule.replace("divide", "self.divide")
    newRule = newRule.replace("multiply", "self.multiply")

    print(newRule)

    newModelPath = PYTHON_WRITER.inject_new_rule(newRule)

    fitness = run_python_based_simulation(newModelPath,complexity, newRule,
                                          MODEL_INIT_DATA["ticks_to_run"], MODEL_INIT_DATA["pop_total"],
                                          MODEL_INIT_DATA["agg_func"])

    remove_model(newModelPath)
    ind_record["Fitness"] = fitness
    ind_record["Rule"] = newRule[:-1]
    ind_record["ModelPath"] = newModelPath
    ind_record = pd.Series(list(ind_record.values()), index=ind_record.keys())

    return ind_record


###############################################################################

def run_python_based_simulation(
    model_path: str,
    complexity: int,
    newRule: str,
    ticks_to_run: int,
    pop_total: int,
    agg_func: Callable = np.mean
) -> pd.DataFrame:
    """
    Run the Python-based ABM simulation.

    :param model_path: str file path or other relevant information for the Python-based model.
    :param steps_to_run: int number of simulation steps to run.
    :param pop_total: int total population parameter for the model.
    :return: List of simulation results.
    """
    # Instantiate your Python-based ABM model

    # print(open(model_path).read())
    module_name = 'custom_model'
    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    # Access the Model class from the loaded module
    Model = model_module.Model

    if "randomly_select_exit" in newRule:
        n = 15
    else:
        n = 2
    result_exit_sim = calculate_stats(Model, n, ticks_to_run, pop_total)
    result_exit_actual = pd.read_csv('results_all.csv')
    fitness = calculate_fitness(result_exit_sim,result_exit_actual)

    return (fitness,)


def calculate_stats(Model, n_runs, num_steps, pop_total):
    all_exits = np.zeros((n_runs, num_steps, 4))

    # Create list of seeds for reproducibility
    seeds = np.random.SeedSequence().generate_state(n_runs)

    def run_simulation(seed):
        """Run single simulation and return exit counts"""
        model = Model(pop_total=pop_total, step_limit=num_steps, random_seed=seed)
        for _ in range(num_steps):
            model.step()
        return np.array([
            [step_data['Exit3'], step_data['Exit4'], step_data['Exit_flow3'], step_data['Exit_flow4']]
            for step_data in model.data_list_sum
        ])

    results = list(map(run_simulation, seeds))

    # Stack results into 3D array
    all_exits = np.stack(results)

    # Calculate statistics across runs (axis=0)
    stats = pd.DataFrame({
        'Time_Step': np.arange(num_steps),
        'Exit3_Mean': all_exits[:, :, 0].mean(axis=0),
        'Exit3_SD': all_exits[:, :, 0].std(axis=0, ddof=1),
        'Exit4_Mean': all_exits[:, :, 1].mean(axis=0),
        'Exit4_SD': all_exits[:, :, 1].std(axis=0, ddof=1),
        'Exit_flow3_Mean': all_exits[:, :, 2].mean(axis=0),
        'Exit_flow3_SD': all_exits[:, :, 2].std(axis=0, ddof=1),
        'Exit_flow4_Mean': all_exits[:, :, 3].mean(axis=0),
        'Exit_flow4_SD': all_exits[:, :, 3].std(axis=0, ddof=1)
    })

    return stats


def calculate_fitness(result_exit_sim, result_exit_actual):
    min_rows = min(len(result_exit_sim), len(result_exit_actual))
    sim = result_exit_sim.iloc[:min_rows]
    actual = result_exit_actual.iloc[:min_rows]

    mean_error3 = np.sqrt(np.mean((sim['Exit3_Mean'] - actual['Exit3_Mean']) ** 2))
    mean_error4 = np.sqrt(np.mean((sim['Exit4_Mean'] - actual['Exit4_Mean']) ** 2))
    sd_error3  = np.sqrt(np.mean((sim['Exit3_SD'] - actual['Exit3_SD']) ** 2))
    sd_error4  = np.sqrt(np.mean((sim['Exit4_SD'] - actual['Exit4_SD']) ** 2))
    rmse_1 = (mean_error3 + mean_error4 + sd_error3 + sd_error4) / 4.0

    mean_error3 = np.sqrt(np.mean((sim['Exit_flow3_Mean'] - actual['Exit_flow3_Mean']) ** 2))
    mean_error4 = np.sqrt(np.mean((sim['Exit_flow4_Mean'] - actual['Exit_flow4_Mean']) ** 2))
    sd_error3 = np.sqrt(np.mean((sim['Exit_flow3_SD'] - actual['Exit_flow3_SD']) ** 2))
    sd_error4 = np.sqrt(np.mean((sim['Exit_flow4_SD'] - actual['Exit_flow4_SD']) ** 2))
    rmse_2 = (mean_error3 + mean_error4 + sd_error3 + sd_error4) / 4.0

    # Weighted components
    combined_score = (
            0.5 * np.clip(rmse_1, 0.0, 1.0) +  # Bound KS stat
            0.5 * np.clip(rmse_2, 0.0, 1.0)
    )
    return combined_score

def load_model_from_path(model_path: str, pop_total: int):
    """
    Load and initialize a Python-based ABM model from a file path.

    :param model_path: str file path to the Python-based model.
    :param pop_total: int total population parameter for the model.
    :return: Initialized model instance.
    """
    try:
        # Load module from file path
        spec = importlib.util.spec_from_file_location("model", model_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)

        # Initialize model instance
        model_instance = model_module.Model(pop_total)
        return model_instance
    except Exception as e:
        # Handle exceptions during module loading or model initialization
        print(f"Error loading or initializing the model: {e}")
        # Optionally, raise the exception to propagate it further
        raise e

############################################################################

def score_factor_presence(
    ind: Union["gp.creator.IndividualMin", "gp.creator.IndividualMax"],
    ModelFactors: "EvolutionaryModelDiscovery.ModelFactors",
) -> Dict[str, int]:
    """
    Scores factor or factor interaction presence as coefficient of simplified rule.

    :param ind: 
    :param ModelFactors: ModelFactors module auto-generated and loaded by EMD.
    :return: Dict[str, int] mapping factor/factor-interaction name to presence score
    """
    factor_interactions = ModelFactors.interactions
    factors = ModelFactors.measureable_factors
    presence_dict = {}
    for factor in factors:
        presence_dict[factor] = 0
    # items on stack represent primitives being processed.
    # items have 3 elements param num considered, primitive(deap.gp.Primitive), and polarity (int)
    stack = [{"param_num": 0, "obj": ind[0], "polarity": 1}]
    interaction = None
    interactionRoot = None
    polarity = 1
    for child in range(1, len(ind)):
        childString = ind[child].name
        childArity = ind[child].arity
        # Update presence counts
        # Check negate
        parent = stack[-1]
        if interaction == None:
            if parent["obj"].name in ModelFactors.negativeOps.keys():
                child_position_polarity = ModelFactors.negativeOps[
                    parent["obj"].name
                ][parent["param_num"]]
                polarity = parent["polarity"] * child_position_polarity
            if childString in factors:
                # Countable
                # print(childString, parent["obj"].name, polarity)
                presence_dict[childString] = (
                    presence_dict[childString] + polarity
                )
            # If interaction found start recording
            if childString in factor_interactions:
                interaction = [childString]
                interactionRoot = len(stack)
        elif type(interaction) == list:
            # if interaction still processing, append
            interaction.append(childString)
        # Process next
        # Tell parent a child has been found...
        stack[-1]["param_num"] = stack[-1]["param_num"] + 1
        parent = stack[-1]
        # Resolve children if any
        if childArity == 0:
            # Terminal found.
            # Travel up the stack and pop any completed parents
            while parent["param_num"] == parent["obj"].arity:
                _ = stack.pop()
                if len(stack) == 0:
                    # root reached
                    break
                parent = stack[-1]
                polarity = parent["polarity"]
                # Now, if all this parent removal revealed an interaction root, process it
                if len(stack) == interactionRoot:
                    # Interaction is done processing
                    interactionString = str(
                        gp.compile(
                            interaction, MODEL_FACTORS.get_DEAP_primitive_set()
                        )
                    )
                    presence_dict[interactionString] = (
                        presence_dict.get(interactionString, 0) + polarity
                    )
                    interaction = None
                    interactionRoot = None
                    polarity = 1
        else:
            # primitive found. Add to family stack with arity
            stack.append(
                {"param_num": 0, "obj": ind[child], "polarity": polarity}
            )
    return presence_dict
