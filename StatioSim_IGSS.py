import math
import warnings
import numpy as np
import os
from scipy.spatial import cKDTree
from collections import Counter
import matplotlib.pyplot as plt

# Dont automatically load seaborn as it isn't needed on the HPC
try:
    from seaborn import kdeplot as sns_kdeplot
except ImportError as e:
    warnings.warn(
        "The seaborn module is not available. If you try to create kde plots for this model (i.e. a wiggle map or density map) then it will fail.")


class Agent:
    '''
    A class representing a generic agent for the StationSim ABM.
    '''

    def __init__(self, model, unique_id):

        self.unique_id = unique_id
        self.selected_gate_out = None
        self.status = 0  # 0 Not Started, 1 Active, 2 Finished
        self.age = np.random.randint(10, 70)
        self.gender = "Male" if np.random.random() < 0.5 else "Female"
        # Location
        perturb = model.gates_space * np.random.uniform(-1, +1)
        gate_in = np.random.randint(model.gates_in)
        self.gate_in = gate_in
        self.loc_start = model.gates_locations[gate_in] + [0, perturb]
        self.location = self.loc_start
        # Speed
        self.speed_max = 0
        while self.speed_max <= model.speed_min:
            self.speed_max = np.random.normal(model.speed_mean, model.speed_std)
        self.speeds = np.arange(self.speed_max, model.speed_min, -model.speed_step)
        self.speed = None
        self.wiggle = min(model.max_wiggle, self.speed_max)

        # randomise the entrance time of the agent from one of the waves
        bins_index= np.searchsorted(model.bins,[np.random.uniform()*model.step_limit],side='right')[0]
        self.steps_activate = model.bins[bins_index-1]+np.random.exponential(model.gates_speed)

        # History
        if model.do_history:
            self.history_locations = []
            self.history_speeds = []
            self.history_wiggles = 0
            self.history_collisions = 0
            self.step_start = None

    ##############################
    def step(self, model):
        '''
        Iterate the agent.

        Description:
            If they are inactive then it checks to see if they should become active.
            If they are active then they move or leave the model.
        '''
        if self.status == 0:
            self.activate(model)
        elif self.status == 1:
            compare_distance = self.compare_distance(model)
            neighbourhood_count_exits = self.neighbourhood_count_exits(model)
            similarity_by_age = self.similarity_by_age(model)
            similarity_by_gender = self.similarity_by_gender(model)
            compare_width = self.compare_width(model)
            ###############################################################################################################
            if abs(self.location[0] - 150) < 5:
                gate_out = self.selected_gate_out
            else:
                # insert evolutionary code here
                # @EMD @EvolveNextLine @Factors-File="Factors.py" @return-type=gate_out
                gate_out = self.randomly_select_exit(model, self.combine(compare_distance,neighbourhood_count_exits))

            self.selected_gate_out = gate_out
            self.loc_desire = model.gates_locations[gate_out]
            #############################################################################################################
            self.move(model)
            self.deactivate(model)

        self.history(model)
    ########################################################################################################################
    def combine(self, a, b):
        a_array = np.array(a)
        b_array = np.array(b)
        result_array = a_array + b_array
        return result_array.tolist()

    def subtract(self, a, b):
        a_array = np.array(a)
        b_array = np.array(b)
        result_array = a_array - b_array
        return result_array.tolist()

    def get_max_select_exit(self, model, probs):
        gate_out = model.gates_in + np.argmax(probs)
        return gate_out

    def get_min_select_exit(self, model, probs):
        gate_out = model.gates_in + np.argmin(probs)
        return gate_out

    def randomly_select_exit(self, model, prob):
        exits = np.arange(model.gates_in, model.gates_in + model.gates_out)
        prob = [x if x > 0 else 0 for x in prob]
        if sum(prob) == 0:
            prob = [0.5, 0.5]
        # prob = np.clip(prob, 1e-10, 1)
        prob = prob / np.sum(prob)
        gate_out = np.random.choice(exits, p=prob)
        return gate_out

    def compare_distance(self, model):
        distances_to_exits = [self.distance(self.location, gate_loc) for gate_loc in
                              model.gates_locations[model.gates_in:]]
        inverse_distances = [1 / d for d in distances_to_exits]
        sum_inverse_distances = sum(inverse_distances)
        exit_probs = [d / sum_inverse_distances if sum_inverse_distances != 0 else 0 for d in inverse_distances]
        return exit_probs

    def compare_width(self, model):
        width_of_exits = model.gates_width
        sum_width_of_exits = sum(width_of_exits)
        exit_probs = [d / sum_width_of_exits if sum_width_of_exits != 0 else 0 for d in width_of_exits]
        return exit_probs

    def neighbourhood_count_exits(self, model):
        exit_counts = []
        for gate_loc in model.gates_locations[model.gates_in:]:
            agents_at_exit = model.tree.query_ball_point(gate_loc, 20)
            active_agent = []
            for neighbouring_agent in agents_at_exit:
                agent = model.agents[neighbouring_agent]
                if agent.status == 1:
                    active_agent.append(agent.unique_id)
            exit_counts.append(len(active_agent))
        if sum(exit_counts) == 0:
            exit_probs = [0.5, 0.5]
        else:
            exit_counts = [x if x != 0 else 0.1 for x in exit_counts]
            exit_probs = 1 / np.array(exit_counts)
            total = np.sum(exit_probs)
            exit_probs = [exit_prob / total for exit_prob in exit_probs]
        return exit_probs

    def similarity_by_age(self, model):
        neighbours_age_exits = []
        for gate_loc in model.gates_locations[model.gates_in:]:
            neighbouring_agents = model.tree.query_ball_point(gate_loc, 20)
            age = []
            for neighbouring_agent in neighbouring_agents:
                agent = model.agents[neighbouring_agent]
                if agent.status == 1:
                    age.append(agent.age)
            if len(age) > 0:
                mean_age_nearby = sum(age) / len(age)
            else:
                mean_age_nearby = 0
            subscore = 1 - abs(self.age - mean_age_nearby) / (max([agent.age for agent in model.agents]))
            neighbours_age_exits.append(subscore)
        neighbours_age_exits = [x if x != 0 else 0.1 for x in neighbours_age_exits]
        total = np.sum(neighbours_age_exits)
        exit_probs = [neighbours_age_exit / total for neighbours_age_exit in neighbours_age_exits]
        return exit_probs

    def similarity_by_gender(self, model):
        neighbours_gender_exits = []
        for gate_loc in model.gates_locations[model.gates_in:]:
            neighbouring_agents = model.tree.query_ball_point(gate_loc, 20)
            gender = []
            for neighbouring_agent in neighbouring_agents:
                agent = model.agents[neighbouring_agent]
                if agent.status == 1:
                    gender.append(agent.gender)
            if len(gender) > 0:
                sum_gender = gender.count(self.gender)
            else:
                sum_gender = 0
            neighbours_gender_exits.append(sum_gender)
        total = sum(neighbours_gender_exits)
        normalized_probabilities_gender = [neighbours_gender / total if total != 0 else 0.5 for neighbours_gender in
                                           neighbours_gender_exits]
        return normalized_probabilities_gender

    def all_potential_exits_locations(self, model):
        return model.gates_locations[model.gates_in:]

    #######################################################################################################################################

    def activate(self, model):
        '''
        Test whether an agent should become active.
        This happens when the model time is greater than the agent's activate time.
        '''
        if model.step_id > self.steps_activate:
            self.status = 1
            model.pop_active += 1
            self.step_start = model.step_id

    @staticmethod
    def distance(loc1, loc2):
        '''
        A helpful function to calculate the distance between two points.
        This simply takes the square root of the sum of the square of the elements.
        This appears to be faster than using np.linalg.norm.
        No doubt the numpy implementation would be faster for large arrays.
        Fortunately, all of our norms are of two-element arrays.
        :param arr:     A numpy array (or array-like DS) with length two.
        :return norm:   The norm of the array.
        '''
        x = loc1[0] - loc2[0]
        y = loc1[1] - loc2[1]
        norm = (x * x + y * y) ** .5
        return norm

    def move(self, model):
        direction = (self.loc_desire - self.location) / self.distance(self.loc_desire, self.location)
        for speed in self.speeds:
            # Direct. Try to move forwards by gradually smaller and smaller amounts
            new_location = self.location + speed * direction
            if self.collision(model, new_location):
                if model.do_history:
                    self.history_collisions += 1
                    model.history_collision_locs.append(new_location)
                    model.history_collision_times.append(model.step_id)

            else:
                break
            # If even the slowest speed results in a colision, then wiggle.
            if speed == self.speeds[-1]:
                # new_location = self.location + [0, -self.wiggle]
                new_location = self.location + [0, self.wiggle*np.random.randint(-1, 1+1)]
                if model.do_history:
                    self.history_wiggles += 1
                    model.history_wiggle_locs.append(new_location)
        # Rebound
        if not model.is_within_bounds(new_location):
            new_location = model.re_bound(new_location)
        # Move
        self.location = new_location
        self.speed = speed

        # calculate the new delay
        steps_exped = self.distance(self.loc_start, self.location) / self.speed_max
        steps_taken = model.step_id - self.step_start
        # model.steps_taken.append(steps_taken)
        steps_delay = steps_taken - steps_exped
        model.steps_delay.append(steps_delay)
        # model.steps_delay.append([0])

    def collision(self, model, new_location):
        '''
        Detects whether a move to the new_location will cause a collision
        (either with the model boundary or another agent).
        '''
        if not model.is_within_bounds(new_location):
            collide = True
        elif self.neighbourhood(model, new_location):
            collide = True
        else:
            collide = False
        return collide

    def neighbourhood(self, model, new_location):
        '''
        This method finds whether or not nearby neighbours are a collision.

        :param model:        the model that this agent is part of
        :param new_location: the proposed new location that the agent will move to
                             (a standard (x,y) floats-tuple)
        '''
        neighbours = False
        neighbouring_agents = model.tree.query_ball_point(new_location, model.separation)
        for neighbouring_agent in neighbouring_agents:
            agent = model.agents[neighbouring_agent]
            if agent.status == 1 and self.unique_id != agent.unique_id and new_location[0] <= agent.location[0]:
                neighbours = True
                break
        return neighbours

    def deactivate(self, model):
        '''
        Determine whether the agent should leave the model and, if so,
        remove them. Otherwise do nothing.
        '''
        if (self.distance(self.location, self.loc_desire) < model.gates_space) and (abs(self.location[1] - self.loc_desire[1]) < 5):
            self.status = 2
            model.pop_active -= 1
            model.pop_finished += 1
            steps_taken_evac = model.step_id - self.step_start
            model.get_evacuation_time.append(steps_taken_evac)
            # Record the exit event in a model-level list
            model.exit_flow_data.append({
                'Step': model.step_id,
                'Agent': self.unique_id,
                'Exit': self.selected_gate_out
            })
            # if model.do_history:
            # steps_exped = (self.distance(self.loc_start, self.loc_desire) - model.gates_space) / self.speeds[0]
            # model.steps_exped.append(steps_exped)
            # steps_taken = model.step_id - self.step_start
            # model.steps_taken.append(steps_taken)
            # steps_delay = steps_taken - steps_exped
            # model.steps_delay.append(steps_delay)

    def history(self, model):
        '''
        Save agent location.
        '''
        if model.do_history:
            if self.status == 1:
                self.history_locations.append(self.location)
            else:
                self.history_locations.append((None, None))


class Model:

    def __init__(self, unique_id=None, **kwargs):
        '''
        Create a new model, reading parameters from a keyword arguement dictionary.
        '''
        self.unique_id = unique_id
        self.status = 1

        # Default Parameters (usually overridden by the caller)
        params = {
            'pop_total': 100,

            'width': 150,
            'height': 100,
            'gates_width': [1, 2],

            'gates_in': 3,
            'gates_out': 2,
            'gates_space': 1,
            'gates_speed': 5,

            'speed_min': .5,
            'speed_max': 1,
            'speed_mean': 1,
            'speed_std': 1,
            'speed_steps': 3,

            'separation': 1,
            'max_wiggle': 1,
            'no_waves': 3,  # number of train arrivals within the study period
            'train_delay': 30,  # train arrival variation in seconds

            'step_limit': 500,

            'do_history': True,
            'do_print': False,

            'random_seed': int.from_bytes(os.urandom(4), byteorder='little')

        }

        if len(kwargs) == 0:
            warnings.warn(
                "No parameters have been passed to the model; using the default parameters: {}".format(params),
                RuntimeWarning
            )
        self.params, self.params_changed = Model._init_kwargs(params, kwargs)
        [setattr(self, key, value) for key, value in self.params.items()]
        # set up number of bins
        self.bins = np.linspace(0, self.step_limit, self.no_waves + 1)

        # Set the random seed
        np.random.seed(self.random_seed)
        # Constants
        self.speed_step = (self.speed_mean - self.speed_min) / self.speed_steps
        self.boundaries = np.array([[0, 0], [self.width, self.height]])
        self.gates_in_locs = [[0, 25], [0, 50], [0, 75]]  # example locations
        self.gates_out_locs = [[150, 20], [150, 80]]
        self.gates_locations = np.concatenate([self.gates_in_locs, self.gates_out_locs])
        # Variables
        self.step_id = 0
        self.pop_active = 0
        self.pop_finished = 0
        # Initialise
        self.agents = [Agent(self, unique_id) for unique_id in range(self.pop_total)]
        if self.do_history:
            self.history_state = []
            self.history_wiggle_locs = []
            self.history_collision_locs = []
            self.history_collision_times = []
            self.steps_taken = []
            self.steps_exped = []
            # self.steps_delay = []
            # Figure Shape Stuff
            self._wid = 8
            self._rel = self._wid / self.width
            self._hei = self._rel * self.height
            self._figsize = (self._wid, self._hei)
            self._dpi = 160
        ######################################
        self.data_list_sum = []
        self.get_evacuation_time = []
        self.exit_flow_data = []

    @staticmethod
    def _gates_init(x, y, n):
        if n == 2:  # Assuming you have 2 exits
            y_coords = [y * 0.3, y * 0.7]  # Specify the desired y-coordinates for the exits
        else:
            y_coords = np.linspace(0, y, n + 2)[1:-1]  # Use the original evenly spaced y-coordinates for other cases
        return np.array([np.full(n, x), y_coords]).T

    def is_within_bounds(self, loc):
        return all(self.boundaries[0] <= loc) and all(loc <= self.boundaries[1])

    def re_bound(self, loc):
        return np.clip(loc, self.boundaries[0], self.boundaries[1])

    @staticmethod
    def _init_kwargs(dict0, dict1):
        '''
        Internal dictionary update tool

        dict0 is updated by dict1 adding no new keys.
        dict2 is the changes excluding 'do_' keys.
        '''
        dict2 = dict()
        for key in dict1.keys():
            if key in dict0:
                if dict0[key] is not dict1[key]:
                    dict0[key] = dict1[key]
                    if 'do_' not in key:
                        dict2[key] = dict1[key]
            else:
                print(f'BadKeyWarning: {key} is not a model parameter.')
        return dict0, dict2

    def step(self):
        '''
        Iterate model forward one step.
        '''
        if self.step_id < self.step_limit and self.status == 1:
            if self.do_print and self.step_id % 100 == 0:
                print(f'\tIteration: {self.step_id}/{self.step_limit}')
            state = self.get_state('location2D')
            self.tree = cKDTree(state)
            self.steps_delay = []
            [agent.step(self) for agent in self.agents]
            #################################################################################
            current_step_data = {'Step': self.step_id}
            for exit_id in [3, 4]:  # assuming exits are labeled 3 and 4
                count = sum(1 for record in self.exit_flow_data if record['Step'] <= self.step_id and record['Exit'] == exit_id)
                current_step_data[f'Exit{exit_id}'] = count
                count2 = sum(1 for record in self.exit_flow_data if record['Step'] == self.step_id and record['Exit'] == exit_id)
                current_step_data[f'Exit_flow{exit_id}'] = count2
            # Optionally, append current_step_data to a list that will be exported later
            self.data_list_sum.append(current_step_data)
            ##################################################################################
            if self.do_history:
                self.history_state.append(state)
            self.step_id += 1
        else:
            if self.do_print and self.status == 1:
                print(f'StationSim {self.unique_id} - Everyone made it!')
                self.status = 0

    # State
    def get_state(self, sensor=None):
        '''
        Convert list of agents in model to state vector.
        '''
        if sensor is None:
            state = [(agent.status, *agent.location, agent.speed) for agent in self.agents]
            state = np.append(self.step_id, np.ravel(state))
        elif sensor is 'location':
            state = [agent.location for agent in self.agents]
            state = np.ravel(state)
        elif sensor is 'location2D':
            state = [agent.location for agent in self.agents]
        return state

    def set_state(self, state, sensor=None):
        '''
        Use state vector to set agent locations.
        '''
        if sensor is None:
            self.step_id = int(state[0])
            state = np.reshape(state[1:], (self.pop_total, 3))
            for i, agent in enumerate(self.agents):
                agent.status = int(state[i, 0])
                agent.location = state[i, 1:]
        elif sensor is 'location':
            state = np.reshape(state, (self.pop_total, 2))
            for i, agent in enumerate(self.agents):
                agent.location = state[i, :]
        elif sensor is 'location2D':
            for i, agent in enumerate(self.agents):
                agent.location = state[i, :]

    # TODO: Deprecated, update PF
    def agents2state(self, do_ravel=True):
        warnings.warn("Replace 'state = agents2state()' with 'state = get_state(sensor='location')'",
                      DeprecationWarning)
        return self.get_state(sensor='location')

    def state2agents(self, state):
        warnings.warn("Replace 'state2agents(state)' with 'set_state(state, sensor='location')'", DeprecationWarning)
        return self.set_state(state, sensor='location')

    # Analytics
    def get_analytics(self, sig_fig=None):
        '''
        A collection of analytics.
        '''
        analytics = {
            'Finish Time': self.step_id,
            'Total': self.pop_total,
            'Active': self.pop_active,
            'Finished': self.pop_finished,
            'Mean Time Taken': np.mean(self.steps_taken),
            'Mean Time Expected': np.mean(self.steps_exped),
            'Mean Time Delay': np.mean(self.steps_delay),
            'Mean Collisions': np.mean([agent.history_collisions for agent in self.agents]),
            'Mean Wiggles': np.mean([agent.history_wiggles for agent in self.agents]),
            # 'GateWiggles': sum(wig[0]<self.gates_space for wig in self.history_wiggle_locs)/self.pop_total
        }
        return analytics


    @classmethod
    def set_random_seed(cls, seed=None):
        """Set a new numpy random seed
        :param seed: the optional seed value (if None then get one from os.urandom)
        """
        new_seed = int.from_bytes(os.urandom(4), byteorder='little') if seed == None else seed
        np.random.seed(new_seed)


if __name__ == '__main__':
    warnings.warn("The stationsim_model.py code should not be run directly. Create a separate script and use that "
                  "to run experimets (e.g. see ABM_DA/experiments/StationSim basic experiment.ipynb )")
    print("Nothing to do")