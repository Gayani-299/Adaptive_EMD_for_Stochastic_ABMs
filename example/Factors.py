import numpy as np

# run by a gate_out
# takes two subscore reporters and concats them with an addition operator
# @EMD @Factor @return-type=comparator @parameter-type=comparator @parameter-type=comparator
def combine(self, a, b):
    a_array = np.array(a)
    b_array = np.array(b)
    result_array = a_array + b_array
    return result_array.tolist()

# run by a gate_out
# takes two subscore reporters and concats them with an addition operator
# @EMD @Factor @return-type=comparator @parameter-type=comparator @parameter-type=comparator
def subtract(self, a, b):
    a_array = np.array(a)
    b_array = np.array(b)
    result_array = a_array - b_array
    return result_array.tolist()

# Select max exit gate
# run by an agents
# takes in callback reporters
# the callback reporters are the composite of the factors in the exit selection
# decision by the agents.
# select the potential exit with the max probability by the factor subscores
# @EMD @Factor @return-type=gate_out @parameter-type=gate_outs @parameter-type=comparator
def get_max_select_exit(self, model, probs):
    gate_out = model.gates_in + np.argmax(probs)
    return gate_out

# Select min exit gate
# run by an agents
# takes in callback reporters
# the callback reporters are the composite of the factors in the exit selection
# decision by the agents.
# select the potential exit with the min probability by the factor subscores
# @EMD @Factor @return-type=gate_out @parameter-type=gate_outs @parameter-type=comparator
def get_min_select_exit(self, model, probs):
    gate_out = model.gates_in + np.argmin(probs)
    return gate_out

# @EMD @Factor @return-type=gate_out @parameter-type=gate_outs @parameter-type=comparator
def randomly_select_exit(self, model ,prob):
    if sum(prob) == 0:
        prob = [1 / len(prob)] * len(prob)
    prob = np.clip(prob, 0, 1)
    prob = prob / np.sum(prob)
    gate_out = np.random.choice(range(model.gates_in, model.gates_in + model.gates_out), p=prob)
    return gate_out
###########################Rational factors############################
# run by a exit called by a agent
# compares its own location to the location of the exits of the calling agents
# @EMD @Factor @return-type=comparator
def compare_distance(self, model):
    distances_to_exits = [self.distance(self.location, gate_loc) for gate_loc in model.gates_locations[model.gates_in:]]
    total_distance = sum(distances_to_exits)
    normalized_probabilities_distance = [distance / total_distance if total_distance != 0 else 0 for distance in
                                         distances_to_exits]
    return normalized_probabilities_distance

# @EMD @Factor @return-type=comparator
def compare_width(self, model):
    width_of_exits = model.gates_width
    sum_width_of_exits = sum(width_of_exits)
    exit_probs = [d / sum_width_of_exits if sum_width_of_exits != 0 else 0 for d in width_of_exits]
    return exit_probs

###########################Social/emotional factors############################
# run by a exit
# reports the number of nearby agent
# @EMD @Factor @return-type=comparator
def neighbourhood_count_exits(self, model):
    neighbours_count_exits = []
    for gate_loc in model.gates_locations[model.gates_in:]:
        nearby_agents = model.tree.query_ball_point(gate_loc, model.separation)
        # print(nearby_agents)
        active_neighbours_count = sum(1 for agent_id in nearby_agents if self.unique_id != agent_id)
        # print(active_neighbours_count)
        neighbours_count_exits.append(active_neighbours_count)
    crowding_at_exits = neighbours_count_exits
    total_crowd = sum(crowding_at_exits)
    normalized_probabilities_crowd = [crowd / total_crowd if total_crowd != 0 else 0 for crowd in crowding_at_exits]
    return normalized_probabilities_crowd

# @EMD @Factor @return-type=comparator
def similarity_by_age(self, model):
        neighbours_age_exits = []
        # Get neighbours nearby
        for gate_loc in model.gates_locations[model.gates_in:]:
            neighbouring_agents = model.tree.query_ball_point(gate_loc, model.separation)
            # print(neighbouring_agents)
            active_neighbours_count = sum(1 for agent_id in neighbouring_agents if self.unique_id != agent_id)
            # print(active_neighbours_count)
            if active_neighbours_count>0:
                age =[]
                for neighbouring_agent in neighbouring_agents:
                    agent = model.agents[neighbouring_agent]
                    # print(agent.age)
                    age.append(agent.age)
                mean_age_nearby = sum(age) / active_neighbours_count
            else:
                mean_age_nearby = 0
            # print(mean_age_nearby)
            # Calculate subscore based on age difference
            subscore = 1 - abs(self.age - mean_age_nearby) / (1e-10 + max([agent.age for agent in model.agents]))
            neighbours_age_exits.append(subscore)
        # print(neighbours_age_exits)
        total = sum(neighbours_age_exits)
        normalized_probabilities_age = [neighbours_age / total if total != 0 else 0 for neighbours_age in neighbours_age_exits]
        # print(normalized_probabilities_age)
        return normalized_probabilities_age

# @EMD @Factor @return-type=comparator
def similarity_by_gender(self, model):
        neighbours_gender_exits = []
        # Get neighbours nearby
        for gate_loc in model.gates_locations[model.gates_in:]:
            neighbouring_agents = model.tree.query_ball_point(gate_loc, model.separation)
            print(neighbouring_agents)
            active_neighbours_count = sum(1 for agent_id in neighbouring_agents if self.unique_id != agent_id)
            print(active_neighbours_count)
            if active_neighbours_count>0:
                gender =[]
                for neighbouring_agent in neighbouring_agents:
                    agent = model.agents[neighbouring_agent]
                    print(agent.gender)
                    gender.append(agent.gender)
                sum_gender = gender.count(self.gender)
                print(self.gender)
            else:
                sum_gender = 0
            # print(mean_gender_nearby)
            neighbours_gender_exits.append(sum_gender)
        print(neighbours_gender_exits)
        total = sum(neighbours_gender_exits)
        normalized_probabilities_gender = [neighbours_gender / total if total != 0 else 0 for neighbours_gender in neighbours_gender_exits]
        print(normalized_probabilities_gender)
        return normalized_probabilities_gender

# run by a agent
# agent knows all potential exits in the room (Full Information)
# @EMD @Factor @return-type=gate_outs
def all_potential_exits_locations(self, model):
    return model.gates_locations[model.gates_in:]
