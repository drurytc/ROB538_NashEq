import numpy as np

class Nash:
    def __init__(self, num_people, bar_capacity, num_nights):
        self.num_people = num_people
        self.bar_capacity = bar_capacity
        self.num_nights = num_nights

    # Local reward function: payoff depends on the number of people attending
    def local_reward(self, num_people_going):
        return num_people_going * np.exp(-num_people_going / self.bar_capacity)

    # Global reward function: sum of local rewards across all nights
    def global_reward(self, people_per_night):
        return np.sum([self.local_reward(x) for x in people_per_night])

    # Difference reward function: marginal contribution of the agent attending or not attending
    def difference_reward(self, num_people_going):
        if num_people_going == 0:
            return 0  # No people attending, so difference reward is 0
        current_reward = num_people_going * np.exp(-num_people_going / self.bar_capacity)
        counterfactual_reward = (num_people_going - 1) * np.exp(-(num_people_going - 1) / self.bar_capacity)
        return current_reward - counterfactual_reward

    # Nash equilibrium: find a stable distribution of agents across nights
    def nash_equilibrium(self, use_global_reward=False, use_difference_reward=False):
        # Initial distribution: Each agent picks a random night
        people_per_night = np.zeros(self.num_nights, dtype=int)
        agent_assignment = np.random.choice(self.num_nights, self.num_people)

        for night in agent_assignment:
            people_per_night[night] += 1

        stable = False
        while not stable:
            stable = True  # Assume stable until an agent improves their reward by switching

            # Check each agent's decision and move if beneficial
            for agent in range(self.num_people):
                current_night = agent_assignment[agent]
                current_people = people_per_night[current_night]

                # Calculate current payoff (local, global, or difference)
                if use_global_reward:
                    current_payoff = self.global_reward(people_per_night)
                elif use_difference_reward:
                    current_payoff = self.difference_reward(current_people)
                else:
                    current_payoff = self.local_reward(current_people)

                # Try moving agent to another night and compare payoffs
                for other_night in range(self.num_nights):
                    if current_night != other_night:
                        # Simulate moving the agent
                        people_per_night[current_night] -= 1
                        people_per_night[other_night] += 1

                        # Calculate the new payoff after the move
                        if use_global_reward:
                            new_payoff = self.global_reward(people_per_night)
                        elif use_difference_reward:
                            new_payoff = self.difference_reward(people_per_night[other_night])
                        else:
                            new_payoff = self.local_reward(people_per_night[other_night])

                        # If new payoff is better, move the agent
                        if new_payoff > current_payoff:
                            agent_assignment[agent] = other_night
                            stable = False
                            break
                        else:
                            # Revert the move if no improvement
                            people_per_night[other_night] -= 1
                            people_per_night[current_night] += 1

        return people_per_night
    

def run_simulation(num_people, bar_capacity, num_nights, num_runs=100):

    # Placeholder for storing results
    locals = []
    globals = []
    differences = []
    bar_problem = Nash(num_people, bar_capacity, num_nights)
    
    for _ in range(num_runs):
        nash_local = bar_problem.nash_equilibrium()
        locals.append(tuple(sorted(nash_local)))
        
        nash_global = bar_problem.nash_equilibrium(use_global_reward=True)
        globals.append(tuple(sorted(nash_global)))
        
        nash_difference = bar_problem.nash_equilibrium(use_difference_reward=True)
        differences.append(tuple(sorted(nash_difference)))

    locals = set(locals)
    globals = set(globals)
    differences = set(differences)
    
    return locals, globals, differences

if __name__ == "__main__":
    num_people = 50
    bar_capacity = 8
    num_nights = 6

    bar_problem = Nash(num_people, bar_capacity, num_nights)

    # Run simulation 100 times and calculate averages
    locals, globals, differences = run_simulation(num_people, bar_capacity, num_nights, num_runs=100)

    # Nash Equilibrium using local reward
    print(f"Nash Equilibrium Distribution using local reward: {locals}")

    for l in locals:
        local_rewards = [bar_problem.local_reward(x) for x in l]
        print(f"Local Rewards (per night) at Nash Equilibrium: {local_rewards}")

    # Nash Equilibrium using global reward
    print(f"Nash Equilibrium Distribution using global reward: {globals}")

    # Calculate global rewards for each night
    for g in globals:
        global_rewards = bar_problem.global_reward(g)
        print(f"Global Rewards (per night) at Nash Equilibrium: {global_rewards}")

    # Nash Equilibrium using difference reward
    print(f"Nash Equilibrium Distribution using difference reward: {differences}")

    for d in differences:
        difference_rewards = [bar_problem.difference_reward(x) for x in d]
        print(f"Difference Rewards (per night) at Nash Equilibrium: {difference_rewards}")  

