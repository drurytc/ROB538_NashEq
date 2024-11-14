import numpy as np
import matplotlib.pyplot as plt

def sample_action(q_values, nights):
    """ Sample an action based on the Q-values as probabilities. """
    # Ensure Q-values are non-negative for exponentiation
    q_values = np.array(q_values)
    # Convert Q-values to probabilities
    probabilities = np.exp(q_values) / np.sum(np.exp(q_values))  # Normalized probabilities
    # Sample an action based on probabilities
    action = np.random.choice(nights, p=probabilities)
    return action

def run_simulation(num_agents, num_nights, b, time_limit=1000, epsilon_init=0.05):
    # Initialize agent action-value estimates and history storage
    agent_values = np.zeros((num_agents, num_nights))  # Q-values for each agent for each night
    attendance_history = np.zeros((time_limit, num_nights))  # Tracks attendance for each night
    system_rewards_history = []  # Tracks system-wide rewards
    local_rewards_history = []  # Tracks local rewards per agent
    difference_rewards_history = []  # Tracks difference rewards per agent
    
    
    epsilon = epsilon_init
    temp = 1.0
    # Simulation loop over weeks
    for week in range(time_limit):
        epsilon = epsilon * 0.99  # Decrease exploration rate over time
        # Step 1: Each agent selects a night based on their action-value estimates (with exploration)
        actions = []
        for i in range(num_agents):
            # Random exploration: choose a random action with probability epsilon
            if np.random.rand() < epsilon:
                # Convert Q-values into a probability distribution using softmax
                action = sample_action(agent_values[i], num_nights)
                actions.append(action)
            else:
                # Exploitation: choose the best action (highest Q-value)
                action = np.argmax(agent_values[i])
            
            actions.append(action)

        # Step 2: Calculate attendance for each night
        attendance = np.zeros(num_nights)
        for action in actions:
            attendance[action] += 1  # Count how many agents attend each night
        attendance_history[week] = attendance  # Store attendance for this week

        # Step 3: Calculate system reward (global reward based on attendance)
        system_reward = np.sum(attendance * np.exp(-attendance / b))
        system_rewards_history.append(system_reward)

        # Step 4: Calculate local rewards for each agent
        local_rewards = np.zeros(num_agents)
        for i in range(num_agents):
            night = actions[i]
            xk = attendance[night]
            local_rewards[i] = xk * np.exp(-xk / b)  # Local reward based on attendance for the night
        local_rewards_history.append(np.mean(local_rewards))  # Store the average local reward

        # Step 5: Calculate difference rewards for each agent
        difference_rewards = np.zeros(num_agents)
        for i in range(num_agents):
            night = actions[i]

            # Compute system reward without the current agent's attendance
            attendance_minus_i = attendance.copy()
            attendance_minus_i[night] -= 1
            new_night = np.argmin(attendance_minus_i)
            #new_night = np.argmax(attendance_minus_i)
            attendance_minus_i[new_night] += 1
            # Select a new night for the agent to go on (targeting less crowded night)
            system_reward_without_i = np.sum(attendance_minus_i * np.exp(-attendance_minus_i / b)) 
            
            # Difference reward is the marginal impact of agent i on the system reward
            difference_rewards[i] = system_reward - system_reward_without_i
        difference_rewards_history.append(np.mean(difference_rewards))  # Store the average difference reward

       # Step 6: Update action-value estimates (Q-values) for each agent
        for i in range(num_agents):
            night = actions[i]
            
            # Initialize arrays to store possible rewards for each night
            possible_loc = np.zeros(num_nights)
            possible_sys = np.zeros(num_nights)
            
            # Compute possible future rewards for all nights
            for j in range(num_nights):
                xk = attendance[j]  # For each possible night j
                possible_loc[j] = xk * np.exp(-xk / b)  # Local reward structure
                possible_sys[j] = np.sum(attendance * np.exp(-attendance / b))  # System reward structure
            
            # Update the agent's value (Q-value) using the chosen reward structure
            # Uncomment the one you want to use:
            
            # Local reward update
            #agent_values[i, night] += (1.0* (local_rewards[i] + 1.0*max(possible_loc) - agent_values[i, night]))

            # Difference reward update 
            #agent_values[i, night] += (1.0 * (difference_rewards[i] + 1.0 * max(possible_loc) - agent_values[i, night]))
            
            # System-wide reward update
            agent_values[i, night] += (2.0 * (system_reward + 0.9 * max(possible_sys) - agent_values[i, night]))
            
            



    # Convert history lists to arrays for easier manipulation and plotting
    system_rewards_history = np.array(system_rewards_history)
    local_rewards_history = np.array(local_rewards_history)
    difference_rewards_history = np.array(difference_rewards_history)

    # Return the results for plotting later
    return system_rewards_history, local_rewards_history, difference_rewards_history, attendance_history


def plot_results(system_rewards_history, local_rewards_history, difference_rewards_history, attendance_history, num_nights):
    # Plot performance of the system, local, and difference rewards over time
    plt.figure(figsize=(10, 6))
    plt.plot(system_rewards_history, label='System Reward', color='blue')
    plt.plot(local_rewards_history, label='Local Reward', color='green')
    plt.plot(difference_rewards_history, label='Difference Reward', color='red')
    plt.xlabel('Weeks')
    plt.ylabel('Average Reward')
    plt.title('Average Performance of Agents Over Time - Case 2')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate mean and standard deviation for attendance across the weeks
    mean_attendance = np.mean(attendance_history, axis=0)
    std_attendance = np.std(attendance_history, axis=0)
    nights = np.arange(num_nights)  # Night indices for x-axis

    # Plot bar graph of average attendance with error bars
    plt.figure(figsize=(10, 5))

    # Set error bars to be the maximum of 0 or mean - std
    error_bars = np.where(mean_attendance - std_attendance < 0, mean_attendance, std_attendance)

    # Plot average attendance with error bars
    plt.bar(nights, mean_attendance, yerr=error_bars, capsize=5, alpha=0.7, color='skyblue', edgecolor='black', label='Average Attendance')

    # Plot attendance profile for the week with the max reward
    max_system_reward = np.max(system_rewards_history)
    max_index = np.argmax(system_rewards_history)
    max_attendance = attendance_history[max_index]

    # Add red circles to indicate max attendance during the max reward week
    plt.plot(nights, max_attendance, 'ro', markersize=10, label='Max Reward Attendance')

    # Add plot title and labels
    plt.title("Average Attendance Over Time")
    plt.xlabel("Nights")
    plt.ylabel("Average Attendance (# of Agents)")
    plt.xticks(nights)
    plt.grid(axis='y')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Case 1: 25 Agents, 7 Nights, b=5
    system_rewards_1, local_rewards_1, difference_rewards_1, attendance_history_1 = run_simulation(
        num_agents=25, num_nights=7, b=5, time_limit=1000)

    # Print Case 1 results
    initial_system_reward_1 = system_rewards_1[0]
    initial_attendance_1 = attendance_history_1[0]
    final_max_system_reward_1 = np.max(system_rewards_1)  # Use np.max to get the max value
    max_system_reward_iteration_1 = np.argmax(system_rewards_1)  # Use np.argmax to get the index
    attendance_at_max_reward_1 = attendance_history_1[max_system_reward_iteration_1]  # Get attendance at max reward

    print("Simulation with 25 agents, 5 capacity, 7 nights, and 1000 weeks")
    print(f"Initial system reward: {initial_system_reward_1}")
    print(f"Initial attendance: {initial_attendance_1}")
    print(f"Max system reward: {final_max_system_reward_1}")
    print(f"Attendance at max reward week: {attendance_at_max_reward_1}")
    print(f"Max system reward at iteration {max_system_reward_iteration_1}")
    print('_________________________________________________________')
    print(f"Final system reward: {system_rewards_1[-1]}")
    print(f"Final attendance: {attendance_history_1[-1]}")

    # Plot results for Case 1
    plot_results(system_rewards_1, local_rewards_1, difference_rewards_1, attendance_history_1, num_nights=7)

    # Case 2: 40 Agents, 6 Nights, b=4
    system_rewards_2, local_rewards_2, difference_rewards_2, attendance_history_2 = run_simulation(
        num_agents=40, num_nights=6, b=4, time_limit=1000)

    # Print Case 2 results
    initial_system_reward_2 = system_rewards_2[0]
    initial_attendance_2 = attendance_history_2[0]
    final_max_system_reward_2 = np.max(system_rewards_2)  # Use np.max to get the max value
    max_system_reward_iteration_2 = np.argmax(system_rewards_2)  # Use np.argmax to get the index
    attendance_at_max_reward_2 = attendance_history_2[max_system_reward_iteration_2]  # Get attendance at max reward

    print("Simulation with 40 agents, 4 capacity, 6 nights, and 1000 weeks")
    print(f"Initial system reward: {initial_system_reward_2}")
    print(f"Initial attendance: {initial_attendance_2}")
    print(f"Final max system reward: {final_max_system_reward_2}")
    print(f"Attendance at max reward week: {attendance_at_max_reward_2}")
    print(f"Max system reward at iteration {max_system_reward_iteration_2}")
    print('_________________________________________________________')
    print(f"Final system reward: {system_rewards_2[-1]}")
    print(f"Final attendance: {attendance_history_2[-1]}")

    # Plot results for Case 2
    plot_results(system_rewards_2, local_rewards_2, difference_rewards_2, attendance_history_2, num_nights=6)