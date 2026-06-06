# Implementation of SARSA Control Algorithm using FrozenLake Environment

## Aim

To implement the SARSA (State-Action-Reward-State-Action) Control algorithm for learning the optimal policy in the FrozenLake environment using Reinforcement Learning.

---

## Algorithm

### SARSA Control Algorithm

1. Import the required libraries.
2. Create the FrozenLake environment using OpenAI Gym.
3. Initialize:
   - Learning rate \( \alpha \)
   - Discount factor \( \gamma \)
   - Exploration rate \( \epsilon \)
   - Q-table \( Q(s,a) \)
4. Define the epsilon-greedy policy:
   - Choose a random action with probability \( \epsilon \).
   - Otherwise choose the action with the highest Q-value.
5. For each episode:
   - Reset the environment.
   - Choose an initial action using the epsilon-greedy policy.
   - Repeat until the episode ends:
     - Take the selected action.
     - Observe:
       - Next state
       - Reward
       - Terminal condition
     - Select the next action using the epsilon-greedy policy.
     - Update the Q-value using SARSA update rule:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \left[
r + \gamma Q(s',a') - Q(s,a)
\right]
\]

     - Update the current state and action.
6. Store rewards for each episode.
7. Reduce epsilon gradually for better exploitation.
8. Display the learned Q-table and reward performance graph.

---

## Program

```python

# ============================================================
# SARSA CONTROL ALGORITHM
# ============================================================

import gym
import numpy as np
import matplotlib.pyplot as plt

# Create Environment
env = gym.make("FrozenLake-v1", is_slippery=False)

# Parameters
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

episodes = 15000

# Q-table
Q = np.zeros((env.observation_space.n,
              env.action_space.n))

# Reward Tracking
rewards_per_episode = []

# Epsilon-Greedy Policy
def choose_action(state):

    if np.random.rand() < epsilon:
        return env.action_space.sample()

    return np.argmax(Q[state])

# SARSA Training
for episode in range(episodes):

    state, _ = env.reset()

    action = choose_action(state)

    done = False

    total_reward = 0

    while not done:

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        next_action = choose_action(next_state)

        # SARSA Update
        Q[state][action] = Q[state][action] + alpha * (
            reward
            + gamma * Q[next_state][next_action]
            - Q[state][action]
        )

        state = next_state
        action = next_action

        total_reward += reward

    # Store rewards
    rewards_per_episode.append(total_reward)

    # Decay epsilon
    epsilon = max(epsilon_min,
                  epsilon * epsilon_decay)

# ============================================================
# OPTIMAL POLICY
# ============================================================

print("\nLearned Optimal Policy:\n")

actions = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP"
}

for state in range(env.observation_space.n):

    best_action = np.argmax(Q[state])

    print(f"State {state} --> {actions[best_action]}")

# ============================================================
# REWARD GRAPH
# ============================================================

moving_avg = np.convolve(
    rewards_per_episode,
    np.ones(100)/100,
    mode='valid'
)

plt.figure(figsize=(10,5))

plt.plot(moving_avg)

plt.xlabel("Episodes")
plt.ylabel("Average Reward")
plt.title("SARSA Learning Progress")

plt.show()


      
```

## Output

<img width="839" height="312" alt="image" src="https://github.com/user-attachments/assets/497ca806-82da-46bc-9bc6-fef27c95efad" />



## Output Graph

<img width="920" height="431" alt="image" src="https://github.com/user-attachments/assets/50306517-d0dd-4381-aa8e-bca60ef90f55" />

The graph shows the reward obtained per episode during SARSA training. As training progresses, the rewards increase, indicating that the agent learns the optimal policy.

---

## Result

Thus, the SARSA Control algorithm was successfully implemented in the FrozenLake environment. The agent learned the optimal action-value function and improved its performance through continuous interaction with the environment using on-policy Temporal Difference learning.
