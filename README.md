# SARSA Learning Algorithm


## AIM
To develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method.


## PROBLEM STATEMENT
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.


## SARSA LEARNING ALGORITHM
1. Initialize the Q-values arbitrarily for all state-action pairs.

2. Repeat for each episode:

   i. Initialize the starting state.

   ii. Repeat for each step of episode:
   
          a. Choose action from state using policy derived from Q (e.g., epsilon-greedy).
   
          b. Take action, observe reward and next state.
   
          c. Choose action from next state using policy derived from Q (e.g., epsilon-greedy).
   
          d. Update Q(s, a) := Q(s, a) + alpha * [R + gamma * Q(s', a') - Q(s, a)]
   
          e. Update the state and action.
   
    iii. Until state is terminal.

3. Until performance converges.

## SARSA LEARNING FUNCTION
### Name: THARUN D
### Register Number: 212223240167

```
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)


    select_action = lambda state, Q, epsilon: \
        np.argmax(Q[state]) \
        if np.random.random() > epsilon \
        else np.random.randint(len(Q[state]))


    alphas = decay_schedule(
        init_alpha,
        min_alpha,
        alpha_decay_ratio,
        n_episodes
    )


    epsilons = decay_schedule(
        init_epsilon,
        min_epsilon,
        epsilon_decay_ratio,
        n_episodes
    )

    for e in tqdm(range(n_episodes), leave=False):

        state, done = env.reset(), False
        action = select_action(state, Q, epsilons[e])

        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = select_action(next_state, Q, epsilons[e])


            td_target = reward + gamma * Q[next_state][next_action] * (not done)

            td_error = td_target - Q[state][action]

            Q[state][action] = Q[state][action] + alphas[e] * td_error

            state, action = next_state, next_action

        Q_track[e] = Q.copy() # Use Q.copy() to save the Q function at the end of each episode.
        pi_track.append(np.argmax(Q, axis=1))

    V = np.max(Q, axis=1)
    pi = lambda s: np.argmax(Q[s])

    return Q, V, pi, Q_track, pi_track
```
## OUTPUT:
![WhatsApp Image 2025-10-25 at 10 45 27_73e43f69](https://github.com/user-attachments/assets/c348f185-1c35-447d-b907-f38b43925078)
![WhatsApp Image 2025-10-25 at 11 28 27_bb72f86d](https://github.com/user-attachments/assets/affc7902-0e61-4ecb-948b-17c856c62eac)

![WhatsApp Image 2025-10-25 at 10 45 27_63765692](https://github.com/user-attachments/assets/a89408e7-861d-4827-8ea5-fbed66e388b0)
![WhatsApp Image 2025-10-25 at 10 45 28_41e09525](https://github.com/user-attachments/assets/85d98b13-d863-4eb4-92f2-909416e64344)


## RESULT:

Thus, SARSA learning successfully trained an agent for optimal policy.

