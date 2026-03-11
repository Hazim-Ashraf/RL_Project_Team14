import numpy as np
import matplotlib.pyplot as plt

n_states = 11
n_actions = 3
gamma = 0.95
theta = 1e-6

# --- Transition matrices from Task A ---
P_hover = np.zeros((n_states,n_states))
P_descent = np.zeros((n_states,n_states))
P_boost = np.zeros((n_states,n_states))

# terminal state
for P in [P_hover,P_descent,P_boost]:
    P[10,10] = 1

# hover
for s in range(10):
    P_hover[s,s] += 0.6
    P_hover[s,min(s+1,10)] += 0.4

# descent
for s in range(10):
    P_descent[s,max(s-1,0)] += 0.48
    P_descent[s,s] += 0.44
    P_descent[s,min(s+1,10)] += 0.08

# emergency boost
for s in range(10):
    target = max(s-3,0)
    if s <= 6:
        P_boost[s,target] = 1
    else:
        P_boost[s,target] = 0.75
        P_boost[s,10] = 0.25

# combine them
P = np.stack([P_hover,P_descent,P_boost])

# --- Reward matrix ---
R = np.zeros((n_states,n_actions))

state_reward = np.zeros(n_states)
state_reward[0:4] = -5
state_reward[4:7] = 20
state_reward[7:10] = -5
state_reward[10] = -500

action_cost = [0,-2,-8]

for s in range(n_states):
    for a in range(n_actions):
        R[s,a] = state_reward[s] + action_cost[a]

# terminal reward
R[10,:] = -500

# --- Value Iteration ---
V = np.zeros(n_states)
diff_history = []

while True:
    V_new = np.zeros(n_states)

    for s in range(n_states):

        action_values = []

        for a in range(n_actions):
            expected_value = np.sum(P[a,s,:] * (R[s,a] + gamma*V))
            action_values.append(expected_value)

        V_new[s] = max(action_values)

    diff = np.max(np.abs(V_new - V))
    diff_history.append(diff)

    V = V_new.copy()

    if diff < theta:
        break

# optimal policy
policy = np.zeros(n_states)

for s in range(n_states):

    action_values = []

    for a in range(n_actions):
        expected_value = np.sum(P[a,s,:]*(R[s,a] + gamma*V))
        action_values.append(expected_value)

    policy[s] = np.argmax(action_values)

print("Optimal Value Function:")
print(V)

print("\nOptimal Policy (0=Hover,1=Descent,2=Boost):")
print(policy)

# convergence plot
plt.plot(diff_history)
plt.xlabel("Iteration")
plt.ylabel("||Vk - Vk-1||")
plt.title("Value Iteration Convergence")
plt.show()


gamma = 0.95
theta = 1e-6

# initial random policy
policy = np.random.randint(0,3,size=11)

V = np.zeros(11)

while True:

    # ---------- POLICY EVALUATION ----------
    while True:

        delta = 0

        for s in range(11):

            a = policy[s]

            v_new = np.sum(P[a,s,:] * (R[s,a] + gamma*V))

            delta = max(delta, abs(v_new - V[s]))

            V[s] = v_new

        if delta < theta:
            break

    # ---------- POLICY IMPROVEMENT ----------
    policy_stable = True

    for s in range(11):

        old_action = policy[s]

        action_values = []

        for a in range(3):

            val = np.sum(P[a,s,:]*(R[s,a] + gamma*V))
            action_values.append(val)

        best_action = np.argmax(action_values)

        policy[s] = best_action

        if old_action != best_action:
            policy_stable = False

    if policy_stable:
        break

print("Optimal Value Function:")
print(V)

print("\nOptimal Policy:")
print(policy)

from matplotlib.patches import Patch

# optimal policy from your result
policy = np.array([0,0,0,0,0,1,1,1,1,1])

states = np.arange(10)

plt.figure(figsize=(8,4))

plt.bar(states, policy)

plt.xlabel("State")
plt.ylabel("Best Action")
plt.title("Optimal Action per State")

plt.xticks(states)
plt.yticks([0,1,2])

# custom legend explaining actions
legend_elements = [
    Patch(facecolor='blue', label='0 = Hover'),
    Patch(facecolor='blue', label='1 = Controlled Descent'),
    Patch(facecolor='blue', label='2 = Emergency Boost')
]

plt.legend(handles=legend_elements, loc="upper right")

plt.show()