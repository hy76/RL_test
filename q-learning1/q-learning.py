import numpy as np
import pandas as pd
import time

N_STATES = 6   # the length of the 1 dimensional world
ACTIONS = ['left', 'right']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 13   # maximum episodes
FRESH_TIME = 0.3    # fresh time for one move


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions,
    )
    return table


def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if(np.random.uniform() > EPSILON) or (state_actions.all()==0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()
    return action_name


def get_env_feedback(state, action):
    if action == 'right':
        if state == N_STATES-2:
            _state = 'terminal'
            reward = 1
        else:
            _state = state + 1
            reward = 0
    else:
        reward=0
        if state == 0:
            _state = state
        else:
            _state = state - 1
    return _state, reward


def update_env(state, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if state == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[state] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        state = 0
        is_terminated = False
        update_env(state, episode, step_counter)
        while not is_terminated:
            action = choose_action(state, q_table)
            _state, reward = get_env_feedback(state, action)
            q_predict = q_table.loc[state, action]
            if _state != 'terminal':
                q_target = reward + GAMMA * q_table.iloc[_state, :].max()
            else:
                q_target = reward
                is_terminated = True

            q_table.loc[state, action] += ALPHA * (q_target - q_predict)
            state = _state

            update_env(state, episode, step_counter+1)

            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)

