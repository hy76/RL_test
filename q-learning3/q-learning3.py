import gym
import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if (np.random.uniform() < self.epsilon):
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.argmax()
        else:
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_, done):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if not done:
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r

        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    agent = QLearningTable(list(range(env.action_space.n)))

    for episode in range(100):
        observation = env.reset()

        t = 0
        while True:
            t += 1
            r = 0
            env.render()
            action = agent.choose_action(str(observation))
            observation_, reward, done, info = env.step(action)
            r += reward
            # print(reward)
            agent.learn(str(observation), action, reward, str(observation_), done)

            observation = observation_

            if done:
                print("Episode{} finished after {} timesteps".format(episode+1, t))
                print(r/t)
                # print(agent.q_table)
                break

    env.close()




