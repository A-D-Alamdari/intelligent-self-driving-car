import gym
import gym_sumo
from dqn_sumo_gym import Agent

if __name__ == '__main__':
    agent = Agent("Agent")
    env = gym.make("sumo-v0", render_mode="human")
    agent.train_RL(env)
