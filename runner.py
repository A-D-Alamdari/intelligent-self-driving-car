import os, sys
from sumo_env import Sumo
import traci
import numpy as np

import gymnasium as gym
from dqn import Agent

if __name__ == '__main__':
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    sumo = Sumo()
    agent = Agent("Agent")
    #agent.train_pole(env)
    agent.train_RL(sumo)


    # sumo = Sumo()
    # sumo.startSumo()
    # sumo.warmup()
    # observations = sumo.reset()
    # step = 0
    # while step < 360000:
    #     action = np.random.choice(3)
    #     next_obs,reward, done, _ = sumo.step(action)
    #     print(reward)
    #     if done:
    #         break;
    #     x, y = traci.vehicle.getPosition('av_0')
    #     traci.gui.setOffset("View #0",x-23.0,y)
    #     #traci.simulationStep()
    #     step += 1

    # traci.close()
