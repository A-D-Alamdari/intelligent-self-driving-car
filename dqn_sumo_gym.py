
import gym
import gym_sumo

import math
import random
import numpy as np 
from collections import namedtuple,deque
from itertools import count

import torch
from torch.utils.tensorboard import SummaryWriter
torch.manual_seed(0)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import matplotlib
import matplotlib.pyplot as plt
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()
#import traci
# implementation link:
#https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

# if GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
import platform
if platform.system() == "Windows":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state','action','next_state','reward'))

step_done = 0


class ReplayMemory(object):
	"""docstring for ReplayMemory"""
	def __init__(self, capacity):
		super(ReplayMemory, self).__init__()
		self.memory = deque([], maxlen=capacity)

	def push(self,*args):
		self.memory.append(Transition(*args))

	def sample(self,batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256,128)
        self.layer5 = nn.Linear(128,n_actions) 

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layer5(x)


class Agent(object):
	"""docstring for Agent"""
	def __init__(self, arg):
		super(Agent, self).__init__()
		self.arg = arg
		self.batch_size = 32
		self.gamma = 0.99
		self.eps_start = 1.0
		self.eps_end = 0.01
		self.eps_decay = 100000
		self.tau = 0.05
		self.lr = 1e-5
		self.n_actions = 5
		self.n_observations = 19
		self.writter = SummaryWriter()
		self.episodic_loss = 0

		self.episode_durations = []

		# create network
		self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
		self.target_net = DQN(self.n_observations,self.n_actions).to(device)
		self.target_net.load_state_dict(self.policy_net.state_dict())

		self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
		self.memory = ReplayMemory(50000)

	def select_action(self,state):
		global step_done
		sample = random.random()
		#print(f'Step: {step_done}')
		eps_threshold = self.eps_end + (self.eps_start-self.eps_end)*math.exp(-1.*step_done/self.eps_decay)
		step_done += 1

		if sample > eps_threshold:
			with torch.no_grad():
				return self.policy_net(state).max(1)[1].view(1,1)
		else:
			return torch.tensor([[np.random.choice(self.n_actions)]], device= device,dtype=torch.long)


	def learn_model(self):
		if len(self.memory) < self.batch_size:
			return

		transitions = self.memory.sample(self.batch_size)
		batch = Transition(*zip(*transitions))

		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
		non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
		state_batch = torch.cat(batch.state)
		action_batch = torch.cat(batch.action)
		reward_batch = torch.cat(batch.reward)
		state_action_values = self.policy_net(state_batch).gather(1, action_batch)
		next_state_values = torch.zeros(self.batch_size, device=device)
		with torch.no_grad():
			next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
		expected_state_action_values = (next_state_values * self.gamma) + reward_batch
		criterion = nn.SmoothL1Loss()
		loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
		self.episodic_loss += loss
		self.optimizer.zero_grad()
		loss.backward()
		#torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
		self.optimizer.step()

	def updateTargetNetwork(self):
		# soft update of the target network's weights
		# θ′ ← τ θ + (1 −τ )θ′
		target_net_state_dict = self.target_net.state_dict()
		policy_net_state_dict = self.policy_net.state_dict()

		for key in policy_net_state_dict:
			target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
		self.target_net.load_state_dict(target_net_state_dict)

	def plot_durations(self,show_result=False):
	    plt.figure(1)
	    durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
	    if show_result:
	        plt.title('Result')
	    else:
	        #plt.clf()
	        plt.title('Training...')
	    plt.xlabel('Episode')
	    plt.ylabel('Duration')
	    plt.plot(durations_t.numpy())
	    plt.savefig("training_reward_test.png")
	    # Take 100 episode averages and plot them too
	    # if len(durations_t) >= 100:
	    #     means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
	    #     means = torch.cat((torch.zeros(99), means))
	    #     plt.plot(means.numpy())

	    # plt.pause(0.001)  # pause a bit so that plots are updated
	    # if is_ipython:
	    #     if not show_result:
	    #         display.display(plt.gcf())
	    #         display.clear_output(wait=True)
	    #     else:
	    #         display.display(plt.gcf())

	def train_pole(self, env):
		for e in range(5000):
			state, info = env.reset()
			print(state)
			r_r = 0
			state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
			for t in count():
				env.render()
				action = self.select_action(state)
				observation, reward, terminated, turncated, _ = env.step(action.item())
				r_r += reward
				reward = torch.tensor([reward], device=device)
				done = terminated or turncated
				if terminated:
					next_state = None
				else:
					next_state = torch.tensor(observation,dtype=torch.float32, device=device).unsqueeze(0)

				self.memory.push(state, action, next_state, reward)
				state = next_state

				self.learn_model()
				self.updateTargetNetwork()

				if done:
					self.episode_durations.append(r_r)
					self.plot_durations()
					break
		self.plot_durations(show_result=True)
		plt.ioff()
		plt.show()


	def train_RL(self, env):
		for e in range(5000):
			state, info = env.reset()
			r_r = 0
			state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
			for t in count():
				#env.render()
				action = self.select_action(state)
				observation, reward, terminated, _ = env.step(action.item())
				r_r += reward
				if reward == -10:
					print(f'Collision: {reward}')
				reward = torch.tensor([reward], device=device)
				done = terminated
				if terminated:
					next_state = None
				else:
					next_state = torch.tensor(observation,dtype=torch.float32, device=device).unsqueeze(0)

				self.memory.push(state, action, next_state, reward)
				state = next_state

				self.learn_model()
				self.updateTargetNetwork()
				if (e+1)%2 == 0:
					torch.save(self.policy_net.state_dict(), "models/model_test.pth")

				if done:
					self.episode_durations.append(r_r)
					self.plot_durations()
					env.closeEnvConnection()
					print(f'Episodes:{e+1}, Reward: {r_r}')
					break
				env.move_gui()
			self.writter.add_scalar("Loss/train", self.episodic_loss, (e+1))
			self.writter.add_scalar("Reward/Train", r_r, (e+1))
			self.writter.flush()
			self.episodic_loss = 0.0
		self.plot_durations(show_result=True)
		plt.ioff()
		plt.show()
		env.closeEnvConnection()
		self.writter.close()






		
