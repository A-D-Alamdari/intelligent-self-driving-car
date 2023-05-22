import os
import gym
from gym import spaces
import pygame
import numpy as np 
import sys
from gym_sumo.envs import env_config as c
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'],'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")
import traci
import sumolib
## add each lane density and mean_speed ##

def creat_observation():
	state_space_list = ['ego_speed','ego_acc','ego_dis_to_leader','leader_speed','leader_acc','dis_to_left_leader','left_leader_speed','left_leader_acc',
	'dis_to_right_leader','right_leader_speed','right_leader_acc']
	for i in range(c.NUM_OF_LANES):
		state_space_list.append("lane_"+str(i)+"_mean_speed")
		state_space_list.append("lane_"+str(i)+"_density")
	#print(state_space_list)
	state_space_low = np.array([c.RL_MIN_SPEED_LIMIT,-c.RL_DCE_RANGE,-c.RL_SENSING_RADIUS,c.RL_MIN_SPEED_LIMIT,-c.RL_DCE_RANGE,-c.RL_SENSING_RADIUS,c.RL_MIN_SPEED_LIMIT,-c.RL_DCE_RANGE,-c.RL_SENSING_RADIUS,c.RL_MIN_SPEED_LIMIT,-c.RL_DCE_RANGE
		,c.RL_MIN_SPEED_LIMIT,c.MIN_LANE_DENSITY,c.RL_MIN_SPEED_LIMIT,c.MIN_LANE_DENSITY,c.RL_MIN_SPEED_LIMIT,c.MIN_LANE_DENSITY,c.RL_MIN_SPEED_LIMIT,c.MIN_LANE_DENSITY])
	state_space_high = np.array([c.RL_MAX_SPEED_LIMIT,c.RL_ACC_RANGE,c.RL_SENSING_RADIUS,c.RL_MAX_SPEED_LIMIT,c.RL_ACC_RANGE,c.RL_SENSING_RADIUS,c.RL_MAX_SPEED_LIMIT,c.RL_ACC_RANGE,c.RL_SENSING_RADIUS,c.RL_MAX_SPEED_LIMIT,c.RL_ACC_RANGE
		,c.RL_MAX_SPEED_LIMIT,c.MAX_LANE_DENSITY,c.RL_MAX_SPEED_LIMIT,c.MAX_LANE_DENSITY,c.RL_MAX_SPEED_LIMIT,c.MAX_LANE_DENSITY,c.RL_MAX_SPEED_LIMIT,c.MAX_LANE_DENSITY])

	obs = spaces.Box(low=state_space_low,high=state_space_high,dtype=np.float64)
	return obs

class SumoEnv(gym.Env):
	"""docstring for SumoEnv"""
	metadata = {"render_modes": ["", "human", "rgb_array"], "render_fps": 4}
	def __init__(self,render_mode=None):
		super(SumoEnv, self).__init__()
		self.action_space = spaces.Discrete(5)
		self.observation_space = creat_observation()

		## class variable
		self.ego = c.EGO_ID
		self.num_of_lanes = c.NUM_OF_LANES
		self.max_speed_limit = c.RL_MAX_SPEED_LIMIT
		self.is_collided = False
		assert render_mode is None or render_mode in self.metadata['render_modes']
		self.render_mode =render_mode
		print(self.render_mode)

	def _getInfo(self):
		return {"current_episode":0}

	def _startSumo(self):
		sumoBinary = "sumo"
		if self.render_mode=="human":
			sumoBinary = "sumo-gui"
		sumoCmd = [sumoBinary, "-c", "sumo_networks/test.sumocfg","--lateral-resolution","3.8",
		 "--start", "true", "--quit-on-end", "true","--no-warnings","True", "--no-step-log", "True"]
		traci.start(sumoCmd)

	def reset(self, seed=None, options=None):
		super().reset(seed=seed)
		self.is_collided = False
		self._startSumo()
		self._warmup()
		obs = np.array(self.observation_space.sample())
		info = self._getInfo()
		return obs, info

	def _getCloseLeader(self, leaders):
		if len(leaders) <= 0:
			return "", -1
		min_dis = float('inf')
		current_leader = None,
		for leader in leaders:
			leader_id, dis = leader
			if dis < min_dis:
				current_leader = leader_id
				min_dis = dis
		return (current_leader, min_dis)

	def _getLaneDensity(self):
		road_id = traci.vehicle.getRoadID(self.ego)
		density = []
		mean_speed = []
		for i in range(self.num_of_lanes):
			density.append(len(traci.lane.getLastStepVehicleIDs(road_id+"_"+str(i))))
			mean_speed.append(traci.lane.getLastStepMeanSpeed(road_id+"_"+str(i)))
		return density, mean_speed

	def _get_rand_obs(self):
		return np.array(self.observation_space.sample())
	def _get_observation(self):
		if self._isEgoRunning()==False:
			return self._get_rand_obs()

		ego_speed = traci.vehicle.getSpeed(self.ego)
		ego_accleration = traci.vehicle.getAccel(self.ego)
		ego_leader = traci.vehicle.getLeader(self.ego)
		if ego_leader is not None:
			leader_id, distance = ego_leader
		else:
			leader_id, distance = "", -1
		l_speed = traci.vehicle.getSpeed(leader_id) if leader_id != "" else 0.01
		l_acc = traci.vehicle.getAccel(leader_id) if leader_id != "" else -2.6
		left_leader, left_l_dis = self._getCloseLeader(traci.vehicle.getLeftLeaders(self.ego, blockingOnly=True))
		left_l_speed = traci.vehicle.getSpeed(left_leader) if left_leader != "" else 0.01
		left_l_acc = traci.vehicle.getAccel(left_leader) if left_leader != "" else -2.6

		right_leader, right_l_dis = self._getCloseLeader(traci.vehicle.getRightLeaders(self.ego, blockingOnly=True))
		right_l_speed = traci.vehicle.getSpeed(right_leader) if right_leader != "" else 0.01
		right_l_acc = traci.vehicle.getAccel(right_leader) if right_leader != "" else -2.6

		states = [ego_speed, ego_accleration, distance, l_speed, l_acc, left_l_dis, left_l_speed, left_l_acc,
			right_l_dis, right_l_speed, right_l_acc]
		density, mean_speed = self._getLaneDensity()
		for i in range(self.num_of_lanes):
			states.append(density[i])
			states.append(mean_speed[i])

		observations = np.array(states)
		return observations

	def _applyAction(self, action):
		if self._isEgoRunning() == False:
			return
		current_lane_index = traci.vehicle.getLaneIndex(self.ego)
		if action == 0:
			# do nothing: stay in the current lane
			pass
		elif action == 1:
			target_lane_index = min(current_lane_index+1, self.num_of_lanes-1)
			traci.vehicle.changeLane(self.ego, target_lane_index, 0.1)
		elif action == 2:
			target_lane_index = max(current_lane_index-1, 0)
			traci.vehicle.changeLane(self.ego, target_lane_index, 0.1)
		elif action == 3:
			traci.vehicle.setAcceleration(self.ego,0.2, 0.1)
		elif action == 4:
			traci.vehicle.setAcceleration(self.ego, -4.5, 0.1)


	def _collision_reward(self):
		collide_vehicles = traci.simulation.getCollidingVehiclesIDList()
		if self.ego in collide_vehicles:
			self.is_collided = True
			return -10
		return 0.0
	def _efficiency(self):
		speed = traci.vehicle.getSpeed(self.ego)
		if speed < 25.0:
			return (speed-25.0)/(self.max_speed_limit-25.0)
		if speed > self.max_speed_limit:
			return (self.max_speed_limit-speed)/(self.max_speed_limit-25.0)
		return speed/self.max_speed_limit
	def _lane_change_reward(self,action):
		if action == 1 or action == 2:
			return -1.0
		return 0

	def _reward(self, action):
		c_reward = self._collision_reward()
		if self.is_collided or self._isEgoRunning()==False:
			return c_reward
		return c_reward + self._efficiency() + self._lane_change_reward(action)



	def step(self, action):
		self._applyAction(action)
		traci.simulationStep()
		reward = self._reward(action)
		observation = self._get_observation()
		done = self.is_collided or (self._isEgoRunning()==False)
		if done == False and traci.simulation.getTime() > 360:
			done = True
		return (observation, reward, done, {})


	def _isEgoRunning(self):
		v_ids_e0 = traci.edge.getLastStepVehicleIDs("E0")
		v_ids_e1 = traci.edge.getLastStepVehicleIDs("E1")
		v_ids_e2 = traci.edge.getLastStepVehicleIDs("E2")
		if "av_0" in v_ids_e0 or "av_0" in v_ids_e1 or "av_0" in v_ids_e2:
			return True
		return False

	def _warmup(self):
		while True:
			v_ids_e0 = traci.edge.getLastStepVehicleIDs("E0")
			v_ids_e1 = traci.edge.getLastStepVehicleIDs("E1")
			v_ids_e2 = traci.edge.getLastStepVehicleIDs("E2")
			if "av_0" in v_ids_e0 or "av_0" in v_ids_e1 or "av_0" in v_ids_e2:
				traci.vehicle.setLaneChangeMode(self.ego,0)
				#traci.vehicle.setSpeedMode(self.ego,0)
				return True
			traci.simulationStep()


	def closeEnvConnection(self):
		traci.close()

	def move_gui(self):
		if self.render_mode == "human":
			x, y = traci.vehicle.getPosition('av_0')
			traci.gui.setOffset("View #0",x-23.0,108.49)
		
