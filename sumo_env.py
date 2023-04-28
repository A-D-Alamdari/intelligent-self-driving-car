import os, sys
import numpy as np
import tensorflow as tf
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci

class Sumo(object):
	"""docstring for Sumo"""
	def __init__(self, ego_id="av_0"):
		super(Sumo, self).__init__()
		self.ego = ego_id
		self.num_of_lanes = 4
		self.max_speed_limit = 55.55
		self.is_collided = False

	def startSumo(self, isGui=False):
		sumoBinary = "sumo"
		if isGui:
			sumoBinary = "sumo-gui"
		sumoCmd = [sumoBinary, "-c", "sumo_networks/test.sumocfg","--lateral-resolution","3.2",
		 "--start", "true", "--quit-on-end", "true","--no-warnings","True", "--no-step-log", "True"]
		traci.start(sumoCmd)

	def closeSumo(self):
		traci.close()

	def reset(self):
		''' Reset states'''
		self.is_collided = False
		ego_speed = np.random.uniform(0.01, 55.55)
		ego_acc = np.random.uniform(-2.6,2.6)
		l_dis = np.random.uniform(0,300)
		l_speed = np.random.uniform(0.01,55.55)
		l_acc = np.random.uniform(-2.6,2.6)

		ll_dis = np.random.uniform(0,300)
		ll_speed = np.random.uniform(0.01,55.55)
		ll_acc = np.random.uniform(-2.6,2.6)

		rl_dis = np.random.uniform(0,300)
		rl_speed = np.random.uniform(0.01,55.55)
		rl_acc = np.random.uniform(-2.6,2.6)

		observations =np.array([ego_speed, ego_acc, l_dis, l_speed, l_acc, ll_dis, ll_speed, ll_acc,rl_dis, rl_speed, rl_acc])

		mu = np.mean(observations)
		std = np.std(observations)
		observations = (observations-mu)/std
		#observations = tf.convert_to_tensor(observations, dtype=tf.float32)
		return observations

	def getCloseLeader(self, leaders):
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


	def getState(self):
		if self.isEgoRunning()==False:
			return self.reset()
		ego_speed = traci.vehicle.getSpeed(self.ego)
		ego_accleration = traci.vehicle.getAccel(self.ego)
		ego_leader = traci.vehicle.getLeader(self.ego)
		if ego_leader is not None:
			leader_id, distance = ego_leader
		else:
			leader_id, distance = "", -1
		l_speed = traci.vehicle.getSpeed(leader_id) if leader_id != "" else 0.01
		l_acc = traci.vehicle.getAccel(leader_id) if leader_id != "" else -2.6
		left_leader, left_l_dis = self.getCloseLeader(traci.vehicle.getLeftLeaders(self.ego))
		left_l_speed = traci.vehicle.getSpeed(left_leader) if left_leader != "" else 0.01
		left_l_acc = traci.vehicle.getAccel(left_leader) if left_leader != "" else -2.6

		right_leader, right_l_dis = self.getCloseLeader(traci.vehicle.getRightLeaders(self.ego))
		right_l_speed = traci.vehicle.getSpeed(right_leader) if right_leader != "" else 0.01
		right_l_acc = traci.vehicle.getAccel(right_leader) if right_leader != "" else -2.6

		observations = np.array([ego_speed, ego_accleration, distance, l_speed, l_acc, left_l_dis, left_l_speed, left_l_acc,
			right_l_dis, right_l_speed, right_l_acc])
		mu = np.mean(observations)
		std = np.std(observations)
		observations = (observations-mu)/std
		#observations = tf.convert_to_tensor(observations,dtype=tf.float32)
		return observations

	def applyAction(self, action):
		if self.isEgoRunning() == False:
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


	def collision_reward(self):
		collide_vehicles = traci.simulation.getCollidingVehiclesIDList()
		if self.ego in collide_vehicles:
			self.is_collided = True
			return -500
		return 0.0
	def efficiency(self):
		return traci.vehicle.getSpeed(self.ego)/self.max_speed_limit
	def lane_change_reward(self,action):
		if action == 1 or action == 2:
			return -1.0
		return 0

	def reward(self, action):
		c_reward = self.collision_reward()
		if self.is_collided or self.isEgoRunning()==False:
			print(f'Is Running: {self.isEgoRunning(), self.is_collided}')
			return c_reward
		return c_reward + self.efficiency() + self.lane_change_reward(action)

	def step(self, action):
		self.applyAction(action)
		traci.simulationStep()
		reward = self.reward(action)
		obs = self.getState()
		done = self.is_collided or (self.isEgoRunning()==False)
		if done == False and traci.simulation.getTime() > 300:
			done = True
		return obs, reward, done, {}


	def isEgoRunning(self):
		v_ids_e0 = traci.edge.getLastStepVehicleIDs("E0")
		v_ids_e1 = traci.edge.getLastStepVehicleIDs("E1")
		v_ids_e2 = traci.edge.getLastStepVehicleIDs("E2")
		if "av_0" in v_ids_e0 or "av_0" in v_ids_e1 or "av_0" in v_ids_e2:
			return True
		return False

	def warmup(self):
		while True:
			v_ids_e0 = traci.edge.getLastStepVehicleIDs("E0")
			v_ids_e1 = traci.edge.getLastStepVehicleIDs("E1")
			v_ids_e2 = traci.edge.getLastStepVehicleIDs("E2")
			if "av_0" in v_ids_e0 or "av_0" in v_ids_e1 or "av_0" in v_ids_e2:
				traci.vehicle.setLaneChangeMode(self.ego,0)
				traci.vehicle.setSpeedMode(self.ego,0)
				return True
			traci.simulationStep()








