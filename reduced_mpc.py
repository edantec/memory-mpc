import matplotlib.pyplot as plt

import pinocchio as pin
import example_robot_data
import numpy as np
import configuration_reduced as conf

from problem_formulation import Problem, sampleRange
import torch
from bullet_Talos import BulletTalos
import sys
import os

sys.path.append(os.getcwd() + '/learn_dataset')
from neural_network import nn_model

# ####### LOAD COMPLETE ROBOT FOR BULLET  ############

URDF_FILENAME = "talos_reduced.urdf"
SRDF_FILENAME = "talos.srdf"
SRDF_SUBPATH = "/talos_data/srdf/" + SRDF_FILENAME
URDF_SUBPATH = "/talos_data/robots/" + URDF_FILENAME
modelPath = example_robot_data.getModelPath(URDF_SUBPATH)

rmodelFreeFlyer, geomModelFreeFlyer, visualModelFreeFlyer = pin.buildModelsFromUrdf(modelPath + URDF_SUBPATH, modelPath, pin.JointModelFreeFlyer())
pin.loadReferenceConfigurations(rmodelFreeFlyer,modelPath + SRDF_SUBPATH, False)
q0FreeFlyer = rmodelFreeFlyer.referenceConfigurations["half_sitting"]

# ####### OCP  ############

ocp = Problem(conf)
#ocp.ddp.setCallbacks([crocoddyl.CallbackVerbose()])
nq = ocp.rmodel.nq
nv = ocp.rmodel.nv

# Pybullet
device = BulletTalos(conf, rmodelFreeFlyer, True)
device.initializeJoints(q0FreeFlyer)
device.showObstacle(conf.obstaclePosition,conf.obstacleHeight,conf.obstacleRadius)
capsule_pose = np.array([-0.025,0,-.225])
device.init_capsule('arm_right_4_joint',capsule_pose)

# Load neural network
state_size = nq + nv
T_horizon = conf.T 
model = nn_model(nq + 3,64, state_size * T_horizon)
model.load_state_dict(torch.load('learn_dataset/nn_model/reduced_nn_obstacle2'))

# Sample target position
x_goals = np.array([0.6,0.9])
y_goals = np.array([0.0,0.6])
z_goals = np.array([0.8,1.2])
target_ranges = [x_goals, y_goals, z_goals]

x0 = ocp.rmodel.defaultState
iterations_warmstart = []
iterations_warmstart_nearest = []
iterations_nowarmstart = []
target_x = []
target_y = []

# Choose random target to reach
targetSample = np.array([0.8, 0.4, 1.1])
#targetSample = sampleRange(target_ranges)
ocp.updateTarget(targetSample)

# Retrieve warm-start from neural network
nn_input = torch.tensor(np.concatenate((targetSample,x0[:nq])), dtype=torch.float32)
trajs = model(nn_input)
trajs = trajs.reshape(1,T_horizon,state_size)[0]
trajs = trajs.detach().cpu().numpy()

# Initial solve
for i in range(T_horizon):
	ocp.ddp.problem.runningModels[i].differential.costs.changeCostStatus("position_RH",False)
ocp.ddp.problem.terminalModel.differential.costs.changeCostStatus("position_RH",False)

xs_init = [x0 for i in range(conf.T + 1)]
us_init = [ocp.ddp.problem.runningModels[0].quasiStatic(ocp.ddp.problem.runningDatas[0], x0) for i in range(conf.T)]
ocp.ddp.solve(xs_init, us_init, 100, False)
xs = ocp.ddp.xs[:]
us = ocp.ddp.us[:]
ddp_cost = ocp.ddp.cost

print(f"Initial solve takes {ocp.ddp.iter} iterations")

duration = 1500
Nc = 10
tracking_RH = T_horizon
end_tracking_RH = duration - 5 * T_horizon
device.showObstacle(conf.obstaclePosition,conf.obstacleHeight,conf.obstacleRadius)
device.showHandToTrack(targetSample,[0,1,0,1])
q_current, v_current = device.measureReducedState()
x_measured = np.concatenate((q_current, v_current))
time_between_memory = 0
use_memory = True
for s in range(duration * Nc):
	if s % 10 == 0:
		# Run high-level control with OCP solve
		# 10 times slower than low-level control
		tracking_RH -= 1
		end_tracking_RH -= 1
		# If beginning of motion, activate tracking cost
		if tracking_RH <= T_horizon and tracking_RH >= 0:
			if tracking_RH == T_horizon:
				ocp.ddp.problem.terminalModel.differential.costs.changeCostStatus("position_RH",True)
				print("Set active tracking RH for terminal AM ")
			else:
				ocp.ddp.problem.runningModels[tracking_RH].differential.costs.changeCostStatus("position_RH",True)
				print("Set active tracking RH for AM " + str(tracking_RH))
		# If end of motion, deactivate tracking cost
		if end_tracking_RH <= T_horizon and end_tracking_RH >= 0:
			if end_tracking_RH == T_horizon:
				ocp.ddp.problem.terminalModel.differential.costs.changeCostStatus("position_RH",False)
				print("Set i,active tracking RH for terminal AM ")
			else:
				ocp.ddp.problem.runningModels[end_tracking_RH].differential.costs.changeCostStatus("position_RH",False)
				print("Set inactive tracking RH for AM " + str(end_tracking_RH))
		
		# Update warm-start in a MPC-like style
		xs = list(xs[1:]) + [xs[-1]] 
		xs[0] = x_measured
		us = list(us[1:]) + [us[-1]] 
		ocp.ddp.problem.x0 = x_measured

		# Retrieve warm-start from neural network
		nn_input = torch.tensor(np.concatenate((targetSample,x_measured[:nq])), dtype=torch.float32)
		trajs = model(nn_input)
		trajs = trajs.reshape(1,T_horizon,state_size)[0]
		trajs = trajs.detach().cpu().numpy()

		xinit_hpp = list(np.concatenate((x_measured.reshape(1,12),trajs)))
		uinit_hpp = [ocp.ddp.problem.runningModels[0].quasiStatic(ocp.ddp.problem.runningDatas[0], xinit_hpp[i]) for i in range(T_horizon)]

		# Refine warm-start with DDP
		ocp.ddp.solve(xinit_hpp,uinit_hpp,1, False)
		xinit_nn = ocp.ddp.xs[:]
		uinit_nn = ocp.ddp.us[:]

		# Compare solutions 
		if ddp_cost > ocp.ddp.cost and time_between_memory > 1 and use_memory:
			print("Warm-start with memory guess")
			ocp.ddp.solve(xinit_nn,uinit_nn,1, False)
			time_between_memory = 0
		else:
			ocp.ddp.solve(xs,us,1, False)
			time_between_memory += 1

		ddp_cost = ocp.ddp.cost
		xs = ocp.ddp.xs[:]
		us = ocp.ddp.us[:]
	
	# Run low-level torque control with Riccati feedback
	torques = ocp.ddp.us[0] + ocp.ddp.K[0] @ (ocp.state.diff(x_measured,ocp.ddp.xs[0]))
	device.execute(torques)
	q_current, v_current = device.measureReducedState()
	x_measured = np.concatenate((q_current, v_current))
	device.set_capsule(capsule_pose)
