import matplotlib.pyplot as plt

import pinocchio as pin
import example_robot_data
import numpy as np
import configuration_reduced as conf

from work import Problem, sampleRange
import time
import torch
from bullet_Talos import BulletTalos
import sys
import os

sys.path.append(os.getcwd() + '/learn_dataset')
from neural_network import nn_model

URDF_FILENAME = "talos_reduced.urdf"
SRDF_FILENAME = "talos.srdf"
SRDF_SUBPATH = "/talos_data/srdf/" + SRDF_FILENAME
URDF_SUBPATH = "/talos_data/robots/" + URDF_FILENAME
modelPath = example_robot_data.getModelPath(URDF_SUBPATH)

rmodelFreeFlyer, geomModelFreeFlyer, visualModelFreeFlyer = pin.buildModelsFromUrdf(modelPath + URDF_SUBPATH, modelPath, pin.JointModelFreeFlyer())
pin.loadReferenceConfigurations(rmodelFreeFlyer,modelPath + SRDF_SUBPATH, False)
q0FreeFlyer = rmodelFreeFlyer.referenceConfigurations["half_sitting"]

# coding: utf8

import numpy as np
import pybullet as pyb

# ####### CONFIGURATION  ############
# ### OCP

ocp = Problem(conf)
#ocp.ddp.setCallbacks([crocoddyl.CallbackVerbose()])
nq = ocp.rmodel.nq
nv = ocp.rmodel.nv

# Pybullet
device = BulletTalos(conf, rmodelFreeFlyer)
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

targetSample = np.array([0.8, 0.4, 1.1])
xs_init = [x0 for i in range(conf.T + 1)]
us_init = [ocp.ddp.problem.runningModels[0].quasiStatic(ocp.ddp.problem.runningDatas[0], x0) for i in range(conf.T)]
ocp.updateTarget(targetSample)
nn_input = torch.tensor(np.concatenate((targetSample,x0[:nq])), dtype=torch.float32)
trajs = model(nn_input)
trajs = trajs.reshape(1,T_horizon,state_size)[0]
trajs = trajs.detach().cpu().numpy()

ocp.ddp.solve(xs_init, us_init, 100, True)
xs1 = ocp.ddp.xs[:]

xinit_hpp = list(np.concatenate((x0.reshape(1,12),trajs)))
uinit_hpp = [ocp.ddp.problem.runningModels[0].quasiStatic(ocp.ddp.problem.runningDatas[0], xinit_hpp[i]) for i in range(conf.T)]
ocp.ddp.solve(xinit_hpp, uinit_hpp, 100, True)
xs2 = ocp.ddp.xs[:]
print(f"Solve with warmstart takes {ocp.ddp.iter} iterations")
test_number = 1
for i in range(test_number):
	print(f"Sample {i}")
	# Solve without warmstart
	#ocp.solveDDP(xSample, targetSample, 100)
	#iterations_nowarmstart.append(ocp.ddp.iter)
	device.showHandToTrack(targetSample,[0,1,0,1])
	for t in range(conf.T):
		time.sleep(0.01)
		device.resetReducedState(xs1[t][:6]) 
		device.set_capsule(capsule_pose)
	for t in range(conf.T):
		time.sleep(0.01)
		device.resetReducedState(xs2[t][:6]) 
		device.set_capsule(capsule_pose)


""" # Plot scatter plot to check regularization
target_x = np.array(target_x)
target_y = np.array(target_y)

# Coordinates at t = 1

fig = plt.figure()
ax1 = plt.subplot(131)
ax1.set_title("With neural network")
im = ax1.scatter(target_x,target_y,c=np.array(iterations_warmstart), cmap='viridis')
ax1.set_ylabel('Y target')
ax1.set_xlabel('X target')
ax2 = plt.subplot(132)
ax2.scatter(target_x,target_y,c=np.array(iterations_nowarmstart), cmap='viridis')
ax2.set_title("Without warmstart")
ax2.set_xlabel('X target')
ax2 = plt.subplot(133)
im = ax2.scatter(target_x,target_y,c=np.array(iterations_warmstart_nearest), cmap='viridis')
ax2.set_title("With nearest neighbours")
ax2.set_xlabel('X target')
cbar = fig.colorbar(im)

plt.show()
	 """
