#!/usr/bin/env python3
"""Hard work here. Not efficient, but hard."""

from typing import Tuple

import pinocchio as pin  # type: ignore
import crocoddyl
import hppfcl
from example_robot_data.robots_loader import TalosLoader
import configuration_reduced as conf
import numpy as np

class FixedTalos(TalosLoader): free_flyer = False

class Problem:
	def __init__(self, configuration):
		fixed_talos = FixedTalos().robot

		self.configuration = configuration
		self.rmodelComplete = fixed_talos.model
		self.basePosition = np.array([0., 0., 1.01927])
		self.q0Complete = fixed_talos.q0 
		rdataComplete = self.rmodelComplete.createData()

		self.pinocchioControlledJoints = [ i for (i,n) in enumerate(self.rmodelComplete.names) if n in configuration.controlled_joints ]
		#1-6 leg_left, 7-12 leg_right, 13-14 torso, 15-21 arm_left, 22 gripper_left, 23-29 arm_right, 30 gripper_right, 31-32 head
		self.JointsToLockId = [ i for i in range(1,self.rmodelComplete.njoints) if i not in self.pinocchioControlledJoints]

		# Create reduced model
		self.rmodel = pin.buildReducedModel(self.rmodelComplete,self.JointsToLockId,self.q0Complete) 
		self.rdata = self.rmodel.createData()

		# Load reference configuration
		q0 = self.q0Complete[np.array(self.pinocchioControlledJoints) - 1]
		self.rmodel.defaultState = np.concatenate((q0, np.zeros(self.rmodel.nv)))
		endEffector = 'gripper_right_joint'
		self.endEffectorId = self.rmodel.getFrameId(endEffector)

		# Model structure
		self.state = crocoddyl.StateMultibody(self.rmodel)
		self.actuation = crocoddyl.ActuationModelFull(self.state)
		self.targetReach = np.array([0.7, .2, 1])

		# Collision geometry
		self.geomModel = pin.GeometryModel()
		RADIUS = 0.09
		LENGTH = 0.45
		self.collision_radius = RADIUS + configuration.obstacleRadius + 0.05
		se3ArmPose = pin.SE3.Identity()
		se3ArmPose.translation = np.matrix([[-0.025],[0.],[-.225]])
		se3ObsPose = pin.SE3.Identity()
		se3ObsPose.translation = configuration.obstaclePosition

		# Add capsule for the arm
		self.ig_arm = self.geomModel.addGeometryObject(pin.GeometryObject("simple_arm",
														self.rmodel.getFrameId("arm_right_4_link"),
														self.rmodel.frames[self.rmodel.getFrameId("arm_right_4_link")].parent,
														hppfcl.Capsule(0, LENGTH),
														se3ArmPose),
														self.rmodel)

		# Add obstacle in the world
		self.ig_obs = self.geomModel.addGeometryObject(pin.GeometryObject("simple_obs",
														self.rmodel.getFrameId("universe"),
														self.rmodel.frames[self.rmodel.getFrameId("universe")].parent,
														hppfcl.Capsule(0, configuration.obstacleHeight),
														se3ObsPose),
														self.rmodel)
        
		self.geomModel.addCollisionPair(pin.CollisionPair(self.ig_arm,self.ig_obs))
        
		self.ddp = self.createProblem()

	def createHandPositionCost(self):
		residualHandPosition = crocoddyl.ResidualModelFrameTranslation(self.state,self.endEffectorId,self.targetReach,self.actuation.nu)
		handPositionCost = crocoddyl.CostModelResidual(self.state,crocoddyl.ActivationModelQuad(3),residualHandPosition)
		
		return handPositionCost
	
	def createHandVelocityCost(self):
		goalMotion = pin.Motion(np.zeros(6))
		handLeftVelocityCost = crocoddyl.CostModelResidual(self.state, crocoddyl.ResidualModelFrameVelocity(
														   self.state, self.endEffectorId, 
														   goalMotion, pin.WORLD, self.actuation.nu))
		
		return handLeftVelocityCost
	
	def createStateRegularization(self):
		xRegCost = crocoddyl.CostModelResidual(self.state,
											   crocoddyl.ActivationModelWeightedQuad(self.configuration.stateWeights),
											   crocoddyl.ResidualModelState(self.state,self.rmodel.defaultState, self.actuation.nu))
		return xRegCost
	
	def createControlRegularization(self):
		uRegCost = crocoddyl.CostModelResidual(self.state,
											   crocoddyl.ActivationModelWeightedQuad(self.configuration.controlWeight),
											   crocoddyl.ResidualModelControl(self.state,self.actuation.nu))
		
		return uRegCost
	
	def createObstacleCost(self):
		residualPairCollision = crocoddyl.ResidualModelPairCollision(self.state,self.actuation.nu, self.geomModel, 0, self.rmodel.getJointId("arm_right_4_joint"))
		obstacleCost = crocoddyl.CostModelResidual(self.state,crocoddyl.ActivationModel2NormBarrier(3, self.collision_radius), residualPairCollision)
		
		return obstacleCost
	
	def createLimitCost(self):
		maxfloat = 1e9
		xlb = np.concatenate((self.configuration.lowKinematicLimits,
						 -maxfloat * np.ones(self.state.nv)))
		xub = np.concatenate((self.configuration.highKinematicLimits,
						maxfloat * np.ones(self.state.nv)))
		bounds = crocoddyl.ActivationBounds(xlb,xub,1.)
		limitCost = crocoddyl.CostModelResidual(self.state,
												crocoddyl.ActivationModelQuadraticBarrier(bounds),
												crocoddyl.ResidualModelState(self.state,np.zeros(self.state.nx), self.actuation.nu))
		
		return limitCost
	
	def createRunningModel(self):
		runningCostModel = crocoddyl.CostModelSum(self.state)
		
		# Target translation cost
		runningCostModel.addCost("position_RH", self.createHandPositionCost(), self.configuration.wHandTranslation)
		
		# Velocity translation cost
		runningCostModel.addCost("velocity_RH", self.createHandVelocityCost(), self.configuration.wHandVelocity)
		
		# State regularization	
		runningCostModel.addCost("state", self.createStateRegularization(), self.configuration.wStateReg)
		
		# Control regularization
		runningCostModel.addCost("control", self.createControlRegularization(), self.configuration.wControlReg)
		
		# Obstacle
		runningCostModel.addCost("obstacle", self.createObstacleCost(), self.configuration.wHandCollision)
		
		# Kinematic limits
		runningCostModel.addCost("limits", self.createLimitCost(), self.configuration.wLimit)
		
		#Integrate differential model
		dmodelRunning = crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, runningCostModel)
		runningModel = crocoddyl.IntegratedActionModelEuler(dmodelRunning, self.configuration.DT)
		
		temp_data = runningModel.createData()
		runningModel.differential.costs.costs["control"].cost.reference = runningModel.quasiStatic(temp_data,self.rmodel.defaultState)
		
		return runningModel

	def createTerminalModel(self):
		terminalCostModel = crocoddyl.CostModelSum(self.state)
		
		# Target translation cost
		terminalCostModel.addCost("position_RH", self.createHandPositionCost(), self.configuration.wHandTranslation)
		
		# Velocity translation cost
		terminalCostModel.addCost("velocity_RH", self.createHandVelocityCost(), self.configuration.wHandVelocity)
		
		# State regularization	
		terminalCostModel.addCost("state", self.createStateRegularization(), self.configuration.wStateReg)
		
		# Obstacle
		terminalCostModel.addCost("obstacle", self.createObstacleCost(), self.configuration.wHandCollision)
		
		# Kinematic limits
		terminalCostModel.addCost("limits", self.createLimitCost(), self.configuration.wLimit)
		
		dmodelTerminal = crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, terminalCostModel)
		terminalModel = crocoddyl.IntegratedActionModelEuler(dmodelTerminal, 0)
		
		return terminalModel
	
	def createProblem(self):
		runningModels = [self.createRunningModel() for i in range(self.configuration.T)]
		problem = crocoddyl.ShootingProblem(self.rmodel.defaultState,runningModels,self.createTerminalModel())
		ddp = crocoddyl.SolverDDP(problem)
		#ddp.setCallbacks([crocoddyl.CallbackVerbose()])
		
		return ddp
	
	def updateTarget(self,new_target):
		for i in range(conf.T):
			self.ddp.problem.runningModels[i].differential.costs.costs["position_RH"].cost.residual.reference = new_target - self.basePosition
		self.ddp.problem.terminalModel.differential.costs.costs["position_RH"].cost.residual.reference = new_target - self.basePosition
	
	def checkCollision(self):
		for i in range(self.ddp.problem.T):
			if self.ddp.problem.runningDatas[i].differential.costs.costs["obstacle"].residual.geometry.distanceResults[0].min_distance < 0.05:
				return False
		if self.ddp.problem.terminalData.differential.costs.costs["obstacle"].residual.geometry.distanceResults[0].min_distance < 0.05:
			return False
		return True
	
	def checkKinematicLimits(self):
		for i in range(self.ddp.problem.T):
			for j in range(self.rmodel.nq):
				if self.ddp.xs[i][j] < self.configuration.lowKinematicLimits[j]:
					print(f"Traj number {i} at joint {j} is below low limit")
					return False
				if self.ddp.xs[i][j] > self.configuration.highKinematicLimits[j]:
					print(f"Traj number {i} at joint {j} is above high limit")
					return False
		return True
		
	def solveDDP(self,x0,target, ddp_iteration = 100):
		self.updateTarget(target)
		self.ddp.problem.x0 = x0
		xs_init = [x0 for i in range(self.configuration.T + 1)]
		us_init = [np.zeros(self.rmodel.nv) for i in range(self.configuration.T)]
		
		self.ddp.solve(xs_init, us_init, ddp_iteration, False)
		
	def quasiStatic(self,trajs_x):
		uquasistatic = []
		rdata = self.ddp.problem.runningDatas[0]
		for i in range(len(trajs_x)):
			uquasistatic.append(self.ddp.problem.runningModels[0].quasiStatic(rdata,trajs_x[i]))
		return uquasistatic

def sampleRange(samples_range):
	samples_vector = []
	for sample_range in samples_range:
		samples_vector.append(np.random.uniform(sample_range[0],sample_range[1]))
	return np.array(samples_vector)
	
if __name__ == "__main__":
	problem = Problem(conf)
	print("OCP created")
