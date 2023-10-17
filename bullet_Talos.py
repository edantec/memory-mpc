#!/usr/bin/env python3
"""
Created on Mon May  9 18:22:56 2022

@author: nvilla
"""

import pybullet_data
import pybullet as p  # PyBullet simulator
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

# import os

class BulletTalos:
    def __init__(self, conf, rmodelComplete):
        p.connect(p.GUI)  # Start the client for PyBullet
        p.setTimeStep(conf.simu_step)
        p.setGravity(*conf.gravity.tolist())  # Set gravity (disabled by default)

        # place CoM of root link ## TODO: check placement
        robotStartPosition = [0.0, 0.0, 1.01927]
        robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        p.setAdditionalSearchPath(conf.modelPath + "/talos_data/robots/")

        self.robotId = p.loadURDF(
            conf.URDF_FILENAME,
            robotStartPosition,
            robotStartOrientation,
            useFixedBase=False,
        )

        # Load horizontal plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        self.localInertiaPos = p.getDynamicsInfo(self.robotId, -1)[
            3
        ]  # of the base link

        # leg_left (45-50), leg_right (52-57), torso (0-1), arm_left (11-17),
        # gripper_left (21), arm_right (28-34), gripper_right (38), head (3,4).
        self.bulletJointNames = [
            p.getJointInfo(self.robotId, i)[1].decode()
            for i in range(p.getNumJoints(self.robotId))
        ]
        self.JointIndicesComplete = [
            self.bulletJointNames.index(rmodelComplete.names[i])
            for i in range(2, rmodelComplete.njoints)
        ]

        # Joints controlled with crocoddyl
        self.bulletControlledJoints = [
            i
            for i in self.JointIndicesComplete
            if p.getJointInfo(self.robotId, i)[1].decode() not in conf.blocked_joints
        ]

        # Disable default position controler in torque controlled joints
        # Default controller will take care of other joints
        p.setJointMotorControlArray(
            self.robotId,
            jointIndices=self.bulletControlledJoints,
            controlMode=p.VELOCITY_CONTROL,
            forces=[0.0 for m in self.bulletControlledJoints],
        )
        self.lines = []
        self.pinLines = []

        # Augment friction to forbid feet sliding
        p.changeDynamics(self.robotId, 50, lateralFriction=100, spinningFriction=30)
        p.changeDynamics(self.robotId, 57, lateralFriction=100, spinningFriction=30)

    def initializeJoints(self, q0CompleteStart):
        # Initialize position in pyBullet
        p.resetBasePositionAndOrientation(
            self.robotId,
            posObj=[
                q0CompleteStart[0] + self.localInertiaPos[0],
                q0CompleteStart[1] + self.localInertiaPos[1],
                q0CompleteStart[2] + self.localInertiaPos[2],
            ],
            ornObj=q0CompleteStart[3:7],
        )
        initial_joint_positions = np.array(q0CompleteStart[7:].flat).tolist()
        for i in range(len(initial_joint_positions)):
            p.enableJointForceTorqueSensor(self.robotId, i, True)
            p.resetJointState(
                self.robotId, self.JointIndicesComplete[i], initial_joint_positions[i]
            )

    def resetState(self, q0Start):
        # Initialize position in pyBullet
        p.resetBasePositionAndOrientation(
            self.robotId,
            posObj=[
                q0Start[0] + self.localInertiaPos[0],
                q0Start[1] + self.localInertiaPos[1],
                q0Start[2] + self.localInertiaPos[2],
            ],
            ornObj=q0Start[3:7],
        )
        for i in range(len(self.bulletControlledJoints)):
            p.resetJointState(
                self.robotId, self.bulletControlledJoints[i], q0Start[i + 7]
            )
    
    def resetReducedState(self, q0Start):
        for i in range(len(self.bulletControlledJoints)):
            p.resetJointState(
                self.robotId, self.bulletControlledJoints[i], q0Start[i]
            )
            
    def addStairs(self, path, position, orientation):
        p.setAdditionalSearchPath(path)
        self.stepId = p.loadURDF("step/step.urdf")
        p.resetBasePositionAndOrientation(
            self.stepId, posObj=position, ornObj=orientation
        )

    def execute(self, torques):
        p.setJointMotorControlArray(
            self.robotId,
            self.bulletControlledJoints,
            controlMode=p.TORQUE_CONTROL,
            forces=torques,
        )
        p.stepSimulation()
        
    def changeCamera(self,cameraDistance,cameraYaw,cameraPitch,cameraTargetPos):
        p.resetDebugVisualizerCamera(cameraDistance,
                                     cameraYaw,
                                     cameraPitch,
                                     cameraTargetPos)

    def measureState(self):
        jointStates = p.getJointStates(
            self.robotId, self.JointIndicesComplete
        )  # State of all joints
        baseState = p.getBasePositionAndOrientation(self.robotId)
        baseVel = p.getBaseVelocity(self.robotId)

        # Joint vector for Pinocchio
        q = np.hstack(
            [
                baseState[0],
                baseState[1],
                [jointStates[i_joint][0] for i_joint in range(len(jointStates))],
            ]
        )
        v = np.hstack(
            [
                baseVel[0],
                baseVel[1],
                [jointStates[i_joint][1] for i_joint in range(len(jointStates))],
            ]
        )
        rotation = R.from_quat(q[3:7])
        q[:3] -= rotation.as_matrix() @ self.localInertiaPos
        return q, v
    
    def addTable(self, path, position):
        p.setAdditionalSearchPath(path)
        self.tableId = p.loadURDF("table/table.urdf")
        p.resetBasePositionAndOrientation(self.tableId, posObj=position,ornObj=[0,0,0,1])

    def showSlope(self, position, orientation):
        visualShapeTarget = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[2, 0.3, 0.01],
            rgbaColor=[0.0, 1.0, 0.0, 1.0],
            specularColor=[0.4, 0.4, 0],
            visualFramePosition=[0.0, 0.0, 0.0],
        )

        self.sphereIdRight = p.createMultiBody(
            baseMass=0.0,
            baseInertialFramePosition=[0, 0, 0],
            baseVisualShapeIndex=visualShapeTarget,
            basePosition=position,
            baseOrientation=orientation,
            useMaximalCoordinates=True,
        )
        
    def display_contact_forces(self,pinocchio_force):

        # Info about contact points with the ground
        #contactPoints_FL = p.getContactPoints(self.robotId, self.planeId, linkIndexA=50)  # Left  foot
        #contactPoints_FR = p.getContactPoints(self.robotId, self.planeId, linkIndexA=57)  # Right foot
        contactPoints_hand = p.getContactPoints(self.robotId, self.tableId, linkIndexA=32)  # Hand

        # Display debug lines for contact forces visualization
        i_line = 0
        f_tmp = [0.0] * 3
        for contact in contactPoints_hand:
            if not isinstance(contact, int):  # type(contact) != type(0):
                start = [contact[6][0], contact[6][1], contact[6][2]+0.04]
                end = [contact[6][0], contact[6][1], contact[6][2]+0.04]
                end_pin = [contact[6][0], contact[6][1], contact[6][2]+0.04]
                K = 0.01
                for i_direction in range(0, 3):
                    f_tmp[i_direction] = (contact[9] * contact[7][i_direction] + contact[10] *
                                          contact[11][i_direction] + contact[12] * contact[13][i_direction])
                    end[i_direction] += K * f_tmp[i_direction]
                    end_pin[i_direction] += K * pinocchio_force[i_direction] 

                if (i_line+1) > len(self.lines):  # If not enough existing lines in line storage a new item is created
                    lineID = p.addUserDebugLine(start, end, lineColorRGB=[1.0, 0.0, 0.0], lineWidth=8)
                    pinLinId = p.addUserDebugLine(start, end_pin, lineColorRGB=[0.0, 1.0, 0.0], lineWidth=8)
                    self.lines.append(lineID)
                    self.pinLines.append(pinLinId)
                else:  # If there is already an existing line item we modify it (to avoid flickering)
                    self.lines[i_line] = p.addUserDebugLine(start, end, lineColorRGB=[
                        1.0, 0.0, 0.0], lineWidth=8, replaceItemUniqueId=self.lines[i_line])
                    self.pinLines[i_line] = p.addUserDebugLine(start, end_pin, lineColorRGB=[
                        0.0, 1.0, 0.0], lineWidth=8, replaceItemUniqueId=self.pinLines[i_line])
                i_line += 1

        # Should be around 21,5 (2.2 kg * 9.81 m^2/s)
        # print("Total ground reaction force: (", f_x, ", ", f_y, ", ", f_z, ")")

        for i_zero in range(i_line, len(self.lines)):
            self.lines[i_zero] = p.addUserDebugLine([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], lineColorRGB=[
                1.0, 0.0, 0.0], lineWidth=8, replaceItemUniqueId=self.lines[i_zero])
            self.pinLines[i_zero] = p.addUserDebugLine([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], lineColorRGB=[
                1.0, 0.0, 0.0], lineWidth=8, replaceItemUniqueId=self.pinLines[i_zero])
        
    def showHandToTrack(self, target_pose, rgba = [0,0,1,1]):
        visualShapeTarget = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.05, 0.05, 0.05],
            rgbaColor=rgba,
            specularColor=[0.4, 0.4, 0],
            visualFramePosition=[0.0, 0.0, 0.0],
        )

        self.sphereIdHand = p.createMultiBody(
            baseMass=0.0,
            baseInertialFramePosition=[0, 0, 0],
            baseVisualShapeIndex=visualShapeTarget,
            basePosition=[
                target_pose[0],
                target_pose[1],
                target_pose[2],
            ],
            useMaximalCoordinates=True,
        )
    
    def showObstacle(self, obstaclePos, obstacleHeight, obstacleRadius):
        ### Obstacle capsule
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_CAPSULE,
            rgbaColor=[1,1,0,1],
            visualFramePosition=[0.0, 0.0, 0.0],
            radius=obstacleRadius, 
            length=obstacleHeight,
            halfExtents = [0.0, 0.0, 0.0])
        obstacleId = p.createMultiBody(baseMass=0.0,
            baseInertialFramePosition=[0, 0, 0],
		    baseVisualShapeIndex=visualShapeId,
		    basePosition=obstaclePos,
		    useMaximalCoordinates=True)
		
        p.resetBasePositionAndOrientation(obstacleId, 
		    posObj=[obstaclePos[0],obstaclePos[1],obstaclePos[2]], 
		    ornObj=np.array([0,0,0,1]))
		
    def showTargetToTrack(self, LF_pose, RF_pose):
        visualShapeTarget = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.1, 0.075, 0.001],
            rgbaColor=[0.0, 0.0, 1.0, 1.0],
            specularColor=[0.4, 0.4, 0],
            visualFramePosition=[0.0, 0.0, 0.0],
        )

        self.sphereIdRight = p.createMultiBody(
            baseMass=0.0,
            baseInertialFramePosition=[0, 0, 0],
            baseVisualShapeIndex=visualShapeTarget,
            basePosition=[
                RF_pose.translation[0],
                RF_pose.translation[1],
                RF_pose.translation[2],
            ],
            useMaximalCoordinates=True,
        )

        self.sphereIdLeft = p.createMultiBody(
            baseMass=0.0,
            baseInertialFramePosition=[0, 0, 0],
            baseVisualShapeIndex=visualShapeTarget,
            basePosition=[
                LF_pose.translation[0],
                LF_pose.translation[1],
                LF_pose.translation[2],
            ],
            useMaximalCoordinates=True,
        )
    
    def set_capsule(self, pos):
        #move the capsule according to current robot's position
        base_pos = np.array(p.getLinkState(self.robotId,self.linkArmId)[0])
        base_inertia_pos = np.array(p.getDynamicsInfo(self.robotId,self.linkArmId)[3])
        base_ori = np.array(p.getLinkState(self.robotId,self.linkArmId)[1])
        
        R_base_ori = np.array(p.getMatrixFromQuaternion(base_ori)).reshape(3,-3)
        true_pos = base_pos + np.dot(R_base_ori, pos-base_inertia_pos)

        p.resetBasePositionAndOrientation(self.capsuleArmId, true_pos, base_ori)
    
    def init_capsule(self,joint_name,pos):
        RADIUS, LENGTH  = 0.09, 0.45
        visualShapeIdcap = p.createVisualShape(shapeType=p.GEOM_CAPSULE,
                                            rgbaColor=[0,0,1,.5],
                                            visualFramePosition=[0.0, 0.0, 0.0], 
                                            radius=RADIUS, 
                                            length=LENGTH,
                                            halfExtents = [0.0, 0.0, 0.0])
        self.capsuleArmId = p.createMultiBody(baseMass=0.0,
                            baseInertialFramePosition=[0, 0, 0],
                            baseVisualShapeIndex=visualShapeIdcap,
                            basePosition=[0.0, 0.0, 0.0],
                            useMaximalCoordinates=True)


        ### Set capsule on robot arm
        self.linkArmId = self.bulletJointNames.index(joint_name)
        self.set_capsule(pos)

    def moveMarkers(self, LF_pose, RF_pose):

        p.resetBasePositionAndOrientation(
            self.sphereIdRight,
            posObj=[
                RF_pose.translation[0],
                RF_pose.translation[1],
                RF_pose.translation[2],
            ],
            ornObj=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        p.resetBasePositionAndOrientation(
            self.sphereIdLeft,
            posObj=[
                LF_pose.translation[0],
                LF_pose.translation[1],
                LF_pose.translation[2],
            ],
            ornObj=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        
    def visualizeTrajectory(self, qs, dt=0.1):
        for q in qs:
            self.resetState(q)
            time.sleep(dt)

    def close(self):
        p.disconnect()


if __name__ == "__main__":

    import configuration as config
    from pin_Talos import PinTalos

    design = PinTalos(config)

    o = BulletTalos(config, design.rmodelComplete)
    o.initializeJoints(design.q0Complete)

    o.close()
