#!/usr/bin/env python3
import example_robot_data
import numpy as np

# PATHS

URDF_FILENAME = "talos_reduced.urdf"
SRDF_FILENAME = "talos.srdf"
SRDF_SUBPATH = "/talos_data/srdf/" + SRDF_FILENAME
URDF_SUBPATH = "/talos_data/robots/" + URDF_FILENAME
modelPath = example_robot_data.getModelPath(URDF_SUBPATH)

# Joint settings

blocked_joints = [
	"universe",
	"leg_left_1_joint",
	"leg_left_2_joint",
	"leg_left_3_joint",
	"leg_left_4_joint",
	"leg_left_5_joint",
	"leg_left_6_joint",
	"leg_right_1_joint",
	"leg_right_2_joint",
	"leg_right_3_joint",
	"leg_right_4_joint",
	"leg_right_5_joint",
	"leg_right_6_joint",
	#"torso_1_joint",
	#"torso_2_joint",
	"arm_left_1_joint",
	"arm_left_2_joint",
	"arm_left_3_joint",
	"arm_left_4_joint",
	"arm_left_5_joint",
	"arm_left_6_joint",
	"arm_left_7_joint",
	#"arm_right_1_joint",
	#"arm_right_2_joint",
	#"arm_right_3_joint",
	#"arm_right_4_joint",
	"arm_right_5_joint",
	"arm_right_6_joint",
	"arm_right_7_joint",
	"gripper_left_joint",
	"gripper_right_joint",
	"head_1_joint",
	"head_2_joint",
]

controlled_joints = [
			"torso_1_joint",
			"torso_2_joint",
			"arm_right_1_joint",
			"arm_right_2_joint",
			"arm_right_3_joint",
			"arm_right_4_joint",
			]

# #### TIMING #####
DT = 1e-2  # Time step of the DDP
T = 100  # Time horizon of the DDP (number of nodes)
TtrackingRHToContactRH = 60
TcontactRHToTrackingLH = 100
TtrackingLH = 400
TcontactRH = 150
simu_step = simu_period = 1e-3  #
ddpIteration = 1

Nc = int(round(DT / simu_step)) 

gravity = np.array([0, 0, -9.81])

mu = 0.3
footSize = 0.05
coneBox = np.array([0.1, 0.05])
minNforce = 200
maxNforce = 1200  # This may be still too low

normal_height = 0.87
omega = np.sqrt(-gravity[2] / normal_height)

# Obstacle

obstacleHeight = 2
obstaclePosition = np.array([0.5,0.0,1])
obstacleRadius = 0.1

# ##### CROCO - CONFIGURATION ########
# relevant frame names

rightFootName = "leg_right_sole_fix_joint"
leftFootName  = "leg_left_sole_fix_joint"
rightHandName = "arm_right_7_link"
leftHandName = "arm_left_7_link"

# Weights for all costs

wHandTranslation = 100
wHandRotation = 0
wHandVelocity = 10
wHandCollision = 10000
wStateReg = 0.1
wControlReg = 0.001
wLimit = 1e3
wWrenchCone = 0.05
wForceHand = 1
wCoM = 0
wDCM = 0

weightArmPos = [10,10, 10,10]  # [z, x, z, y, z, x, y]
weightArmVel = [10, 10, 10, 10]  # [z, x, z, y, z, x, y]
weightTorsoPos = [10,1000]  # [z, y]
weightTorsoVel = [10,0]  # [z, y]
stateWeights = np.array(
	weightTorsoPos
	+ weightArmPos
	+ weightTorsoVel
	+ weightArmVel
)

weightuArm = [1, 1, 1, 1]
weightuTorso = [1, 1]
controlWeight = np.array(weightuTorso 
						+ weightuArm
)
lowKinematicLimits = np.array([-1.3,-0.2, # torso
						   #-1.57,0.1,-2.44,-2.1  # left arm
                           -0.52,-2.88,-2.44,-2.1])  # right arm
highKinematicLimits = np.array([1.3,0.5,  # torso     
						   #0.52,2.88,2.44,0]) # left arm
                           1.57,-0.2,2.44,0]) # right arm      

th_stop = 1e-6  # threshold for stopping criterion
th_grad = 1e-9  # threshold for zero gradient.
simulator = "bullet"
