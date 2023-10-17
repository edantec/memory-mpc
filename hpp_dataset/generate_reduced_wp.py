# /
import numpy as np
from hpp import Transform
from hpp.gepetto import PathPlayer
from hpp.corbaserver.manipulation import ConstraintGraph, ProblemSolver, newProblem, Constraints
from hpp.gepetto.manipulation import ViewerFactory
from hpp.corbaserver.manipulation.robot import HumanoidRobot
import pickle

class Box(object):
    handles = ["handle1", "handle2", "handle3", "handle4"]
    contacts = ["front_surface", "rear_surface"]
    rootJointType = "freeflyer"
    urdfFilename = "package://gerard_bauzil/urdf/plank_of_wood2.urdf"
    srdfFilename = "package://gerard_bauzil/srdf/plank_of_wood2.srdf"
    def __init__(self, name, vf):
        self.name = name
        vf.loadObjectModel(self.__class__, name)
        self.handles = [name + "/" + h for h in self.__class__.handles]
        self.contacts = [name + "/" + h for h in self.__class__.contacts]

newProblem()

# Import the Talos robot

print("reading generic URDF")
HumanoidRobot.urdfFilename = "package://talos_data/urdf/pyrene.urdf"
HumanoidRobot.srdfFilename = "package://talos_data/srdf/pyrene.srdf"

objects = list()
robot = HumanoidRobot("talos", "talos", rootJointType="freeflyer", client=None)
robot.insertRobotSRDFModel("talos", "package://agimus_demos/srdf/contact_surface_on_the_arms.srdf")
robot.leftAnkle = "talos/leg_left_6_joint"
robot.rightAnkle = "talos/leg_right_6_joint"

robot.setJointBounds("talos/root_joint", [-1, 1, -1, 1, 0.5, 1.5])

# Create path planning solver

ps = ProblemSolver(robot)
ps.selectPathProjector("Progressive", 0.2)
ps.setErrorThreshold(1e-3)
ps.setMaxIterProjection(40)

# Create viewer object and obstacle boxes

vf = ViewerFactory(ps)
v=vf.createViewer()
box = Box(name="box", vf=vf)
box1 = Box(name="box1", vf=vf)
box2 = Box(name="box2", vf=vf)
box3 = Box(name="box3", vf=vf)
robot.setJointBounds("box/root_joint"  , [-2, 2, -2, 2, -2, 2])
robot.setJointBounds("box1/root_joint"  , [-2, 2, -2, 2, -2, 2])
robot.setJointBounds("box2/root_joint"  , [-2, 2, -2, 2, -2, 2])
robot.setJointBounds("box3/root_joint"  , [-2, 2, -2, 2, -2, 2])

# Initial robot pose

half_sitting = [
    0,
    0,
    1.0192720229567027,
    0,
    0,
    0,
    1,  # root_joint
    0.0,
    0.0,
    -0.411354,
    0.859395,
    -0.448041,
    -0.001708,  # leg_left
    0.0,
    0.0,
    -0.411354,
    0.859395,
    -0.448041,
    -0.001708,  # leg_right
    0,
    0.006761,  # torso
    0.25847,
    0.173046,
    -0.0002,
    -0.525366,
    0,
    0,
    0.1,  # arm_left
    0,
    0,
    0,
    0,
    0,
    0,
    0,  # gripper_left
    -0.25847,
    -0.173046,
    0.0002,
    -0.525366,
    0,
    0,
    0.1,  # arm_right
    0,
    0,
    0,
    0,
    0,
    0,
    0,  # gripper_right
    0,
    0,  # head
]

# Box constraint
xBox = 0.5
yBox = 0
zBox  = 1
ps.createLockedJoint("locked_box","box/root_joint",[xBox,yBox,zBox - 0.25,0, np.sqrt(2)/2,0,np.sqrt(2)/2])
ps.setConstantRightHandSide("locked_box", True)

ps.createLockedJoint("locked_box1","box1/root_joint",[xBox,yBox,zBox,0, np.sqrt(2)/2,0,np.sqrt(2)/2])
ps.setConstantRightHandSide("locked_box1", True)

ps.createLockedJoint("locked_box2","box2/root_joint",[xBox,yBox,zBox + 0.25,0, np.sqrt(2)/2,0,np.sqrt(2)/2])
ps.setConstantRightHandSide("locked_box2", True)

ps.createLockedJoint("locked_box3","box3/root_joint",[xBox,yBox,zBox + 0.5,0, np.sqrt(2)/2,0,np.sqrt(2)/2])
ps.setConstantRightHandSide("locked_box3", True)

# Initial position for boxes must be the same as constrained position
half_sitting += [xBox,yBox,0.75,0,np.sqrt(2)/2,0,np.sqrt(2)/2]  
half_sitting += [xBox,yBox,1,0,np.sqrt(2)/2,0,np.sqrt(2)/2]  
half_sitting += [xBox,yBox,1.25,0,np.sqrt(2)/2,0,np.sqrt(2)/2]  
half_sitting += [xBox,yBox,1.5,0,np.sqrt(2)/2,0,np.sqrt(2)/2]  

q_init = half_sitting[:] #robot.getCurrentConfig()
robot.setCurrentConfig(half_sitting)


joints_locked = []
talosLockedJoint = ['talos/leg_left_1_joint',
                      'talos/leg_left_2_joint',
                      'talos/leg_left_3_joint',
                      'talos/leg_left_4_joint',
                      'talos/leg_left_5_joint',
                      'talos/leg_left_6_joint',
                      'talos/leg_right_1_joint',
                      'talos/leg_right_2_joint',
                      'talos/leg_right_3_joint',
                      'talos/leg_right_4_joint',
                      'talos/leg_right_5_joint',
                      'talos/leg_right_6_joint',
                      'talos/arm_left_1_joint',
                      'talos/arm_left_2_joint',
                      'talos/arm_left_3_joint',
                      'talos/arm_left_4_joint',
                      'talos/arm_left_5_joint',
                      'talos/arm_left_6_joint',
                      'talos/arm_left_7_joint',
                      'talos/gripper_left_joint',
                      'talos/arm_right_5_joint',
                      'talos/arm_right_6_joint',
                      'talos/arm_right_7_joint',
                      'talos/gripper_right_joint',
                      'talos/head_1_joint',
                      'talos/head_2_joint'
]

talosControlledJoint = ['talos/torso_1_joint',
                        'talos/torso_2_joint',
                        'talos/arm_right_1_joint',
                        'talos/arm_right_2_joint',
                        'talos/arm_right_3_joint',
                        'talos/arm_right_4_joint',                     
]

ids_controlled = []

for n in talosLockedJoint:
    s = robot.getJointConfigSize(n)
    r = robot.rankInConfiguration[n]
    ps.createLockedJoint(n, n, half_sitting[r : r + s])
    ps.setConstantRightHandSide(n, True)
    joints_locked.append(n)

for n in talosControlledJoint:
    r = robot.rankInConfiguration[n]
    ids_controlled.append(r)

# Static stability constraint
# CoM constraint is created but not used in reduced setup
ps.addPartialCom("talos", ["talos/root_joint"])
robot.createStaticStabilityConstraint(
     "balance/", "talos", robot.leftAnkle, robot.rightAnkle, half_sitting,
     (True, True, False)
)
com_constraint = ["balance/relative-com",]
footPlacement = ["balance/pose-left-foot", "balance/pose-right-foot"]
footPlacementComplement = []

robot.setCurrentConfig(half_sitting)
com_wf = np.array(robot.getCenterOfMass())
tf_la = Transform(robot.getJointPosition(robot.leftAnkle))
com_la = tf_la.inverse().transform(com_wf)

ps.createRelativeComConstraint(
    "com_talos", "talos", robot.leftAnkle, com_la.tolist(), (True, True, True)
)

commonStateConstraints = Constraints(numConstraints = footPlacement + # ['com_talos'] +\ 
                                    joints_locked)
commonTransitionConstraints = Constraints(numConstraints = footPlacementComplement + 
                                          ['locked_box', 'locked_box1', 'locked_box2', 'locked_box3'])

# Sample a goal config not in collision
i = 0
wps_data = []
target_data = []
timings_data = []
numberTrial = 100
ceiling = 1000
# Sample target
def sampleRange(samples_range):
	samples_vector = []
	for sample_range in samples_range:
		samples_vector.append(np.random.uniform(sample_range[0],sample_range[1]))
	return samples_vector

x_goals = np.array([0.6,0.9])
y_goals = np.array([-0.5,0.5])
z_goals = np.array([0.8,1.2])
target_ranges = [x_goals, y_goals, z_goals]

ps.setParameter("SimpleTimeParameterization/safety", 0.5)
ps.setParameter("SimpleTimeParameterization/order", 2)
ps.setParameter("SimpleTimeParameterization/maxAcceleration", 1.0)

#ps.addPathOptimizer("SimpleTimeParameterization")
ps.addPathOptimizer ("EnforceTransitionSemantic")
ps.addPathOptimizer ("RandomShortcut")
pp = PathPlayer (v)

for t in range(numberTrial):
    targetSample = sampleRange(target_ranges)
    ps.createTransformationConstraint(
        'hand_pose',
        'universe',
        'talos/gripper_right_joint',
        targetSample + [1.0, 0.0, 0.0, 0.0], # Position of the target to reach with right hand
        [True, True, True, False, False, False],
    )  
    # Create graph of constraint
    graph = ConstraintGraph(robot, "graph" + str(t))
    print(f"Created graph{t}")
    # Create nodes and edges of the graph
    graph.createNode(['reach', 'free'])

    graph.createEdge('free', 'reach', 'to_reach', 1,
                        isInNode='free')
    graph.createEdge('reach', 'free', 'to_free', 1,
                        isInNode='reach')
    graph.createEdge('free', 'free', 'Loop | f', 1, isInNode='free')

    # Set constraints in states and transitions
    graph.addConstraints(node='free', constraints = commonStateConstraints)
    graph.addConstraints(node='reach', 
                        constraints = commonStateConstraints + \
                        Constraints(numConstraints = ['hand_pose']))

    graph.addConstraints(edge='to_reach', 
                        constraints=commonTransitionConstraints)
    graph.addConstraints(edge='to_free', 
                        constraints=commonTransitionConstraints)
    graph.addConstraints(edge='Loop | f', 
                        constraints=commonTransitionConstraints)

    graph.initialize ()
    q1 = q_init[:]
    
    # Sample robot configuration
    i = 0
    finished = False
    while not finished and i < ceiling:
        i += 1
        #print(f"test {i}")
        q = robot.shootRandomConfig() 
        res, q0, err = graph.generateTargetConfig("Loop | f",
                                                  q_init, q)
        if not res: continue
        res, msg = robot.isConfigValid(q0)
        if not res:
            #print("config not valid")
            continue
        print("Sample q0.")
        finished = res

    i = 0
    finished = False
    while not finished and i < ceiling:
        i += 1
        #print(f"test {i}")
        q = robot.shootRandomConfig() 
        res, q1, err = graph.generateTargetConfig("to_reach",
                                                  q_init, q)
        v(q1)
        if not res: continue
        res, msg = robot.isConfigValid(q1)
        if not res:
            #print("config not valid")
            continue
        print("Found q1.")
        finished = res
    
    if i == ceiling: continue
    print("Found good goal configuration")

    ps.resetGoalConfigs()
    ps.clearRoadmap()
    #ps.resetGoalConstraints()
    #ps.resetConstraintMap()
    #ps.resetConstraints()

    ps.setInitialConfig (q0)
    ps.addGoalConfig (q1)

    print (ps.solve ()) 

    tn = ps.numberPaths()
    if tn < 3:
        continue
    
    trajs = ps.getWaypoints(2)[0]
    timing = ps.getWaypoints(2)[1]

    print("Play path")

    v (q0)
    pp(2)

    wps = []
    for traj in trajs:
        traj = np.array(traj)
        wps.append(traj[ids_controlled])
    
    wps_data.append(np.array(wps))
    timings_data.append(np.array(timing))
    target_data.append(np.array(targetSample))

    del graph
    for j in range(tn):
         ps.erasePath(0)

data_dict = {}
data_dict['waypoints'] = wps_data
data_dict['timings'] = timings_data
data_dict['targets'] = target_data

with open('reduced_trajectories.pkl', 'wb') as f: 
	pickle.dump(data_dict, f)
print("Data stored in reduced_trajectories.pkl")