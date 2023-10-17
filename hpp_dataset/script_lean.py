from math import sqrt
import numpy as np
from hpp import Transform
from hpp.corbaserver import loadServerPlugin
from hpp.gepetto import PathPlayer
from hpp.corbaserver.manipulation import Constraints, ConstraintGraph, \
    ProblemSolver, newProblem
from hpp.corbaserver.manipulation.robot import HumanoidRobot
from hpp.gepetto.manipulation import ViewerFactory
from agimus_demos.tools_hpp import concatenatePaths
from agimus_demos.talos.tools_hpp import shootRandomArmConfig

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

class Table(object):
    contacts = ["top"]
    handles = []
    rootJointType = "freeflyer"
    urdfFilename = "package://gerard_bauzil/urdf/table_140_70_73.urdf"
    srdfFilename = "package://gerard_bauzil/srdf/table_140_70_73.srdf"
    def __init__(self, name, vf):
        self.name = name
        vf.loadObjectModel(self.__class__, name)
        self.handles = [name + "/" + h for h in self.__class__.handles]
        self.contacts = [name + "/" + h for h in self.__class__.contacts]

def shrinkJointRange (robot, ratio):
    for j in robot.jointNames:
        if j [:6] == "talos/" and j [:13] != "talos/gripper":
            bounds = robot.getJointBounds (j)
            if len (bounds) == 2:
                width = bounds [1] - bounds [0]
                mean = .5 * (bounds [1] + bounds [0])
                m = mean - .5 * ratio * width
                M = mean + .5 * ratio * width
                robot.setJointBounds (j, [m, M])

newProblem()

# Load robot via URDF

print("reading generic URDF")
HumanoidRobot.urdfFilename = "package://talos_data/urdf/pyrene.urdf"
HumanoidRobot.srdfFilename = "package://talos_data/srdf/pyrene.srdf"

objects = list()
robot = HumanoidRobot("talos", "talos", rootJointType="freeflyer", client=None)
robot.insertRobotSRDFModel("talos", "package://agimus_demos/srdf/contact_surface_on_the_arms.srdf")
robot.leftAnkle = "talos/leg_left_6_joint"
robot.rightAnkle = "talos/leg_right_6_joint"

robot.setJointBounds("talos/root_joint", [-1, 1, -1, 1, 0.5, 1.5])
shrinkJointRange (robot, 0.95)

# Create Problem Solver object to solve for path
ps = ProblemSolver(robot)
ps.selectPathProjector("Progressive", 0.2)
ps.setErrorThreshold(1e-3)
ps.setMaxIterProjection(40)

vf = ViewerFactory(ps)

table = Table(name="table", vf=vf)
box   = Box  (name="box",   vf=vf)
objects.append(box)

robot.setJointBounds("table/root_joint", [-2, 2, -2, 2, -2, 2])
robot.setJointBounds("box/root_joint"  , [-2, 2, -2, 2, -2, 2])

initConf = [
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
# Pose of the table
initConf += [1.,0.,0,0,0,0,1]
# Pose of the box
initConf += [1.5,0,0.8325,0,sqrt(2)/2,0,sqrt(2)/2]

# Create Constraint graph to define motion
graph = ConstraintGraph(robot, "graph")

# Create constraints
ps.createQPStabilityConstraint('contact_on_left_arm/force', 'talos/root_joint',
        ['talos/left_arm','talos/left_sole','talos/right_sole'])
ps.createQPStabilityConstraint('stand_on_feet/force', 'talos/root_joint',
    ['talos/left_sole','talos/right_sole'])

# Fix table and box position
nameTable = table.name + "/root_joint"
s = ps.robot.getJointConfigSize(nameTable)
r = ps.robot.rankInConfiguration[nameTable]
tableLockedJoint = [nameTable]
ps.createLockedJoint(nameTable, nameTable, initConf[r : r + s])
ps.setConstantRightHandSide(nameTable, False)

boxPlace, boxPrePlace = ps.createPlacementConstraints\
    ('place_box', ['box/front_surface'], ['table/top'], width=.15)
graph.createPreGrasp('pregrasp_box', 'talos/right_gripper', 'box/handle3')

# Lock non controlled joints
joints_locked = []
talosLockedJoint = ['talos/arm_left_5_joint',
                    'talos/arm_left_6_joint',
                    #'talos/arm_left_7_joint',
                    'talos/gripper_left_joint',
                    'talos/arm_right_5_joint',
                    'talos/arm_right_6_joint',
                    'talos/arm_right_7_joint',
                    'talos/gripper_right_joint',
                    'talos/head_1_joint',
                    'talos/head_2_joint'
]

for n in talosLockedJoint:
    s = robot.getJointConfigSize(n)
    r = robot.rankInConfiguration[n]
    ps.createLockedJoint(n, n, initConf[r : r + s])
    ps.setConstantRightHandSide(n, True)
    joints_locked.append(n)

leftWristLock = 'talos/arm_left_7_joint'
r = ps.robot.rankInConfiguration[leftWristLock]
ps.createLockedJoint(leftWristLock, leftWristLock, [-0.6])
ps.setConstantRightHandSide(leftWristLock, True)

# Placement constraints of foot, com and hands
ps.addPartialCom("talos", ["talos/root_joint"])
robot.createStaticStabilityConstraint(
    "balance/", "talos", robot.leftAnkle, robot.rightAnkle, initConf,
    (True, True, False)
)
comRelativeConstraint = ["balance/relative-com",]
footPlacement = ["balance/pose-left-foot", "balance/pose-right-foot"]
footPlacementComplement = []

leftPlace, leftPrePlace = ps.createPlacementConstraints\
    ('contact_on_left_arm/pose', ['talos/left_arm'], ['table/top'],
        width=.1)

ps.createTransformationConstraint(
    'hand_pose',
    'universe',
    'talos/gripper_right_joint',
    [0.6, -0.3, 1., 1.0, 0.0, 0.0, 0.0], # Position of the target to reach with right hand
    [True, True, True, False, False, False],
)

# Default CoM at init is array([ 0.00773126, -0.08618008,  0.79385158])
com_la = np.array([ 0.10773126, -0.08618008,  0.69385158])
ps.createRelativeComConstraint(
    "com_talos", "talos", robot.leftAnkle, com_la.tolist(), (True, True, True))


# Constraints that are applied to all states
commonStateConstraints = Constraints(numConstraints = footPlacement + ["com_talos"] +\
                                        talosLockedJoint +\
                                        [boxPlace,])
# Constraints that are applied to all transitions
commonTransitionConstraints = Constraints(numConstraints = \
    footPlacementComplement + tableLockedJoint + ['place_box/complement'])

# Create states
graph.createNode(['reach', 'lean_on_left_arm', 'free'])

# Create transitions
graph.createEdge('free', 'lean_on_left_arm', 'go_to_left_contact', 1,
                    isInNode='free')
graph.createEdge('lean_on_left_arm', 'free', 'release_left_contact', 1,
                    isInNode='free')
graph.createEdge('lean_on_left_arm', 'reach', 'to_reach', 1,
                    isInNode='lean_on_left_arm')
graph.createEdge('reach', 'lean_on_left_arm', 'from_reach', 1,
                    isInNode='lean_on_left_arm')

graph.createEdge('free', 'free', 'Loop | f', 1, isInNode='free')
graph.createEdge('lean_on_left_arm', 'lean_on_left_arm',
                    'Loop | contact_on_left_arm', 1,
                    isInNode='lean_on_left_arm')

# Add force constraint at lower priority level
# after initialization since the solvers need to be created
cgraph = ps.client.basic.problem.getProblem().getConstraintGraph()
nc = ps.client.basic.problem.getConstraint('stand_on_feet/force')
state = cgraph.get(graph.nodes['free'])
state.addNumericalCost(nc)

nc = ps.client.basic.problem.getConstraint('contact_on_left_arm/force')
state = cgraph.get(graph.nodes['lean_on_left_arm'])
state.addNumericalCost(nc)
state.setSolveLevelByLevel(True)
for eid in graph.edges.values():
    edge = cgraph.get(eid)
    edge.setSolveLevelByLevel(True) 

# Set constraints in states and transitions
graph.addConstraints(node='free', constraints = commonStateConstraints +
                     Constraints(numConstraints = ['stand_on_feet/force']))
graph.addConstraints(node='lean_on_left_arm',
    constraints = commonStateConstraints + Constraints(numConstraints = \
    [leftPlace, leftWristLock, 'contact_on_left_arm/force'])) #'contact_on_left_arm/force' 'hand_pose'

graph.addConstraints(
    node='reach',
    constraints = commonStateConstraints + \
    Constraints(numConstraints = [leftPlace, 'contact_on_left_arm/force',
    leftWristLock,'hand_pose']))

graph.addConstraints(edge='go_to_left_contact', constraints =\
                    commonTransitionConstraints)
graph.addConstraints(edge='Loop | f', constraints =\
                        commonTransitionConstraints)
graph.addConstraints(edge='release_left_contact', constraints =\
                    commonTransitionConstraints)

graph.addConstraints(
    edge='to_reach', constraints = Constraints(
        numConstraints=['contact_on_left_arm/pose/complement',]) +
        commonTransitionConstraints)
graph.addConstraints(
    edge='from_reach', constraints = Constraints(
        numConstraints=['contact_on_left_arm/pose/complement',]) +
        commonTransitionConstraints)
graph.addConstraints(
    edge='Loop | contact_on_left_arm', constraints = Constraints(
        numConstraints=['contact_on_left_arm/pose/complement',]) +
        commonTransitionConstraints)
graph.initialize ()

finished = False
v = vf.createViewer()
i = 0
while not finished:
    i += 1
    print(f"Try number {i}")
    q = robot.shootRandomConfig()
    res, q1, err = graph.generateTargetConfig("go_to_left_contact",
                                              initConf, q)
    v(q1)
    if not res: continue
    res, msg = robot.isConfigValid(q1)
    if not res: continue
    print("Found q1.")
    finished = True

# Find path between initial and contact config

from agimus_demos import InStatePlanner
planner = InStatePlanner(ps, graph)
planner.maxIterPathPlanning = 500

planner.setEdge('Loop | f')
p0 = None
try:
    p0 = planner.computePath(initConf, [q1])
except:
    print("Failed to compute path between {} and {}".format(initConf,q1))
    print("number of nodes: {}".format(planner.croadmap.getNbNodes()))

if p0:
    ps.client.basic.problem.addPath(p0)

v (initConf)
pp = PathPlayer (v)
pp(0)

""" finished = False
i = 0
while not finished:
    i += 1
    print(f"Try number {i}")
    q = shootRandomArmConfig(robot, 'right', q1)
    res, q2, err = graph.generateTargetConfig("to_reach", q1, q)
    v(q2)
    if not res: continue
    res, msg = robot.isConfigValid(q2)
    if not res: continue
    print("Found q2")
    finished = res

p1 = None
try:
    p1 = planner.computePath(q1, [q2])
except:
    print("Failed to compute path between {} and {}".format(q1,q2))
    print("number of nodes: {}".format(planner.croadmap.getNbNodes()))
if p1:
    ps.client.basic.problem.addPath(concatenatePaths([p0, p1]))
 """