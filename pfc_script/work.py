#!/usr/bin/env python3
"""Hard work here. Not efficient, but hard."""

import numpy as np
from hpp import Transform
from hpp.corbaserver.manipulation import ConstraintGraph, ProblemSolver, newProblem, Constraints
from hpp.gepetto.manipulation import ViewerFactory
from hpp.corbaserver.manipulation.robot import HumanoidRobot

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

class Problem:
    def __init__(self):
        newProblem()

        # Import the Talos robot

        print("reading generic URDF")
        HumanoidRobot.urdfFilename = "package://talos_data/urdf/pyrene.urdf"
        HumanoidRobot.srdfFilename = "package://talos_data/srdf/pyrene.srdf"

        objects = list()
        self.robot = HumanoidRobot("talos", "talos", rootJointType="freeflyer", client=None)
        self.robot.leftAnkle = "talos/leg_left_6_joint"
        self.robot.rightAnkle = "talos/leg_right_6_joint"

        self.robot.setJointBounds("talos/root_joint", [-1, 1, -1, 1, 0.5, 1.5])

        # Create path planning solver

        self.ps = ProblemSolver(self.robot)
        self.ps.selectPathProjector("Progressive", 0.2)
        self.ps.setErrorThreshold(1e-3)
        self.ps.setMaxIterProjection(40)

        # Create viewer object and obstacle boxes

        self.vf = ViewerFactory(self.ps)
        box = Box(name="box", vf=self.vf)
        box1 = Box(name="box1", vf=self.vf)
        box2 = Box(name="box2", vf=self.vf)
        box3 = Box(name="box3", vf=self.vf)
        self.robot.setJointBounds("box/root_joint"  , [-2, 2, -2, 2, -2, 2])
        self.robot.setJointBounds("box1/root_joint"  , [-2, 2, -2, 2, -2, 2])
        self.robot.setJointBounds("box2/root_joint"  , [-2, 2, -2, 2, -2, 2])
        self.robot.setJointBounds("box3/root_joint"  , [-2, 2, -2, 2, -2, 2])

        # Initial robot pose

        self.half_sitting = [
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
        self.ps.createLockedJoint("locked_box","box/root_joint",[xBox,yBox,zBox - 0.25,0, np.sqrt(2)/2,0,np.sqrt(2)/2])
        self.ps.setConstantRightHandSide("locked_box", True)

        self.ps.createLockedJoint("locked_box1","box1/root_joint",[xBox,yBox,zBox,0, np.sqrt(2)/2,0,np.sqrt(2)/2])
        self.ps.setConstantRightHandSide("locked_box1", True)

        self.ps.createLockedJoint("locked_box2","box2/root_joint",[xBox,yBox,zBox + 0.25,0, np.sqrt(2)/2,0,np.sqrt(2)/2])
        self.ps.setConstantRightHandSide("locked_box2", True)

        self.ps.createLockedJoint("locked_box3","box3/root_joint",[xBox,yBox,zBox + 0.5,0, np.sqrt(2)/2,0,np.sqrt(2)/2])
        self.ps.setConstantRightHandSide("locked_box3", True)

        # Initial position for boxes must be the same as constrained position
        self.half_sitting += [xBox,yBox,0.75,0,np.sqrt(2)/2,0,np.sqrt(2)/2]  
        self.half_sitting += [xBox,yBox,1,0,np.sqrt(2)/2,0,np.sqrt(2)/2]  
        self.half_sitting += [xBox,yBox,1.25,0,np.sqrt(2)/2,0,np.sqrt(2)/2]  
        self.half_sitting += [xBox,yBox,1.5,0,np.sqrt(2)/2,0,np.sqrt(2)/2]  

        self.robot.setCurrentConfig(self.half_sitting)

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

        self.ids_controlled = []

        for n in talosLockedJoint:
            s = self.robot.getJointConfigSize(n)
            r = self.robot.rankInConfiguration[n]
            self.ps.createLockedJoint(n, n, self.half_sitting[r : r + s])
            self.ps.setConstantRightHandSide(n, True)
            joints_locked.append(n)

        for n in talosControlledJoint:
            r = self.robot.rankInConfiguration[n]
            self.ids_controlled.append(r)

        # Static stability constraint
        # CoM constraint is created but not used in reduced setup
        self.ps.addPartialCom("talos", ["talos/root_joint"])
        self.robot.createStaticStabilityConstraint(
            "balance/", "talos", self.robot.leftAnkle, self.robot.rightAnkle, self.half_sitting,
            (True, True, False)
        )
        com_constraint = ["balance/relative-com",]
        footPlacement = ["balance/pose-left-foot", "balance/pose-right-foot"]
        footPlacementComplement = []

        self.robot.setCurrentConfig(self.half_sitting)
        com_wf = np.array(self.robot.getCenterOfMass())
        tf_la = Transform(self.robot.getJointPosition(self.robot.leftAnkle))
        com_la = tf_la.inverse().transform(com_wf)

        self.ps.createRelativeComConstraint(
            "com_talos", "talos", self.robot.leftAnkle, com_la.tolist(), (True, True, True)
        )

        self.commonStateConstraints = Constraints(numConstraints = footPlacement + # ['com_talos'] +\ 
                                            joints_locked)
        self.commonTransitionConstraints = Constraints(numConstraints = footPlacementComplement + 
                                                ['locked_box', 'locked_box1', 'locked_box2', 'locked_box3'])
        
        self.ps.setParameter("SimpleTimeParameterization/safety", 0.5)
        self.ps.setParameter("SimpleTimeParameterization/order", 2)
        self.ps.setParameter("SimpleTimeParameterization/maxAcceleration", 1.0)

        #self.ps.addPathOptimizer("SimpleTimeParameterization")
        self.ps.addPathOptimizer ("EnforceTransitionSemantic")
        self.ps.addPathOptimizer ("RandomShortcut")
        
        # Data container and counting
        self.ceiling = 1000

    def createAndSolveGraph(self, targetSample, n):
        self.ps.createTransformationConstraint(
            'hand_pose',
            'universe',
            'talos/gripper_right_joint',
            targetSample + [1.0, 0.0, 0.0, 0.0], # Position of the target to reach with right hand
            [True, True, True, False, False, False],
        )  
        # Create graph of constraint
        graph = ConstraintGraph(self.robot, "graph" + str(n))
        # Create nodes and edges of the graph
        graph.createNode(['reach', 'free'])

        graph.createEdge('free', 'reach', 'to_reach', 1,
                            isInNode='free')
        graph.createEdge('reach', 'free', 'to_free', 1,
                            isInNode='reach')
        graph.createEdge('free', 'free', 'Loop | f', 1, isInNode='free')

        # Set constraints in states and transitions
        graph.addConstraints(node='free', constraints = self.commonStateConstraints)
        graph.addConstraints(node='reach', 
                            constraints = self.commonStateConstraints + \
                            Constraints(numConstraints = ['hand_pose']))

        graph.addConstraints(edge='to_reach', 
                            constraints=self.commonTransitionConstraints)
        graph.addConstraints(edge='to_free', 
                            constraints=self.commonTransitionConstraints)
        graph.addConstraints(edge='Loop | f', 
                            constraints=self.commonTransitionConstraints)

        graph.initialize ()

        # Sample robot initial configuration
        q0, res0 = self.sampleAndProject(graph, "Loop | f")
        if not(res0): return None, None, None

        q1, res1 = self.sampleAndProject(graph, "to_reach")
        if not(res1): return None, None, None
        
        # Find path between q0 and q1
        self.ps.resetGoalConfigs()
        self.ps.clearRoadmap()

        self.ps.setInitialConfig (q0)
        self.ps.addGoalConfig (q1)

        print (self.ps.solve ()) 

        tn = self.ps.numberPaths()
        if tn < 3: return None
        
        trajs = self.ps.getWaypoints(2)[0]
        timing = self.ps.getWaypoints(2)[1]

        wps = []
        for traj in trajs:
            traj = np.array(traj)
            wps.append(traj[self.ids_controlled])

        del graph
        for j in range(tn):
            self.ps.erasePath(0)
        
        return wps, timing, targetSample
    
    def sampleAndProject(self, graph, edge_name):
        i = 0
        finished = False
        qs = None
        while not finished and i < self.ceiling:
            i += 1
            q = self.robot.shootRandomConfig() 
            res, qs, err = graph.generateTargetConfig(edge_name,
                                                    self.half_sitting, q)
            if not res: continue
            res, msg = self.robot.isConfigValid(qs)
            if not res: continue
            finished = res
        return qs, finished

if __name__ == "__main__":
    problem = Problem()
