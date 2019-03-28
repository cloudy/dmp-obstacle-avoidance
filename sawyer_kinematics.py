#!/usr/bin/env python

# Updated Sawyer Robot Class by Michail Theofanidis

# import libraries
import numpy as np
import sympy as sp
import math
import rospy
import roslib
import tf
import geometry_msgs.msg
import intera_interface

from std_msgs.msg import Header
from sensor_msgs.msg import JointState

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header
from sensor_msgs.msg import JointState

from intera_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)


# Decleration of the Sawyer Robot Class
class Sawyer:
    def __init__(self):

        # Number of joints, links and offsets
        self.num_joints = 7
        self.num_links = 8
        self.num_offsets = 7

        # Arrays that store the link parameters and offset parameters
        self.L = np.array([0.0794732, 0.237, 0.142537, 0.259989, 0.126442, 0.274653, 0.105515, 0.0695])
        self.d = np.array([0.0814619, 0.0499419, 0.140042, 0.0419592, 0.1224936, 0.031188, 0.109824])

        # Joint Names array
        self.joint_names = ['right_j%i' % i for i in range(self.num_joints)]

        # Joint variable
        self.q = [sp.Symbol('q%i' % i) for i in range(self.num_joints)]

        # Homogeneous Transformation Matrices of the Robot
        self.T_R_01 = sp.Matrix([[sp.cos(self.q[0]), -sp.sin(self.q[0]), 0, 0],
                                 [sp.sin(self.q[0]), sp.cos(self.q[0]), 0, 0],
                                 [0, 0, 1, self.L[0]],
                                 [0, 0, 0, 1]])

        self.T_R_12 = sp.Matrix([[sp.cos(self.q[1]), -sp.sin(self.q[1]), 0, self.d[0]],
                                 [0, 0, 1, self.d[1]],
                                 [-sp.sin(self.q[1]), -sp.cos(self.q[1]), 0, self.L[1]],
                                 [0, 0, 0, 1]])

        self.T_R_23 = sp.Matrix([[sp.cos(self.q[2]), -sp.sin(self.q[2]), 0, 0],
                                 [0, 0, -1, -self.d[2]],
                                 [sp.sin(self.q[2]), sp.cos(self.q[2]), 0, self.L[2]],
                                 [0, 0, 0, 1]])

        self.T_R_34 = sp.Matrix([[sp.cos(self.q[3]), -sp.sin(self.q[3]), 0, 0],
                                 [0, 0, 1, -self.d[3]],
                                 [-sp.sin(self.q[3]), -sp.cos(self.q[3]), 0, self.L[3]],
                                 [0, 0, 0, 1]])

        self.T_R_45 = sp.Matrix([[sp.cos(self.q[4]), -sp.sin(self.q[4]), 0, 0],
                                 [0, 0, -1, -self.d[4]],
                                 [sp.sin(self.q[4]), sp.cos(self.q[4]), 0, -self.L[4]],
                                 [0, 0, 0, 1]])

        self.T_R_56 = sp.Matrix([[sp.cos(self.q[5]), -sp.sin(self.q[5]), 0, 0],
                                 [0, 0, 1, self.d[5]],
                                 [-sp.sin(self.q[5]), -sp.cos(self.q[5]), 0, self.L[5]],
                                 [0, 0, 0, 1]])

        self.T_R_67 = sp.Matrix([[sp.cos(self.q[6]), -sp.sin(self.q[6]), 0, 0],
                                 [0, 0, -1, -self.d[6]],
                                 [sp.sin(self.q[6]), sp.cos(self.q[6]), 0, self.L[6]],
                                 [0, 0, 0, 1]])

        self.T_R_89 = sp.Matrix([[0, -1, 0, 0],
                                 [1, 0, 0, 0],
                                 [0, 0, 1, 0.0245],
                                 [0, 0, 0, 1]])

        self.T_R_9e = sp.Matrix([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0.0450],
                                 [0, 0, 0, 1]])

        # Trasformation Tree of the Robot
        self.T_R = [self.T_R_01, self.T_R_12, self.T_R_23, self.T_R_34, self.T_R_45, self.T_R_56, self.T_R_67,
                    self.T_R_89, self.T_R_9e]

    # Forward Kinematics for the Robot
    def get_T_f(self):

        self.Tf_R = self.Forward_Kinematics(self.T_R)

        return self.Tf_R

    # Method to adjust the offset in joint space
    def JointOffset(self, angles):

        angles[1] = angles[1] + math.radians(90)
        angles[6] = angles[6] + math.radians(170)# + math.radians(90)

        return angles

    # Function that performs the Forward Kinematic Equations
    def Forward_Kinematics(self, trans):

        self.temp = [trans[0]]
        counter = -1;

        # Traverse through the transformation tree
        for i in trans[1:]:  #

            counter = counter + 1
            self.temp.append(self.temp[counter] * i)  #

        return self.temp
    
    
    # Function that return the IK solution given a position and orientation
    def Inverse_Kinematics(self, limb, coordinates, orientation, prev_solution = None):
        angles=limb.joint_angles()
        ns = "ExternalTools/right/PositionKinematicsNode/IKService"
        iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        ikreq = SolvePositionIKRequest()
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        quaternion = tf.transformations.quaternion_from_euler(*np.deg2rad(orientation))
        
        poses = {
                'right': PoseStamped(
                    header=hdr,
    		pose=Pose(
    		position=Point(
    		x=coordinates[0],
    		y=coordinates[1],
    		z=coordinates[2],
    		),
    		orientation=Quaternion(
    		x=quaternion[0],
    		y=quaternion[1],
    		z=quaternion[2],
    		w=quaternion[3],),
    		),),}
    
        ikreq.pose_stamp.append(poses['right'])
        ikreq.tip_names.append('right_hand')
        ikreq.seed_mode = ikreq.SEED_USER
        
        seed = JointState()
        seed.name = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']
        seed.position = prev_solution if prev_solution is not None else [angles[a] for a in seed.name]

        ikreq.seed_angles.append(seed)
        
        # Optimize the null space in terms of joint configuration
        #ikreq.use_nullspace_goal.append(True)
        #goal = JointState()
        #goal.name = ['right_j2']
        #goal.position = [0]
        #ikreq.nullspace_goal.append(goal)
        #ikreq.nullspace_gain.append(0.4)
        
        try:
        	rospy.wait_for_service(ns, 5.0)
        	resp = iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException) as e:
        	rospy.logerr("Service call failed: %s" % (e,))
        
        limb_joints = angles 
        # Check if result valid, and type of seed ultimately used to get solution
        if (resp.result_type[0] > 0):
        	seed_str = {ikreq.SEED_USER: 'User Provided Seed',
        		    ikreq.SEED_CURRENT: 'Current Joint Angles',
        		    ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
        		    }.get(resp.result_type[0], 'None')
        	limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
        else:
            print('Failed to find solution!')

        return limb_joints

