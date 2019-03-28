#!/usr/bin/env python2

#TODO: Seperate DMP for position and orientation

import argparse
from pyrdmp.dmp import DynamicMovementPrimitive as DMP
import pyrdmp.plots as plot
from pyrdmp.utils import *
import numpy as np
import sympy as sp
from sawyer_kinematics import Sawyer
import rospy
import intera_interface


def fix_angle(previous, current):
    k = np.round((current - previous)/np.pi)
    if previous >= 0.9*np.pi:
        if k > 0.5: return current + k*np.pi
        if k < -0.5: return current - k*np.pi
    if previous <= -0.9*np.pi: 
        if k > 0.5: return current - k*np.pi
        if k < -0.5: return current + k*np.pi
    return current


def main(arg):
    rospy.init_node("Sawyer_DMP")
    limb = intera_interface.Limb('right')
    
    # Initialize the DMP class
    my_dmp = DMP(arg.gain, arg.num_gaussians, arg.stabilization, arg.obstacle_avoidance)

    # Load the demo data
    data = load_demo(arg.input_file)

    # Obtain the joint position data and the time vector
    t, q = parse_demo(data)

    # Get the phase from the time vector
    s = my_dmp.phase(t)

    # Get the Gaussian
    psv = my_dmp.distributions(s)

    # Normalize the time vector
    t = normalize_vector(t)
    
    print('Calculating forward kinematics...')
    robot = Sawyer()
    Te = sp.lambdify(robot.q, robot.get_T_f()[-1])
    ht = np.array([Te(*robot.JointOffset(qi)) for qi in q])
    pose, orientation = np.zeros((len(t), 6)), np.zeros((len(t), 3))

    for i, h in enumerate(ht):
        c = h[:3,3]
        p = np.arctan2(-h[2,0], np.sqrt(h[0,0]**2 + h[1,0]**2))
        y = np.arctan2(h[1,0]/np.cos(p), h[0,0]/np.cos(p))
        r = np.arctan2(h[2,1]/np.cos(p), h[2,2]/np.cos(p))
        
        if i > 0:
            y = fix_angle(pose[i-1,5], y)
            p = fix_angle(pose[i-1,4], p)
            r = fix_angle(pose[i-1,3], r)

        pose[i] = np.concatenate((c,np.stack((r,p,y))))
    
    # Set static obstacles
    obstacles = np.array([pose[len(pose)//2]])#,:3]])
    
    print('Smoothing and filtering cartesian trajectory...')
    # Compute velocity and acceleration
    dpose, ddpose = np.zeros((2, pose.shape[0], pose.shape[1]))
    for i in range(pose.shape[1]):
        pose[:, i] = smooth_trajectory(pose[:, i], arg.window)
        dpose[:, i] = vel(pose[:, i], t)
        ddpose[:, i] = vel(dpose[:, i], t)

    # Filter the position velocity and acceleration signals
    f_pose, f_dpose, f_ddpose = np.zeros((3, pose.shape[0], pose.shape[1]))

    for i in range(pose.shape[1]):
        f_pose[:, i] = blend_trajectory(pose[:, i], dpose[:, i], t, arg.blends)
        f_dpose[:, i] = vel(f_pose[:, i], t)
        f_ddpose[:, i] = vel(f_dpose[:, i], t)

    print('DMP: Imitating trajectory...')
    ftarget, w = my_dmp.imitate(pose, dpose, ddpose, t, s, psv)
    
    print('DMP: Generating trajectory...')
    ddx, dx, x = my_dmp.generate(w, f_pose[0], f_pose[-1], t, s, psv, obstacles)

    print('Calculating inverse kinematics from DMP-cartesian trajectory...')
    xc = np.zeros(q.shape)
    prev_sol = q[0]
    for i,xi in enumerate(x):
        if i > 0:
            prev_sol = xc[i - 1]

        ik = robot.Inverse_Kinematics(limb, xi[:3], np.rad2deg(xi[3:]), prev_sol )
        xc[i] = [ik[j] for j in limb.joint_names()]

    pose_mod = np.array([Te(*qi) for qi in xc])[:,:3,3]
  
    # Plot functions
    if arg.show_plots:
        plot.comparison(t, q, xc, None, labels=['Original q', 'cartesian-DMP', 'None'])
        plot.position(t, x[:,3:], pose[:,3:], title='Orientation')
        plot.cartesian_history([x[:,:3], pose[:,:3], obstacles])
        plot.show_all()

    print('Saving joint angle solutions to: %s' % (arg.output_file + '_trajectory.txt'))
    traj_final = np.concatenate((xc, np.multiply(np.ones((xc.shape[0], 1)), 0.0402075604203)), axis=1)
    time = np.linspace(0, t[-1], xc.shape[0]).reshape((xc.shape[0], 1))
    traj_final = np.concatenate((t.reshape((-1, 1)), traj_final), axis=1)
    header = 'time,right_j0,right_j1,right_j2,right_j3,right_j4,right_j5,right_j6,right_gripper'
    np.savetxt(arg.output_file + '_trajectory.txt', traj_final, delimiter=',', header=header, comments='', fmt="%1.12f")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Use Reinforced DMP to adapt to new goals")
    parser.add_argument('-ga', '--gain', type=float, default=20.0,
                        help="Set the gain of the DMP transformation system.")
    parser.add_argument('-ng', '--num-gaussians', type=int, default=20,
                        help="Number of Gaussians")
    parser.add_argument('-sb', '--stabilization', type=bool, default=False,
                        help="Add a stability term at end of trajectory")
    parser.add_argument('-if', '--input-file', type=str, default='data/demo14.txt',
                        help="Input trajectory file")
    parser.add_argument('-of', '--output-file', type=str, default='output',
                        help="Output plot file")
    parser.add_argument('-p', '--show-plots', dest='show_plots', action='store_true',
                        help="Show plots at end of computation")
    parser.add_argument('-np', '--no-plots', dest='show_plots', action='store_false',
                        help="Don't show plots at end of computation")
    parser.add_argument('-w', '--window', type=int, default=5,
                        help="Window size for filtering")
    parser.add_argument('-b', '--blends', type=int, default=10,
                        help="Number of blends for filtering")
    parser.add_argument('-s', '--samples', type=int, default=10,
                        help="Number of paths for exploration")
    parser.add_argument('-r', '--rate', type=float, default=0.5,
                        help="Number of possible paths to keep")
    parser.add_argument('-obs', '--obstacles', dest='obstacle_avoidance', action='store_true',
                        help="Use obstacle avoidance dynamics")
    parser.add_argument('-g', '--goal', nargs='+', type=float, 
                        default=[-2.7, 3.4, 0.6, -0.3, 1.8, -2.7, -1.35],
                        help="New position goal (joint space)")
    parser.set_defaults(show_plots=True, obstacle_avoidance=False)
    arg = parser.parse_args()

    main(arg)

