#!/usr/bin/env python2

#TODO: Seperate DMP for position and orientation

from pyrdmp.dmp import DynamicMovementPrimitive as DMP
import pyrdmp.plots as plot
from pyrdmp.utils import *
import numpy as np
import sympy as sp
from sawyer_kinematics import Sawyer
import rospy
import intera_interface
import argparse


def main(arg):
    rospy.init_node("Sawyer_DMP")
    limb = intera_interface.Limb('right')
    
    position_dmp = DMP(arg.gain, arg.num_gaussians, arg.stabilization, arg.obstacle_avoidance)
    orientation_dmp = DMP(arg.gain, arg.num_gaussians, arg.stabilization, False) # no obstacle avoidance on orientation

    data = load_demo(arg.input_file)
    t, q = parse_demo(data)

    s = position_dmp.phase(t)
    psv = position_dmp.distributions(s)
    t = normalize_vector(t)
    
    print('Calculating forward kinematics...')
    robot = Sawyer()
    Te = sp.lambdify(robot.q, robot.get_T_f()[-1])
    ht = np.array([Te(*robot.JointOffset(qi)) for qi in q])
    position, orientation = np.zeros((len(t), 3)), np.zeros((len(t), 3))

    for i, h in enumerate(ht):
        c = h[:3,3]
        p = np.arctan2(-h[2,0], np.sqrt(h[0,0]**2 + h[1,0]**2))
        y = np.arctan2(h[1,0]/np.cos(p), h[0,0]/np.cos(p))
        r = np.arctan2(h[2,1]/np.cos(p), h[2,2]/np.cos(p))
        
        if i > 0:
            y = fix_angle(orientation[i-1,2], y)
            p = fix_angle(orientation[i-1,1], p)
            r = fix_angle(orientation[i-1,0], r)

        position[i] = c
        orientation[i] = np.stack((r,p,y))
    
    
    # Set static obstacles
    #obstacles = np.array([np.concatenate((position[len(position)//3], orientation[len(position)//3]))])#,:3]])
    #obstacles = np.array([np.random.uniform(l,h,size=(2,)) for l,h in [(-0.2,0.6),(-0.6,0.2),(-0.1,0.2),(0,0),(0,0),(0,0)]]).T
    ob_angles = [-1.931447265625, -0.3394013671875, 0.4098779296875, 1.296037109375, -0.550625, 0.7161005859375, 0.5338125] 
    obstacles = np.array([np.concatenate((Te(*robot.JointOffset(ob_angles))[:3,3], [0,0,0]))])
    #obstacles = np.array([[0.4134, -0.5456, 0.109, 0, 0, 0]])
    obstacles = np.array([[ 0.2, -0.69431369, -0.05,0,0,0]])#-0.08348392, 0,0,0]] 

    print('Smoothing and filtering cartesian trajectory...')
    dposition, ddposition = np.zeros((2, position.shape[0], position.shape[1]))
    for i in range(position.shape[1]):
        position[:, i] = smooth_trajectory(position[:, i], arg.window)
        dposition[:, i] = vel(position[:, i], t)
        ddposition[:, i] = vel(dposition[:, i], t)
    
    dorientation, ddorientation = np.zeros((2, orientation.shape[0], orientation.shape[1]))
    for i in range(position.shape[1]):
        orientation[:, i] = smooth_trajectory(orientation[:, i], arg.window)
        dorientation[:, i] = vel(orientation[:, i], t)
        ddorientation[:, i] = vel(dorientation[:, i], t)

    # Filter the position velocity and acceleration signals
    f_position, f_dposition, f_ddposition = np.zeros((3, position.shape[0], position.shape[1]))
    for i in range(position.shape[1]):
        f_position[:, i] = blend_trajectory(position[:, i], dposition[:, i], t, arg.blends)
        f_dposition[:, i] = vel(f_position[:, i], t)
        f_ddposition[:, i] = vel(f_dposition[:, i], t)
    
    f_orientation, f_dorientation, f_ddorientation = np.zeros((3, orientation.shape[0], orientation.shape[1]))
    for i in range(position.shape[1]):
        f_orientation[:, i] = blend_trajectory(orientation[:, i], dorientation[:, i], t, arg.blends)
        f_dorientation[:, i] = vel(f_orientation[:, i], t)
        f_ddorientation[:, i] = vel(f_dorientation[:, i], t)

    print('DMP: Imitating trajectory...')
    ftarget_position, w_position = position_dmp.imitate(position, dposition, ddposition, t, s, psv)
    ftarget_orientation, w_orientation = orientation_dmp.imitate(orientation, dorientation, ddorientation, t, s, psv)
    
    print('DMP: Generating trajectory...')
    ddxp, dxp, xp = position_dmp.generate(w_position, f_position[0], f_position[-1], t, s, psv, obstacles)
    ddxo, dxo, xo = position_dmp.generate(w_orientation, f_orientation[0], f_orientation[-1], t, s, psv, obstacles)

    print('Calculating inverse kinematics from DMP-cartesian trajectory...')
    xc = np.zeros(q.shape)
    prev_sol = q[0]
    for i,xi in enumerate(xp):
        if i > 0:
            prev_sol = xc[i - 1]

        ik = robot.Inverse_Kinematics(limb, xi, np.rad2deg(xo[i]), prev_sol )
        xc[i] = [ik[j] for j in limb.joint_names()]

    real_sol = np.concatenate((xp,xo),axis=1)
    gen_sol = np.concatenate((position,orientation), axis=1)
   
    if arg.show_plots:
        #plot.comparison(t, q, xc, None, labels=['Original q', 'cartesian-DMP', 'None'])
        #plot.position(t, xo, position[:,3:], title='Orientation')
        plot.cartesian_history([real_sol, gen_sol, obstacles], [0.2,0.2,100.0])
        plot.show_all()

    print('Saving joint angle solutions to: %s' % (arg.output_file + '_trajectory.txt'))
    traj_final = np.concatenate((xc, np.multiply(np.ones((xc.shape[0], 1)), 0.0402075604203)), axis=1)
    time = np.linspace(0, t[-1], xc.shape[0]).reshape((xc.shape[0], 1))
    traj_final = np.concatenate((t.reshape((-1, 1)), traj_final), axis=1)
    header = 'time,right_j0,right_j1,right_j2,right_j3,right_j4,right_j5,right_j6,right_gripper'
    np.savetxt(arg.output_file + '_trajectory.txt', traj_final, delimiter=',', header=header, comments='', fmt="%1.12f")


def fix_angle(previous, current):
    k = np.round((current - previous)/np.pi)
    if previous >= 0.9*np.pi:
        if k > 0.5: return current + k*np.pi
        if k < -0.5: return current - k*np.pi
    if previous <= -0.9*np.pi: 
        if k > 0.5: return current - k*np.pi
        if k < -0.5: return current + k*np.pi
    return current


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Use Reinforced DMP to adapt to new goals")
    parser.add_argument('-ga', '--gain', type=float, default=20.0,
                        help="Set the gain of the DMP transformation system.")
    parser.add_argument('-ng', '--num-gaussians', type=int, default=500,
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

