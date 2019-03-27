#!/usr/bin/env python2

import argparse
from pyrdmp.dmp import DynamicMovementPrimitive as DMP
import pyrdmp.plots as plot
from pyrdmp.utils import *
import numpy as np
import sympy as sp
from sawyer_kinematics import Sawyer
import rospy
import intera_interface

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

def main():
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
    parser.add_argument('-g', '--goal', nargs='+', type=float, 
                        default=[-2.7, 3.4, 0.6, -0.3, 1.8, -2.7, -1.35],
                        help="New position goal (joint space)")
    parser.set_defaults(show_plots=True)
    arg = parser.parse_args()

    rospy.init_node("Sawyer_DMP")
    limb = intera_interface.Limb('right')
    
    # Initialize the DMP class
    my_dmp = DMP(arg.gain, arg.num_gaussians, arg.stabilization)

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

    #Do FK
    robot = Sawyer()
    Te = sp.lambdify(robot.q, robot.get_T_f()[-1])
    ht = np.array([Te(*qi) for qi in q])
    
    cart = np.zeros((len(t), 6))
    for i, h in enumerate(ht):
        c = h[:3,3][[0,1,2]]
        #y = np.arctan2(-h[1,2], h[2,2])
        #p = np.arcsin(h[0,2])
        #r = np.arctan2(-h[0,1], h[0,0])
        p = np.arctan2(-h[2,0], np.sqrt(h[0,0]**2 + h[1,0]**2))
        y = np.arctan2(h[1,0]/np.cos(p), h[0,0]/np.cos(p))
        r = np.arctan2(h[2,1]/np.cos(p), h[2,2]/np.cos(p))
        #y = np.arctan2(h[2,0], h[2,1])
        #p = np.arccos(h[2,2])
        #r = -np.arctan2(h[0,2],h[1,2])
        cart[i] = np.concatenate((c,np.stack((r,p,y))))

    

    #print(cart.shape)
    #print(cart)
    
    #q = cart
    # Compute velocity and acceleration
    dcart, ddcart = np.zeros((2, cart.shape[0], cart.shape[1]))

    for i in range(cart.shape[1]):
        #cart[:, i] = smooth_trajectory(cart[:, i], arg.window)
        dcart[:, i] = vel(cart[:, i], t)
        ddcart[:, i] = vel(dcart[:, i], t)

    # Filter the position velocity and acceleration signals
    f_q, f_dq, f_ddq = np.zeros((3, q.shape[0], q.shape[1]))

    for i in range(cart.shape[1]):
        f_q[:, i] = blend_trajectory(cart[:, i], dcart[:, i], t, arg.blends)
        f_dq[:, i] = vel(f_q[:, i], t)
        f_ddq[:, i] = vel(f_dq[:, i], t)

    #ftarget, w = my_dmp.imitate(f_q, f_dq, f_ddq, t, s, psv)
    ftarget, w = my_dmp.imitate(cart, dcart, ddcart, t, s, psv)

    print('Imitation done')

    # Generate the Learned trajectory
    x, dx, ddx = np.zeros((3, cart.shape[0], cart.shape[1]))

    for i in range(cart.shape[1]):
        ddx[:, i], dx[:, i], x[:, i] = my_dmp.generate(w[:, i], f_q[0, i], 
                f_q[-1, i], t, s, psv)

    # Adapt using Reinforcement Learning
    #print('Adaptation start')
    #x_r, dx_r, ddx_r = np.zeros((3, cart.shape[0], cart.shape[1]))
    #w_a = np.zeros((my_dmp.ng, cart.shape[1]))
    #gain = []
    #for i in range(cart.shape[1]):
    #    ddx_r[:, i], dx_r[:, i], x_r[:, i], w_a[:, i], g = my_dmp.adapt(w[:, i], 
    #            x[0, i], arg.goal[i], t, s, psv, arg.samples, arg.rate)
    #    gain.append(g)
    #print('Adaptation complete')
     
    xc = np.zeros((len(x), 7))
    prev_sol = q[0]
    for i,xi in enumerate(x):
        #print('!!!!CURRENTLY AT:', i, ' OF ', len(x))
        if i > 0:
            prev_sol = xc[i - 1]

        ik = robot.Inverse_Kinematics(limb, xi[:3], np.rad2deg(xi[3:]), prev_sol )
        xc[i] = [ik[j] for j in limb.joint_names()]

    #x = np.array([robot.Inverse_Kinematics(xi[:3], np.rad2deg(xi[3:])) for xi in x])
    
    cart_mod = np.array([Te(*qi) for qi in xc])[:,:3,3]

    print('FIRST Q:', q[0])
    print('FIRST C:', cart[0])

    print('MOD Q:', xc[0])
    print('MOD C:', cart_mod[0]) 

    # Plot functions
    if arg.show_plots:
        plot.comparison(t, q, xc, None, labels=['Original q', 'Cartesian-DMP', 'None'])
        #plot.gaussian(s, psv, w, "Initial Gaussian")
        #plot.gaussian(s, psv, w_a, "Altered Gaussian")
        #plot.expected_return(gain)
        plot.cartesian_history([x[:,:3], cart_mod])
        plot.show_all()


    traj_final = np.concatenate((xc, np.multiply(np.ones((xc.shape[0], 1)), 0.0402075604203)), axis=1)
    time = np.linspace(0, t[-1], xc.shape[0]).reshape((xc.shape[0], 1))
    traj_final = np.concatenate((t.reshape((-1, 1)), traj_final), axis=1)
    header = 'time,right_j0,right_j1,right_j2,right_j3,right_j4,right_j5,right_j6,right_gripper'
    np.savetxt(arg.output_file + '_trajectory.txt', traj_final, delimiter=',', header=header, comments='', fmt="%1.12f")

if __name__ == '__main__':
    main()

