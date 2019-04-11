#!/usr/bin/env python2

from pyrdmp.dmp import DynamicMovementPrimitive as DMP
import pyrdmp.plots as plot
from pyrdmp.utils import *
import cv2
import tensorflow as tf
from utils import detector_utils as detector_utils
import numpy as np
import sympy as sp
from sawyer_kinematics import Sawyer
import rospy
import intera_interface
import argparse


# Camera calibration values - Specific to C930e 
CAMERAMATRIX = np.array([[506.857008, 0.000000, 311.541447], 
                         [0.000000, 511.072198, 257.798417], 
                         [0.000000, 0.000000, 1.000000]])
DISTORTION = np.array([0.047441, -0.104070, 0.006161, 0.000338, 0.000000])

CARTIM = [[835, 417], [322, 608]] #[[178, 448], [173, 355]]  # [[XX],[YY]] of the calibration points on table
CARTBOT = [[0.3,-0.3], [-0.4,-0.8]] # [[XX],[YY]] for the cartesian EE table values
ZLOW = -.05#-0.16187#-0.065 # Pick up height
ZHIGH = 0.66845#0.26 # Drop off height (to reach over lip of box)
ZHPIXELS = 315.0*376.0#*1.7
ZLPIXELS = 115.0*160.0#*1.2
CAMWIDTH, CAMHEIGHT = 1280, 720

BLUELOWER = np.array([110, 100, 100])
BLUEUPPER = np.array([120, 255, 255])

# Determines noise clear for morph
KERNELOPEN = np.ones((5, 5))
KERNELCLOSE = np.ones((5, 5))

# Font details for display windows
FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONTSCALE = 1
FONTCOLOR = (255, 255, 255)

def main(arg):
    rospy.init_node("Sawyer_DMP")
    limb = intera_interface.Limb('right')
    lights = intera_interface.Lights() 
    lights.set_light_state('right_hand_red_light')

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMWIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMHEIGHT)
    if not cap.isOpened():
        exit(1)
    
    detection_graph, sess = detector_utils.load_inference_graph()
    
    position_dmp = DMP(arg.gain, arg.num_gaussians, arg.stabilization, arg.obstacle_avoidance, arg.obstacle_gain)
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
    prev = None
    for i, h in enumerate(ht):
        position[i] = h[:3,3]
        prev = orientation[i] = transform_to_euler(h, prev)
    
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
    
    lights.set_light_state('right_hand_green_light')
    lights.set_light_state('right_hand_red_light', False)

    obstacles, image_o = detect_hand(cap, detection_graph, sess, cap.get(3), cap.get(4))
    target, image_t = detect_block(cap)

    if arg.show_camera:
        for i in range(20):
            obstacles, image_o = detect_hand(cap, detection_graph, sess, cap.get(3), cap.get(4))
            target, image_t = detect_block(cap)
            cv2.imshow('hand', cv2.cvtColor(image_o,cv2.COLOR_RGB2BGR))
            cv2.imshow('blocks', image_t)
            cv2.imwrite(arg.output_file + '-obstacle-' +str(i) + '.png',cv2.cvtColor(image_o, cv2.COLOR_RGB2BGR))
            cv2.imwrite(arg.output_file + '-block-' +str(i) + '.png',image_t)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    
    lights.set_light_state('right_hand_red_light')
    lights.set_light_state('right_hand_green_light', False)

    goal_pose = f_position[-1] if arg.original_point else [target[0][0], target[0][1], f_position[-1][2]] 
    obstacles = np.array([[0,0,0,0,0,0]]) if len(obstacles) is 0 else obstacles
    
    print('DMP: Generating trajectory...')
    ddxp, dxp, xp = position_dmp.generate(w_position, f_position[0], goal_pose, t, s, psv, obstacles)
    ddxo, dxo, xo = position_dmp.generate(w_orientation, f_orientation[0], f_orientation[-1], t, s, psv, obstacles)

    print('Calculating inverse kinematics from DMP-cartesian trajectory...')
    xc = np.zeros(q.shape)
    prev_sol = q[0]
    for i,xi in enumerate(xp):
        if i > 0:
            prev_sol = xc[i - 1]

        ik = robot.Inverse_Kinematics(limb, xi, np.rad2deg(xo[i]), prev_sol )
        xc[i] = [ik[j] for j in limb.joint_names()]

    gen_sol = np.concatenate((xp,xo),axis=1)
    real_sol = np.concatenate((position,orientation), axis=1)
 
    w = np.hstack((w_position, w_orientation))
    w_init = np.ones(w.shape)
    ygaussianlabels = ['x', 'y', 'z', '$\\alpha$', '$\\beta$', '$\\gamma$']
    
    
    # Mean squared error
    mse = np.mean((real_sol - gen_sol)**2, axis=0)
    print('Mean squared error:', mse)
    
    print('start:', real_sol[0,:3])
    print('end:', gen_sol[-1,:3])
    print('goal:', goal_pose)
    print('obstacles:', obstacles)
    if arg.show_plots:
        plot.comparison(t, None, q, xc, labels=[',', 'Original', 'Adapted'], directory=arg.output_file)
        plot.gaussian(s, psv, w_init, "Initial Gaussian", directory=arg.output_file, ylabel=ygaussianlabels)
        plot.gaussian(s, psv, w, "Altered Gaussian", directory=arg.output_file, ylabel=ygaussianlabels)
        plot.phase(s, directory=arg.output_file)
        #plot.comparison(t, q, xc, None, labels=['Original q', 'cartesian-DMP', 'None'])
        #plot.position(t, xo, position[:,3:], title='Orientation')
        plot.cartesian_history([gen_sol, real_sol, obstacles], [0.2,0.2,100.0], directory=arg.output_file)
        #plot.show_all()

    print('Saving joint angle solutions to: %s' % (arg.output_file + '_trajectory.txt'))
    traj_final = np.concatenate((xc, np.multiply(np.ones((xc.shape[0], 1)), 0.0402075604203)), axis=1)
    time = np.linspace(0, t[-1], xc.shape[0]).reshape((xc.shape[0], 1))
    traj_final = np.concatenate((t.reshape((-1, 1)), traj_final), axis=1)
    header = 'time,right_j0,right_j1,right_j2,right_j3,right_j4,right_j5,right_j6,right_gripper'
    np.savetxt(arg.output_file + '_trajectory.txt', traj_final, delimiter=',', header=header, comments='', fmt="%1.12f")
    lights.set_light_state('right_hand_red_light', False)

# Filters blocks out of image and returns a list of x-y pairs in relation to the end-effector
def detect_hand(cap, detection_graph, sess, imw, imh, num_hands=1, thresh=0.15):
    for i in range(5): cap.grab() # Disregard old frames
    ret, image_np = cap.read()
    while not ret: # In case an image is not captured
        ret, image_np = cap.read()

    #image_np = cv2.undistort(image_np, CAMERAMATRIX, DISTORTION)
    try:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    except:
        print("Error converting to RGB")

    box, sc = detector_utils.detect_objects(image_np, detection_graph, sess)
    detector_utils.draw_box_on_image(num_hands, thresh, sc, box, imw, imh, image_np)
    conts = detector_utils.get_box_size(num_hands, thresh, sc, box, imw, imh, image_np)
    ret = np.array([np.concatenate([pixels_to_cartesian(*get_center(*c)), [get_height(*c),0,0,0]]) for c in conts])
    return ret, image_np

# Filters blocks out of image and returns a list of x-y pairs in relation to the end-effector
def detect_block(cap):
    for i in range(5): cap.grab() # Disregard old frames

    ret_val, im = cap.read()
    while not ret_val: # In case an image is not captured
        ret_val, im = cap.read()

    #und_im = cv2.undistort(im, CAMERAMATRIX, DISTORTION) # Remove distortions
    und_im = im
    imHSV = cv2.cvtColor(und_im, cv2.COLOR_BGR2HSV)
        
    mask = cv2.inRange(imHSV, BLUELOWER, BLUEUPPER) # Masking out blue cylinders
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNELOPEN)
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, KERNELCLOSE)
   
    #TODO: Get Michail to update installation of OpenCV...
    _, conts, h = cv2.findContours(mask_close.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(und_im, conts, -1, (255, 255, 0), 1) # Helpful for visualization
    

    centers = [get_center(*cv2.boundingRect(c),box=False) for c in conts] # Calc center of each cylinder
    return [pixels_to_cartesian(*c) for c in centers], und_im # Return centers (in cartesian instead of pixels)


def get_height(x, y, w, h):
    m = (ZHIGH - ZLOW)/(ZHPIXELS - ZLPIXELS)
    b = ZLOW
    x = (y-x)*(h-w)
    return m*x + b


# Returns center of block based on bounding box
def get_center(x, y, w, h, box=True):
    if box:
        return ((int)(x + 0.5*(y-x))), ((int)(w + 0.5*(h-w)))
    return ((int)(x + 0.5*w)), ((int)(y + 0.5*h))

# Returns x,y coordinates based on linear relationship to pixel values.
def pixels_to_cartesian(cx, cy):
    a_y = (CARTBOT[1][0]-CARTBOT[1][1])/(CARTIM[1][1]-CARTIM[1][0])
    b_y = CARTBOT[1][1]-a_y*CARTIM[1][0]
    y = a_y*cy+b_y
    a_x = (CARTBOT[0][0]-CARTBOT[0][1])/(CARTIM[0][1]-CARTIM[0][0])
    b_x = CARTBOT[0][1]-a_x*CARTIM[0][0]
    x = a_x*cx+b_x
    return [x, y]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Use Reinforced DMP to adapt to new goals")
    parser.add_argument('-ga', '--gain', type=float, default=20.0,
                        help="Set the gain of the DMP transformation system.")
    parser.add_argument('-og', '--obstacle-gain', type=float, default=500,
                        help="Set the obstacle gain term")
    parser.add_argument('-ng', '--num-gaussians', type=int, default=100,
                        help="Number of Gaussians")
    parser.add_argument('-sb', '--stabilization', type=bool, default=False,
                        help="Add a stability term at end of trajectory")
    parser.add_argument('-if', '--input-file', type=str, default='data/demo14.txt',
                        help="Input trajectory file")
    parser.add_argument('-of', '--output-file', type=str, default='output',
                        help="Output plot file")
    parser.add_argument('-op', '--orig-point', dest='original_point', action='store_true',
                        help="Use end of trajectory point")
    parser.add_argument('-mp', '--mod-point', dest='original_point', action='store_false',
                        help="Use point from camera tracking")
    parser.add_argument('-p', '--show-plots', dest='show_plots', action='store_true',
                        help="Show plots at end of computation")
    parser.add_argument('-np', '--no-plots', dest='show_plots', action='store_false',
                        help="Don't show plots at end of computation")
    parser.add_argument('-c', '--show-camera', dest='show_camera', action='store_true',
                        help="Show camera at end of computation")
    parser.add_argument('-nc', '--no-camera', dest='show_camera', action='store_false',
                        help="Don't show camera at end of computation")
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
    parser.set_defaults(show_plots=True, obstacle_avoidance=False, show_camera=False, original_point=False)
    arg = parser.parse_args()

    main(arg)

