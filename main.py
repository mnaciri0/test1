#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 00:07:05 2018

@author: eert

States y = [q1, q2, qd1, qd2]'
Control u: Torque on first joint
"""

import numpy as np 
from numpy import sin, cos, pi
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
#m1 = 1
#m2 = 1
#l1 = 1
#l2 = 1
#g = 9.81

# Parameters
par = {'g':  9.81,  #[m/s²]
       'l1': 0.2,   #[m]
       'l2': 0.2,   #[m]
       'm1': 0.2,   #[kg]
       'm2': 0.1,   #[m]
       'c1':-0.06,  #[m]
       'c2': 0.08,  #[m]
       'I1': 0.09,  #[kgm^2]
       'I2': 0.0002,#[kgm^2]
       'b1': 5.,    #[kgs^-1]
       'b2': 0.0003,#[kgs^-1]
       'km': 60.,    #[Nm]
       'te': 0.05}  #[s]

T = 10.
nDisControls = 5
u_dis = [0., 3., 1., 2., 5.]
t_dis = np.linspace(0., T, nDisControls)

def der_fun(t, y):
    #u = np.interp(t_dis, u_dis, t)
    u = 0.
    
    q1 = y[0]
    q2 = y[1]
    qd1 = y[2]
    qd2 = y[3]
    
    P1 = par['m1']*par['c1']**2 + par['m2']*par['l1']**2 + par['I1']
    P2 = par['m2']*par['c2']**2 + par['I2']
    P3 = par['m2']*par['l1']*par['c2']
    
    g1 =(par['m1']*par['c1'] + par['m2']*par['l1'])*par['g']
    g2 = par['m2']*par['c2']*par['g']
    
    M = np.array([[P1 + P2 + 2*P3*cos(q2), P2 + P3*cos(q2)], [P2 + P3*cos(q2), P2]])
    
    C = np.array([[par['b1'] - P3*qd2*sin(q2), -P3*(qd1+qd2)*sin(q2)], [P3*qd1*sin(q2), par['b2']]])
 
    G = np.array([[-g1*sin(q1) - g2*sin(q1+q2)], [-g2*sin(q1 + q2)]])
 
    qdd = np.linalg.solve(M, np.array([[u, 0]]).T - C @ np.array([[qd1, qd2]]).T - G)
    
    return [qd1, qd2, qdd[0], qdd[1]]

# Solve ODE
y0 = [pi, pi/2, 0, 0]
sol = integrate.solve_ivp(der_fun, [0, 10], y0)

# Positions
x1 = -par['l1'] * sin(sol.y[0,:])
y1 =  par['l1'] * cos(sol.y[0,:])
x2 = -par['l2'] * sin(sol.y[1,:]) + x1
y2 =  par['l2'] * cos(sol.y[1,:]) + y1

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(min(min(x1), min(x2)), max(max(x1), max(x2))), 
                     ylim=(min(min(y1), min(y2)), max(max(y1), max(y2))))
ax.grid()
line, = ax.plot([], [], 'o-', lw=2)

def init():
    line.set_data([], [])
    return line,

def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]
    line.set_data(thisx, thisy)
    return line,

anim = animation.FuncAnimation(fig, animate, frames = len(sol.y[0]),
                              interval=1000*np.diff(sol.t).mean(), blit=True, init_func=init)

plt.show()



