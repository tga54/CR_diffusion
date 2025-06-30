#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 17:50:25 2025

@author: tga
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import h5py
import gc

# Parameters
Dl = 0.1                   # Diffusion length scale 0.1 pc
n_particles = 100000      # Number of particles
n_steps = 100000             # Number of time steps
dt = 1000                 # Time step year   

sigma_C = 255e-27  # cm^2
c_cms = 3e10 # speed of light in cm s^-1
c_pcyr = 0.307 # speed of light in pc yr ^-1
grammage = np.zeros(n_particles)

# Initialize particles at center
positions = np.zeros((n_particles, int(n_steps/100), 3))
pos_temp = np.zeros((n_particles, 3))
# Initial velocity vectors of particles
c_theta_v = np.random.uniform(-1.0, 1.0, n_particles)
s_theta_v = np.sqrt(1.0-c_theta_v**2.0)

phi_v = np.random.uniform(0, np.pi*2, n_particles)
vx = s_theta_v*np.cos(phi_v)
vy = s_theta_v*np.sin(phi_v)
vz = c_theta_v

decay_flag = np.zeros(n_particles)

t1 = time.time()
# Perform the random walk
for i in range(1, n_steps): # time loop
    # r2 = np.sum(positions[:,i-1]**2,axis=1)
    # nH = 1 / (1 + r2)**1.5        # broken powerlaw gas distribution cm^{-3}
    # grammage += c_pcyr*dt * nH   # grammage of particles after this time step

    # t_decay = 1.0 / (c_pcyr * sigma_C * nH) / 3.15e7
    
    p_scatter = np.random.uniform(0,1,n_particles) # random no. to check if scatters
    p_decay = np.random.uniform(0,1,n_particles) # random no. to check it decays
    
    ind_scatter = np.where(p_scatter < c_pcyr*dt / Dl)          # probability to scatter
    ind_noscatter = np.where(p_scatter >= c_pcyr*dt / Dl) 
    
    # ind_decay = np.where(p_decay < dt / t_decay)          # probability to scatter
    # decay_flag[ind_decay] += 1  
    
    n_scatter = len(ind_scatter[0])           # index of scattered particles
    n_noscatter = len(ind_noscatter[0])       # index of unscattered particles

    c_theta_v_temp = np.random.uniform(-1.0, 1.0, n_scatter)
    s_theta_v_temp = np.sqrt(1.0-c_theta_v_temp**2.0)

    phi_v_temp = np.random.uniform(0, np.pi*2, n_scatter)
    vx[ind_scatter] = s_theta_v_temp*np.cos(phi_v_temp)
    vy[ind_scatter] = s_theta_v_temp*np.sin(phi_v_temp)
    vz[ind_scatter] = c_theta_v_temp
    pos_temp = pos_temp + np.array([vx, vy, vz]).T * dt
    if not i % 100:
        positions[:,int(i/100)] = pos_temp   # particle position after this time step
            

t2 = time.time()
print(t2-t1)

with h5py.File("particle_positions_Tstep100.h5", "w") as hf:
    hf.create_dataset("positions", data=positions, compression="gzip", compression_opts=4)

print("Saved positions array to particle_positions.h5")
del positions       # Delete the positions array
gc.collect() 
print("done !")
# ax = plt.figure().add_subplot(projection='3d')
# x0, y0, z0 = positions[0,:,0], positions[0,:,1], positions[0,:,2]
# x1, y1, z1 = positions[1,:,0], positions[1,:,1], positions[1,:,2]
# x2, y2, z2 = positions[2,:,0], positions[2,:,1], positions[2,:,2]

# ax.plot(x0, y0, z0, color='blue', linewidth=2, label='Trajectory 0')
# ax.plot(x1, y1, z1, color='orange', linewidth=2, label='Trajectory 1')
# ax.plot(x2, y2, z2, color='yellow', linewidth=2, label='Trajectory 2')

# ax.scatter(x0[0], y0[0], z0[0], color='red', s=50, label='Start')
# # ax.scatter(x0[-1], y0[-1], z0[-1], color='green', s=50, label='End')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Trajectory')
# ax.legend()
# plt.tight_layout()
# plt.show()