import time
from datetime import datetime

import brax.mjx.pipeline
from etils import epath
import functools
from IPython.display import HTML
from typing import Any, Dict, Sequence, Tuple, Union
import os
from ml_collections import config_dict


import jax
from jax import numpy as jp
import numpy as np
from flax.training import orbax_utils
from flax import struct
from matplotlib import pyplot as plt
import mediapy as media
from orbax import checkpoint as ocp

import mujoco
from mujoco import viewer
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

import glfw

from random import random as rand
from math import sin
from math import sqrt
from math import pi

Kp = 3.0
Kd = 1.0

batch_size = 8

def pd_control(m, d):
    for i in range(m.nu):
        joint_id = m.jnt_dofadr[i]

        #Proportional term (difference between target and current position)
        pos_err = m.key_qpos[i] - d.qpos[joint_id]

        # Derivative term (difference between target and current velocity)
        vel_err = -d.qvel[i]

        # PD control law
        d.ctrl[i] = Kp * pos_err + Kd * vel_err

#Options
RENDER_VIDEO = True

mj_model = mujoco.MjModel.from_xml_path("./qutee.xml")
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model, width=800, height=600)


#############Controller and Callback################################
controllers = []
for j in range(batch_size):
    controllers.append([])
    for i in range(mj_model.nu):
        a = rand()*mj_model.actuator_ctrlrange[i][1]
        dc = (mj_model.actuator_ctrlrange[i][1] - a) * rand()
        if rand() > 0.5:
            dc = -dc
        controllers[j].append([a, (rand()*20.0), rand(), dc])
        #controllers.append([rand() * mj_model.actuator_ctrlrange[i][1], rand() * 20.0 - 10.0, rand() * 2.0 * pi, rand() * 10.0 - 5.0]) #Sine controller from main.py
controllers = jax.numpy.array(controllers)

def sine_control(d, cont):
    dn = d
    ctrl = ((cont[:, 2] * d.time) + cont[:,1]) * 2 * pi
    ctrl = jax.numpy.sin(ctrl)
    ctrl = (ctrl * cont[:,0]) + cont[:,3]
    dn = dn.replace(ctrl=ctrl)
    return dn

def tan_control(d, cont):
    dn = d
    ctrl = ((cont[:,2]*d.time)+cont[:,1]) * 2*pi
    ctrl = 4 * jax.numpy.sin(ctrl)
    ctrl = jax.numpy.tanh((ctrl*cont[:,0]) + cont[:,3])
    dn = dn.replace(ctrl=ctrl)
    return dn

#    for i in range(m.nu):
#        c = controllers[i]
#        d.ctrl = d.ctrl.at[i].set(c[0]*sin((c[1]*pi*time)+c[2])+c[3])


################################################

mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

print(jax.devices())

rng = jax.random.PRNGKey(0)
rng = jax.random.split(rng, batch_size)
mjx_data = jax.vmap(lambda rng: mjx_data)(rng)


# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

duration = 10  # (seconds)
framerate = 60.0  # (Hz)

mujoco.mj_resetData(mj_model, mj_data)

view = None
if not RENDER_VIDEO:
    view = viewer.launch_passive(mj_model, mj_data)

#mujoco.set_mjcb_control(sine_control)
#jit_sine = jax.jit(sine_control)
jit_sine = jax.jit(jax.vmap(sine_control,[0,0]))
jit_tan = jax.jit(jax.vmap(tan_control, [0,0]))

#jit_step = jax.jit(mjx.step)
jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))

#print(mjx_model.nu, mjx_model.nv, mjx_model.na, mjx_model.actuator_ctrlrange)

timecalc = time.time()
def run_sim(m, d, duration_i, renderer=None, view=None ,fps=60):
    frames = []
    x_model = mjx.put_model(m)
    while d.time[0] < duration_i:
        simstart = d.time[0]
        while (d.time[0] - simstart) < (1.0/fps):
            #print(mjx_data.time)
            d = jit_sine(d, controllers)
            d = jit_step(x_model, d)
            #print(mjx_data.ctrl)
        print("Progress", (d.time[0] / duration_i) * 100, "%")
        if renderer:
            batch_mj_data = mjx.get_data(m, d)
            renderer.update_scene(batch_mj_data[0], scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)
        elif view:
            view.sync()
    return m, d, frames

mjx_data, frames = run_sim(mj_model, mjx_data, duration, renderer)[1:3]

print("Simulation of", batch_size, "robots took", (time.time()-timecalc)/60, "minutes")
distances = []
maxdist = 0.0
best_individual = None
for i in range(mjx_data.xpos.__len__()):
    el = mjx_data.xpos[i];
    dist = sqrt(el[1][0]**2 + el[1][1]**2 + el[1][2])
    distances.append(dist)
    if dist > maxdist:
        best_individual = controllers[i]
        maxdist = dist
print("Longest traversal was", maxdist)
np.set_printoptions(suppress=True)
print("Best traversal by controller:", best_individual)


# Simulate and display video.
if RENDER_VIDEO:
    media.write_video("output.mp4", frames)
