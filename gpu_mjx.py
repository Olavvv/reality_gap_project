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

batch_size = 512

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
RENDER_VIDEO = False

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
        controllers[j].append([a, (rand()*20.0-10.0), rand()*2.0*pi, dc])
        #controllers.append([rand() * mj_model.actuator_ctrlrange[i][1], rand() * 20.0 - 10.0, rand() * 2.0 * pi, rand() * 10.0 - 5.0]) #Sine controller from main.py
controllers = jax.numpy.array(controllers)

def sine_control(m, d, cont):
    dn = d
    ctrl = ((cont[:, 2] * d.time) + cont[:,1])
    ctrl = jax.numpy.sin(ctrl)
    ctrl = (ctrl * cont[:,0]) + cont[:,3]
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

frames = []
mujoco.mj_resetData(mj_model, mj_data)

view = None
if not RENDER_VIDEO:
    view = viewer.launch_passive(mj_model, mj_data)

mujoco.set_mjcb_control(sine_control)
#jit_sine = jax.jit(sine_control)
jit_sine = jax.jit(jax.vmap(sine_control,[None,0,0]))

#jit_step = jax.jit(mjx.step)
jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))

#print(mjx_model.nu, mjx_model.nv, mjx_model.na, mjx_model.actuator_ctrlrange)

timecalc = time.time()
while mjx_data.time[0] < duration:
    simstart = mjx_data.time[0]
    while (mjx_data.time[0] - simstart) < (1.0/framerate):
        #print(mjx_data.time)
        mjx_data = jit_sine(mjx_model, mjx_data, controllers)
        mjx_data = jit_step(mjx_model, mjx_data)
        #print(mjx_data.ctrl)
    print("Progress", (mjx_data.time[0] / duration) * 100, "%")
    if RENDER_VIDEO:
        pass
        #mj_data = mjx.get_data(mj_model,mjx_data)
        #renderer.update_scene(mj_data, scene_option=scene_option)
        #pixels = renderer.render()
        #frames.append(pixels)
    #else:
    #    view.sync()
print("Simulation of", batch_size, "robots took", (time.time()-timecalc)/60, "minutes")
distances = []
maxdist = 0.0
for el in mjx_data.xpos:
    dist = sqrt(el[1][0]**2 + el[1][1]**2 + el[1][2])
    distances.append(dist)
    if dist > maxdist:
        maxdist = dist
print("Longest traversal was", maxdist)

# Simulate and display video.
if RENDER_VIDEO:
    media.write_video("output.mp4", frames)
