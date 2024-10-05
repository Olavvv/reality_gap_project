import time
from datetime import datetime

#import brax.mjx.pipeline
#from etils import epath
#import functools
#from IPython.display import HTML
#from typing import Any, Dict, Sequence, Tuple, Union
#import os
#from ml_collections import config_dict


import jax
from jax import numpy as jp
import numpy as np
#from flax.training import orbax_utils
#from flax import struct
#from matplotlib import pyplot as plt
import mediapy as media
#from orbax import checkpoint as ocp

import mujoco
from mujoco import viewer
from mujoco import mjx

#from brax import base
#from brax import envs
#from brax import math
#from brax.base import Base, Motion, Transform
#from brax.envs.base import Env, PipelineEnv, State
#from brax.mjx.base import State as MjxState
#from brax.training.agents.ppo import train as ppo
#from brax.training.agents.ppo import networks as ppo_networks
#from brax.io import html, mjcf, model

import glfw
glfw.init()

from random import random as rand
from math import sin
from math import sqrt
from math import pi
from math import tanh

def generate_controllers(num, model):
    controllers = []
    for j in range(num):
        controllers.append([])
        for i in range(model.nu):
            a = rand() * model.actuator_ctrlrange[i][1]
            dc = (model.actuator_ctrlrange[i][1] - a) * rand()
            if rand() > 0.5:
                dc = -dc
            controllers[j].append([a, (rand() * 20.0), rand(), dc])
    return np.array(controllers)


def tan_control_mjx(d, cont):
    dn = d
    ctrl = ((cont[:,2]*d.time)+cont[:,1]) * 2*pi
    ctrl = 4 * jax.numpy.sin(ctrl)
    ctrl = jax.numpy.tanh((ctrl*cont[:,0]) + cont[:,3])
    dn = dn.replace(ctrl=ctrl)
    return dn


def tan_control(m, d, cont):
    for i in range(m.nu):
        d.ctrl[i] = tanh(cont[i][0]*4*sin(((cont[i][2]*d.time)+cont[i][1]) * 2 * pi)+cont[i][3])

jit_tan = jax.jit(jax.vmap(tan_control_mjx,(0,0)))
jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))


def run_sim_batch(m, d, duration, controllers,fps=60):
    timecalc = time.time()
    while d.time[0] < duration:
        simstart = d.time[0]
        while (d.time[0] - simstart) < (1.0/fps):
            #print(mjx_data.time)
            d = jit_tan(d, controllers)
            d = jit_step(m, d)
            #print(mjx_data.ctrl)
        print("Progress", (d.time[0] / duration) * 100, "%")
    print("Simulation of", d.xpos.__len__(), "robots took", (time.time() - timecalc) / 60, "minutes")
    return m, d


def batch_info(d):
    distances = []
    maxdist = 0.0
    best_individual = None
    for i in range(d.xpos.__len__()):
        el = d.xpos[i];
        dist = sqrt(el[1][0] ** 2 + el[1][1] ** 2 + el[1][2])
        distances.append(dist)
        if dist > maxdist:
            best_individual = i
            maxdist = dist
    print("Longest traversal was", maxdist)
    print("Best traversal by controller:", best_individual)
    return distances, best_individual


def run_sim(m, d, duration, controller, fps=60, view=None, scene_option=None, do_render=True):
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    rend = mujoco.Renderer(m, width=800, height=600)
    frames = []

    while d.time < duration:
        simstart = d.time
        while (d.time - simstart) < (1.0 / fps):

            tan_control(m, d, controller)
            mujoco.mj_step(m, d)
            # print(mjx_data.ctrl)
        print("Progress", (d.time / duration) * 100, "%")
        #print(d.time)
        rend.update_scene(d, scene_option=scene_option)
        pixels = rend.render()
        frames.append(pixels)
    return m, d, frames



#########################################################################


def main():
    batch_size = 2
    duration = 10
    mj_model = mujoco.MjModel.from_xml_path("./qutee.xml")
    mj_data = mujoco.MjData(mj_model)

    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, batch_size)
    batchify = jax.vmap(lambda rng: mjx_data)
    mjx_data = batchify(rng)

    controllers = generate_controllers(batch_size, mj_model)

    mjx_model, mjx_data = run_sim_batch(mjx_model, mjx_data, duration, controllers)

    distances, best = batch_info(mjx_data)

    frames = run_sim(mj_model, mj_data, duration, controllers[best], do_render=True)[2]
    media.write_video("output.mp4", frames)

print(jax.devices())
if input("Continue with run? (y/n)") == "y":
    main()
else:
    print("Exiting")
