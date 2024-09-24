#Importing dependencies.
import numpy as np
import matplotlib.pyplot as plt
from math import sin

"""
QUTEE MODEL JOINT CALCULATIONS

    Use sine-wave controller for actuators, x = A*sin(f*t + phi).
    There are 4 legs on the robot each leg has 3 actuators, 
    which each needs 2 parameters optimized (we set the frequency the same for each actuator).
    That means we need 4(legs)*3(actuators) = 12x2 size array to store the parameters for the sine wave controller.
    In total 24 parameters.
"""

class individual:
    """
    The class 'individual' will contain all parameters for each actuator for 1 individual (Qutee robot).
    And also some methods to access fitness, etc.
    """

    def __init__(self, n_actuators: int, n_parameters: int, low: float, high: float):
        self.fitness = 0
        self.n_acuators = n_actuators
        self.parameters = n_parameters
        self.parameters_array = np.random.uniform(low=low, high=high, size=(n_actuators, n_parameters))

    def sine_wave_controller(self, A, f, t, phi):
        pass
        

    def performance_calculation(self):
        pass
        


def sine_wave(A, f, t, phi):
    out = np.array(())

    for x in t:
        y = A*sin(f*x+phi)
        out = np.append(out, y)

    return out

x = np.arange(0,10,.1)
y = sine_wave(.78,5,x,0.2)

plt.plot(x,y)
plt.show()


ind1 = individual(12,2,0.05,0.78)

print(ind1.parameters_array)
