import numpy as np


#WIP!!
class Robot:
    def __init__(self):
        #Two params per joint, 12 joints in total.
        self.params = np.zeros((12,2))


    def update_parameters(self, new_parameters):
        self.params = new_parameters