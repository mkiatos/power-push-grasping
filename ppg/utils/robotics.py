"""
Robotics Utilities
==================

"""


class Trajectory:
    """
    Implements a 5-order trajectory

    Parameters
    ----------
    t : tuple
        A tuple with the initial and final time of the trajectory
    pos : tuple
        A tuple with the initial and final position of the trajectory
    vel : tuple
        A tuple with the initial and final velocity of the trajectory. Defaults to [0, 0]
    acc : tuple
        A tuple with the initial and final accelaration of the trajectory. Defaults to [0, 0]
    """
    def __init__(self, t, pos, vel=[0, 0], acc=[0, 0]):
        self.T = t[1] - t[0]
        self.ti = t[0]
        self.tf = t[1]
        self.a0 = pos[0]
        self.a1 = vel[0]
        self.a2 = 0.5 * acc[0]
        self.a3 = (1/(2 * pow(self.T, 3))) * (20 * (pos[1] - pos[0]) - (8 * vel[1] + 12 * vel[0]) * self.T - (3 * acc[0] - acc[1]) * pow(self.T, 2))
        self.a4 = (1/(2 * pow(self.T, 4))) * (30 * (pos[0] - pos[1]) + (14 * vel[1] + 16 * vel[0]) * self.T + (3 * acc[0] - 2 * acc[1]) * pow(self.T, 2))
        self.a5 = (1/(2 * pow(self.T, 5))) * (12 * (pos[1] - pos[0]) - 6 * (vel[1] + vel[0]) * self.T - (acc[0] - acc[1]) * pow(self.T, 2))

    def pos(self, time):
        """
        Return the position of the trajectory for the given time. If the time
        is thresholded in initial and final.
        """
        if (time < self.ti):
            t = 0
        elif (time > self.tf):
            t = self.tf - self.ti
        else:
            t = time - self.ti
        return self.a0 + self.a1 * t + self.a2 * pow(t, 2) + self.a3 * pow(t, 3) + self.a4 * pow(t, 4) + self.a5 * pow(t, 5)

    def vel(self, time):
        """
        Return the velocity of the trajectory for the given time. If the time
        is thresholded in initial and final.
        """
        if (time < self.ti):
            t = 0
        elif (time > self.tf):
            t = self.tf - self.ti
        else:
            t = time - self.ti
        return self.a1 + 2.0 * self.a2 * t + 3.0 * self.a3 * pow(t, 2) + 4.0 * self.a4 * pow(t, 3) + 5.0 * self.a5 * pow(t, 4)