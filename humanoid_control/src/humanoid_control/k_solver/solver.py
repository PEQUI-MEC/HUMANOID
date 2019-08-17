"""Solvers."""

from functools import reduce

import autograd.numpy as np

from .component import Joint, Link


class FKSolver(object):
    """A forward kinematics solver."""

    def __init__(self, components, center_of_mass_shifts = None):
        """Generate a FK solver from link and joint instances."""
        joint_indexes = [
            i for i, c in enumerate(components) if isinstance(c, Joint)
        ]

        if center_of_mass_shifts != None and len(center_of_mass_shifts) != len(components)-len(joint_indexes):
            #if len(components)-len(joint_indexes) != len(joint_indexes)+1:
            #    raise Exception("Invalid Actuator: Each joint must be between two links, and to N joints needs N+1 links.")
            raise Exception("Invalid Actuator: All links needs a CoM shift position.")

        def matrices(angles):
            joints = dict(zip(joint_indexes, angles))
            a = [joints.get(i, None) for i in range(len(components))]
            return [c.matrix(a[i]) for i, c in enumerate(components)]

        self._com_shifts = []
        if center_of_mass_shifts != None:
            self._com_shifts = np.flip(np.array(center_of_mass_shifts))
            self._com_shifts = np.concatenate((center_of_mass_shifts, np.array([[1]*len(center_of_mass_shifts)]).transpose()), axis = 1)
        self._matrices = matrices
        self._types = [0]*len(components)
        for it in joint_indexes:
            self._types[it] = 1

        self._types = np.flip(self._types)

    def solve(self, angles):
        """Calculate a position of the end-effector and return it."""
        return reduce(
            lambda a, m: np.dot(m, a),
            reversed(self._matrices(angles)),
            np.array([0., 0., 0., 1.])
        )[:3]

    def center_of_mass_parts (self, angles):
        if (len(self._com_shifts) != 0):
            points = np.array([[0.], [0.], [0.], [1.]])
            it = 0
            for i, mat in enumerate(reversed(self._matrices(angles))):
                points = np.dot(mat, points)
                if (self._types[i] == 0): # link
                    points = np.concatenate((points, np.array([self._com_shifts[it]]).transpose()), axis = 1)
                    it += 1
            return np.flip(points.transpose()[1:,:3], axis = 0)
        else:
            raise Exception("Paramters not found: Insert center of mass of actuator links on Actuator constructor.")


class IKSolver(object):
    """An inverse kinematics solver."""

    def __init__(self, fk_solver, optimizer):
        """Generate an IK solver from a FK solver instance."""
        def distance_squared(angles, target):
            x = target - fk_solver.solve(angles)
            return np.sum(np.power(x, 2))

        optimizer.prepare(distance_squared)
        self.optimizer = optimizer

    def solve(self, angles0, target):
        """Calculate joint angles and returns it."""
        return self.optimizer.optimize(np.array(angles0), target)