from __future__ import division

import sys
import numpy as np


class stretchcombo:
    """Constrain a linear combination of bond-length . """
    """bondlenght[i] =[[ atom_i_i, atom_i_j, weight-factor_i]]"""

    def __init__(self, a, bondlist, atoms, invert=False):
        xyz = atoms.get_positions()
        self.invert = invert
        self.dim = len(xyz)
        self.xyz = np.copy(xyz)
        self.cdim = len(bondlist)
        self.thr = 1.0e-12
        self.bondlist = list(bondlist)
        self.projected_force = 0.0
        self.f_thresh = 0.0
        self.full_force = 0.0
        self.update_constraint(a, xyz)

    def constraint_to_dimermethod(self):
        displacement_vector = [[0.0, 0.0, 0.0] for ii in range(self.dim)]
        maskding = [0 for i in range(self.dim)]
        for schieber in self.bondlist:
            diff_vec = schieber[2] * (self.xyz[schieber[1]] - self.xyz[schieber[0]])
            ii = 0
            for at in [schieber[0], schieber[1]]:
                displacement_vector[at] = (-1) ** ii * diff_vec
                maskding[at] = 1
                ii += 1
        displacement_vector /= np.linalg.norm(displacement_vector)
        return maskding, displacement_vector

    def update_constraint(self, a, xyz):
        self.a = a
        self.val0 = 0.0
        self.update(xyz)
        try:
            self.val0 = float(a)
            print('set constrained coordinate to ' + str(self.val0) + ' from current value ' + str(self.val))
            oldxyz = np.copy(xyz)
            self.adjust_positions(oldxyz, xyz)
        except:
            self.val0 = self.val
            self.a = self.val0
            print('fix constrained coordinate to initial value ' + str(self.val0))

    def reset_a(self, a, atoms):
        self.update_constraint(a, atoms.get_positions())
        return self.xyz

    def update(self, xyz):
        self.val = 0.0
        self.xyz = xyz
        self.dir = np.zeros([self.dim, 3])
        self.bond = np.zeros(self.cdim)
        self.involved_atoms = []
        ii = -1
        for i, j, fac in self.bondlist:
            ii += 1
            if i not in self.involved_atoms:
                self.involved_atoms += [i]
            if j not in self.involved_atoms:
                self.involved_atoms += [j]
            self.bond[ii] = np.linalg.norm(xyz[i] - xyz[j])
            self.val += fac * self.bond[ii]
            self.dir[i] += fac * (xyz[i] - xyz[j]) / self.bond[ii]
            self.dir[j] -= fac * (xyz[i] - xyz[j]) / self.bond[ii]
        self.der = np.linalg.norm(self.dir)
        self.maxdir = max([np.linalg.norm(ddd) for ddd in self.dir])
        self.dir = self.dir / self.der

    def return_adjusted_positions(self):
        xyz = self.xyz
        oldxyz = np.copy(xyz)
        self.adjust_positions(oldxyz, xyz)
        return self.xyz

    def adjust_positions(self, atoms, newpositions):
        if self.invert:
            return
        try:
            oldpositions = atoms.get_positions()
        except:
            oldpositions = atoms
        self.update(oldpositions)
        step = np.zeros([self.dim, 3])
        for i in range(0, self.dim):
            step[i] = newpositions[i] - oldpositions[i]
        proj = 0.0
        for i in range(0, self.dim):
            proj += np.dot(step[i], self.dir[i])
        for i in range(0, self.dim):
            newpositions[i] = newpositions[i] - self.dir[i] * proj
        self.update(newpositions)

        i_geo = 0
        imax_geo = 100
        while abs(self.val - self.val0) > self.thr:
            i_geo += 1
            if i_geo > imax_geo:
                print('failed to prepare appropriate structure')
                sys.exit(1)
            correction_direction = self.dir.copy()
            tonorm = np.linalg.norm(correction_direction)

            for i in range(0, self.dim):
                newpositions[i] = newpositions[i] - (self.val - self.val0) * correction_direction[i] / (
                            tonorm * self.der)
            self.update(newpositions)

        if i_geo != 0:
            print('... adjusted constraint to new val = ' + str(self.val) + ' required ' + str(i_geo) + ' iterations')
            # is called in practise twice --> apears in output.
            # reason: when, in scan_constraints, atoms.set_positions() is called, this calls again
            # adjust position. no harm..

        if (abs(self.val - self.val0) > self.thr):
            print('primitive first-come-first-served correction from ' + str(self.val) + ' to ' + str(self.val0))

            for i in range(0, self.dim):
                newpositions[i] = newpositions[i] - self.dir[i] * (self.val - self.val0) * self.der
                self.update(newpositions)
            print('... done: new val ' + str(self.val))

    def adjust_forces(self, atoms, forces):
        positions = atoms.get_positions()
        self.update(positions)
        proj = 0.0
        self.forces_weighted_dir = []
        for i in range(0, len(forces)):
            proj += np.dot(forces[i], self.dir[i])
            self.forces_weighted_dir += [self.dir[i] * proj * np.dot(forces[i], self.dir[i])]
        self.projected_force = proj / self.der
        self.f_thresh = self.projected_force * self.der ** 2 / (self.maxdir * float(len(self.involved_atoms)))
        self.full_force = max([np.linalg.norm(ff) for ff in forces])
        self.projected_forces = []
        if self.invert:
            proj *= 2.0
        for i in range(0, len(forces)):
            forces[i] = forces[i] - self.dir[i] * proj
            self.projected_forces += [self.dir[i] * proj]
