#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 16:14:18 2023

@author: a
"""

import numpy as np
from jax_md import space
from jax import vmap, jit

import os
cur_dir = os.getcwd()
input_file_path = os.path.join(cur_dir, 'XDATCAR')
input_file_poscar = os.path.join(cur_dir, 'POSCAR')
output_file_path = os.path.join(cur_dir, 'output.txt')
counts_output_file_path = os.path.join(cur_dir, 'counts_output.txt')

# Open the file for reading
with open(input_file_poscar, 'r') as file:
    # Read lines up to the 7th line
    for i in range(7):
        line = file.readline()
    
    # Split the 7th line into parts and convert them to integers
    values = map(int, line.split())
    
    # Sum the values and store in a variable
    total_n_a = sum(values)

# Output the sum
print(f"Total number of atoms {total_n_a}")


with open(input_file_path, 'r') as f:
    cells = []
    poss = []
    for line in f:
        next(f)
        cell = []
        pos = []
        for i in range(3):
            cell.append([float(j) for j in next(f).split()])
        cells.append(cell)
        next(f)
        next(f)
        next(f)
        for i in range(total_n_a):
            pos.append([float(j) for j in next(f).split()])
        poss.append(pos)

poss = np.array(poss)
cells = np.array(cells)   

mats = []

for cell, pos in zip(cells, poss):
    displacement, shift = space.periodic_general(cell.T,fractional_coordinates=True)
    distances = vmap(displacement, (None,0))
    distance_matrix = vmap(distances, (0,None))
    mat = np.linalg.norm(distance_matrix(pos[0:],pos[0:]), axis=-1)
    mat = (np.where(np.tril(mat) == 0, np.inf, mat))
    mats.append(mat)
mats = np.array(mats)

threshold = 2

ts, at_1, at_2 = (mats < threshold).nonzero()

counts_per_step = {}

with open(output_file_path, 'w') as output_file:
    for t, a1, a2 in zip(ts, at_1, at_2):
        output_file.write(f"{t} {a1+0} {a2+0} {mats[t,a1,a2]}\n")

# Increment the count for this time step
        counts_per_step[t] = counts_per_step.get(t, 0) + 1

with open(counts_output_file_path, 'w') as counts_output_file:
    for t, count in counts_per_step.items():
        counts_output_file.write(f"Time step {t} {count} pairs with distance < {threshold} \n")

