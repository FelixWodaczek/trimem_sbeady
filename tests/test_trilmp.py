import pytest

import numpy as np

from util import icosphere
from trimem.mc.trilmp import TriLmp
from trimesh import Trimesh

def test_trilmp_simplerun():
    sigma_membrane = 1.0

    resolution = 2
    vertices, faces = icosphere(resolution)

    mesh = Trimesh(vertices=vertices, faces=faces)
    # rescaling it so that we start from the right distances
    desired_average_distance = 2**(1.0/6.0) * sigma_membrane
    current_average_distance = np.mean(mesh.edges_unique_length)
    scaling = desired_average_distance/current_average_distance
    mesh.vertices *= scaling

    (xlo,xhi,ylo,yhi,zlo,zhi) = (-50, 50, -50, 50, -50, 50)

    trilmp = TriLmp(
        box=(xlo,xhi,ylo,yhi,zlo,zhi)
    )
    trilmp.run(1)