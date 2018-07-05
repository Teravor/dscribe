"""
Defines a set of regressions tests that should be run succesfully after all
major modification to the code.
"""
import math
import numpy as np
import unittest

from describe.descriptors import GGrid

import ase.build
from ase import Atoms
# from ase.visualize import view
import ase.data


H2O = Atoms(
    cell=[
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0]
    ],
    positions=[
        [0, 0, 0],
        [0.95, 0, 0],
        [0.95*(1+math.cos(76/180*math.pi)), 0.95*math.sin(76/180*math.pi), 0.0]
    ],
    symbols=["H", "O", "H"],
)

# NaCl system
system = ase.build.bulk(
    "NaCl",
    "rocksalt",
    5.64,
)


class GGridTests(unittest.TestCase):

    def test_constructor(self):
        """Tests different valid and invalid constructor values.
        """

    def test_number_of_features(self):
        """Tests that the reported number of features is correct.
        """
        desc = GGrid(
            a=5,
            n=10,
            channels=[{"Na": {"amplitude": 1, "std": 1}, "Cl": {"amplitude": 1, "std": 1}}],
            threshold=1e-2,
            flatten=False,
        )
        n_features = desc.get_number_of_features()
        self.assertEqual(n_features, 1*10**3)

    def test_visual(self):
        """Tests that the reported number of features is correct.
        """

        elements = ["Na", "Cl", "H", "O"]
        stds = {x: ase.data.covalent_radii[ase.data.atomic_numbers[x]]/2 for x in elements}
        amplitudes = {x: stds[x]*np.sqrt(2*np.pi) for x in elements}
        channels = [{x: {"amplitude": amplitudes[x], "std": stds[x]} for x in elements}]

        a = 10
        desc = GGrid(
            a=a,
            n=40,
            channels=channels,
            threshold=1e-2,
            flatten=False,
            periodic=True,
        )

        res = desc.create(system, offset=np.array((a/2, a/2, a/2)))
        # print(res.shape)

        # a = 10
        # desc = GGrid(
            # a=a,
            # n=40,
            # channels=channels,
            # threshold=1e-2,
            # flatten=False,
            # offset=np.array((a/2, a/2,2020 a/2))-H2O.get_center_of_mass(),
            # periodic=False,
        # )

        # res = desc.create(H2O)

        # Visualize as plotly isosurface
        # from skimage import measure
        # from plotly.offline import download_plotlyjs, plot
        # import plotly.plotly as py
        # import plotly.graph_objs as go
        # import plotly.figure_factory as ff
        # verts, faces, _, _ = measure.marching_cubes_lewiner(sums, 0.5, spacing=(0.25, 0.25, 0.25))

        # # Plotly isosurface
        # x, y, z = zip(*verts)
        # colormap = ['rgb(255,105,180)', 'rgb(255,255,51)', 'rgb(0,191,255)']
        # fig = ff.create_trisurf(x=x,
                                # y=y,
                                # z=z,
                                # plot_edges=False,
                                # colormap=colormap,
                                # simplices=faces,
                                # title="Isosurface")
        # plot(fig)

        # Visualize as 2D slice
        # from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as mpl
        mpl.matshow(res[0, :, :, 19], cmap=mpl.cm.gray)
        # plt.clim(0, 0.5)
        mpl.colorbar()
        mpl.show()


    # def test_flatten(self):
        # """Tests the flattening.
        # """
        # # Unflattened
        # desc = GGrid(n_atoms_max=5, flatten=False)
        # cm = desc.create(H2O)
        # self.assertEqual(cm.shape, (5, 5))

        # # Flattened
        # desc = CoulombMatrix(n_atoms_max=5, flatten=True)
        # cm = desc.create(H2O)
        # self.assertEqual(cm.shape, (25,))

    # def test_features(self):
        # """Tests that the correct features are present in the desciptor.
        # """
        # desc = CoulombMatrix(n_atoms_max=5, flatten=False)
        # cm = desc.create(H2O)

        # # Test against assumed values
        # q = H2O.get_atomic_numbers()
        # p = H2O.get_positions()
        # norm = np.linalg.norm
        # assumed = np.array(
            # [
                # [0.5*q[0]**2.4,              q[0]*q[1]/(norm(p[0]-p[1])),  q[0]*q[2]/(norm(p[0]-p[2]))],
                # [q[1]*q[0]/(norm(p[1]-p[0])), 0.5*q[1]**2.4,               q[1]*q[2]/(norm(p[1]-p[2]))],
                # [q[2]*q[0]/(norm(p[2]-p[0])), q[2]*q[1]/(norm(p[2]-p[1])), 0.5*q[2]**2.4],
            # ]
        # )
        # zeros = np.zeros((5, 5))
        # zeros[:3, :3] = assumed
        # assumed = zeros

        # self.assertTrue(np.array_equal(cm, assumed))


if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(GGridTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
