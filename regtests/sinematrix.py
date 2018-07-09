import math
import numpy as np
import unittest

from describe.descriptors import SineMatrix

from ase import Atoms


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


class SineMatrixTests(unittest.TestCase):

    def test_constructor(self):
        """Tests different valid and invalid constructor values.
        """
        with self.assertRaises(ValueError):
            SineMatrix(n_atoms_max=5, permutation="unknown")
        with self.assertRaises(ValueError):
            SineMatrix(n_atoms_max=-1)

    def test_number_of_features(self):
        """Tests that the reported number of features is correct.
        """
        desc = SineMatrix(n_atoms_max=5, permutation="none", flatten=False)
        n_features = desc.get_number_of_features()
        self.assertEqual(n_features, 25)

    def test_flatten(self):
        """Tests the flattening.
        """
        # Unflattened
        desc = SineMatrix(n_atoms_max=5, permutation="none", flatten=False)
        cm = desc.create(H2O)
        self.assertEqual(cm.shape, (5, 5))

        # Flattened
        desc = SineMatrix(n_atoms_max=5, permutation="none", flatten=True)
        cm = desc.create(H2O)
        self.assertEqual(cm.shape, (25,))

    def test_features(self):
        """Tests that the correct features are present in the desciptor.
        """
        desc = SineMatrix(n_atoms_max=2, permutation="none", flatten=False)

        # Test that without cell the matrix cannot be calculated
        system = Atoms(
            positions=[[0, 0, 0], [1.0, 1.0, 1.0]],
            symbols=["H", "H"],
        )
        with self.assertRaises(ValueError):
            desc.create(system)

        # Test that periodic boundaries are considered by seeing that an atom
        # in the origin is replicated to the  corners
        system = Atoms(
            cell=[
                [10, 10, 0],
                [0, 10, 0],
                [0, 0, 10],
            ],
            scaled_positions=[[0, 0, 0], [1.0, 1.0, 1.0]],
            symbols=["H", "H"],
            pbc=True,
        )
        # from ase.visualize import view
        # view(system)
        matrix = desc.create(system)

        # The interaction between atoms 1 and 2 should be infinite due to
        # periodic boundaries.
        self.assertEqual(matrix[0, 1], float("Inf"))

        # The interaction of an atom with itself is always 0.5*Z**2.4
        atomic_numbers = system.get_atomic_numbers()
        for i, i_diag in enumerate(np.diag(matrix)):
            self.assertEqual(i_diag, 0.5*atomic_numbers[i]**2.4)

    # def test_visual(self):
        # import matplotlib.pyplot as mpl
        # """Plot the
        # """
        # test_sys = Atoms(
            # cell=[[1, 0.0, 0.0], [1, 1, 0.0], [0.0, 0.0, 1.0]],
            # positions=[[0, 0, 0], [2, 1, 1]],
            # symbols=["H", "H"],
        # )
        # test_sys.charges = np.array([1, 1])

        # desc = SineMatrix(n_atoms_max=5, flatten=False)

        # # Create a graph of the interaction in a 2D slice
        # size = 100
        # x_min = 0.0
        # x_max = 3
        # y_min = 0.0
        # y_max = 3
        # x_axis = np.linspace(x_min, x_max, size)
        # y_axis = np.linspace(y_min, y_max, size)
        # interaction = np.empty((size, size))
        # for i, x in enumerate(x_axis):
            # for j, y in enumerate(y_axis):
                # temp_sys = Atoms(
                    # cell=[[1, 0.0, 0.0], [1, 1, 0.0], [0.0, 0.0, 1.0]],
                    # positions=[[0, 0, 0], [x, y, 0]],
                    # symbols=["H", "H"],
                # )
                # temp_sys.set_initial_charges(np.array([1, 1]))
                # value = desc.create(temp_sys)
                # interaction[i, j] = value[0, 1]

        # mpl.imshow(interaction, cmap='RdBu', vmin=0, vmax=100,
                # extent=[x_min, x_max, y_min, y_max],
                # interpolation='nearest', origin='lower')
        # mpl.colorbar()
        # mpl.show()

if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(SineMatrixTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
