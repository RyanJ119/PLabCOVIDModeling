import unittest
import numpy as np
from hpv_pde import HPVModel, Parameters


class SolverTests(unittest.TestCase):
    def test_population_balance_and_positivity(self):
        model = HPVModel(Parameters(n_age=15, n_x=20, beta_peak=1,
            vaccination_portion_per_year=(0.3, 0.2, 0.1, 0.1, 0.1), newborn_rate=0.01))
        _, states, rows = model.simulate(3, 0.5)
        self.assertGreaterEqual(states.min(), -1e-14)
        np.testing.assert_allclose(rows[:, 1]+rows[:, 3]+rows[:, 4], 1+rows[:, 5], atol=1e-12)
        self.assertGreater(rows[-1, 3], 0)

    def test_exact_one_cell_transport(self):
        # da=1; zero disease velocity; dt=1 gives exactly one age-cell shift.
        model = HPVModel(Parameters(max_age=5, n_age=5, n_x=4,
            beta_peak=0, clearance=(0,)*6, cfl=1))
        state = np.zeros((6, 5, 4))
        state[5, 1, 2] = 1
        updated, dt, cancer, _ = model.step(state, 1)
        expected = np.zeros_like(state)
        expected[5, 2, 2] = 1
        np.testing.assert_allclose(updated, expected)
        self.assertEqual(dt, 1)
        self.assertEqual(cancer.sum(), 0)

    def test_vaccination_source_and_healthy_retention(self):
        model = HPVModel(Parameters(n_age=5, n_x=10, beta_peak=0,
            vaccination_portion_per_year=(0.2, 0, 0, 0, 0)))
        state = np.zeros((6, 5, 10))
        state[5, 0, 0] = 1
        updated, dt, cancer, age_loss = model.step(state, 0.01)
        self.assertAlmostEqual(updated[0].sum(), 0.2*dt)
        self.assertAlmostEqual(updated.sum(), 1)
        self.assertEqual(cancer.sum(), 0)
        self.assertEqual(age_loss, 0)

    def test_burden_is_not_normalized(self):
        model = HPVModel()
        state = model.initial_state()
        self.assertAlmostEqual(model.burden(2*state), 2*model.burden(state))


if __name__ == '__main__':
    unittest.main()
