"""Verification for cohort bookkeeping and controlled scenario comparisons."""
import unittest
import numpy as np
from hpv_pde import HPVModel, Parameters
from scenario_comparison import run_scenarios


class AdditionalFigureTests(unittest.TestCase):
    def test_tag_does_not_change_total_solution(self):
        p = Parameters(n_age=20, n_x=20, vaccination_portion_per_year=(.05,)*5)
        tagged = HPVModel(p)
        time, density, summary = tagged.simulate(3, track_initial=True)
        _, plain, plain_summary = HPVModel(p).simulate(3)
        np.testing.assert_array_equal(density, plain)
        np.testing.assert_array_equal(summary, plain_summary)
        age_total = density.sum(axis=(1, 3))*tagged.dx
        np.testing.assert_allclose(tagged.initial_cohort_age_density[0], age_total[0])
        self.assertTrue((tagged.initial_cohort_age_density <= age_total+1e-12).all())
        self.assertGreaterEqual(tagged.initial_cohort_age_density.min(), 0)
        # Complement integrates to exactly the later-born tracked population.
        split = tagged.initial_cohort_age_density.sum(axis=1)*tagged.da
        later = (age_total-tagged.initial_cohort_age_density).sum(axis=1)*tagged.da
        np.testing.assert_allclose(split+later, summary[:, 1])

    def test_no_births_tag_equals_total(self):
        m = HPVModel(Parameters(n_age=20, n_x=20, newborn_rate=0))
        _, density, _ = m.simulate(3, track_initial=True)
        np.testing.assert_allclose(m.initial_cohort_age_density,
                                  density.sum(axis=(1, 3))*m.dx)

    def test_scenarios_same_start_and_balance(self):
        p = Parameters(n_age=20, n_x=20, vaccination_portion_per_year=(.05,)*5)
        time, _, summary = HPVModel(p).simulate(3)
        results = run_scenarios(p, 3, 1,
            {'No new vaccination': (0,)*5, 'Moderate': (.05,)*5, 'High': (.5,)*5},
            time, summary)
        self.assertIs(results[1]['summary'], summary)
        for item in results:
            rows = item['summary']
            np.testing.assert_array_equal(rows[0], summary[0])
            np.testing.assert_allclose(rows[:, 1]+rows[:, 3]+rows[:, 4]+rows[:, 18],
                                       rows[0, 1]+rows[:, 5], atol=1e-12)
            rates = np.diff(rows[:, 5]-rows[:, 3]-rows[:, 4]-rows[:, 18])/np.diff(time)
            np.testing.assert_allclose(rates, np.diff(rows[:, 1])/np.diff(time), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
