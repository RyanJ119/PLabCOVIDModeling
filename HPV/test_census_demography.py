import unittest
import csv
from pathlib import Path
import numpy as np
from census_demography import age_counts_on_grid, birth_inflow
from hpv_pde import HPVModel, Parameters

DATA = Path(__file__).resolve().parent/'census_data'/'us_2025_age_counts.csv'

class CensusTests(unittest.TestCase):
    def test_exact_single_year_counts(self):
        with DATA.open() as f:
            counts = np.array([float(r['population']) for r in csv.DictReader(f)])
        np.testing.assert_array_equal(age_counts_on_grid(DATA,100,100),counts)
        self.assertEqual(counts.sum(),341665675)

    def test_remapping_conserves_counts(self):
        for n in (20,137,200):
            self.assertAlmostEqual(age_counts_on_grid(DATA,100,n).sum(),341665675,places=5)
        self.assertAlmostEqual(age_counts_on_grid(DATA,1,2).sum(),3618898)
        with self.assertRaises(ValueError):
            age_counts_on_grid(DATA,110,110)

    def test_birth_scaling(self):
        rate=birth_inflow(DATA,3620461,1,100,100)
        self.assertAlmostEqual(rate,0.01059650197521305)
        self.assertAlmostEqual(birth_inflow(DATA,3620461,2,100,100),2*rate)

    def test_initial_profile_and_population_balance(self):
        m=HPVModel(Parameters(initial_age_data_file=str(DATA),
            newborn_rate=birth_inflow(DATA,3620461,1,100,100)))
        initial=m.initial_state()
        np.testing.assert_allclose(initial.sum(axis=(0,2))*m.da*m.dx,
            age_counts_on_grid(DATA,100,100)/341665675)
        np.testing.assert_allclose(initial.sum(axis=(1,2))*m.da*m.dx,
                                  m.par.initial_group_fractions)
        t,p,s=m.simulate(2)
        self.assertGreaterEqual(p.min(),0)
        np.testing.assert_allclose(s[:,1]+s[:,3]+s[:,4]+s[:,18],s[0,1]+s[:,5],atol=1e-12)

if __name__=='__main__':
    unittest.main()
