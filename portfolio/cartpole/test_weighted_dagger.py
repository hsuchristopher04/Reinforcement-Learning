import unittest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import weighted_dagger as w


class WeightingChecks(unittest.TestCase):
    def test_zero_excludes_new_labels_and_one_matches_uniform(self):
        rng=np.random.default_rng(2026)
        x=rng.normal(size=(300,4)).astype(np.float32); y=(x[:,2]>0).astype(int)
        nx=rng.normal(size=(500,4)).astype(np.float32); ny=(nx[:,0]>0).astype(int)
        _,zero=w.fit_weighted(x,y,nx,ny,0)
        _,one=w.fit_weighted(x,y,nx,ny,1)
        baseline=DecisionTreeClassifier(max_depth=2,random_state=2026).fit(x,y)
        uniform=DecisionTreeClassifier(max_depth=2,random_state=2026).fit(np.r_[x,nx],np.r_[y,ny])
        self.assertEqual(zero,w.d.export_model(baseline))
        self.assertEqual(one,w.d.export_model(uniform))

    def test_selection_excludes_zero_control_and_breaks_ties(self):
        rows=[dict(weight=0.,completion_rate=1.,mean_return=500),
              dict(weight=.1,completion_rate=.7,mean_return=480),
              dict(weight=.25,completion_rate=.7,mean_return=480),
              dict(weight=1.,completion_rate=0.,mean_return=12)]
        self.assertEqual(w.choose_weight(rows)['weight'],.1)


if __name__=='__main__': unittest.main()
