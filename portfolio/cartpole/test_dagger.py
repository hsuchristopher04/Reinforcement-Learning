import unittest
from unittest.mock import patch
import numpy as np
import dagger


class AggregationChecks(unittest.TestCase):
    def test_teacher_labels_but_learner_drives(self):
        class Env:
            x_threshold=2.4
            theta_threshold_radians=.21
            def __init__(self): self.unwrapped=self; self.actions=[]
            def reset(self,seed): self.i=0; return np.zeros(4,dtype=np.float32),{}
            def step(self,action):
                self.actions.append(action); self.i+=1
                return np.array([0,0,.01*self.i,0],dtype=np.float32),1,False,self.i==3,{}
            def close(self): pass
        env=Env()
        with patch.object(dagger.c.gym,'make',return_value=env), \
             patch.object(dagger.c,'select_action',return_value=1), \
             patch.object(dagger.d,'tree_action',return_value=0):
            batch,outcomes=dagger.collect_labeled(None,{},2,11001000)
        self.assertEqual(env.actions,[0]*6)
        np.testing.assert_array_equal(batch['teacher_actions'],[1]*6)
        np.testing.assert_array_equal(batch['executed_actions'],[0]*6)
        np.testing.assert_array_equal(batch['steps'],[0,1,2,0,1,2])
        np.testing.assert_array_equal(batch['reset_seeds'],[11001000]*3+[11001001]*3)
        self.assertEqual(len(outcomes),2)

    def test_selection_uses_completion_before_mean(self):
        rows=[dict(round=1,completion_rate=.9,mean_return=470),
              dict(round=2,completion_rate=.8,mean_return=490),
              dict(round=3,completion_rate=.9,mean_return=475),
              dict(round=4,completion_rate=.9,mean_return=475)]
        self.assertEqual(dagger.choose_round(rows)['round'],3)


if __name__=='__main__': unittest.main()
