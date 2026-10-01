import unittest
from capacity import select_candidate


class SelectionChecks(unittest.TestCase):
    def row(self,n,rate,mean,dataset='original'):
        return dict(leaves=n,completion_rate=rate,mean_return=mean,dataset=dataset)
    def test_smallest_qualifying_before_best_score(self):
        rows=[self.row(4,.8,480),self.row(8,.98,498),self.row(16,1,500)]
        selected,qualified=select_candidate(rows)
        self.assertTrue(qualified);self.assertEqual(selected['leaves'],8)
    def test_ties_and_fallback(self):
        rows=[self.row(8,.95,490),self.row(8,.95,499,'augmented')]
        self.assertEqual(select_candidate(rows)[0]['dataset'],'augmented')
        chosen,qualified=select_candidate([self.row(4,.7,490),self.row(8,.8,480)])
        self.assertFalse(qualified);self.assertEqual(chosen['leaves'],8)


if __name__=='__main__': unittest.main()
