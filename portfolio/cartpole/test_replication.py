import unittest
from replication import summarize, fingerprint, COLLECTION_SEEDS, EVAL_SEED


class ReplicationChecks(unittest.TestCase):
    def test_dataset_means_are_replication_units(self):
        rows=[dict(dataset=1,completion_rate=1,mean_return=500,fingerprint='a')]*3
        rows += [dict(dataset=2,completion_rate=0,mean_return=10,fingerprint='b')]
        result=summarize(rows)
        self.assertEqual(result['mean_dataset_completion'],.5)
        self.assertEqual(result['distinct_trees'],2)
        self.assertEqual(result['datasets'][0]['distinct_trees'],1)
    def test_seed_ranges_disjoint(self):
        ranges=[set(range(s,s+200)) for s in COLLECTION_SEEDS]+[set(range(EVAL_SEED,EVAL_SEED+500))]
        for i,a in enumerate(ranges):
            for b in ranges[i+1:]:self.assertFalse(a&b)
    def test_fingerprint_is_order_invariant_but_value_sensitive(self):
        self.assertEqual(fingerprint({'a':1,'b':2}),fingerprint({'b':2,'a':1}))
        self.assertNotEqual(fingerprint({'a':1}),fingerprint({'a':2}))


if __name__=='__main__':unittest.main()
