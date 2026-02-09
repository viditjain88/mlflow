import unittest
from hmdss.core.fusion import reciprocal_rank_fusion

class TestFusion(unittest.TestCase):
    def test_rrf(self):
        list1 = [
            {"content": "A", "score": 10},
            {"content": "B", "score": 9},
            {"content": "C", "score": 8}
        ]
        list2 = [
            {"content": "B", "score": 100}, # B is top here
            {"content": "D", "score": 50},
            {"content": "A", "score": 10}
        ]

        # k=60
        # A: 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 ~ 0.0322
        # B: 1/(60+2) + 1/(60+1) = 0.01612 + 0.01639 ~ 0.0325
        # C: 1/(60+3) = 0.01587
        # D: 1/(60+2) = 0.01612

        # Expected order: B, A, D, C (roughly)

        fused = reciprocal_rank_fusion([list1, list2], k=60)
        print("Fused Scores:", [d['rrf_score'] for d in fused])
        print("Fused Contents:", [d['content'] for d in fused])

        self.assertEqual(fused[0]['content'], "B")
        self.assertEqual(fused[1]['content'], "A")
        self.assertEqual(len(fused), 4)

if __name__ == '__main__':
    unittest.main()
