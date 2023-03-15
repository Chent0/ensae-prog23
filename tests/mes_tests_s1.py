import sys 
sys.path.append("delivery_network/")

import unittest 
from graph import Graph, graph_from_file

'''Test de get_path_with_power'''

class Test_Reachability(unittest.TestCase):
    def test_network1(self):
        g = graph_from_file("input/network.01.in")
        self.assertEqual(g.get_path_with_power(4, 7, 2), [4, 5, 7])
        self.assertEqual(g.get_path_with_power(1, 5, 10), None)

    def test_network3(self):
        g = graph_from_file("input/network.03.in")
        self.assertIn(g.get_path_with_power(1, 4, 11), [[1, 4], [1, 2, 3, 4]])
        self.assertEqual(g.get_path_with_power(1, 3, 9), None)

'''Test de min_power'''

class Test_MinimalPower(unittest.TestCase):
    def test_network1(self):
        g = graph_from_file("input/network.01.in")
        self.assertEqual(g.min_power(4, 7)[1], 1)
        self.assertEqual(g.min_power(1, 3)[1], 1)

    def test_network2(self):
        g = graph_from_file("input/network.02.in")
        self.assertEqual(g.min_power(1, 3)[1], 4)
        self.assertEqual(g.min_power(1, 2)[1], 4)

if __name__ == '__main__':
    unittest.main()

