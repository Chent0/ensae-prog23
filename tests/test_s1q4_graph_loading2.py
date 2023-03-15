import sys 
sys.path.append("delivery_network/")

import unittest 
from graph import Graph, graph_from_file

class Test_GraphLoadingv2(unittest.TestCase):
    def test_network4(self):
        g = graph_from_file("input/network.04.in")
        self.assertEqual(g.nb_nodes, 10)
        self.assertEqual(g.nb_edges, 4)
        self.assertEqual(g.graph[3][0][1],4)

if __name__ == '__main__':
    unittest.main()
    