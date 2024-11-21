""" edge.py

Definition of the Edge object, which is a connection between two nodes

"""

import sys, os, re, json


try:
    pass
    #local imports with . before
except:
    pass
    #local imports without . before



class Edge():
    """
    This object represents an edge in between two nodes in a graph
    """

    def __init__(self, nodeA, nodeB, data={}, opt={}):
        """
        Create edge. Edge has 2 nodes and data (dictionary)
        """
        self.nodeA = nodeA
        self.nodeB = nodeB
