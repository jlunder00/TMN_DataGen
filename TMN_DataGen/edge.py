""" edge.py

Definition of the Edge object, which is a connection between two nodes

"""
#this may not be necessary anymore

import sys, os, re, json, typing


try:
    pass
    # local imports with . before
    from .node import Node
except ImportError:
    pass
    # local imports without . before
    from node import Node


class Edge():
    """
    This object represents an edge in between two nodes in a graph
    """

    def __init__(self, node: Node, data: dict = {}, opt: dict = {}):
        """
        Create edge.
        Nodes have a list of edges as their child nodes. 
        also has  wata (dictionary)
        """
        self.node = node
        self.data = data
        self.opt = opt

    def reassign(self, newNode):
        self.node = newNode

    def updateData(self, key, value):
        self.data[key] = value

    def getData(self, key):
        return self.data[key]
