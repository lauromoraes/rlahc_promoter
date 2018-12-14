#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:54:31 2018

@author: fnord
"""

def LocalSearch(object):
    def __init__(self, neighbour_structure):
        self.neighbour_structure = neighbour_structure

    def search(self, solution):
        pass

def FirstImprovementSearch(LocalSearch):
    def __init__(self):
        neighbour_structure = FirstImprovementNeighbour()
        super(FirstImprovement, self).__init__(neighbour_structure)