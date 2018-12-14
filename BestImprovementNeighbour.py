#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:54:31 2018

@author: fnord
"""

from Neighbour import Neighbour

class BestImprovementNeighbour(Neighbour):
    def __init__(self, max_iterations=0):
        super(BestImprovementNeighbour, self).__init__(max_iterations)

    def perform_search(self, solution):
        self.running_flag = True
        self.improved = True
        while self.running_flag and self.improved:
            neighbour_objective_function, neighbour_position = self.find_neighbour(solution)
            solution = self.update_solution(solution, neighbour_objective_function, neighbour_position)
            # print('improved', self.improved)
        return solution

    def running_condition(self):
        if self.iteration_stop_condition():
            return False
        return True