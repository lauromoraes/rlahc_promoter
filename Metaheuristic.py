#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:57:09 2018

@author: fnord
"""

from Solution import Solution

class Metaheuristic(object):
    def __init__(self, max_iterations=100):
        self.star_solution = None
        self.current_solution = None
        self.stop_flag = False
        self.current_iteration = 0
        self.last_update_iteration = 0
        self.max_iterations = max_iterations

    def set_star_solution(self, solution):
        self.star_solution = solution

    def set_current_solution(self, solution):
        self.current_solution = solution

    def get_star_objective_function(self):
        return self.star_solution.objective_function

    def get_current_objective_function(self):
        return self.current_solution.objective_function

    def accept_solution(self, solution):
        if self.get_star_objective_function() > solution.objective_function:
            self.star_solution = solution
            self.last_update_iteration = self.current_iteration
            return True
        return False

    def running_condition(self):
        if self.current_iteration - self.last_update_iteration >= self.max_iterations:
            self.running_flag = False
        return self.running_flag

    def run(self):
        pass