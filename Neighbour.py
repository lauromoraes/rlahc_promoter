#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:54:31 2018

@author: fnord
"""

import random

class Neighbour(object):
    def __init__(self, max_iterations=0):
        self.max_iterations = max_iterations
        self.running_flag = False
        self.current_iteration = 0
        self.last_update_iteration = 0
        self.best_objective_function = -1.*float('inf')
        self.improved = False

    def running_condition(self):
        raise NotImplementedError('Please implement this method')

    def iteration_stop_condition(self):
        if self.max_iterations and self.current_iteration - self.last_update_iteration >= self.max_iterations:
            self.running_flag = False
            return True
        return False

    def update_solution(self, solution, neighbour_objective_function, neighbour_position):
        print(neighbour_objective_function)
        if self.best_objective_function < neighbour_objective_function:
            print('Updated: {} to {}'.format(self.best_objective_function, neighbour_objective_function))
            solution.objective_function = neighbour_objective_function
            solution.swap_bit(neighbour_position)
            self.best_objective_function = neighbour_objective_function
            self.improved = True
        else:
            self.improved = False
        return solution

    def find_neighbour(self, solution):
        # Setup objective function
        best_objective_function = solution.objective_function
        # Number of possibles positions
        N = len(solution.representation)
        # Create shuffled list with all possible positions
        positions = range(0,N)
        random.shuffle(positions)
        # Setup best position
        best_position = positions[0]
        cnt=0
        # For each position
        for i in range(N):
            # Update iterator counter
            self.current_iteration += 1
            cnt+=1
            # Setup current position
            current_position = positions[i]
            # Setup auxiliary objective function - to restore value without evaluation
            aux_objective_function = solution.objective_function
            # Apply movement
            solution.swap_bit(current_position)
            # print('Testing:', solution.representation)
            # Evaluate solution after movement
            objective_function = solution.eval()
            print(cnt, current_position, 'neighbour objective_function', objective_function, best_objective_function)
            # Update best objetive function and best position
            if best_objective_function < objective_function:
                # print('new best neigbhour', best_objective_function, objective_function)
                best_position = current_position
                best_objective_function = objective_function
                self.last_update_iteration = self.current_iteration
            # Undo movement
            solution.swap_bit(current_position)
            # Restore objective function
            solution.objective_function = aux_objective_function
            # Verify stop condition
            if not self.running_condition():
                break
        return best_objective_function, best_position