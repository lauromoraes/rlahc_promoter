#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:54:31 2018

@author: fnord
"""

from Neighbour import Neighbour
import random
from bitstring import BitArray
import copy

class RNLAHC(object):
    def __init__(self, max_iterations=10, hl=4):
        self.max_iterations = max_iterations
        self.running_flag = False
        self.current_iteration = 0
        self.last_update_iteration = 0
        self.best_objective_function = -1.*float('inf')
        self.improved = False
        self.hl=hl

    def perform_search(self, solution):
        self.running_flag = True
        self.improved = True
        self.restricted_set = set()
        # Define best solution
        best_solution_representation = copy.deepcopy(solution.representation)
        # Setup objective function
        best_objective_function = solution.objective_function
        # Create historic list
        self.historic_list = [ best_objective_function for _ in range(self.hl) ]
        # Number of possibles positions
        N = len(solution.representation)
        # Tabu list
        self.tabu = [ 0 for _ in range(N) ]
        # Iterators counter
        I = 0
        idle = 0
        fos = []
        sols = []
        freeze_time = 7
        # while I < self.max_iterations or idle < I*0.1:
        while I < self.max_iterations:
            # Setup auxiliary objective function - to restore value without evaluation
            aux_objective_function = solution.objective_function

            fos.append(best_objective_function)

            # Create neighbour that satisfy restric set
            flag = True
            while flag and I+len(self.restricted_set) < self.max_iterations:
                # Setup bit position
                neighbour_bit_position = random.randint(0, N-1)
                if I >= self.tabu[neighbour_bit_position]:
                    self.tabu[neighbour_bit_position] = I+freeze_time
                    # Apply movement
                    solution.swap_bit(neighbour_bit_position)
                    # Get integer representation
                    int_repr = BitArray(solution.representation).uint
                    # Verify restriction
                    if int_repr in self.restricted_set:
                        # Undo movement
                        solution.swap_bit(neighbour_bit_position)
                    else:
                        self.restricted_set.add(int_repr)
                        flag = False
                else:
                    print('TABU')

            # Evaluate solution after movement
            neighbour_objective_function = solution.eval()

            sols.append(neighbour_objective_function)

            print('Neighbour FO {} | Best FO: {}'.format(neighbour_objective_function, best_objective_function))

            if neighbour_objective_function <= aux_objective_function:
                idle+=1
            else:
                idle = 0

            if neighbour_objective_function > best_objective_function or (neighbour_objective_function == best_objective_function and sum(solution.representation) < sum(best_solution_representation)):
                # Update best solution
                del best_solution_representation
                best_solution_representation = copy.deepcopy(solution.representation)
                best_objective_function = neighbour_objective_function

            # Calculate actual position on historical list
            v = I % self.hl

            if solution.objective_function > self.historic_list[v] and neighbour_objective_function < aux_objective_function:
                print('\n>>> Accepted worse: neighbour {} | historical {} | actual {}\n'.format(solution.objective_function, self.historic_list[v], aux_objective_function) )

            # If worse than historical and current solution
            if neighbour_objective_function <= self.historic_list[v] and neighbour_objective_function < aux_objective_function:
                # Apply movement
                solution.swap_bit(neighbour_bit_position)
                # Restore objective function
                solution.objective_function = aux_objective_function
                

            if solution.objective_function > self.historic_list[v]:
                self.historic_list[v] = solution.objective_function

            # Increment I
            I+=1

        solution.representation = best_solution_representation
        solution.objective_function = best_objective_function
        print(solution)
        return solution, fos, sols