#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:54:31 2018

@author: fnord
"""

import numpy as np
import random
from MLModel import MLModel

class Solution(object):

    def __init__(self, args, solution_length=0):
        self.solution_length = solution_length
        self.objective_function = -1.*float('inf')
        self.representation = None
        self.args = args
        self.model = MLModel(args)

    def __str__(self):
        msg = '{of} - {rep}'.format(of=round(self.objective_function,4), rep=self.representation)
        return msg

    def set_random_solution(self):
        self.representation = self.gen_random_solution()

    def set_full_solution(self):
        self.representation = self.gen_full_solution()

    def gen_random_solution(self):
        return np.random.randint(2, size=self.solution_length)

    def gen_full_solution(self):
        return np.zeros(self.solution_length, dtype='int')+1

    def swap_bit(self, position):
        self.representation[position] = 1 - self.representation[position]

    def eval(self):
        mcc = self.model.eval(mask=self.representation)
        mcc = round(mcc, 4)
        self.objective_function = mcc
        print('mcc', mcc)
        return mcc