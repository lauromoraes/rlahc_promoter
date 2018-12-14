#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:57:09 2018

@author: fnord
"""

from Solution import Solution
from Neighbour import Neighbour
from FirstImprovementNeighbour import FirstImprovementNeighbour
from BestImprovementNeighbour import BestImprovementNeighbour
from RNLAHC import RNLAHC

def parse_agrs():
    import argparse
    desc = "Metaheuristics for Attibute Selection in Promoter Region Classifier Problem."
    parser = argparse.ArgumentParser(description=desc)
    
    # Experiment
    parser.add_argument('-o', '--organism', default='Bacillus',
                        help="The organism used for test. Generate auto path for fasta files. Should be specified when testing")
    parser.add_argument('--partitions', default=1, type=int,
                        help="Number of partitions for Cross-Validation test.")
    # Model
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=70, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--loss_type', default=0, type=int)
    parser.add_argument('--optimizer_type', default=0, type=int)
    # Utils        
    parser.add_argument('--save_dir', default='./results')
    parser.add_argument('--debug', default=1, type=int)  # debug>0 will save weights by TensorBoard
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    import time
    import os
    import shutil
    import datetime
    import glob
    import numpy as np
    args = parse_agrs()

    HLs = [1, 3, 5, 10]

    for hl in HLs:
        
        values = []
        FOS=[]
        SOLS=[]
        for i in range(10):
            print('Testing HL {} - Test {}'.format(hl, i))

            solution = Solution(args, solution_length=80)
            # solution.set_random_solution()
            solution.set_full_solution()
            solution.eval()
            print('initial solution')
            print(solution)
            base_mcc = solution.objective_function
            
            neighbour = RNLAHC(max_iterations=50, hl=hl)
            # neighbour = BestImprovementNeighbour(max_iterations=0)
            start = time.time()
            solution, fos, sols = neighbour.perform_search(solution)

            done = time.time()
            elapsed = done - start
            print('Elapsed:', str(datetime.timedelta(seconds=elapsed)))

            print(solution)
            print(base_mcc, solution.objective_function)

            
            filelist=glob.glob(args.save_dir+"/*.h5")
            for file in filelist:
                os.remove(file)

            values.append((i, base_mcc, solution.objective_function, elapsed))
            FOS.append(fos)
            SOLS.append(sols)

        T = np.median(SOLS, axis=0)
        np.savetxt('sols-{}.csv'.format(hl), np.array(T), delimiter=';')

        T = np.median(FOS, axis=0)
        np.savetxt('fos-{}.csv'.format(hl), np.array(T), delimiter=';')

        f = open('hl-{}.csv'.format(hl), 'w')
        for j in range(len(values[0])):
            line = ';'.join([ str(v[j]) for v in values ])
            f.write(line+'\n')
        f.close()

    print('< END >')