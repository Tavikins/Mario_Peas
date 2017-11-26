#!/usr/bin/env python

### IMPORTS ###
import sys, os
import multiprocessing as mp
from functools import partial
import pickle
import numpy as np


sys.path.append(os.path.join(os.path.split(__file__)[0],'..','..')) 
from peas.methods.neat import NEATPopulation, NEATGenotype

from peas.networks.rnn import NeuralNetwork
# from peas.methods.neatpythonwrapper import NEATPythonPopulation
import gym
import gym_pull

SAVE_POP = False
LOAD_POP = False


env = gym.make('meta-SuperMarioBros-Tiles-v0')




# Create a population
genotype = lambda: NEATGenotype(inputs=208, 
                                outputs=6,
                                weight_range=(-50., 50.), 
                                types=['tanh'])

pop = NEATPopulation(genotype, popsize=150)

if LOAD_POP:
    with open('pop.p', 'rb') as pf:
        pop = pickle.load(pf)
    
def ui(i):
    return np.unravel_index(i,(2,2,2,2,2,2))

class MarioTask(object):
    
    def __init__(self, save = True, load = False):
        self.save = save
        self.load = load
        self.Valid_Inputs = [i for i in range(64) if (ui(i)[0] + ui(i)[1] + ui(i)[2] + ui(i)[3]) <= 1]
        self.starting_state = env.reset()
        
        
    def _loop(self, network):
        
        steps  = 0
        done = False
        state = self.starting_state
        while not done:
            steps += 1
            net_input = state.flatten()
            actions = network.feed(net_input)[-7:-1]
            action = [0 if actions[i] < 0.5 else 1 for i in range(actions.size)]
            state, reward, done, info = env.step(action)
            if done or type(state)==type(None) or float(info['distance'])/float(steps) <= 0.5:
                env.close()
                env.reset()
                break
        distance = info['distance']
        

        
        return steps, distance
        
        
    def evaluate(self, network):
        """ Perform a single run of this task """
        # Convert to a network if it is not.

        if not isinstance(network, NeuralNetwork):
            network = NeuralNetwork(network)
            
        if self.save:
            with open('pop.p', 'wb') as pf:
                pickle.dump(pop, pf)
            
        steps, distance = self._loop(network)
        
        return {'fitness': distance, 'steps': steps}
        
        
    def solve(self, network):
        """ This function should measure whether the network passes some
            threshold of "solving" the task. Returns False/0 if the 
            network 'fails' the task. 
        """
        # Convert to a network if it is not.
        if not isinstance(network, NeuralNetwork):
            network = NeuralNetwork(network)
        
        steps, distance = self._loop(network)
        if distance < 3200:
            print "Failed test: %d" % distance
            return 0
        return 1
        
        
        
        
# Create a factory for genotypes (i.e. a function that returns a new 
# instance each time it is called)

    
# Create a task
task = MarioTask(SAVE_POP,LOAD_POP)

# Run the evolution, tell it to use the task as an evaluator
pop.epoch(generations=100, evaluator=task, solution=task)