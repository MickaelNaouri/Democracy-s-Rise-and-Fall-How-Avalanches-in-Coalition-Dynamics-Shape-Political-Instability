import random
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class Tree:
    
    def __init__(self, key, children = []):
        self.key = key
        self.children = children
        
        if len(self.children) == 0:
            self.isleaf = True
            self.weight = 1
            
        else:
            self.isleaf = False
            self.weight = 0
        
            for child in self.children:
                self.weight += child.weight
                
    def __str__(self):
        return f'({self.key}: {len(self.children)})'

          
    def __repr__(self):
        return f'({self.key}: {len(self.children)})'
    
    
def division(forest, node):
    if forest[node].isleaf:
        return (False, forest)
    
    newForest = []
    
    for child in forest[node].children:
        newForest.append(child)
        
    for key, tree in enumerate(forest):
        if key != node:
            newForest.append(tree)
            
    return (True, newForest) 
            
    
def hierarchization(forest, node1, node2):
    newTree = Tree(-1, [forest[node1], forest[node2]])
    newForest = [newTree]
    
    for key, tree in enumerate(forest):
        if key == node1 or key == node2:
            continue
            
        newForest.append(tree)

    return newForest
        
    
def unification(forest, node1, node2):
    if forest[node1].isleaf or forest[node2].isleaf:
        return (False, forest)
    
    newTree = Tree(-1, forest[node1].children + forest[node2].children) 
    newForest = [newtree]
    
    for key, tree in enumerate(forest):
        if key == node1 or key == node2:
            continue
            
        newForest.append(tree)
  
    return (True, newforest)
            

def computeVote(forest, votes):
    
    outcome = 0
    for tree in forest:
        outcome += tree.weight * getTreeVote(votes, tree)
        
    if outcome == 0:
        return (random.random() > 0.5)*2 - 1
    
    if outcome >= 1:
        return 1
    
    else:
        return -1
    
def getTreeVote(votes, tree):
    
    if tree.isleaf:
        return votes[tree.key]
    
    outcome = 0
    for child in tree.children:
        outcome += child.weight * getTreeVote(votes, child)
        
    if outcome == 0:
        return (random.random() > 0.5)*2 - 1
    
    if outcome >= 1:
        return 1
    
    else:
        return -1

def directSample(nbVoters, forest, voter, choice, samples):
    result = 0
    for _ in range(samples):
        outcome = 0
        votes = [2*(random.random() > 0.5) - 1 for _ in range(nbVoters)]
        for key, tree in enumerate(forest):
            if key != voter:
                outcome += tree.weight * getTreeVote(votes, tree)

            else:
                outcome += tree.weight * choice
                
        if outcome == 0:
            if random.random() >= 0.5:
                result = 1

        if outcome >= 1:
            result += 1

        else:
            pass

    return result / samples


def computeVotingPower(nbVoters, forest, voter, samples):
    return directSample(nbVoters, forest, voter, 1, samples) - directSample(nbVoters, forest, voter, -1, samples)


def calculateAverageCoalitionDuration(forest):
    coalition_durations = []
    current_coalition = set()

    for i, tree in enumerate(forest):
        if tree.isleaf:
            current_coalition.add(tree.key)
        else:
            for child in tree.children:
                current_coalition.add(child.key)

        if i == len(forest) - 1 or len(current_coalition) > 1:
            coalition_durations.append(len(current_coalition))
            current_coalition.clear()

    return sum(coalition_durations) / len(coalition_durations)


def calculateAverageCoalitionSize(forest):
    coalition_sizes = []

    for tree in forest:
        coalition_size = tree.weight
        coalition_sizes.append(coalition_size)

    return sum(coalition_sizes) / len(coalition_sizes)


def forwardSimulation(nbVoters, initialForestState, samples, iterations):
    forest = initialForestState
    
    nbCoalitions = 0
    coalitions = [0]
    
    averageCoalitionWeight = [1]
    averageCoalitionComplexity = [0]
    averageCoalitionDuration = []
    averageCoalitionSize = []
    
    
    for _ in tqdm(range(iterations)):
        weights = np.array([tree.weight for tree in forest])
        weights = weights / np.sum(weights)
        r = random.random()
        
        if r < 1/2:
            ''' Division'''
            node = np.random.choice(list(range(len(forest))), size = 1, p = weights, replace = False)[0]
            oldVP = computeVotingPower(nbVoters, forest, node, samples)
            possible, newForest = division(forest, node)
            if not possible:
                continue
            else:
                newVP = 0
                for treeIndex in range(len(forest[node].children)):
                    newVP += computeVotingPower(nbVoters, newForest, treeIndex, samples)
                    
                newVP /= len(forest[node].children)
                
                if oldVP == 0 or random.random() < newVP / oldVP:
                    '''accept'''
                    nbMetaNode -= 1
                    forest = newForest
                
                else:
                    '''refuse'''
                    continue    
        
        elif r < 3/4:
            '''Hierarchization'''
            if len(forest) <= 1:
                continue
            node1, node2 =  np.random.choice(list(range(len(forest))), size = 2, p = weights, replace = False)
            oldVotingPower = 0.5 * computeVotingPower(nbVoters, forest, node1, samples) + 0.5 * computeVotingPower(nbVoters, forest, node2, samples) 
            newForest = hierarchization(forest, node1, node2)
            newVotingPower = computeVotingPower(nbVoters, forest, 0, samples)
            
            if oldVotingPower == 0 or random.random() < newVP / oldVP:
                '''accept'''
                nbCoalitions += 1
                forest = newForest
                
            else:
                '''refuse'''
                continue
            
        else:
            '''Unification'''
            if len(forest) <= 1:
                continue
            node1, node2 =  np.random.choice(list(range(len(forest))), size = 2, p = weights, replace = False)
            oldVP = 0.5 * computeVotingPower(nbVoters, forest, node1, samples) + 0.5 * computeVotingPower(nbVoters, forest, node2, samples) 
            possible, newForest = unification(forest, node1, node2)
            
            if not possible:
                continue
            else:
                newVotingPower = computeVotingPower(nbVoters, newForest, 0, samples)
                
                if oldVotingPower == 0 or random.random() < newVP / oldVP:
                    '''accept'''
                    nbCoalitions -= 1
                    forest = newForest
                
                else:
                    '''refuse'''
                    continue
                    
        coalitions.append(nbCoalitions)
        
        averageCoalitionDuration.append(calculateAverageCoalitionDuration(forest))
        averageCoalitionSize.append(calculateAverageCoalitionSize(forest))
        averageCoalitionWeight.append(np.mean([tree.weight for tree in forest]))
        averageCoalitionComplexity.append(nbMetaNode / len(forest))
        
        
    return forest, coalitions, averageCoalitionWeight, averageCoalitionComplexity, averageCoalitionDuration, averageCoalitionSize



N = 100
initialForest = []

for i in range(N):
    initialForest.append(Tree(i))

forest, coalitions,averageCoalitionWeight, averageCoalitionComplexity, averageCoalitionDuration, averageCoalitionSize = forwardSimulation(N, initialForest, 100, 20000)
