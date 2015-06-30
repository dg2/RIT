'''
Random Intersection Trees

http://jmlr.org/papers/volume15/shah14a/shah14a.pdf
'''
import numpy as np
import pandas as pd
from collections import defaultdict


class Node():
    '''
    A node in an intersection tree
    '''
    def __init__(self, parent):
        if parent is None:
            self.active_set = None
            self.depth = 0
        else:
            self.active_set = parent.active_set
            self.depth = parent.depth + 1

    def activate(self, data):
        if self.active_set is None:
            self.active_set = frozenset(data)
        else:
            self.active_set = self.active_set.intersection(frozenset(data))

    def contrast(self, negative_example):
        if self.active_set.issubset(negative_example):
            self.active_set = set()
            
    def is_terminal(self):
        if self.active_set == set():
            return True
        return False

    def get_depth(self):
        return self.depth


class IntersectionTree():
    '''
    Single Intersection Tree

    TODO: Can be easily turned into an online method
    by just assigning a new data point to a new active
    node, i.e. a children of an already existing node

    The standard way to convert a RIT into a binary
    classification method is to compute the (approximate)
    probability of a particular interaction set existing
    in the opposite class, and stops that process whenevar
    that probability exceeds some threshold (since very
    subset will have at least that same probability)

    I believe that this can also be extended to a two-class
    setting by having two types of nodes: inclusion and
    exclusion nodes. Inclusion nodes are the standard ones,
    which take the union of its parent active set and a
    random data point. Exclusion nodes takes set differences
    between the power sets of the parent active set and a
    data point from the opposing class. Or probably the power
    set thing is overkill and we should take standard set
    differences
    '''
    def __init__(self, depth, branching, neg_sampling=10):
        '''
        Create a intersection tree of given depth and branching factor
        (i.e., number of children nodes per parent)
        '''
        self.depth = depth
        self.branching = branching
        self.nodes = []
        self.layers = []
        self.active_nodes = []
        self._counter = 0
        self.leaf_nodes = set()
        self.one_class = None
        self.neg_sampling = neg_sampling

    def _add_node(self, node):
        self.nodes.append(node)
        self._counter += 1
        return self._counter - 1

    def fit(self, data, labels=None):
        # Create all the tree structure
        # Simple breadth first traversal
        if isinstance(labels, list):
            labels = np.array(labels)

        if labels is None:
            self.one_class = True
            N_pos = len(data)
            N_neg = 0
        else:
            aux = (labels == 1)
            pos_indices = np.nonzero(aux)[0]
            neg_indices = np.nonzero(~aux)[0]
            N_pos = len(pos_indices)
            N_neg = len(neg_indices)
            if N_neg > 0:
                self.one_class = False
            else:
                self.one_clase = True

        for layer in range(self.depth + 1):
            self.layers.append([])
            if layer == 0:
                node_idx = self._add_node(Node(None))
                self.layers[0] = [node_idx]
            else:
                for parent_idx in self.layers[layer - 1]:
                    parent = self.nodes[parent_idx]
                    all_child_terminal = True
                    for child_id in range(self.branching):
                        random_pos = np.random.randint(N_pos)
                        if self.one_class is False:
                            random_pos = pos_indices[random_pos]
                        child = Node(parent)
                        child.activate(data[random_pos])
                        if self.one_class is False:
                            for i in range(self.neg_sampling):
                                random_neg = neg_indices[np.random.randint(N_neg)]
                                child.contrast(data[random_neg])
                        if not child.is_terminal():
                            node_idx = self._add_node(child)
                            self.layers[layer].append(node_idx)
                            all_child_terminal = False

                    if all_child_terminal:
                        self.leaf_nodes.add(parent_idx)
        
        # Add nodes in the last layer to the list of leafs
        self.leaf_nodes.update(self.layers[layer])
    
    def fit_online(self, data_generator):
        for data in data_generator:
            # Find a node which can be a parent i.e. has less than
            # self.branching children already and its
            # active set is not null
            raise NotImplementedError()
            parent = TODO
            # Create a new node which is a child of that parent
            child = Node(parent)
            child.activate(data[idx])
            node_idx = self._add_node(child)
            self.layers[child.layer].append(node_idx)

    def get_leafs(self):
        return [self.nodes[idx] for idx in self.leaf_nodes]


class RIT():
    def __init__(self, num_trees, depth, branching, hashing=False, neg_sampling=10):
        self.num_trees = num_trees
        self.depth = depth
        self.branching = branching
        self.trees = []
        self.fit_ = False
        self.hashing = hashing
        self.neg_sampling = neg_sampling
        # Initialize the trees
        for i in range(num_trees):
            self.trees.append(IntersectionTree(depth, branching, neg_sampling=neg_sampling))

    def fit(self, data, labels=None):
        # We should do this in parallel
        for i in range(self.num_trees):
            self.trees[i].fit(data, labels)
        self.fit_ = True

    def candidates(self, min_depth=1):
        assert(self.fit_ is True)
        candidates = []
        for tree in self.trees:
            for node in [node for node in tree.get_leafs() if node.depth >= min_depth]:
                candidates.append(node.active_set)
        return candidates

    def ranked_candidates(self, min_depth=1, normalize=False):
        if min_depth > self.depth:
            raise ValueError('Maximum tree depth is %i' % self.depth)

        candidates = self.candidates(min_depth)
        N = float(len(candidates))
        if N == 0:
            print 'No surviving candidates'
            return None

        aux = defaultdict(lambda: 0)
        for candidate in candidates:
            aux[candidate] += 1
        aux = dict(aux)
        # Alternatively, we can return a sorted version
        # of the dictionary:
        # return sorted(aux, key=aux.get, reverse=True)
        out = pd.DataFrame(index=aux.keys(),
                           data=aux.values()).sort(0, ascending=False)

        if normalize is True:
            out /= N

        return out


def test_data_generator(N, K, d, p, l, r):
    '''
    N -> Number of samples
    K -> Number of symbols in the alphabet
    d -> Average number of symbols in a data point
    p -> Probability of observing a given special interaction in class 1
    l -> Length of the special interaction
    r -> Class ratio
    '''
    out = []
    point_sizes = np.random.randint(d / 2.0, d * 2, N)
    special_set = range(l)
    aux_v = np.random.rand(N)
    aux_c = np.random.rand(N)
    labels = np.zeros(N)
    for idx in range(N):
        aux = list(np.random.randint(0, K, point_sizes[idx]))
        if aux_c[idx] < r:
            labels[idx] = 1
            if aux_v[idx] < p:
                aux.extend(special_set)
        out.append(aux)
    return (out, labels)
