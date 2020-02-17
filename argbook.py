''' This module is responsible for ARG classes'''
import math
from sortedcontainers import SortedSet
import msprime
import bintrees
import treeSequence
import random
import pickle
import pandas as pd
import numpy as np
import collections
import time
import copy
from tqdm import tqdm
# import compress_pickle
# import zipfile
import sys
import os
sys.setrecursionlimit(40000)

class Segment(object):

    """
    A class representing a single segment. Each segment has a left and right, denoting
    the loci over which it spans, a node giving the node to which it belongs in an ARG,
     a prev and next, giving the previous and next segments in the chain,
     and samples representing the samples underneath the segment.
    """
    def __init__(self):
        self.left = None
        self.right = None
        self.node = None
        self.prev = None
        self.next = None
        self.samples = bintrees.AVLTree()

    def __str__(self):
        s = "({}:{}-{}->{}: prev={} next={})".format(
            self.left, self.right, self.node.index, self.samples,repr(self.prev),
            repr(self.next))
        return s

    def __lt__(self, other):
        return ((self.left, self.right)
                < (other.left, other.right))

    def contains(self, x):
        return x >= self.left and x < self.right

    def is_mrca(self, sample_size):#V3
        assert len(self.samples) <= sample_size
        return len(self.samples) == sample_size

    def equal_samples(self, other):
        '''is self.samples == other.samples'''
        # self.samples.is_subset(other.samples) and other.samples.is_subset(self.samples)
        return sorted(self.samples) == sorted(other.samples)

    def union_samples(self, other):
        # s = self.samples[:]
        # f = s[:]
        # s.extend(other.samples)
        return self.samples.union(other.samples)

    def equal(self, other):
        if self.node is None and other.node is None:
            return self.left == other.left and \
                   self.right == other.right and self.node == other.node and \
                   sorted(self.samples) == sorted(other.samples)
        elif self.node is not None and other.node is not None:
            return self.left == other.left and \
                   self.right == other.right and self.node.index == other.node.index and \
                   sorted(self.samples) == sorted(other.samples)
        else:
            return False

    def defrag_segment_chain(self):
        y = self
        while y.prev is not None:
            x = y.prev
            if x.right == y.left and x.node.index == y.node.index and y.equal_samples(x):
                x.right = y.right
                x.next = y.next
                if y.next is not None:
                    y.next.prev = x
            y = x

    def get_first_segment(self):
        '''get the fist segment in a chain of segments'''

        seg = self
        while seg.prev is not None:
            seg = seg.prev
        return seg

    def get_variants(self, data):
        '''get all snps of data that are in  self
        TODO: find an efficient way
        '''
        seg_variants = bintrees.AVLTree()
        for item in data.keys():
            if self.contains(item):
                seg_variants[item] = item
        return seg_variants

    def add_snps(self, node , data):
        '''add snps to node that are in this segment'''
        for snp in data.keys():
            if self.contains(snp) and \
                    sorted(self.samples) == sorted(data[snp]):
                node.snps[snp] = snp

    def get_seg_variants(self, data):
        '''TODO: implement efficiently'''
        assert self is not None
        seg_variants = bintrees.AVLTree()
        for item in data.keys():
            if self.contains(item):
                seg_variants[item] = item
        return seg_variants

    def get_intersect(self, start, end):
        '''
        return the intersection of self and [start, end)
        '''
        ret = []
        if self.left <= start and self.right > start:
            if self.right<= end:
                ret = [start, self.right]
            else:
                ret = [start, end]
        elif self.left > start and self.left < end:
            if self.right<=end:
                ret = [self.left, self.right]
            else:
                ret= [self.left, end]
        return ret

class Node(object):
    """
    A class representing a single node. Each node has a left and right child,
    and a left and right parent. If a node arises from a recombination, then
    left_child == right_child, while if it ends in a coalescence, then
    left_parent == right_parent. Each node also has a time at which it
    appears in the tree, and a list of segments of ancestral material, and a list of
    snps. The snps represent mutations arising on the branch of which the node is a
    child. The fact that snps are stored on a single branch is for computational
    convenience; the MCMC algorithm marginalises over all contiguous branches which
    subtend the same leaves at the site of the snp, and hence the snp could just as well
    be stored on any one of them.
    """

    def __init__(self, index):
        self.left_child = None
        self.right_child = None
        self.left_parent = None
        self.right_parent = None
        self.first_segment = None
        self.snps = bintrees.AVLTree()
        self.time = None
        self.breakpoint = None
        self.index = index

    def contains(self, x):
        seg = self.first_segment
        while seg is not None:
            if seg.contains(x):
                return True
            seg = seg.next
        return False

    def x_segment(self, x):
        '''return the segment containing x
        given that we know self contains x'''
        seg = self.first_segment
        while seg is not None:
            if seg.contains(x):
                return seg
            seg = seg.next
        raise ValueError("x is not in node")

    def num_links(self):
        seg = self.first_segment
        left = seg.left
        while seg.next is not None:
            seg = seg.next
        return seg.right - left -1

    def is_leaf(self):
        return self.left_child == None

    def is_root(self):
        return  self.left_parent == None

    def equal(self, other):
        '''
        two nodes are exactly the same, to verify if
        the original node changes after some updating.
        '''
        if self is not None and other is not None:
            seg = self.first_segment
            sego = other.first_segment
            while seg is not None and sego is not None:
                if not seg.equal(sego):
                    return False
                seg = seg.next
                sego = sego.next
            if seg is None and sego is None:
                if sorted(self.snps) == sorted(other.snps):
                    return True
                else:
                    return False
            else:
                return False
        else:
            raise ValueError("one or both nodes are NONE")

    def arg_node_age(self):
        '''the arg branch length of a node '''
        if self.left_parent is not None:
            return self.left_parent.time - self.time
        else:
            return 0

    def upward_path(self, x):
        '''for position x check if we can move upward.
        this is used in finding the branch length at
        position x in a tree'''
        if self.left_parent is None:
            block = True
            return self, block
        elif self.left_parent.index is not self.right_parent.index:
            if self.left_parent.contains(x) + self.right_parent.contains(x) != 1:
                print("in upward_path x is", x, "left_aprent", self.left_parent.index,
                      "right parent",self.right_parent.index, "node", self.index)

            assert self.left_parent.contains(x) + self.right_parent.contains(x) == 1
            block = False
            if self.left_parent.contains(x):
                return self.left_parent, block
            else:
                return self.right_parent, block
        else: # CA
            sib = self.sibling()
            #--- after spr before clean up, sib might be NAM
            if sib.first_segment != None and sib.contains(x):
                block = True
                return self.left_parent, block
            else:
                block = False
                return self.left_parent, block

    def tree_node_age(self, x):
        '''
        the tree branch length of
        node self, at position x
         '''
        node = self
        child_time = node.time
        block = False
        while not block:
            node, block = node.upward_path(x)
        assert node.time - child_time >= 0
        return node.time - child_time

    def sibling(self):
        '''
        Find and return the sibling node of u
        where u is a child of a CA
        '''
        assert self.left_parent is not None
        assert self.left_parent.index == self.right_parent.index
        p = self.left_parent
        v = p.left_child
        if v.index == self.index:
            v = p.right_child
        assert v.index is not self.index
        return v

    def push_snp_down(self, x):
        # Push the snp at position x down one branch from node to one of its children
        # provided only one is ancestral at x.
        if self.left_child is None:
            block = True
            return self, block
        elif self.left_child is not self.right_child:
            if self.left_child.contains(x) and self.right_child.contains(x):
                block = True
                return self, block
            elif self.left_child.contains(x):
                self.left_child.snps.__setitem__(x, x)
                self.snps.discard(x)
                block = False
                return self.left_child, block
            else:
                self.right_child.snps.__setitem__(x, x)
                self.snps.discard(x)
                block = False
                return self.right_child, block
        else:# rec
            self.left_child.snps.__setitem__(x, x)
            self.snps.discard(x)
            block = False
            return self.left_child, block

    def get_tail(self):
        seg = self.first_segment
        while seg.next is not None:
            seg = seg.next
        return seg

    def get_variants(self, data):
        '''get all snps in data lay in the self segments
        TODO: an efficient way
        it is not efficient ot loop over data SNPs for each segment
        '''
        node_variants = bintrees.AVLTree()
        seg = self.first_segment
        while seg is not None:
            for item in data.keys():
                if seg.contains(item):
                    node_variants[item] = item
            seg = seg.next
        return node_variants

    def update_child(self, oldchild, newchild):
        '''update self child from oldchild to newchild'''
        if self.left_child.index == oldchild.index:
            self.left_child = newchild
        if self.right_child.index == oldchild.index:
            self.right_child = newchild

class ARG(object):
    '''
    Ancestral Recombination Graph
    '''

    def __init__(self):
        self.nodes = {}
        self.roots = bintrees.AVLTree()# root indexes
        self.rec = bintrees.AVLTree() # arg rec parents nodes
        self.coal = bintrees.AVLTree() # arg CA parent node
        self.num_ancestral_recomb = 0
        self.num_nonancestral_recomb = 0
        self.branch_length = 0
        self.nextname = 1 # next node index
        self.available_names = SortedSet()

    def __iter__(self):
        '''iterate over nodes in the arg'''
        return list(self.nodes)

    def __len__(self):
        '''number of nodes'''
        return len(self.nodes)

    def __getitem__(self, index):
        '''returns node by key: item'''
        return self.nodes[index]

    def __setitem__(self, index, node):
        '''adds a node to the ARG'''
        node.index = index
        self.add(node)

    def __contains__(self, index):
        '''if ARG contains node key '''
        return index in self.nodes

    def equal(self, other):
        '''if self is equal with other (structural equality)'''
        for node in self.nodes.values():
            if node.index not in other:
                return False
            if not node.equal(other[node.index]):
                return False
        return True

    def set_roots(self):
        self.roots.clear()
        for node in self.nodes.values():
            if node.left_parent is None:
                self.roots[node.index] = node.index

    def get_times(self):
        '''return a sorted set of the ARG node.time'''
        times = SortedSet()
        for node in self.nodes.values():
            times.add(node.time)
        return times

    def get_higher_nodes(self, t):
        ''':return nodes.index of nodes with node.time >= t
        TODO: a more efficient search option
        '''
        return [key for key in self.nodes if self.nodes[key].time >= t]

    #==========================
    # node manipulation

    def alloc_segment(self, left = None, right = None, node = None,
                      samples = bintrees.AVLTree(), prev = None, next = None):
        """
        alloc a new segment
        """
        s = Segment()
        s.left = left
        s.right = right
        s.node = node
        s.samples = samples
        s.next = next
        s.prev = prev
        return s

    def alloc_node(self, index = None, time = None,
                    left_child = None, right_child = None):
        """
        alloc a new Node
        """
        node = Node(index)
        node.time = time
        node.first_segment = None
        node.left_child = left_child
        node.right_child = right_child
        node.left_parent = None
        node.right_parent = None
        node.breakpoint = None
        node.snps = bintrees.AVLTree()
        return node

    def store_node(self, segment, node):
        '''store node with segments: segment'''
        x = segment
        if x is not None:
            while x.prev is not None:
                x = x.prev
            s = self.alloc_segment(x.left, x.right, node, x.samples.copy())
            node.first_segment = s
            x.node = node
            x = x.next
            while x is not None:
                s = self.alloc_segment(x.left, x.right, node, x.samples.copy(), s)
                s.prev.next = s
                x.node = node
                x = x.next
        else:#
            node.first_segment = None
        self.nodes[node.index] = node

    def copy_node_segments(self, node):
        '''
        copy the segments of a node,
        in CA event or Rec events, we need to copy the first node
        in order to make changes on them
        '''
        x = node.first_segment
        if x is None:
            return None
        else:
            assert x.prev is None
            s = self.alloc_segment(x.left, x.right, node, x.samples.copy())
            x.node = node
            x = x.next
            while x is not None:
                s = self.alloc_segment(x.left, x.right, node, x.samples.copy(), s)
                s.prev.next = s
                x.node = node
                x = x.next
            return s

    def get_available_names(self):
        '''get free names from 0 to max(nodes)'''
        self.available_names = SortedSet()
        current_names = SortedSet(self.__iter__())
        counter = 0
        prev = current_names[0]
        while counter < len(current_names):
            if current_names[counter] != prev + 1:
                self.available_names.update(range(prev+1, current_names[counter]))
            prev = current_names[counter]
            counter += 1

    def new_name(self):
        '''returns a new name for a node'''
        if self.available_names:
            name = self.available_names.pop()
        else:
            name = self.nextname
            self.nextname += 1
        return name

    def add(self, node):
        ''' add a ready node to the ARG:
        '''
        self.nodes[node.index] = node
        return node

    def rename(self, oldindex, newindex):
        '''renames a node in the ARG'''
        node = self.nodes[oldindex]
        node.index = newindex
        del self.nodes[oldindex]
        self.nodes[newindex] = node

    def total_branch_length(self):
        '''the ARG total branch length'''
        total_material = 0
        for node in self.nodes.values():
            if node.left_parent is not None:
                age = node.left_parent.time - node.time
                seg = node.first_segment
                while seg is not None:
                    total_material += ((seg.right - seg.left)* age)
                    seg = seg.next
        return total_material

    #=======================
    #spr related

    def detach(self, node, sib):
        '''
        Detaches a specified coalescence node from the rest of the ARG
        '''
        print("Detach()",node.index, "sib", sib.index, "p",node.left_parent.index)
        assert node.left_parent.index == node.right_parent.index
        parent = node.left_parent
        sib.left_parent = parent.left_parent
        sib.right_parent = parent.right_parent
        sib.breakpoint = parent.breakpoint
        grandparent = parent.left_parent
        if grandparent is not None:
            grandparent.update_child(parent, sib)
            grandparent = parent.right_parent
            grandparent.update_child(parent, sib)

    def reattach(self, u, v, t, new_names):
        # Reattaches node u above node v at time t, new_names is a avltree of all
        #new nodes.index in a new ARG in mcmc
        assert t > v.time
        # assert v.left_parent == None or t < v.left_parent.time
        if u.left_parent is None:# new_name
            new_name = self.new_name()
            new_names[new_name] = new_name
            # self.coal[new_name] = new_name # add the new CA parent to the ARG.coal
            parent = self.add(self.alloc_node(new_name))
            parent.left_child = u
            u.left_parent = parent
            u.right_parent = parent
        else:
            assert u.left_parent.index == u.right_parent.index
            parent = u.left_parent
        parent.time = t
        parent.breakpoint = v.breakpoint
        v.breakpoint = None
        parent.left_parent = v.left_parent
        grandparent = v.left_parent
        if grandparent is not None:
            grandparent.update_child(v, parent)
        parent.right_parent = v.right_parent
        grandparent = v.right_parent
        if grandparent is not None:
            grandparent.update_child(v, parent)
        v.left_parent = parent
        v.right_parent = parent
        if parent.left_child.index == u.index:
            parent.right_child = v
        else:
            parent.left_child = v
        return new_names

    def push_mutation_down(self, node, x):
        '''
        for a given node push the mutation (at x) as down as possible
        normally mutations automatically should stay at their
        lowest possible position. This might be useful for initial ARG
        '''
        block = False
        while not block:
            node, block = node.push_snp_down(x)

    def push_all_mutations_down(self, node):
        '''push down all mutations on node as low as possible'''
        snp_keys = [k for k in node.snps]
        for x in snp_keys:
            self.push_mutation_down(node, x)
        # iter = len(node.snps)
        # i = 0
        #
        # while iter > 0:
        #     x = node.snps[i]
        #     self.push_mutation_down(node, x)
        #     iter -= 1
        #     if node.snps and len(node.snps) > i:
        #         if node.snps[i] == x:
        #             i += 1

    def find_tmrca(self, node, x):
        '''
        check the parent of node to see
        if it is mrca for site x
        '''
        if node.left_parent is None:
            block = True
            return node, block
        elif node.left_parent.index is not node.right_parent.index:
            assert node.left_parent.contains(x) + node.right_parent.contains(x) == 1
            block = False
            if node.left_parent.contains(x):
                return node.left_parent, block
            else:
                return node.right_parent, block
        elif node.left_parent.contains(x):
            block = False
            return node.left_parent, block
        else:# it is mrca for x
            block = True
            return node.left_parent, block

    def tmrca(self, x):
        '''tmrca for site x
        1. start from a leaf
        2. follow the path of x until its mrca
        '''
        node = self.__getitem__(0)
        block = False

        while not block:
            node, block = self.find_tmrca(node, x)
        return node.time

    @property
    def breakpoints(self):
        br = bintrees.AVLTree()
        for node in self.nodes.values():
            if node.breakpoint != None:
                br[node.breakpoint] = node.breakpoint
        return br

    #========== probabilites
    def log_likelihood(self, theta, data):
        '''
        log_likelihood of mutations on a given ARG up to a normalising constant
         that depends on the pattern of observed mutations, but not on the ARG
         or the mutation rate.
         Note after spr and berfore clean up we might have NAM lineages,
         this method covers take this into account.
         :param m : is number of snps
         '''
        snp_nodes = [] # nodes with len(snps) > 0
        total_material = 0
        number_of_mutations = 0
        #get total matereial and nodes with snps
        for node in self.nodes.values():
            if node.first_segment != None:
                assert node.left_parent != None
                age = node.left_parent.time - node.time
                seg = node.first_segment
                while seg is not None:
                    total_material += ((seg.right - seg.left)* age)
                    seg = seg.next
                if node.snps:
                    number_of_mutations += len(node.snps)
                    snp_nodes.append(node)
        self.branch_length = total_material
        print("number_of_mutations", number_of_mutations, "m", len(data))
        assert number_of_mutations == len(data) # num of snps
        if theta == 0:
            if number_of_mutations == 0:
                ret = 0
            else:
                ret = -float("inf")
        else:
            ret = number_of_mutations * math.log(total_material * theta) -\
                (total_material * theta)
        # now calc prob of having this particular mutation pattern
        for node in snp_nodes:
            for x in node.snps:
                potential_branch_length = node.tree_node_age(x)
                ret += math.log(potential_branch_length / total_material)
            # # verify the mutation is on the correct spot
            verify_mutation_node(node, data)
        return ret

    def log_prior(self, sample_size, sequence_length, rho, Ne,
                  NAM = True, new_roots = False , kuhner = False):
        '''probability of the ARG under coalescen with recombination
        this is after a move and before clean up. then there might be some
         extra NAM lineages, we ignore them.
         :param NAM: no-ancestral material node. If NAm node is allowed. note after spr and
            before clean up step there might be some NAM in the ARG which is ok. But after clean up
            or on the initial ARG there should not be any.
         '''
        # order nodes by time
        #TODO: find an efficient way to order nodes
        ordered_nodes = [v for k, v in sorted(self.nodes.items(),
                                     key = lambda item: item[1].time)]
        number_of_lineages = sample_size
        number_of_links = number_of_lineages * (sequence_length - 1)
        number_of_nodes = self.__len__()
        counter = sample_size
        time  = 0
        ret = 0
        rec_count = 0
        coal_count = 0
        roots = bintrees.AVLTree()
        new_coal = bintrees.AVLTree()
        if kuhner:
            self.rec.clear()
        self.num_ancestral_recomb = 0
        self.num_nonancestral_recomb = 0
        while counter < number_of_nodes:
            node = ordered_nodes[counter]
            assert node.time >= time # make sure it is ordered
            rate = (number_of_lineages * (number_of_lineages - 1)
                    / (4*Ne)) + (number_of_links * (rho))
            # ret -= rate * (node.time - time)
            if node.left_child.index == node.right_child.index: #rec
                assert node.left_child.first_segment != None
                assert node.left_child.left_parent.first_segment != None
                assert node.left_child.right_parent.first_segment != None
                ret -= rate * (node.time - time)
                gap = node.left_child.num_links()-\
                      (node.left_child.left_parent.num_links() +
                       node.left_child.right_parent.num_links())
                ret += math.log(rho)
                assert gap >= 1
                if gap == 1:
                    self.num_ancestral_recomb += 1
                else:
                    self.num_nonancestral_recomb += 1
                number_of_links -= gap
                number_of_lineages += 1
                if kuhner:# add rec
                    self.rec[node.index] = node.index
                    self.rec[ordered_nodes[counter+1].index] = ordered_nodes[counter+1].index
                counter += 2
                time = node.time
                rec_count += 1
            elif node.left_child.first_segment != None and\
                        node.right_child.first_segment != None:
                ret -= rate * (node.time - time)
                ret -=  math.log(2*Ne)
                if node.first_segment == None:
                    node_numlink = 0
                    number_of_lineages -= 2
                    counter += 1
                    if new_roots:
                        roots[node.index] = node.index
                else:
                    node_numlink = node.num_links()
                    number_of_lineages -= 1
                    counter += 1
                lchild_numlink = node.left_child.num_links()
                rchild_numlink = node.right_child.num_links()
                number_of_links -= (lchild_numlink + rchild_numlink) - node_numlink
                time = node.time
                coal_count += 1
                if new_roots:
                    new_coal[node.index] = node.index
            else:
                counter += 1
            if not NAM:
                assert node.left_child.first_segment != None
                assert node.right_child.first_segment != None
        if new_roots:
            return ret, roots, new_coal
        else:
            return ret

    def dump(self, path = ' ', file_name = 'arg.arg'):
        output = path + "/" + file_name
        with open(output, "wb") as file:
            pickle.dump(self, file)

    def load(self, path = ' ', file_name = 'arg.arg'):
        output = path + "/" + file_name
        with open(output, "rb") as file:
            return pickle.load(file)

    def verify(self):
        '''
        verify arg:
        1. a node with parent must have seg
        2. a node with no parent a. must be in roots b. different child
        3. node.parent_time > node.time
        4. arg name == node.index
        5. recomb parent must have self.snps.empty()
        6. nodes with child = None must be leaf
        7. number coal + rec + roots check
        8. seg.samples is not empty, seg.left < seg.right
        '''
        for node in self.nodes.values():
            assert self.nodes[node.index].index == node.index
            if node.left_parent is None:# roots
                if node.first_segment is not None:
                    print("iv verrify node is ", node.index)
                    self.print_state()
                assert node.first_segment == None
                assert node.index in self.roots
                assert node.breakpoint == None
                assert node.left_child.index != node.right_child.index
                assert node.right_parent == None
                assert node.index in self.coal
                assert node.time > node.left_child.time
                assert node.time > node.right_child.time
            else: # rest
                assert node.first_segment != None
                assert node.index not in self.roots
                assert node.left_parent.time > node.time
                if node.left_child is None: #leaves
                    assert node.right_child is None
                    assert node.time == 0
                if node.left_parent.index != node.right_parent.index:
                    assert node.breakpoint != None
                    assert node.left_parent.left_child.index ==\
                           node.left_parent.right_child.index
                    assert node.right_parent.left_child.index ==\
                        node.right_parent.right_child.index
                    assert node.right_parent.left_child.index == node.index
                    assert not node.left_parent.snps
                    assert not node.right_parent.snps
                    assert node.left_parent.time == node.right_parent.time
                    assert node.left_parent.index in self.rec
                    assert node.right_parent.index in self.rec
                    if node.left_parent.first_segment.left > node.right_parent.first_segment.left:
                        print("in verify node", node.index)
                        print("node.left_parent", node.left_parent.index)
                        print("node.right_parent", node.right_parent.index)
                    assert node.left_parent.first_segment.left < node.right_parent.first_segment.left
                else:
                    assert node.left_parent.index in self.coal
                    assert node.left_parent.left_child.index !=\
                           node.left_parent.right_child.index
                    assert node.breakpoint == None
            if node.first_segment is not None:
                seg = node.first_segment
                assert seg.prev is None
                while seg is not None:
                    assert seg.samples
                    assert seg.left < seg.right
                    assert seg.node.index == node.index
                    seg = seg.next

    def print_state(self):
        print("self.arg.coal", self.coal)
        print("self.arg.rec", self.rec)
        print("self.arg.roots", self.roots)
        print("node", "time", "left", "right", "l_chi", "r_chi", "l_par", "r_par",
              "l_bp", "snps", "fir_seg_sam",
              sep="\t")
        for j in self.nodes:
            node = self.__getitem__(j)
            if node.left_parent is not None or node.left_child is not None:

                s = node.first_segment
                if s is None:
                    print(j, "%.5f" % node.time, "root", "root",
                              node.left_child.index,
                              node.right_child.index,
                              node.left_parent,node.right_parent,
                              node.breakpoint,
                              node.snps ,None, sep="\t")

                while s is not None:
                    l = s.left
                    r = s.right
                    if node.left_child is None:
                        print(j, "%.5f" % node.time, l,r, "Leaf", "Leaf",
                              node.left_parent.index,node.right_parent.index,
                              node.breakpoint,
                              node.snps,  s.samples,  sep="\t")#
                    elif  node.left_parent is None:
                        print(j, "%.5f" % node.time, l, r,
                              node.left_child.index,
                              node.right_child.index, "Root", "Root",
                              node.breakpoint,
                              node.snps ,s.samples, sep="\t")
                    else:
                        print( j, "%.5f" % node.time, l, r,
                             node.left_child.index, node.right_child.index,
                              node.left_parent.index, node.right_parent.index,
                              node.breakpoint,
                              node.snps, s.samples, sep="\t")
                    s = s.next

#====== verificatio

def verify_mutation_node(node, data):
    '''
    verify node is the lowest possible position
    the mutation can sit on.
    '''
    for x in node.snps:
        # bth children have x
        # left_child is not right_child
        # for the segment containing x on node, samples == data[x]
        if node.left_child is not None:
            assert node.left_child.index is not node.right_child.index
            assert node.left_child.contains(x) and node.right_child.contains(x)
        node_samples = node.x_segment(x).samples
        assert sorted(node_samples) == sorted(data[x])

class TransProb(object):
    '''transition probability calculation'''

    def __init__(self):
        self.log_prob_forward = 0
        self.log_prob_reverse = 0

    def spr_choose_detach(self, numCoals, forward = True):
        '''
        choose a coal parent randomly and
        then choose one child with half prob
        '''
        if forward:
            self.log_prob_forward += math.log((1/numCoals) *(1/2))
        else: #reverse prob
            self.log_prob_reverse += math.log((1/numCoals) *(1/2))

    def spr_choose_reattach(self, numReattaches, forward = True):
        '''choose a reattach node among all possible nodes: numReattaches'''
        if forward:
            self.log_prob_forward += math.log(1/numReattaches)
        else: #reverse
            self.log_prob_reverse += math.log(1/numReattaches)

    def spr_reattach_time(self, new_time,lower_bound = 0, upper_bound = 0,
                          reattach_root = True, forward = True, lambd = 10000):
        '''if reattach_root: new time  is lower_bound + exp(1)
        else: new_time is  lower_bound  + U(lower_bound, upper_bound)
        '''
        if forward:
            if reattach_root: #expo
                print("forward expo:", new_time- lower_bound)
                self.log_prob_forward += math.log(lambd) - \
                                         (lambd * (new_time- lower_bound))
            else: #uniform
                print("forward uniform:", upper_bound -lower_bound)
                self.log_prob_forward += math.log(1/(upper_bound -lower_bound))
        else:
            if reattach_root:
                print("reverse expo:", new_time- lower_bound)
                self.log_prob_reverse += math.log(lambd)- (lambd * (new_time- lower_bound))
            else:
                print("reverse uniform:", upper_bound -lower_bound)
                self.log_prob_reverse += math.log(1/(upper_bound -lower_bound))

    def spr_recomb_simulate(self, l_break, r_break, forward = True):
        '''simulate the recomb breakpoint if a window is given
        there are four scenarios on the recombination:
        1. ancestral to non-ancestral: no forward, yes reverse
        2. ancestral to ancestral: no forward, no reverse
        3. non ancestral to ancestral: yes forward, no reverse
        4. non ancestral to non ancestral: yes forward, yes reverse
        If a rec parent is NAM or rec child is NAM: No transition +
            no chnage of breakpoints
        :param l_break: left_breakpoint
        :param r_break: right_breakpoint
        :param forward: Forward transition if True, else, reverse.
        '''
        if forward:
            self.log_prob_forward += math.log(1/(r_break - l_break +1))
        else:
            self.log_prob_reverse += math.log(1/(r_break - l_break +1))

    def rem_choose_remParent(self, numRecPs, forward = True):
        '''choose a rec parent to remove'''
        if forward:
            self.log_prob_forward += math.log(1/numRecPs)
        else:
            self.log_prob_reverse += math.log(1/numRecPs)

    def add_choose_node(self, numnodes, forward = True):
        ''' prob of choosing a node to which add a rec'''
        if forward:
            self.log_prob_forward += math.log(1/numnodes)
        else:
            self.log_prob_reverse += math.log(1/numnodes)

    def add_choose_breakpoint(self, numlinks, forward = True):
        '''choose a genomic position as recomb breakpoint'''
        assert numlinks > 0
        if forward:
            self.log_prob_forward += math.log(1/numlinks)
        else:
            self.log_prob_reverse += math.log(1/numlinks)

    def add_choose_node_to_float(self, forward = True):
        '''randomly choose one of new parents to
        follow the child path, the other flow'''
        if forward:
            self.log_prob_forward += math.log(1/2)
        else:
            self.log_prob_reverse += math.log(1/2)

    def kuhner_num_nodes(self, num_nodes,forward =  True):
        '''resiprocal of the number of nodes in the ARG excluding the roots'''
        if forward:
            self.log_prob_forward = math.log(1/num_nodes)
        else:
            self.log_prob_reverse = math.log(1/num_nodes)

class MCMC(object):

    def __init__(self, data = {}, outpath = os.getcwd()):

        self.data = data #a dict, key: snp_position- values: seqs with derived allele
        self.arg = ARG()
        self.theta = 1e-8 #mu
        self.rho = 1e-8 #r
        self.n = 10 # sample size
        self.Ne = 5000
        self.seq_length = 4e5 # sequence length
        self.m = 0 # number of snps
        self.log_lk = 0
        self.log_prior = 0
        self.outpath = outpath
        self.transition_prob = TransProb()
        self.floatings = bintrees.AVLTree()#key: time, value: node index, d in the document
        self.floatings_to_ckeck = bintrees.AVLTree()#key index, value: index; this is for checking new roots
        self.new_names = bintrees.AVLTree()# index:index, of the new names (new parent indexes)
        self.lambd = 1/(self.Ne) # lambd in expovariate
        self.NAM_recParent = bintrees.AVLTree() # rec parent with seg = None
        self.NAM_coalParent = bintrees.AVLTree() # coal parent with seg = None
        self.coal_to_cleanup = bintrees.AVLTree()# ca parent with atleast a child.seg = None
        self.accept = False
        #------- To start get msprime output as initial
        self.get_initial_arg()
        self.arg.nextname = max(self.arg.nodes) + 1
        self.summary = pd.DataFrame(columns=('likelihood', 'prior', "posterior",
                                             'ancestral recomb', 'non ancestral recomb',
                                                'branch length', 'setup'))
        np.save(os.getcwd()+"/true_values.npy", [self.log_lk, self.log_prior, self.log_lk + self.log_prior,
                                                 self.arg.branch_length,self.arg.num_ancestral_recomb,
                                                 self.arg.num_nonancestral_recomb])
        #---- kuhner
        self.floats = bintrees.AVLTree()#floatings: index:index
        self.partial_floatings = collections.defaultdict(list)# index:[a, b, num]
        self.need_to_visit = bintrees.AVLTree()#index:index, all the nodes we need to visit
        self.higher_times = collections.defaultdict(set)#time: (index)
        self.active_nodes = bintrees.AVLTree()#node.index
        self.active_links = 0
        #--- test
        self.detail_acceptance = collections.defaultdict(list)# transiton:[total, accepted]

    def get_initial_arg(self):
        '''
        TODO: build an ARG for the given data.
        '''
        ts_full = msprime.simulate(sample_size = self.n, Ne = self.Ne,
                                   length = self.seq_length, mutation_rate = self.theta,
                                   recombination_rate = self.rho,
                                   random_seed = 20, record_full_arg = True)
        tsarg = treeSequence.TreeSeq(ts_full)
        tsarg.ts_to_argnode()
        self.arg = tsarg.arg
        self.data = treeSequence.get_arg_genotype(ts_full)
        self.m = len(self.data)
        self.log_lk = self.arg.log_likelihood(self.theta, self.data)
        self.log_prior = self.arg.log_prior(self.n, self.seq_length, self.rho, self.Ne, False)

        # --------

    def truncated_expo(self, a, b, lambd):
        '''
        generate a random number from trucated exponential with rate lambd  in (a, b)
        '''
        assert b > a
        u= random.random()
        trunc_number = -(1/lambd) * (math.log(math.exp(-lambd*a) -\
                                              (u*(math.exp(-lambd*a)- math.exp(-lambd*b)))))
        if not (trunc_number < b) and not (a<trunc_number):
            return self.truncated_expo(a = a, b = b, lambd = lambd)
        else:
            return trunc_number

    def Metropolis_Hastings(self, new_log_lk, new_log_prior,
                            trans_prob = True, kuhner = False):
        '''if trans_prob: the ratio includes likelihood,
            prior and transiton probabilities,
            Otherwaise: only likelihood. This is
            for cases where the transition is based on the prior
        '''
        if trans_prob:
            ratio = new_log_lk + new_log_prior + \
                    self.transition_prob.log_prob_reverse - \
                    (self.log_lk + self.log_prior +
                     self.transition_prob.log_prob_forward)
        else:
            ratio = new_log_lk - self.log_lk
            if kuhner:
                ratio += (self.transition_prob.log_prob_reverse -\
                          self.transition_prob.log_prob_forward)
        print("forward_prob:", self.transition_prob.log_prob_forward)
        print("reverse_prob:", self.transition_prob.log_prob_reverse)
        print("ratio:", ratio)
        print("new_log_lk", new_log_lk, "new_log_prior", new_log_prior)
        print("old.log_lk", self.log_lk,"old.log_prior", self.log_prior)
        if math.log(random.random()) <= ratio: # accept
            self.log_lk = new_log_lk
            self.log_prior = new_log_prior
            self.accept = True

    def find_break_seg(self, z, break_point):
        '''z is the chain of segments
        return: the segment that includes breakpoint
            or the fist segment after breakpoint
            if None, then  second parent is empty
            '''
        while z.prev is not None:
            z = z.prev
        while z is not None:
            if z.contains(break_point):
                return z
            elif z.prev is not None and z.prev.right <= break_point and\
                    break_point < z.left:
                return z
            z = z.next
        return None

    def real_parent(self, node):
        ''' to be used in spr_reattachment_nodes
        return the original parent of a node
        if exist: return it otherwise None
        also this assumes that all rec parents
         are original nodes
        '''
        original_parent = None
        while node.left_parent != None:
            if self.new_names.__contains__(node.index):
                node = node.left_parent
            else:
                original_parent = node
                break
        return original_parent

    def spr_reattachment_nodes(self, detach_time, forward = True):
        '''
        return all possible reattachment nodes
        for detach
        return those exist at or after detach_time  and their
        segment != None,
        since we already update ARG, then if a segment is None,
        we should reattach to them
        :param forward: if false, for clean up step, when we check the new roots to calc the
            reverse prob, we need to also include those original nodes that will be floating
        '''
        reattach_nodes = bintrees.AVLTree()
        if forward:
            for node in self.arg.nodes.values():
                if node.time > detach_time:
                    if node.first_segment != None: # make sure it is not a NAM
                        reattach_nodes[node.index] = node.index
                elif node.left_parent != None and node.left_parent.time > detach_time and\
                        node.first_segment != None:
                    reattach_nodes[node.index] = node.index
                elif not self.floatings.is_empty(): #BUG2 in notes
                    reattach_nodes.update({v:v for v in self.floatings.values()})
        else:
            for node in self.arg.nodes.values():
                if node.time > detach_time:
                    if node.first_segment != None:
                        reattach_nodes[node.index] = node.index
                    elif not self.new_names.__contains__(node.index):
                        # that is a original node that will float in reverse
                        reattach_nodes[node.index] =  node.index
                elif node.left_parent != None:
                    if node.left_parent.time > detach_time:
                        if node.first_segment != None:
                            reattach_nodes[node.index] = node.index
                        elif not self.new_names.__contains__(node.index):
                            reattach_nodes[node.index] = node.index
                    else:# BUG5 NOTES
                        #its parent might not be its original
                        # (a new name with lower time than its original)
                        original_parent = self.real_parent(node.left_parent)
                        if original_parent != None and original_parent.time > detach_time:
                            if node.first_segment != None:
                                reattach_nodes[node.index] = node.index
                            elif not self.new_names.__contains__(node.index):
                                reattach_nodes[node.index] = node.index
        return reattach_nodes

    def incompatibility_check(self,node,  S1, S2, s, detach_snps, completed_snps):
        '''
        if the coalescence of child 1 and child 2 compatible for this snp.
        All are AVLTrees()
        S1: samples for first child segment
        S2: samples for the second child segment
        s:  the focal SNP
        node: the parent node
        '''
        ret = True
        D = self.data[s]
        # symmetric difference between S1 union S2  and D
        A = S1.union(S2)
        symA_D = A.difference(D)
        if len(symA_D) == 0:# subset or equal
            if len(A) == len(D): #put the mutation on this node
                node.snps.__setitem__(s, s)
                # delete s from S_F
                detach_snps.discard(s)
                # add to completed_snps
                completed_snps[s] = s
        elif len(symA_D) == len(A): # distinct
            pass
        else:#
            symD_A = D.difference(A)
            if len(symD_A) > 0: # incompatible
                ret = False
        return ret, detach_snps, completed_snps

    def update_ancestral_material(self, node_index, nodes_to_update,
                                  nodesToUpdateTimes, backtrack = False):
        '''update the materials of the parent(s) of node
        do not change any parent/child node, only update materials
        if backtrack = True: retrieve to the original ARG
        '''
        s = nodes_to_update.pop(node_index)
        node = self.arg.__getitem__(node_index)
        if node.left_parent is None:
            if not backtrack:
                raise ValueError("The root node shouldn't be added "
                             "to node_to_check at the first place")
        elif node.left_parent.index == node.right_parent.index:
            #common ancestor event
            node_sib = node.sibling()
            if nodes_to_update.__contains__(node_sib.index):
                sib_segs = nodes_to_update.pop(node_sib.index)
            else:# copy them to remain intact
                sib_segs = self.arg.copy_node_segments(node_sib)
            if s is not None and sib_segs is not None:
                x = s.get_first_segment()
                y = sib_segs.get_first_segment()
                assert x is not None
                assert y is not None
                z = None
                defrag_required = False
                if backtrack:
                    node.left_parent.snps.clear()
                while x is not None or y is not None:
                    alpha = None
                    if x is None or y is None:
                        if x is not None:
                            alpha = x
                            x = None
                        if y is not None:
                            alpha = y
                            y = None
                    else:
                        if y.left < x.left:
                            beta = x
                            x = y
                            y = beta
                        if x.right <= y.left:
                            alpha = x
                            x = x.next
                            alpha.next = None
                        elif x.left != y.left:
                            alpha = self.arg.alloc_segment(x.left, y.left,
                                                    node.left_parent, x.samples)
                            x.left = y.left
                        else:
                            left = x.left
                            r_max = min(x.right, y.right)
                            right = r_max
                            alpha = self.arg.alloc_segment(left, right,
                                                    node.left_parent, x.union_samples(y))
                            if alpha.is_mrca(self.n):
                                alpha = None
                            if backtrack and alpha != None:
                                # put mutations on the node if any
                                alpha.add_snps(node.left_parent, self.data)
                            if x.right == right:
                                x = x.next
                            else:
                                x.left = right
                            if y.right == right:
                                y = y.next
                            else:
                                y.left = right
                    if alpha is not None:
                        if z is not None:
                            defrag_required |= z.right == alpha.left
                            z.next = alpha
                        alpha.prev = z
                        z = alpha
                if defrag_required:
                    z.defrag_segment_chain()
                if node.left_parent.left_parent is not None:# if not Root
                        nodes_to_update[node.left_parent.index] = z
                        nodesToUpdateTimes[node.left_parent.left_parent.time].add(node.left_parent.index)
                if z is not None:
                    z = z.get_first_segment()
                    #--- this is where we should add to floatings
                    if not backtrack:
                        if node.left_parent.left_parent is None:
                            # this is floating
                            self.floatings[node.left_parent.time] = node.left_parent.index
                            self.floatings_to_ckeck[node.left_parent.index] = node.left_parent.index
                    node.left_parent.first_segment = z
                    self.arg.store_node(z, node.left_parent)
                    self.NAM_coalParent.discard(node.left_parent.index)
                else:
                    node.left_parent.first_segment = None
                    if backtrack:
                        if node.left_parent.left_parent != None:
                            assert node.left_parent.left_parent.index == node.left_parent.right_parent.index
                            parentSib = node.left_parent.sibling()
                            if not self.new_names.__contains__(parentSib.index):# BUG4_2 NOTES
                                if not nodes_to_update.__contains__(parentSib.index):
                                    nodes_to_update[parentSib.index] = self.arg.copy_node_segments(parentSib)
                                nodesToUpdateTimes[parentSib.left_parent.time].add(parentSib.index)
                            if nodes_to_update.__contains__(node.left_parent.index):
                                del nodes_to_update[node.left_parent.index]
                            self.arg.detach(node.left_parent, node.left_parent.sibling())
                            del self.arg.nodes[node.left_parent.left_parent.index]
                            node.left_parent.left_parent = None
                            node.left_parent.right_parent = None
                    self.NAM_coalParent[node.left_parent.index] = node.left_parent.index
                self.coal_to_cleanup.discard(node.left_parent.index)
            elif s is None and sib_segs is None:
                if backtrack:
                    raise ValueError("there shouldnt be a NAM while we backtrack")
                node.left_parent.first_segment = None
                if node.left_parent.left_parent is not None:
                    nodes_to_update[node.left_parent.index] = None
                    nodesToUpdateTimes[node.left_parent.left_parent.time].add(node.left_parent.index)
                self.NAM_coalParent[node.left_parent.index] = node.left_parent.index
                self.coal_to_cleanup[node.left_parent.index] = node.left_parent.index
            else: # s is  None or  sib_seg is None
                if sib_segs is None:
                    z = s.get_first_segment()
                else:# s is None
                    z = sib_segs.get_first_segment()
                if node.left_parent.left_parent is not None:
                    nodes_to_update[node.left_parent.index] = z
                    nodesToUpdateTimes[node.left_parent.left_parent.time].add(node.left_parent.index)
                else:
                    self.floatings[node.left_parent.time] = node.left_parent.index
                    self.floatings_to_ckeck[node.left_parent.index] = node.left_parent.index
                node.left_parent.first_segment = z
                self.arg.store_node(z, node.left_parent)
                self.NAM_coalParent.discard(node.left_parent.index)
                self.coal_to_cleanup[node.left_parent.index] = node.left_parent.index
        else:
            #recomb event--> don't forget both transitions
            if s is None: # both parents are None
                z = None
                lhs_tail = None
            else:
                y = self.find_break_seg(s, node.breakpoint)
                if y is not None: # parent2 is not empty
                    x = y.prev
                    if y.left < node.breakpoint < y.right:#y.contains(node.breakpoint):# new is ancestral
                        # no forward + no reverse
                        z = self.arg.alloc_segment(node.breakpoint, y.right,
                                                   node, y.samples, None, y.next)
                        assert node.breakpoint < y.right
                        if y.next is not None:
                            y.next.prev = z
                        y.next = None
                        y.right = node.breakpoint
                        assert y.left < node.breakpoint
                        lhs_tail = y
                    elif x is not None:
                        # no forward+ yes reverse
                        # assert x.right is not y.left
                        assert x.right <= node.breakpoint <= y.left
                        x.next = None
                        y.prev = None
                        z = y
                        lhs_tail = x
                    else: # first parent is empty
                        # no update of breakpoint + no transition
                        z = y
                        lhs_tail = None
                else: # second parent is empty
                    # dont change the breakpoint no transition
                    z = None
                    lhs_tail = s
            if node.left_parent.left_parent is not None:# BUG1 in NOTES
                nodes_to_update[node.left_parent.index] = lhs_tail
                nodesToUpdateTimes[node.left_parent.left_parent.time].add(node.left_parent.index)
            else:
                #must be in self.floatings
                assert node.left_parent.index in [ind for ind in self.floatings.values()]
            if node.right_parent.left_parent is not None:# BUG1 in NOTES
                nodes_to_update[node.right_parent.index] = z
                nodesToUpdateTimes[node.right_parent.left_parent.time].add(node.right_parent.index)
            else:
                #must be in self.floatings
                assert node.right_parent.index in [ind for ind in self.floatings.values()]
            # TODO: Implement a way to check whether a recombination is removable, and
            # delete those recombination nodes which are not. Will have to be done once the
            # ARG has been constructed by checking which ones will violate snps
            # node = self.arg.alloc_node(self.arg.new_name(), time, lhs_tail.node, lhs_tail.node)
            # lhs_tail.node.left_parent = node
            if backtrack:
                if lhs_tail is  None or z is  None:
                    print("in backtrack node is ", node.index, "node.left_parent:", node.left_parent.index)
                    print("node.right_parent", node.right_parent.index, "node.breakpoint", node.breakpoint)
                    print("node.head.left", node.first_segment.left, "node.tail.right", node.get_tail().right)
                assert lhs_tail is not None and z is not None
            self.arg.store_node(lhs_tail, node.left_parent)
            # node = self.arg.add(node1)
            # self.R[node.index] = node.index
            # self.arg.rec[node.index] = node.index
            # node = self.arg.alloc_node(self.arg.new_name(), time,  z.node, z.node)
            # z.node.right_parent = node
            self.arg.store_node(z, node.right_parent)
            # if NAM put them in NAM_recParent
            if lhs_tail is None:
                self.NAM_recParent[node.left_parent.index] = node.left_parent.index
            else:
                self.NAM_recParent.discard(node.left_parent.index)
            if z is None:
                self.NAM_recParent[node.right_parent.index] = node.right_parent.index
            else:
                self.NAM_recParent.discard(node.right_parent.index)
        return nodes_to_update, nodesToUpdateTimes

    def get_detach_SF(self, detach, sub_interval=[]):
        '''get snps from detach
         that we need to check for incompatibility
          1. for a seg find all snps within.
          2. for each snp check if the mut has happened or will happen in future
          3. if in future add to detach_snps
          :param sub_interval: only check those in [start, end) interval
                    this is currenctly using in adjust_breakpoint algorithm
          '''
        detach_snps = bintrees.AVLTree()
        seg = detach.first_segment
        while seg is not None:
            seg_snps =[]
            if sub_interval:# adjust breakpoint
                intersect = seg.get_intersect(sub_interval[0], sub_interval[1])
                if intersect:
                    seg_snps = [key for key in self.data.keys() if intersect[0] <= key < intersect[1]]
            else:
                seg_snps = [key for key in self.data.keys() if seg.left <= key < seg.right]
            for item in seg_snps:
                D = self.data[item]
                A = seg.samples
                symA_D = A.difference(D)
                if len(symA_D) == 0: # A is equal or subset
                    if len(A) == len(D):
                        pass # its on detach or already occured
                        # assert detach.snps.__contains__(item)
                    else:# A is subset+ mut in future
                        detach_snps[item] = item
                elif len(symA_D) == len(A):# distinct
                    # A is all ancestral allel + might in future
                    # if self.n - len(D) < len(A): # if not all ancestral are here
                    detach_snps[item] = item
                elif len(D.difference(A)) > 0:# len(A.symD) >0 and len(D.symA)>0
                    raise ValueError("The original ARG was incompatible")
                else:# len(D.symmetric_difference(A)) == 0#mutation already happened
                    pass
            seg = seg.next
        return detach_snps

    def update_all_ancestral_material(self, node, backtrack = False):
        '''update the ancestral materials of all the ancestors of node
        if backtrack = True: we retrieve the original ARG
        :param node: a list of nodes [node]
        :param rem, if the move is  remove_recombination'''
        nodes_to_update = {}
        their_times = collections.defaultdict(set) #key:time, value (set of indexes)
        for n in node:
            if n.left_parent is not None:
                nodes_to_update[n.index] = self.arg.copy_node_segments(n)
                their_times[n.left_parent.time].add(n.index)
        while nodes_to_update:
            next_ind = their_times.pop(min(their_times))
            while next_ind:
                next_index = next_ind.pop()
                if nodes_to_update.__contains__(next_index):# BUG4_2
                    if self.arg.__contains__(next_index):# might pruned already in backtrack
                        nodes_to_update, nodes_times = self.update_ancestral_material(next_index,
                                                        nodes_to_update, their_times, backtrack)
                    else:
                        del nodes_to_update[next_index]

    def find_sc_original_parent(self, node):
        '''find second child of original_parent the original parent'''
        node = node.left_parent # this is original parent
        sc_original_parent = None; valid = True
        while node.left_parent is not None:
            if node.left_parent.index != node.right_parent.index:
                if node.left_parent.first_segment == None or\
                    node.right_parent.first_segment == None:
                    valid = False
                    break
                assert not self.new_names.__contains__(node.left_parent.index)
                sc_original_parent = node.left_parent
                break
            elif self.new_names.__contains__(node.left_parent.index):
                node = node.left_parent
            else:
                sc_original_parent = node.left_parent
                break
        return sc_original_parent, valid

    def find_original_parent(self, node):
        '''find the original parent for node, if any. Original parent is
        the parent of node in the original ARG'''
        original_parent = None; valid = True; second_child = None
        path =[node.index]
        while node.left_parent != None:
            if node.left_parent.index != node.right_parent.index:
                if node.left_parent.first_segment == None or\
                    node.right_parent.first_segment == None:
                    valid = False
                    break
                assert not self.new_names.__contains__(node.left_parent.index)
                original_parent = node.left_parent
                if node.left_child.index == path[-1]:
                    #sec child is the second child of the last node berfore original_p
                    second_child = node.right_child
                else:
                    second_child = node.left_child
                break
            elif self.new_names.__contains__(node.left_parent.index):
                path.append(node.index)
                node = node.left_parent
            else:
                original_parent = node.left_parent
                assert original_parent.left_child != original_parent.right_child
                if node.index == original_parent.left_child.index:
                    second_child = original_parent.right_child
                else:
                    second_child = original_parent.left_child
                break
        return original_parent, valid, second_child

    def find_original_child(self, node, second_child = False):
        '''find the original child for node, the child in the original ARG
        :param second_child if we find original child for second child of original parent.
            if False: find original child for new root
        '''
        original_child = None
        if not second_child:
            while node.left_child != None:
                if not self.floatings_to_ckeck.__contains__(node.left_child.index) and \
                    not self.floatings_to_ckeck.__contains__(node.right_child.index):
                    original_child = node
                    break
                elif self.floatings_to_ckeck.__contains__(node.left_child.index) and \
                    self.floatings_to_ckeck.__contains__(node.right_child.index):
                    break
                elif self.floatings_to_ckeck.__contains__(node.left_child.index):
                    node = node.right_child
                else:
                    node = node.left_child
        else:
            if not self.new_names.__contains__(node.index):# BUG3 NOTES
                    original_child = node
            else:
                while node.left_child != None:
                    if not self.new_names.__contains__(node.left_child.index) and \
                        not self.new_names.__contains__(node.right_child.index):
                        original_child = node
                        break
                    elif self.new_names.__contains__(node.left_child.index) and \
                        self.new_names.__contains__(node.right_child.index):
                        print("node is", node.index, "node.left_child.index", node.left_child.index,
                              "node.right_child.index", node.right_child.index)
                        print("self.new_names", self.new_names)
                        raise ValueError("There must be an original "
                                         "child for the second child.")
                        break
                    elif self.new_names.__contains__(node.left_child.index):
                        node = node.right_child
                    else:
                        node = node.left_child
        return original_child

    def is_new_root(self, node):
        '''
        is the node a new root  in  new ARG, so that will be floating in
         the reverse move?
         return original_parent and original_child. if both are not None, then yes node is
         a new root and need to calculate the reverse prob for that. otherwise, No!
         by original parent or original child, I mean the parent/child that
         exist in the original ARG.
         If a node doesnt have an original parent:
            1. it was a root in G_{j}
            2. it is a result of floating lineages, then no need to calc reverse

         '''
        original_child = None
        original_parent = None
        valid = True
        second_child = None
        if node.left_child.first_segment !=  None and\
                node.right_child.first_segment != None:
            # find original parent
            original_parent, valid, second_child = self.find_original_parent(node)
            if valid and original_parent != None:
                #now find original child
                original_child = self.find_original_child(node)
        return original_parent, original_child, valid, second_child

    def spr_reattach_floatings(self, detach, sib, old_merger_time):
        '''reattach all the floatings including detach'''
        while self.floatings:
            min_time = self.floatings.min_key()
            node = self.arg.nodes[self.floatings[min_time]]
            self.floatings.discard(min_time)
            # check if it is still floating
            still_floating = False
            if node.index == detach.index:
                still_floating = True
            elif node.first_segment is not None and node.left_parent is None:
                still_floating = True
            if not still_floating:
                self.floatings.discard(min_time)
            else:
                # first check if the move is valid ---> do this in cleanup
                # if  node.left_child.first_segment is None or \
                #     node.right_child.first_segment is None:
                #     # this move is not valid
                #     pass
                # --- reattach
                # 1. find potential reattachment
                all_reattachment_nodes = self.spr_reattachment_nodes(min_time)
                if node.left_parent is not None: # only F
                    all_reattachment_nodes.discard(node.left_parent.index)
                    #C is not added to all_reattach_nodes: because t_c<t_F
                    # and sib.left_parent = None ---after detach if P is root
                    all_reattachment_nodes[sib.index] = sib.index
                all_reattachment_nodes.discard(node.index)
                if all_reattachment_nodes.is_empty(): # BUG2 in notes, the grand root
                    print("node", node.index)
                    print("self.floatings", self.floatings)
                    raise ValueError("all_reattachment_nodes is empty."
                                     " There must be atleast one node to reattach.")
                reattach = self.arg.__getitem__(random.choice(list(all_reattachment_nodes)))
                print("node", node.index, "rejoins to ", reattach.index)
                #---trans_prob for choose reattach
                self.transition_prob.spr_choose_reattach(len(all_reattachment_nodes))
                max_time = max(min_time, reattach.time)
                if node.index == detach.index and reattach.index == sib.index:
                    self.floatings.discard(old_merger_time) # this is for removing sib
                if reattach.left_parent is None:
                    new_merger_time = max_time + random.expovariate(self.lambd)
                    print("new time is from an exponential distribution"
                          " + reattach.time:", reattach.time)
                    print("new_merger_time", new_merger_time)
                    self.transition_prob.spr_reattach_time(new_merger_time, max_time, 0,
                                                           True, True, self.lambd)
                else:
                    new_merger_time = random.uniform(max_time, reattach.left_parent.time)
                    # new_merger_time = self.truncated_expo(max_time, reattach.left_parent.time, self.lambd)
                    self.transition_prob.spr_reattach_time(new_merger_time, max_time,
                                                   reattach.left_parent.time, False)
                    print("new_merger_time", new_merger_time)
                #-- reattach
                self.new_names = self.arg.reattach(node, reattach, new_merger_time, self.new_names)
                #---- update
                self.update_all_ancestral_material([node])
                #---
                self.floatings.discard(reattach.time) #if any

    def revert_spr(self, detach, sib, old_merger_time, oldparent_ind):
        '''revert to the original ARG'''
        new_sib = detach.sibling()
        self.arg.detach(detach, new_sib)
        new_names = self.arg.reattach(detach, sib, old_merger_time, self.new_names)
        # do we need to check the grand parent too? No
        # update materials
        if self.new_names.__contains__(new_sib.index):# BUG4 NOTES
            self.update_all_ancestral_material([detach], True)
        else:
            self.update_all_ancestral_material([detach, new_sib], True)
        if detach.left_parent.index != oldparent_ind:# BUG6
            new_name = detach.left_parent.index
            self.arg.rename(new_name, oldparent_ind)
            self.arg.coal.discard(new_name)

    def clean_up(self, coal_to_cleanup):
        '''clean up the Accepted ARG. the ancestral material and snps
        has already set up. This method is only for cleaning the ARG from NAM lineages
        NAM is No ancestral Material nodes that are not a root.
        order the nodes by time and then clean up'''
        def reconnect(child, node):# BUG7
            '''from child--> node--> parent: TO child ---> parent '''
            leftparent = node.left_parent
            rightparent = node.right_parent
            child.left_parent = leftparent
            child.right_parent = rightparent
            child.breakpoint = node.breakpoint
            leftparent.update_child(node, child)
            rightparent.update_child(node, child)

        while coal_to_cleanup:
            node = self.arg.__getitem__(coal_to_cleanup.pop(coal_to_cleanup.min_key()))
            if node.left_child == None and node.right_child == None:
                if node.left_parent is not None:
                    assert node.left_parent.index == node.right_parent.index
                    node.left_parent.update_child(node, None)
            elif node.left_child != None and node.right_child is None:
                if node.left_parent is not None:
                    reconnect(node.left_child, node)
            elif node.right_child != None and node.left_child is None:
                if node.left_parent is not None:
                    reconnect(node.right_child, node)
            else: # both not None
                if node.left_child.first_segment == None and\
                        node.right_child.first_segment == None:
                    if node.left_parent is not None:
                        assert node.first_segment is  None
                        assert node.left_parent.index == node.right_parent.index
                        node.left_child.left_parent = None
                        node.left_child.right_parent = None
                        node.right_child.left_parent = None
                        node.right_child.right_parent = None
                        node.left_parent.update_child(node, None)
                elif node.left_child.first_segment != None and\
                        node.right_child.first_segment is None:
                    assert node.left_parent is not None
                    reconnect(node.left_child, node)
                elif node.right_child.first_segment != None and\
                        node.left_child.first_segment is None:
                    assert node.left_parent is not None
                    reconnect(node.right_child, node)
                else: # non None
                    raise ValueError("both children have seg, so "
                                     "this shouldn't be in coal_to_clean")
            del self.arg.nodes[node.index]

    def spr_validity_check(self, node,  clean_nodes,
                           detach_snps, detach, completed_snps, reverse_done):
        '''after rejoining all the floating, it is time to check if
        the changes cancels any recombination.
        Also, if there is a new root (node.segment == None and node.left_parent!=None),
        calculate the reverse prob for them.
        In addition, whether there is a incompatibility.
        This method is responsible for checking the above mentioned for node.
        :param node: the node we need to check its validability and revers prob (if applicable)
        :param detach: the detach node in spr
        :param clean_nodes: a dict of k: time, v: node.indexes the nodes we need to check for validity
        '''
        valid = True
        if node.first_segment != None:# not a new root
            assert node.left_parent != None
            # check incompatibility if both children have segments
            if node.left_child.index != node.right_child.index:
                # first delete all the detach_snps in node.snps
                node.snps = node.snps.difference(detach_snps)
                node.snps = node.snps.difference(completed_snps)
                #find all SNP on left_child
                lc_variants = node.left_child.get_variants(self.data)
                #find all SNP on right child
                rc_variants = node.right_child.get_variants(self.data)
                # find the intersection of them
                intersect_variants = detach_snps.intersection(lc_variants, rc_variants)
                for snp in intersect_variants:
                    S1 = node.left_child.x_segment(snp).samples
                    S2 = node.right_child.x_segment(snp).samples
                    valid, detach_snps, completed_snps = \
                        self.incompatibility_check(node, S1, S2, snp, detach_snps,
                                                   completed_snps)
                    if not valid:
                        break
            else:
                assert len(node.snps) == 0
            clean_nodes[node.left_parent.time].add(node.left_parent.index)
            clean_nodes[node.right_parent.time].add(node.right_parent.index)
            #add the parents to the clean_node
        elif node.left_parent != None:
            if node.left_parent.index != node.right_parent.index:
                valid = False
            elif node.left_child.index == node.right_child.index:
                valid = False
            else:
                # might be new root
                # is it really new root?
                original_parent, original_child,\
                valid, second_child = self.is_new_root(node)
                # if we have original parent and original child
                # the floating will be from node.time and rejoins to original.parent.time
                # reverse: choose original parent, choose time on
                # original_parent's second_child.time to original parent.left_parent.time
                # then add what to clean_up? node.left_parent, since we need to check incompatibility
                # without considering incompatibility then original_parent.
                #second child is the second child of the original_parent. we need to find its original child
                # and original parent (if any) for the time prob calculation
                if valid and original_parent != None and original_child != None and\
                        not reverse_done.__contains__(node.index):# NOTE AFTER BUG5
                    all_reattachment_nodes = self.spr_reattachment_nodes(node.time, False)
                    all_reattachment_nodes.discard(original_parent.index)
                    all_reattachment_nodes.discard(node.index)
                    # ---- reverse of choosin g a lineage to rejoin
                    self.transition_prob.spr_choose_reattach(len(all_reattachment_nodes), False)
                    #----- reverse prob time
                    if node.index != detach.index: # already done for detach
                        # find the origin child and origin parent of second child (if any)
                        sc_origin_child = self.find_original_child(second_child, True)
                        assert sc_origin_child is not None
                        if second_child.left_parent.index == original_parent.index:
                            # find original parent for second child
                            sc_original_parent, valid = self.find_sc_original_parent(second_child)
                        else:
                            # original parent is a parent of a rec then:
                            sc_original_parent = original_parent
                        if valid:
                            assert node.index != second_child.index
                            if sc_original_parent is None:
                                self.transition_prob.spr_reattach_time(original_parent.time,
                                        max(node.time, second_child.time), 0 , True, False, self.lambd)
                            else:
                                self.transition_prob.spr_reattach_time(original_parent.time,
                                        max(node.time, second_child.time), sc_original_parent.time
                                                                       ,False, False)
                            reverse_done[second_child.index] = second_child.index
                # add nodeleft parent ot cleanup
                clean_nodes[node.left_parent.time].add(node.left_parent.index)
        return valid, clean_nodes, detach_snps, completed_snps, reverse_done

    def all_validity_check(self, clean_nodes, detach_snps, detach):
        '''do spr_validity_check()for all the needed nodes
        '''
        valid = True # move is valid
        completed_snps = bintrees.AVLTree() # those of detach_snps that completed already
        reverse_done = bintrees.AVLTree()# reverse prob in done for them
        while valid and clean_nodes:
            # get the min_time one
            nodes = clean_nodes.pop(min(clean_nodes))
            assert 0 < len(nodes) <= 2
            if len(nodes) == 2:# two rec parents
                nodes = [self.arg.__getitem__(nodes.pop()), self.arg.__getitem__(nodes.pop())]
                assert nodes[0].left_child.index == nodes[0].right_child.index
                assert nodes[1].left_child.index == nodes[1].right_child.index
                assert nodes[0].left_child.index == nodes[1].left_child.index
                if nodes[0].first_segment is None or nodes[1].first_segment is None:
                    valid = False # cancels a rec
                    break
            else:
                assert len(nodes) == 1
                nodes = [self.arg.__getitem__(nodes.pop())]
            while nodes:
                node = nodes.pop(0)
                valid, clean_nodes, detach_snps, completed_snps, reverse_done = \
                    self.spr_validity_check(node, clean_nodes, detach_snps,
                                            detach, completed_snps, reverse_done)
                if not valid:
                    break
        return valid

    def spr(self):
        '''
        Transition number 1
        perform an SPR move on the ARG
        '''
        assert self.arg.__len__() == (len(self.arg.coal) + len(self.arg.rec) + self.n)
        # Choose a random coalescence node, and one of its children to detach.
        # Record the current sibling and merger time in case move is rejected.
        # TODO: We need a better way to sample a uniform choice from an AVL tree, or
        # a better container.
        detach = self.arg.__getitem__(random.choice(list(self.arg.coal)))
        if random.random() < 0.5:
            detach = detach.left_child
        else:
            detach = detach.right_child
        oldparent_ind = detach.left_parent.index
        self.floatings[detach.time] = detach.index
        self.floatings_to_ckeck[detach.index] = detach.index
        self.new_names[detach.left_parent.index] = detach.left_parent.index
        #---- forward trans prob of choosing detach
        self.transition_prob.spr_choose_detach(len(self.arg.coal))#1
        #---------------
        old_merger_time = detach.left_parent.time
        print("detach node", detach.index, " time", detach.time)
        sib = detach.sibling()
        print("sibling node", sib.index, "parent", detach.left_parent.index)
        #-- reverse transition for time
        if detach.left_parent.left_parent is None:
            self.transition_prob.spr_reattach_time(old_merger_time, max(detach.time, sib.time),
                                                   0, True, False, self.lambd)
            self.floatings[sib.left_parent.time] = sib.index
            self.floatings_to_ckeck[sib.index] = sib.index
        else:
            self.transition_prob.spr_reattach_time(old_merger_time, max(detach.time, sib.time),
                                                sib.left_parent.left_parent.time, False, False)
        #---detach
        self.arg.detach(detach, sib)
        #--- update arg
        if sib.left_parent is not None:
            self.update_all_ancestral_material([sib])
        #---- reattach all
        self.spr_reattach_floatings(detach, sib, old_merger_time)
        #update arg.coal---> what about new ones? added in reattach
        # coal_to_cleanup = self.update_arg_coal()
        if self.NAM_recParent: # rec is canceled, reject
            print("not valid due to removing rec")
            valid = False
        else:
            #=--- Vlidity check, do mutations, and revers probs
            #--- snps we need to check for incompatibility
            detach_snps = self.get_detach_SF(detach)
            all_reattachment_nodes = self.spr_reattachment_nodes(detach.time, False)
            all_reattachment_nodes.discard(detach.left_parent.index)
            all_reattachment_nodes.discard(detach.index)
            clean_nodes = collections.defaultdict(set) #key: time, value: nodes
            clean_nodes[detach.left_parent.time].add(detach.left_parent.index)
            if sib.left_parent is not None and \
                    sib.left_parent.index != detach.left_parent.index:
                clean_nodes[sib.left_parent.time].add(sib.left_parent.index)
                clean_nodes[sib.right_parent.time].add(sib.right_parent.index)# if rec
            else: # sib is a root with seg = None and left_parent  = None
                all_reattachment_nodes[sib.index] = sib.index
            self.transition_prob.spr_choose_reattach(len(all_reattachment_nodes), False)
            valid = self.all_validity_check(clean_nodes, detach_snps, detach)
        print("valid is ", valid)
        if valid:
            #-- calc prior and likelihood and then M-H
            old_branch_length = self.arg.branch_length
            old_anc_recomb = self.arg.num_ancestral_recomb
            old_nonancestral_recomb = self.arg.num_nonancestral_recomb
            new_log_lk = self.arg.log_likelihood(self.theta, self.data)
            new_log_prior, new_roots, new_coals = \
                self.arg.log_prior(self.n,self.seq_length, self.rho,
                                   self.Ne, True, True)
            #--- reverse prob choose detach
            self.transition_prob.spr_choose_detach(len(new_coals), False)
            self.Metropolis_Hastings(new_log_lk, new_log_prior)
            print("mh_accept ", self.accept)
            #if acceptt
            if self.accept: # clean up
                # the ancestral material already set up. we just need to
                # remove the NAM nodes.
                self.clean_up(self.coal_to_cleanup)
                self.check_root_parents(new_roots)
                self.arg.roots = new_roots # update roots
                self.arg.coal = new_coals
            else: # rejected: retrieve the original- backtrack--> no changes on arg.coal
                assert old_anc_recomb + old_nonancestral_recomb ==\
                       self.arg.num_ancestral_recomb + self.arg.num_nonancestral_recomb
                self.arg.branch_length = old_branch_length
                self.arg.num_ancestral_recomb = old_anc_recomb
                self.arg.num_nonancestral_recomb = old_nonancestral_recomb
                self.revert_spr(detach, sib, old_merger_time, oldparent_ind)
        else:
            self.revert_spr(detach, sib, old_merger_time, oldparent_ind)
        self.empty_containers()

    #============
    # remove recombination

    def empty_containers(self):
        '''empty all the containers'''
        self.floatings.clear()
        self.NAM_recParent.clear()
        self.NAM_coalParent.clear()
        self.coal_to_cleanup.clear()
        self.new_names.clear()
        self.floatings_to_ckeck.clear()
        self.transition_prob.log_prob_forward = 0
        self.transition_prob.log_prob_reverse = 0
        self.arg.nextname = max(self.arg.nodes) + 1
        self.arg.get_available_names()

    def detach_otherParent(self, remParent, otherParent, child):
        '''child ---rec---(remPrent, otherParent), now that we've removed
        remParent, rejoin child to otherParent.left_parent and detach otherParent
        a. If remParent and otherParent coalesce back: parent of both rem and other is equal 
            then we should rejoin child to  otherParent.left_parent.left_parent (must exist)
        b. Otherwise:
            rejoin child to otherParent.left_parent
        '''
        if remParent.left_parent.index == otherParent.left_parent.index:
            invisible = True # rec is invisible
            assert remParent.left_parent.left_parent is not None
            othergrandparent = otherParent.left_parent
            child.left_parent = othergrandparent.left_parent
            child.right_parent = othergrandparent.right_parent
            child.breakpoint = othergrandparent.breakpoint
            parent = othergrandparent.left_parent
            parent.update_child(othergrandparent, child)
            parent = othergrandparent.right_parent
            parent.update_child(othergrandparent, child)
        else:
            invisible = False# rec is not invisible
            child.left_parent = otherParent.left_parent
            child.right_parent = otherParent.right_parent
            child.breakpoint = otherParent.breakpoint
            parent = otherParent.left_parent
            parent.update_child(otherParent, child)
            parent = otherParent.right_parent
            parent.update_child(otherParent, child)
        return invisible

    def revert_remove_recombination(self, remParent, otherParent,
                                child, remGrandparent, remPsib,
                                old_child_bp, invisible):
        '''revert REMOVE rec to the original ARG'''
        self.arg.nodes[remParent.index] = remParent
        self.arg.nodes[otherParent.index] = otherParent
        self.arg.nodes[remGrandparent.index] = remGrandparent
        if invisible:
            assert remParent.left_parent.index == remGrandparent.index
            assert otherParent.left_parent.index == remGrandparent.index
            assert remParent.left_child.index == child.index
            assert otherParent.left_child.index == child.index
            child.left_parent.update_child(child, remGrandparent)
            child.right_parent.update_child(child, remGrandparent)
            remGrandparent.breakpoint = child.breakpoint
        else:
            new_names = self.arg.reattach(remParent, remPsib,
                             remParent.left_parent.time, self.new_names)
            # reattach other parent to child
            assert otherParent.left_child.index == child.index
            assert remParent.left_child.index == child.index
            otherParent.left_parent = child.left_parent
            otherParent.right_parent = child.right_parent
            child.left_parent.update_child(child, otherParent)
            child.right_parent.update_child(child, otherParent)
            otherParent.breakpoint = child.breakpoint
        child.left_parent = remParent
        child.right_parent = otherParent
        if remParent.first_segment.left > otherParent.first_segment.left:
            child.right_parent = remParent
            child.left_parent = otherParent
        child.breakpoint = old_child_bp
        #ancestral material
        self.update_all_ancestral_material([remParent, otherParent], True)

    def remove_recombination(self):
        '''
        transition number 2
        remove a recombination event
        1. randomly choose a rec parent (remParent) and calc the forward prob
        2. if remParent.left_parent!= remParent.right_parent: reject, else:
            detach(remNode), detach(otherParent),
        3. update ancestral material
        4. reattach the floatings (if any)
        5. check validity, compatibility and reverse prob
        6. if not valid, revert the move
        '''
        assert self.arg.__len__() == (len(self.arg.coal) + len(self.arg.rec) + self.n)
        assert not self.arg.rec.is_empty()
        #1. choose a rec parent
        remParent = self.arg.__getitem__(random.choice(list(self.arg.rec.keys())))
        assert remParent.left_child == remParent.right_child
        #-- forward transition prob
        self.transition_prob.rem_choose_remParent(len(self.arg.rec))#1
        if remParent.left_parent.index != remParent.right_parent.index:
            valid = False
        else:
            #--- detach R:
            remPsib = remParent.sibling()
            remGrandparent = remParent.left_parent
            #-----
            child = remParent.left_child
            old_child_bp = child.breakpoint
            otherParent = child.left_parent
            if otherParent.index == remParent.index:
                otherParent = child.right_parent
            if remGrandparent.left_parent is None:
                self.floatings[remPsib.left_parent.time] = remPsib.index
                self.floatings_to_ckeck[remPsib.index] = remPsib.index
            if remParent.left_parent.index != otherParent.left_parent.index:#visible
                self.arg.detach(remParent, remPsib)
            #--- detach otherParent
            print("remPrent",remParent.index, "otherParent",
                  otherParent.index,"child", child.index)
            print("remPsib", remPsib.index, "remGrandparent",
                  remGrandparent.index)
            print("otherparentlp", otherParent.left_parent.index,
                  "otherparentrp", otherParent.right_parent.index)
            print("childlp,", child.left_parent.index, "childrp",
                  child.right_parent.index)
            invisible = self.detach_otherParent(remParent, otherParent, child)
            print("invisible", invisible)
            #--- remove them from ARG
            remParent = self.arg.nodes.pop(remParent.index)
            otherParent = self.arg.nodes.pop(otherParent.index)
            remGrandparent = self.arg.nodes.pop(remGrandparent.index)
            #--- update ancestral material
            if invisible or remGrandparent.left_parent is None: #otherParent == remPsib
                self.update_all_ancestral_material([child])
            else:#
                self.update_all_ancestral_material([child, remPsib])
            #--- reattach all the floatings (if any) NOTE: the *args are fake:
            self.spr_reattach_floatings(remParent, otherParent, remParent.time)
            # is there any canceled rec?
            if self.NAM_recParent:
                print("not valid due to removing rec")
                valid = False
            else:
                #--- check validity, compatibility and mutations, reverse prob
                remParent_snps = self.get_detach_SF(remParent)
                clean_nodes = collections.defaultdict(set) #key: time, value: nodes
                assert child.left_parent is not None
                clean_nodes[child.left_parent.time].add(child.left_parent.index)
                clean_nodes[child.right_parent.time].add(child.right_parent.index)
                if not invisible and remPsib.left_parent != None: #not invisible rec
                    clean_nodes[remPsib.left_parent.time].add(remPsib.left_parent.index)
                    clean_nodes[remPsib.right_parent.time].add(remPsib.right_parent.index)
                valid = self.all_validity_check(clean_nodes, remParent_snps, remParent)#third *arg is fake
            print("valid is ", valid)
            if valid:
                #-- calc prior and likelihood and then M-H
                old_branch_length = self.arg.branch_length
                old_anc_recomb = self.arg.num_ancestral_recomb
                old_nonancestral_recomb = self.arg.num_nonancestral_recomb
                new_log_lk = self.arg.log_likelihood(self.theta, self.data)
                new_log_prior, new_roots, new_coals = self.arg.log_prior(self.n,
                                                self.seq_length, self.rho, self.Ne, True, True)
                #== reverse probability (ADD REC):
                #1.choose a lin to add rec on
                numnodes = self.arg.__len__()
                empty_nodes = len(self.NAM_coalParent.union(new_roots))#exclude roots
                self.transition_prob.add_choose_node(numnodes - empty_nodes, False)
                #2. choose a time on child to add rec
                self.transition_prob.spr_reattach_time(remParent.time,child.time,
                                        child.left_parent.time, False, False)
                #3. choose a breakpoint on child
                self.transition_prob.add_choose_breakpoint(child.num_links(), False)
                #4. choose one to float
                self.transition_prob.add_choose_node_to_float(False)
                #5. reattach remParent to remPsib
                all_reattachment_nodes = self.spr_reattachment_nodes(remParent.time)
                #6. time of reaattachment
                if remGrandparent.left_parent is None:
                    self.transition_prob.spr_reattach_time(remGrandparent.time,
                                                           max(remParent.time, remPsib.time),
                                                           0, True, False, self.lambd)
                else:
                    self.transition_prob.spr_reattach_time(remGrandparent.time,
                                                           max(remParent.time, remPsib.time),
                                                           remGrandparent.left_parent.time,
                                                           False, False)
                assert not all_reattachment_nodes.__contains__(remParent.index)
                assert not all_reattachment_nodes.__contains__(remGrandparent.index)
                assert all_reattachment_nodes.__contains__(child.index)
                self.transition_prob.spr_choose_reattach(len(all_reattachment_nodes), False)
                #--- now mh
                self.Metropolis_Hastings(new_log_lk, new_log_prior)
                print("mh_accept ", self.accept)
                if self.accept:
                    #update coal, and rec
                    self.clean_up(self.coal_to_cleanup)
                    self.check_root_parents(new_roots)
                    self.arg.roots = new_roots # update roots
                    self.arg.coal = new_coals
                    self.arg.rec.discard(remParent.index)
                    self.arg.rec.discard(otherParent.index)
                else:
                    assert old_anc_recomb + old_nonancestral_recomb - 1 ==\
                        self.arg.num_ancestral_recomb + self.arg.num_nonancestral_recomb
                    self.arg.branch_length = old_branch_length
                    self.arg.num_ancestral_recomb = old_anc_recomb
                    self.arg.num_nonancestral_recomb = old_nonancestral_recomb
                    self.revert_remove_recombination(remParent, otherParent, child,
                                            remGrandparent, remPsib, old_child_bp, invisible)
            else:
                self.revert_remove_recombination(remParent, otherParent, child,
                                            remGrandparent, remPsib, old_child_bp, invisible)
        self.empty_containers()

    #=============
    #  ADD recombination

    def check_root_parents(self, new_roots):
        '''after mh acceptance, make sure all the new roots dont have any parent
        This is not taken care of in clean_up(), because coal_to_clean does
        not contain nodes without seg where both children have segments (roots)
        '''
        for ind in new_roots:
            root = self.arg.__getitem__(ind)
            assert root.first_segment is None
            if root.left_parent is not None:
                assert not self.arg.nodes.__contains__(root.left_parent.index)
                root.left_parent = None
                root.right_parent = None
            else:
                root.right_parent = None

    def revert_add_recombination(self, child, followParent, detachParent, oldbr):
        '''revert ADD rec move'''
        # detach detachParent
        sib = detachParent.sibling()
        self.arg.detach(detachParent, sib)
        child.left_parent = followParent.left_parent
        child.right_parent = followParent.right_parent
        child.breakpoint = oldbr
        followParent.left_parent.update_child(followParent, child)
        followParent.right_parent.update_child(followParent, child)
        self.arg.rec.discard(followParent.index)
        self.arg.rec.discard(detachParent.index)
        del self.arg.nodes[followParent.index]
        del self.arg.nodes[detachParent.index]
        del self.arg.nodes[detachParent.left_parent.index]
        if sib.index == followParent.index or\
                self.new_names.__contains__(sib.index):#invisible rec #BUG4 NOTES
            self.update_all_ancestral_material([child], True)
        else:
            self.update_all_ancestral_material([child, sib], True)

    def add_choose_child(self):
        '''choose a node to put a recombination on it'''
        ind = random.choice(list(self.arg.nodes.keys()))
        if not self.arg.roots.__contains__(ind):
            return self.arg.__getitem__(ind)
        else:
            return self.add_choose_child()

    def split_node(self, child, k, t):
        '''split a node (child) to two parent node from k at time t
        and add the parents to the arg
        '''
        s = self.arg.copy_node_segments(child) #child segments
        y = self.find_break_seg(s, k)
        if y is None:
            print("in split node", "child is", child.index, "k:", k,
                  "child_head", child.first_segment.left,
                  "child tail right", child.get_tail().right, "y", y)
        assert y is not None
        x = y.prev
        if y.left < k:
            assert k < y.right
            z = self.arg.alloc_segment(k, y.right, y.node,
                                       y.samples, None, y.next)
            if y.next is not None:
                y.next.prev = z
            y.next = None
            y.right = k
            lhs_tail = y
        else:
            assert x is not None
            x.next = None
            y.prev = None
            z = y
            lhs_tail = x
        leftparent = self.arg.alloc_node(self.arg.new_name(),
                                         t, lhs_tail.node, lhs_tail.node)
        lhs_tail.node.left_parent = leftparent
        self.arg.store_node(lhs_tail, leftparent)
        rightparent = self.arg.alloc_node(self.arg.new_name(),
                                          t, z.node, z.node)
        z.node.right_parent = rightparent
        self.arg.store_node(z, rightparent)
        #--- update breakpoint
        child.breakpoint = k
        #--- add to rec
        self.arg.rec[leftparent.index] = leftparent.index
        self.arg.rec[rightparent.index] = rightparent.index
        return leftparent, rightparent

    def add_recombination(self):
        '''
        Transition number 3
        add a recombination to the ARG
        1. randomly choose a node excluding roots
        2. randomly choose a time on the node
        3. randomly choose a breakpoint on the node and split the node to two
        4. randomly choose a parent to follow the path, the other to float
        5. update ancestral material
        6. do spr on the floating node to reattach it to the ARG
            a) randomly choose the potential node to rejoin the detach
            b) rejoin the floating and update ancestral material
            c) rejoin all the floatings
            d) check the validity and compatibility
            e) m-h
        forward transition: also reattach detachPrent, choose time to reattach
        '''
        assert self.arg.__len__() == (len(self.arg.coal) + len(self.arg.rec) + self.n)
        child = self.add_choose_child()
        assert child.first_segment != None and child.left_parent != None
        self.transition_prob.add_choose_node(len(self.arg.nodes) - len(self.arg.roots))#1
        head = child.first_segment
        tail = child.get_tail()
        if tail.right - head.left <= 1:# no link
            print("the node has no link")
            valid = False
        else:
            #breakpoint and time
            self.transition_prob.add_choose_breakpoint(tail.right - head.left - 1)#2
            break_point = random.choice(range(head.left + 1, tail.right))
            new_rec_time = random.uniform(child.time, child.left_parent.time)
            self.transition_prob.spr_reattach_time(new_rec_time, child.time,
                                                   child.left_parent.time, False)#3
            oldleftparent = child.left_parent
            oldrightparent = child.right_parent
            oldbreakpoint = child.breakpoint
            newleftparent, newrightparent = self.split_node(child, break_point, new_rec_time)
            #--- choose one to follow child path
            followParent = random.choice([newleftparent, newrightparent])
            if followParent.index == newleftparent.index:
                detachParent = newrightparent
            else:
                detachParent = newleftparent
            self.transition_prob.add_choose_node_to_float()#4
            print("child", child.index, "followparent", followParent.index, "detach", detachParent.index,
                  "old_leftpanre", oldleftparent.index, "oldrightparent", oldrightparent.index)
            #----
            followParent.breakpoint = oldbreakpoint
            followParent.left_parent = oldleftparent
            followParent.right_parent = oldrightparent
            oldleftparent.update_child(child, followParent)
            oldrightparent.update_child(child, followParent)
            #now update ancestral material
            print("child.left_parent", child.left_parent.index)
            print("childrp", child.right_parent.index)
            self.update_all_ancestral_material([followParent])
            #--- reattach detachParent
            self.floatings[detachParent.time] = detachParent.index
            self.floatings_to_ckeck[detachParent.index] = detachParent.index
            self.spr_reattach_floatings(child, child, child.time) # fake *args
            if self.NAM_recParent: # rec is canceled, reject
                print("not valid due to removing rec")
                valid = False
            else:
                #validability, compatibility
                detach_snps = self.get_detach_SF(detachParent)
                clean_nodes = collections.defaultdict(set) #key: time, value: nodes
                clean_nodes[child.left_parent.time].add(child.left_parent.index)
                clean_nodes[child.right_parent.time].add(child.right_parent.index)
                valid = self.all_validity_check(clean_nodes, detach_snps, child)#thirs *arg fake
            if valid:
                old_branch_length = self.arg.branch_length
                old_anc_recomb = self.arg.num_ancestral_recomb
                old_nonancestral_recomb = self.arg.num_nonancestral_recomb
                #-- calc prior and likelihood and then M-H
                new_log_lk = self.arg.log_likelihood(self.theta, self.data)
                new_log_prior, new_roots, new_coals = self.arg.log_prior(self.n,
                                                self.seq_length, self.rho, self.Ne, True, True)
                #--- reverse prob--
                self.transition_prob.rem_choose_remParent(len(self.arg.rec), False)
                self.Metropolis_Hastings(new_log_lk, new_log_prior)
                print("mh_accept ", self.accept)
                if self.accept:
                    #update coal, and rec
                    self.clean_up(self.coal_to_cleanup)
                    self.check_root_parents(new_roots)
                    self.arg.roots = new_roots # update roots
                    self.arg.coal = new_coals
                else:
                    assert old_anc_recomb + old_nonancestral_recomb + 1 ==\
                        self.arg.num_nonancestral_recomb + self.arg.num_ancestral_recomb
                    self.arg.branch_length = old_branch_length
                    self.arg.num_ancestral_recomb = old_anc_recomb
                    self.arg.num_nonancestral_recomb = old_nonancestral_recomb
                    self.revert_add_recombination(child, followParent,
                                                 detachParent, oldbreakpoint)
            else:
                self.revert_add_recombination(child, followParent,
                                                detachParent, oldbreakpoint)
        self.empty_containers()
    #=========
    # adjust times move

    def adjust_times(self, calc_prior = True):
        '''
        Transition number 4
        modify the node times according to CwR
        also calculate the prior in place if calc_prior = true
        '''
        ordered_nodes = [v for k, v in sorted(self.arg.nodes.items(),
                                     key = lambda item: item[1].time)]
        number_of_lineages = self.n
        number_of_links = number_of_lineages * (self.seq_length - 1)
        number_of_nodes = self.arg.__len__()
        counter = self.n
        prev_t  = 0
        log_prior = 0
        original_t = [0 for i in range(number_of_nodes)] # for reverting the move
        while counter < number_of_nodes:
            node = ordered_nodes[counter]
            rate = (number_of_lineages * (number_of_lineages - 1)
                    / (4*self.Ne)) + (number_of_links * (self.rho))
            t = prev_t + random.expovariate(rate)
            # ret -= rate * (node.time - time)
            if node.left_child.index == node.right_child.index: #rec
                original_t[counter] = node.time
                original_t[counter +1] = node.time
                node.time = t
                ordered_nodes[counter+1].time = t
                gap = node.left_child.num_links()-\
                          (node.left_child.left_parent.num_links() +
                           node.left_child.right_parent.num_links())
                if calc_prior:
                    log_prior -= rate * (t - prev_t)
                    log_prior += math.log(self.rho)
                number_of_links -= gap
                number_of_lineages += 1
                counter += 2
            else: #CA
                original_t[counter] = node.time
                node.time = t
                if calc_prior:
                    log_prior -= rate * (t - prev_t)
                    log_prior -=  math.log(2*self.Ne)
                if node.first_segment == None:
                    node_numlink = 0
                    number_of_lineages -= 2
                    counter += 1
                else:
                    node_numlink = node.num_links()
                    number_of_lineages -= 1
                    counter += 1
                lchild_numlink = node.left_child.num_links()
                rchild_numlink = node.right_child.num_links()
                number_of_links -= (lchild_numlink + rchild_numlink) - node_numlink
            prev_t = t
        # m-h, without prior because it get cancels out with transition probs
        old_branch_length = self.arg.branch_length
        new_log_lk = self.arg.log_likelihood(self.theta, self.data)
        self.Metropolis_Hastings(new_log_lk, log_prior, trans_prob = False)
        print("mh_accept ", self.accept)
        if not self.accept:
            self.arg.branch_length = old_branch_length
            self.revert_adjust_times(ordered_nodes, original_t)
        self.empty_containers()

    def revert_adjust_times(self, ordered_nodes, original_t):
        '''revert the proposed ARG by
        adjust_times move to its orgiginal
        '''
        counter = self.n
        number_of_nodes = len(ordered_nodes)
        while counter < number_of_nodes:
            node = ordered_nodes[counter]
            node.time = original_t[counter]
            if node.left_child.index == node.right_child.index: #rec
                ordered_nodes[counter + 1].time = original_t[counter + 1]
                counter += 2
            else:
                counter += 1

    #==============
    #transition 5: adjust recombination position

    def adjust_breakpoint(self):
        '''transition number 5
        change the  breakpoint of
        an existing recombination
        1. randomly choose a recombination event
        2. simulate a new breakpoint for the rec
        3. update ancestral material
        4. if floating: reattach
        5. compatibility/ validity check
        transition would only be for floatings
        '''
        assert not self.arg.rec.is_empty()
        recparent = self.arg.__getitem__(random.choice(list(self.arg.rec.keys())))
        assert recparent.left_child.index == recparent.right_child.index
        child = recparent.left_child
        old_breakpoint = child.breakpoint
        print("child is ", child.index)
        # TODO complete this
        leftparent = child.left_parent
        rightparent  = child.right_parent
        assert leftparent.index != rightparent.index
        # simulate a new breakpoint
        child_head = child.first_segment
        child_tail = child.get_tail()
        new_breakpoint = random.choice(range(child_head.left + 1, child_tail.right))
        print("old_breakpoint is",  child.breakpoint)
        print("new_breakpoint:", new_breakpoint)
        y = self.find_break_seg(child_head, child.breakpoint)
        if y.prev is not None:
            print("x.right", y.prev.right)
        print("y left", y.left, "y.right", y.right)
        if new_breakpoint == old_breakpoint or\
                (not y.contains(old_breakpoint) and\
                 y.prev.right <= old_breakpoint<= y.left and\
                 y.prev.right <= new_breakpoint<= y.left):
            print("new_breakpoint is still non ancestral and at the same interval")
            # no prob changes, the breakpoint is still in the same non ancestral int
            child.breakpoint = new_breakpoint
            self.accept = True
        else:
            assert old_breakpoint != new_breakpoint
            start = old_breakpoint
            end = new_breakpoint
            if new_breakpoint < old_breakpoint:#
                start = new_breakpoint
                end = old_breakpoint
            # update ancestral material
            child.breakpoint = new_breakpoint
            self.update_all_ancestral_material([child])
            print("self.floating", self.floatings)
            #reattach flaotings if any
            self.spr_reattach_floatings(child, child, child.time)#fake *args
            # is there any canceled rec?
            if self.NAM_recParent:
                print("not valid due to removing rec")
                valid = False
            else: # check validity
                # get the affected snps from [start, end) interval
                child_snps = self.get_detach_SF(child, [start, end])
                clean_nodes = collections.defaultdict(set) #key: time, value: nodes
                clean_nodes[child.left_parent.time].add(child.left_parent.index)
                clean_nodes[child.right_parent.time].add(child.right_parent.index)
                valid = self.all_validity_check(clean_nodes, child_snps, child)#third *arg is fake
            if valid:
                #-- calc prior and likelihood and then M-H
                old_branch_length = self.arg.branch_length
                old_anc_recomb = self.arg.num_ancestral_recomb
                old_nonancestral_recomb = self.arg.num_nonancestral_recomb
                new_log_lk = self.arg.log_likelihood(self.theta, self.data)
                new_log_prior, new_roots, new_coals = self.arg.log_prior(self.n,
                                                self.seq_length, self.rho, self.Ne, True, True)
                #--- now mh
                self.Metropolis_Hastings(new_log_lk, new_log_prior)
                print("mh_accept ", self.accept)
                if self.accept:
                    #update coal, and rec
                    self.clean_up(self.coal_to_cleanup)
                    self.check_root_parents(new_roots)
                    self.arg.roots = new_roots # update roots
                    self.arg.coal = new_coals
                else:#revert
                    self.arg.branch_length = old_branch_length
                    self.arg.num_ancestral_recomb = old_anc_recomb
                    self.arg.num_nonancestral_recomb = old_nonancestral_recomb
                    child.breakpoint = old_breakpoint
                    self.update_all_ancestral_material([child], True)
            else:
                child.breakpoint = old_breakpoint
                self.update_all_ancestral_material([child], True)
        self.empty_containers()
    #===================
    #transition 6 : Kuhner move

    def find_active_nodes(self, t):
        '''nodes higher than t and also active nodes immediately after t'''
        for node in self.arg.nodes.values():
            if node.time >t:
                self.higher_times[node.time].add(node.index)
            elif node.left_parent != None and node.left_parent.time > t:
                self.active_nodes[node.index] = node.index

    def update_prune_parent(self, prune, node, deleted_nodes, parents):
        '''if prune is a child of a CA event:
        easily detach it and remove its parent.
        if sibling is root, depending on its time add to float
        If prune is a child of a rec, delete both parents,
        and continue deleting them till reach a CA, then detach
        '''
        if node.left_parent is None:
            # is possible only if node is  parent of a rec,
            # so it should have already remove in else option
            assert not self.arg.__contains__(node.index)
            assert not self.floats.__contains__(node.index)
            assert not self.partial_floatings.__contains__(node.index)
            assert not self.active_nodes.__contains__(node.index)
        elif node.left_parent.index == node.right_parent.index:
            sib = node.sibling()
            if deleted_nodes.__contains__(sib.index):
                if node.left_parent.left_parent != None:
                    parents[node.left_parent.index] = node.left_parent
                    if parents.__contains__(sib.index):
                        del parents[sib.index]
                        self.need_to_visit.discard(sib.index)
            else:
                self.arg.detach(node, sib)
                self.need_to_visit[sib.index] = sib.index
                if sib.left_parent is None:
                    assert sib.left_parent == None
                    if sib.time <= prune.time:
                        assert sib.first_segment != None
                        self.floats[sib.index] = sib.index
                        self.active_links += sib.num_links()
                        self.active_nodes.discard(sib.index)
                        if self.partial_floatings.__contains__(sib.index):#B
                            ch = self.partial_floatings.pop(sib.index)
                            self.active_links -= ch[2]
                    if self.floats.__contains__(node.left_parent.index):#B
                        self.floats.discard(node.left_parent.index)
                        self.active_links -= node.left_parent.num_links()
                    self.need_to_visit.discard(node.left_parent.index)
            deleted_nodes[node.left_parent.index] = node.left_parent.index
            self.need_to_visit.discard(node.left_parent.index)
            if self.arg.__contains__(node.left_parent.index):
                del self.arg.nodes[node.left_parent.index]
                del self.higher_times[node.left_parent.time]
            else:
                assert not self.higher_times.__contains__(node.left_parent.time)
        else:
            parents[node.left_parent.index] = node.left_parent
            parents[node.right_parent.index] = node.right_parent
            deleted_nodes[node.left_parent.index] = node.left_parent.index
            deleted_nodes[node.right_parent.index] = node.right_parent.index
            if self.floats.__contains__(node.left_parent.index):
                self.floats.discard(node.left_parent.index)
                self.active_links -= node.left_parent.num_links()
            self.need_to_visit.discard(node.left_parent.index)
            if self.floats.__contains__(node.left_parent.index):
                self.floats.discard(node.right_parent.index)
                self.active_links -= node.right_parent.num_links()
            self.need_to_visit.discard(node.right_parent.index)
            del self.higher_times[node.left_parent.time]
            del self.arg.nodes[node.left_parent.index]
            del self.arg.nodes[node.right_parent.index]
        return deleted_nodes, parents

    def update_prune_parents(self, prune):
        parents = {prune.index: prune}
        deleted_nodes = bintrees.AVLTree()
        while parents:
            node = parents.pop(min(parents))
            deleted_nodes, parents = self.update_prune_parent(prune, node,
                                                              deleted_nodes, parents)
        prune.left_parent = None
        prune.right_parent = None
        prune.breakpoint = None

    def get_active_links(self):
        self.active_links = 0
        for ind in self.floats:
            num_link = self.arg.__getitem__(ind).num_links()
            self.active_links += num_link
        for ind in self.partial_floatings:
            self.active_links += self.partial_floatings[ind][2]

    def new_event_time(self, lower_time = 0, upper_time = 1, passed_gmrca= False):
        '''
        simulate the time of a new event
        given a time interval (lower, upper), if the new time is in between,
        there is a new event. Three types of events:
            1. coal between two floatings
            2. coal between a floating and one from the rest actve nodes
            3. a rec on a floating lineage
        if passed_gmrca is True: we already passed the original GMRCA,
            so there is no upper_time and any time is acceptable. and the
            events are 1. a coal between two floatings or a rec on a floating
        :return:
        '''
        assert len(self.floats) != 0 or self.active_links != 0
        assert self.floats.is_disjoint(self.active_nodes)
        coalrate_bothF = (len(self.floats) * (len(self.floats) - 1)/2)/(2*self.Ne)
        coalrate_1F1rest = (len(self.floats) * len(self.active_nodes))/(2*self.Ne)
        recrate = self.rho * self.active_links
        totrate = coalrate_bothF + coalrate_1F1rest + recrate
        new_time = lower_time + random.expovariate(totrate)
        if not passed_gmrca:
            if new_time  >= upper_time:
                # no new event in the time interval
                return False
            else:
                if random.random() < (recrate/totrate):
                    return ["REC", new_time]
                elif random.random() < (coalrate_bothF/(coalrate_1F1rest+coalrate_bothF)):
                    return ["CABF", new_time]
                else:
                    return ["CA1F", new_time]
        else:
            assert coalrate_1F1rest == 0
            assert len(self.floats) > 1
            if random.random() < (recrate/totrate):
                return ["REC", new_time]
            else:
                return ["CABF", new_time]

    def general_incompatibility_check(self,node,  S, s):
        '''
        if the coalescence of child 1 and child 2 compatible for this snp.
        All are AVLTrees()
        this is applicable to Kuhner and initial
        S: is the parent samples for this segment
        s:  the focal SNP
        node: the parent node
        '''
        ret = True
        D = self.data[s]
        # symmetric difference between S1  and D
        A = S
        symA_D = A.difference(D)
        if len(symA_D) == 0:# subset or equal
            if len(A) == len(D): #put the mutation on this node
                node.snps.__setitem__(s, s)
                # delete s from S_F
                # detach_snps.discard(s)
                # # add to completed_snps
                # completed_snps[s] = s
        elif len(symA_D) == len(A): # distinct
            pass
        else:#
            symD_A = D.difference(A)
            if len(symD_A) > 0: # incompatible
                ret = False
        return ret

    def merge(self, leftchild, rightchild, parent = None):
        '''CA event between two lineages,
        also check compatibility and put mut on node
        :param parent: if None, the parent node is a new node in ARG
            else: the parent already exists in the ARG and we need to update
             the segments and the snps
        '''
        x = self.arg.copy_node_segments(leftchild)
        y = self.arg.copy_node_segments(rightchild)
        x = x.get_first_segment()
        y = y.get_first_segment()
        assert x is not None
        assert y is not None
        index =  self.arg.new_name()
        if parent == None:
            node = self.arg.alloc_node(index, time, x.node, y.node)
        else:
            node = parent
            node.snps.clear()
        self.arg.coal[node.index] = node.index
        x.node.left_parent = node
        x.node.right_parent = node
        y.node.left_parent = node
        y.node.right_parent = node
        z = None
        defrag_required = False
        valid = True
        while x is not None or y is not None:
            alpha = None
            if x is None or y is None:
                if x is not None:
                    alpha = x
                    x = None
                    assert alpha.left < alpha.right
                if y is not None:
                    alpha = y
                    y = None
                    assert alpha.left < alpha.right
            else:
                if y.left < x.left:
                    beta = x
                    x = y
                    y = beta
                if x.right <= y.left:
                    alpha = x
                    x = x.next
                    alpha.next = None
                    assert alpha.left < alpha.right
                elif x.left != y.left:
                    alpha = self.arg.alloc_segment(x.left, y.left, node, x.samples)
                    x.left = y.left
                    assert alpha.left < alpha.right
                else:
                    left = x.left
                    r_max = min(x.right, y.right)
                    right = r_max
                    alpha = self.arg.alloc_segment(left, right, node, x.union_samples(y))
                    assert alpha.left < alpha.right
                    if alpha.is_mrca(self.n):
                        alpha = None
                    else:# check compatibility, add snps
                        seg_snps = alpha.get_seg_variants(self.data)
                        for snp in seg_snps:#intersect_variants:
                            valid = self.general_incompatibility_check(node,  alpha.samples, snp)
                            if not valid:# break for
                                break
                        if not valid:# break while
                            break
                    if x.right == right:
                        x = x.next
                    else:
                        x.left = right
                    if y.right == right:
                        y = y.next
                    else:
                        y.left = right
            if alpha is not None:
                # if z is None:
                #     self.parent_nodes[p] = alpha
                if z is not None:
                    defrag_required |= z.right == alpha.left
                    z.next = alpha
                alpha.prev = z
                z = alpha
        if defrag_required:
            z.defrag_segment_chain()
        assert node is not None
        if z is not None:
            z = z.get_first_segment()
            node.first_segment = z
            if parent == None:
                self.arg.store_node(z, node)
            else:# already exist in ARG, update segments
                node.first_segment = z
                while z is not None:
                    z.node = node
                    z = z.next
        else:
            if parent == None:
                self.arg.add(node)
            else:
                node.first_segment = None
            self.arg.roots[node.index] = node.index
        return valid, node

    def make_float(self, child, oldbreakpoint, oldleftparent, oldrightparent,
                       floatParent, followparent, old_left, old_right, ch):
        '''to be used in new_recombination'''
        followparent.breakpoint = oldbreakpoint
        followparent.left_parent = oldleftparent
        followparent.right_parent = oldrightparent
        oldleftparent.update_child(child, followparent)
        oldrightparent.update_child(child, followparent)
        #update partial floating
        rphead = followparent.first_segment
        rptail = followparent.get_tail()
        self.add_to_partial_floating(followparent, old_left,
                old_right, rphead.left, rptail.right)
        self.floats[floatParent.index] = floatParent.index
        # active links
        diff = ch[2] - floatParent.num_links()
        self.active_links -= diff
        self.need_to_visit[floatParent.index] = floatParent.index
        self.need_to_visit[followparent.index] = followparent.index
        self.need_to_visit.discard(child.index)
        self.active_nodes.discard(child.index)

    def new_recombination(self, t):
        '''
        The new event is a recomb at t
        1. choose a lineage to put the rec on from
            self.floats and self.partial_floatings,
            proportional to their num of links
        2. if a floating is chosen, randomly choose a breakpoint
                it and split to two and put both parents in self.floats
            Else: it is from a partial floating,
                choose a breakpoint randomly from the new sites (a', a), (b, b')
                split the lineage to two. The parent with all new sites will
                float and the other follows the child path.
        '''
        valid = True
        partial_links = [self.partial_floatings[item][2] for item in self.partial_floatings]
        float_keys = list(self.floats.keys())
        float_links  = [self.arg.__getitem__(item).num_links() for item in float_keys]
        assert sum(partial_links) + sum(float_links) == self.active_links
        type = "f" #from floating
        if self.partial_floatings:
            type = random.choices(["pf", "f"], [sum(partial_links), sum(float_links)])[0]
        if type == "f": #from floating
            child_ind = random.choices(float_keys, float_links)[0]
            child = self.arg.__getitem__(child_ind)
            # choose a breakpoint on child
            head = child.first_segment
            tail = child.get_tail()
            if tail.right - head.left>1:
                break_point = random.choice(range(head.left + 1,
                                                  tail.right))
                leftparent, rightparent = self.split_node(child, break_point, t)
                self.floats.discard(child_ind)
                self.floats[leftparent.index] = leftparent.index
                self.floats[rightparent.index] = rightparent.index
                self.need_to_visit.discard(child_ind)
                self.need_to_visit[leftparent.index] = leftparent.index
                self.need_to_visit[rightparent.index] = rightparent.index
                self.active_links -= (child.num_links() -\
                                      (leftparent.num_links() + rightparent.num_links()))
            else:
                valid = False
        else: # rec on one of the partials
            child_ind = random.choices(list(self.partial_floatings.keys()), partial_links)[0]
            ch = self.partial_floatings.pop(child_ind)
            if ch[2]>1:
                bp = random.choice(range(ch[2]))
                child = self.arg.__getitem__(child_ind)
                head = child.first_segment
                tail = child.get_tail()
                a = ch[0]
                b = ch[1]
                oldleftparent = child.left_parent
                oldrightparent = child.right_parent
                oldbreakpoint = child.breakpoint
                if a != None and b != None:
                    if bp < a - head.left:
                        # the left parent is floating
                        break_point = head.left + bp + 1
                        assert head.left < break_point <= a
                        leftparent, rightparent = self.split_node(child, break_point, t)
                        self.make_float(child, oldbreakpoint, oldleftparent, oldrightparent,
                                            leftparent, rightparent, a, b, ch)
                    else:
                        #right parent is floating
                        break_point = bp - (a- head.left) +b
                        assert b<= break_point < tail.right
                        leftparent, rightparent = self.split_node(child, break_point, t)
                        self.make_float(child, oldbreakpoint, oldleftparent, oldrightparent,
                                                rightparent, leftparent, a, b, ch)
                elif a!= None:
                    break_point = head.left + bp + 1
                    # assert head.left < break_point <= a
                    leftparent, rightparent = self.split_node(child, break_point, t)
                    self.make_float(child, oldbreakpoint, oldleftparent, oldrightparent,
                       leftparent, rightparent, a, tail.right, ch)
                elif b!= None:
                    #right parent is floating
                    break_point = bp + b
                    assert b<= break_point < tail.right
                    leftparent, rightparent = self.split_node(child, break_point, t)
                    self.make_float(child, oldbreakpoint, oldleftparent, oldrightparent,
                                rightparent, leftparent, head.left, b, ch)
                else:
                    raise ValueError("both a and b are None")
            else:
                valid = False
        return valid

    def coal_bothF(self, t):
        '''
        coalescence of two floating lineages at t
        1. choose two lineages in self.floats
        2. update materials and check compatibility
            also put new mut on parent node if any
        '''
        valid = True
        assert len(self.floats) > 1
        inds = random.sample(list(self.floats), 2)
        leftchild = self.arg.__getitem__(inds[0])
        rightchild = self.arg.__getitem__(inds[1])
        assert leftchild.breakpoint == None
        assert rightchild.breakpoint == None
        valid, parent = self.merge(leftchild, rightchild)
        if valid:
            parent.time = t
            if parent.first_segment != None:
                self.floats[parent.index] = parent.index
                self.need_to_visit[parent.index] = parent.index
                self.active_links -= leftchild.num_links() +\
                                     rightchild.num_links() - parent.num_links()
            else:
                self.active_links -= leftchild.num_links() + rightchild.num_links()
            self.floats.discard(inds[0])
            self.floats.discard(inds[1])
            self.need_to_visit.discard(inds[0])
            self.need_to_visit.discard(inds[1])
        return valid

    def coal_1F_1active(self, t):
        '''
        coalescence between one floating node
        and one active node
        1. choose one from active, one from flaots
        '''
        valid = True
        active_ind = random.choice(list(self.active_nodes))
        float_ind = random.choice(list(self.floats))
        print("CA1F", "floats:", float_ind, "reattaches to", active_ind)
        activenode = self.arg.__getitem__(active_ind)
        floatnode = self.arg.__getitem__(float_ind)
        #--- start
        active_head = activenode.first_segment
        active_tail = activenode.get_tail()
        float_head = floatnode.first_segment
        float_tail = floatnode.get_tail()
        oldleftparent = activenode.left_parent
        oldrightparent = activenode.right_parent
        oldbreakpoint = activenode.breakpoint
        activenode.breakpoint = None
        floatnode.breakpoint = None
        valid, parent = self.merge(activenode, floatnode)
        parent.time = t
        if valid:
            parent.breakpoint = oldbreakpoint
            parent.left_parent = oldleftparent
            parent.right_parent = oldrightparent
            oldleftparent.update_child(activenode, parent)
            oldrightparent.update_child(activenode, parent)
            if parent.first_segment!= None:
                parent_head = parent.first_segment
                parent_tail = parent.get_tail()
                if parent.left_parent == None: #future floating
                    self.floats[parent.index] = parent.index
                    self.active_links -= float_tail.right - float_head.left -1
                    self.active_links += parent_tail.right - parent_head.left-1
                    if self.partial_floatings.__contains__(active_ind):
                        assert self.partial_floatings[active_ind][2] > 0
                        self.active_links -= self.partial_floatings[active_ind][2]
                        del self.partial_floatings[active_ind]
                elif not self.partial_floatings.__contains__(active_ind):
                    self.add_to_partial_floating(parent, active_head.left,
                                active_tail.right, parent_head.left, parent_tail.right)
                    self.active_links -= float_tail.right - float_head.left -1
                else: # active_ind in partial floating
                    ch = self.partial_floatings.pop(active_ind)
                    if ch[0] != None and ch[1] != None:
                        self.add_to_partial_floating(parent, ch[0],
                                ch[1], parent_head.left, parent_tail.right)
                    elif ch[0] == None and ch[1] != None:
                        self.add_to_partial_floating(parent, active_head.left,
                                ch[1], parent_head.left, parent_tail.right)
                    elif ch[0] != None and ch[1] == None:
                        self.add_to_partial_floating(parent, ch[0],
                                active_tail.right, parent_head.left, parent_tail.right)
                    else:
                        raise ValueError("both ch[0] and ch[1] are None, not a partial float")
                    self.active_links -= float_tail.right - float_head.left -1
                    self.active_links -= ch[2]
                self.need_to_visit[parent.index] = parent.index
            elif parent.left_parent != None:
                # delete its parent--> same way we did for prune
                self.update_prune_parents(parent)
                self.need_to_visit.discard(parent.index)
                self.active_links -= float_tail.right - float_head.left -1
                if self.partial_floatings.__contains__(active_ind):
                    assert self.partial_floatings[active_ind][2] > 0
                    self.active_links -= self.partial_floatings[active_ind][2]
                    del self.partial_floatings[active_ind]
            else:# root
                self.active_links -= float_tail.right - float_head.left -1
                if self.partial_floatings.__contains__(active_ind):
                    assert self.partial_floatings[active_ind][2] > 0
                    self.active_links -= self.partial_floatings[active_ind][2]
                    del self.partial_floatings[active_ind]
            self.active_nodes.discard(active_ind)
            self.floats.discard(float_ind)
            self.need_to_visit.discard(active_ind)
            self.need_to_visit.discard(float_ind)
        return valid

    def apply_new_event(self, newevent_type, new_time):
        valid = True
        if newevent_type == "REC":
            valid = self.new_recombination(new_time)
        elif newevent_type == "CABF":
            valid = self.coal_bothF(new_time)
        else:#CA1F
            valid = self.coal_1F_1active(new_time)
        return valid

    def split_node_kuhner(self, child, left_parent, right_parent):
        '''
        update the materials of leftparent and rightparent
        this is for an exsting recombination event in the ARG
        example: in kuhner when there is no new event.
        '''
        s = self.arg.copy_node_segments(child) #child segments
        y = self.find_break_seg(s, child.breakpoint)
        if y != None:
            x = y.prev
            if y.left < child.breakpoint < y.right:
                z = self.arg.alloc_segment(child.breakpoint, y.right, y.node,
                                           y.samples, None, y.next)
                if y.next is not None:
                    y.next.prev = z
                y.next = None
                y.right = child.breakpoint
                lhs_tail = y
            elif x is not None:
                assert x.right <= child.breakpoint <= y.left
                x.next = None
                y.prev = None
                z = y
                lhs_tail = x
            else: # first parent is empty
                z = y
                lhs_tail = None
        else: # second parent is empty
            z = None
            lhs_tail = s
        #=======
        # the parents already in ARG just update segments
        if lhs_tail != None:
            seg = lhs_tail.get_first_segment()
            left_parent.first_segment = seg
            while seg is not None:
                seg.node = left_parent
                seg = seg.next
        else:
            left_parent.first_segment = None
        if z != None:
            seg = z.get_first_segment()
            right_parent.first_segment = z
            while seg is not None:
                seg.node = right_parent
                seg = seg.next
        else:
            right_parent.first_segment = None
        return left_parent, right_parent

    def add_to_partial_floating(self,node, old_left,
                                old_right,new_left, new_right):
        '''check and add the node to partial floating'''
        assert node.first_segment is not None
        if node.left_parent is None:
            # add to floats
            assert not self.floats.__contains__(node.index)
            self.floats[node.index] = node.index
            self.active_links += node.num_links()
        else:
            a = None
            b = None
            new_links = 0
            if new_left < old_left and new_right > old_left:
                if new_right <= old_right:
                    a = old_left
                    new_links += old_left - new_left
                else:
                    a = old_left
                    b= old_right
                    new_links += (old_left - new_left) + (new_right - old_right)
            elif new_left < old_left  and new_right <= old_left:
                new_links += new_right - new_left - 1
                if new_links > 0:
                    a= new_left
            elif new_left >= old_left and new_left< old_right:
                if new_right > old_right:
                    b = old_right
                    new_links += new_right - old_right
            elif new_left >= old_left and new_left >= old_right:
                new_links += new_right - new_left -1
                if new_links > 0:
                    a= new_left
            if a != None or b != None:
                self.partial_floatings[node.index] = [a, b, new_links]
                self.active_links += new_links
            self.active_nodes[node.index] = node.index

    def check_material(self, leftchild, rightchild, leftparent, rightparent):
        '''
         for a alredy existing event, update the ancestral material
         this is for kuhner move, the cases with no event between a time interval
        '''
        valid = True
        if leftparent.index != rightparent.index:# rec
            assert leftchild.index == rightchild.index
            child = leftchild
            oldleftparent_left = leftparent.first_segment.left
            oldleftparent_right = leftparent.get_tail().right
            oldrightparent_left = rightparent.first_segment.left
            oldrightparent_right = rightparent.get_tail().right
            leftparent, rightparent = self.split_node_kuhner(child, leftparent, rightparent)
            if leftparent.first_segment != None and\
                    rightparent.first_segment != None:
                if self.partial_floatings.__contains__(child.index):
                    leftparent_head = leftparent.first_segment
                    leftparent_tail = leftparent.get_tail()
                    rightparent_head = rightparent.first_segment
                    rightparent_tail = rightparent.get_tail()
                    self.add_to_partial_floating(leftparent, oldleftparent_left,
                                    oldleftparent_right,leftparent_head.left, leftparent_tail.right)
                    self.add_to_partial_floating(rightparent, oldrightparent_left,
                                    oldrightparent_right,rightparent_head.left, rightparent_tail.right)
                    ch = self.partial_floatings.pop(child.index)
                    self.active_links -= ch[2]
                elif leftparent.left_parent != None and rightparent.left_parent != None:
                    self.active_nodes[leftparent.index] = leftparent.index
                    self.active_nodes[rightparent.index] = rightparent.index
                elif leftparent.left_parent != None:# right parent was future floating
                    assert not self.floats.__contains__(rightparent.index)
                    self.floats[rightparent.index] = rightparent.index
                    self.active_links += rightparent.num_links()
                    self.active_nodes[leftparent.index] = leftparent.index
                elif rightparent.left_parent != None:
                    assert not self.floats.__contains__(leftparent.index)
                    self.floats[leftparent.index] = leftparent.index
                    self.active_links += leftparent.num_links()
                    self.active_nodes[rightparent.index] = rightparent.index
                else:# both are None
                    self.floats[leftparent.index] = leftparent.index
                    self.active_links += leftparent.num_links()
                    self.floats[rightparent.index] = rightparent.index
                    self.active_links += rightparent.num_links()
                self.need_to_visit[leftparent.index] = leftparent.index
                self.need_to_visit[rightparent.index] = rightparent.index
                self.need_to_visit.discard(child.index)
                self.active_nodes.discard(child.index)
            elif leftparent.first_segment != None:
                child.left_parent = leftparent.left_parent
                child.right_parent = leftparent.right_parent
                child.breakpoint = leftparent.breakpoint
                parent = leftparent.left_parent
                if parent is None:
                    #child will be floating
                    self.floats[child.index] = child.index
                    self.active_links += child.num_links()
                    self.active_nodes.discard(child.index)
                    if self.partial_floatings.__contains__(child.index):
                        ch = self.partial_floatings.pop(child.index)
                        self.active_links -= ch[2]
                    assert self.need_to_visit.__contains__(child.index)
                else:
                    parent.update_child(leftparent, child)
                    parent = leftparent.right_parent
                    parent.update_child(leftparent, child)
                    # child stays active and if in partial-->stays same
                del self.arg.nodes[leftparent.index]
                del self.arg.nodes[rightparent.index]
                self.update_prune_parents(rightparent)
                #parents might alredy added to need_to_visit
                self.need_to_visit.discard(rightparent.index)
                self.need_to_visit.discard(leftparent.index)
            elif rightparent.first_segment != None:
                child.left_parent = rightparent.left_parent
                child.right_parent = rightparent.right_parent
                child.breakpoint = rightparent.breakpoint
                parent = rightparent.left_parent
                if parent is None:
                    #child will be floating
                    self.floats[child.index] = child.index
                    self.active_links += child.num_links()
                    self.active_nodes.discard(child.index)
                    if self.partial_floatings.__contains__(child.index):
                        ch = self.partial_floatings.pop(child.index)
                        self.active_links -= ch[2]
                    assert self.need_to_visit.__contains__(child.index)
                else:
                    parent.update_child(rightparent, child)
                    parent = rightparent.right_parent
                    parent.update_child(rightparent, child)
                    # child stays active
                del self.arg.nodes[leftparent.index]
                del self.arg.nodes[rightparent.index]
                self.update_prune_parents(leftparent)
                self.need_to_visit.discard(leftparent.index)
                self.need_to_visit.discard(rightparent.index)
            else: # both None
                raise ValueError("at least one parent must have segment")
        else: #coal
            assert leftchild.index != rightchild.index
            parent = leftparent
            if parent.first_segment != None:
                oldparent_left = parent.first_segment.left
                oldparent_right = parent.get_tail().right
            else:
                oldparent_left= None
                oldparent_right= None
            valid, parent = self.merge(leftchild, rightchild, parent)
            if valid:
                if parent.first_segment!= None:
                    parent_head = parent.first_segment
                    parent_tail = parent.get_tail()
                    if parent.left_parent == None: #future floating
                        self.floats[parent.index] = parent.index
                        self.active_links += parent_tail.right - parent_head.left-1
                    else:
                        assert oldparent_left != None
                        assert oldparent_right != None
                        self.add_to_partial_floating(parent, oldparent_left, oldparent_right,
                                                     parent_head.left, parent_tail.right)
                    self.need_to_visit[parent.index] = parent.index
                elif parent.left_parent != None:
                    self.update_prune_parents(parent)
                    self.need_to_visit.discard(parent.index)
                else:# parent is root
                    self.need_to_visit.discard(parent.index) # might have been added for future float
                #----
                if self.partial_floatings.__contains__(leftchild.index):
                    self.active_links -= self.partial_floatings[leftchild.index][2]
                    del self.partial_floatings[leftchild.index]
                if self.partial_floatings.__contains__(rightchild.index):
                    self.active_links -= self.partial_floatings[rightchild.index][2]
                    del self.partial_floatings[rightchild.index]
                self.active_nodes.discard(leftchild.index)
                self.active_nodes.discard(rightchild.index)
                self.need_to_visit.discard(leftchild.index)
                self.need_to_visit.discard(rightchild.index)
        return valid

    def kuhner(self):
        '''
        1.randomly choose a lineage otherthan the root,
            a.put it in need_check, floats
        2. find all the time and nodes greater than prune.time
        3. find mutations need to be check in future
        4. if prune is a child of CA:
            a) detach prune
            b)remove P
            c) put C in need_check,
            d) if P root, put C in future floating
            else: prune parents, and all the further parents
        '''
        # self.print_state()
        #---- forward transition
        self.transition_prob.kuhner_num_nodes(self.arg.__len__() - len(self.arg.roots))
        valid = True
        prune = self.add_choose_child()
        print("prune is ", prune.index)
        assert prune.left_parent != None
        self.floats[prune.index] = prune.index
        self.need_to_visit[prune.index] = prune.index
        self.find_active_nodes(prune.time)# 2
        self.update_prune_parents(prune)# 4
        self.get_active_links()# active links
        self.active_nodes = self.active_nodes.difference(self.floats)
        lower_time = prune.time
        while self.need_to_visit:
            if self.higher_times:# time interval
                upper_time = min(self.higher_times)
                parent_indexes = self.higher_times[upper_time]
                # check for future floating
                if self.active_links > 0 or len(self.floats) > 0:
                    new_event = self.new_event_time(lower_time, upper_time)
                else:
                    new_event = False
                if new_event:
                    new_time = new_event[1]; newevent_type = new_event[0]
                    print("new event type", newevent_type, "at time", new_time)
                    valid = self.apply_new_event(newevent_type, new_time)
                    lower_time = new_time
                    if not valid:
                        break
                else: # no new event
                    del self.higher_times[upper_time]
                    assert 0 < len(parent_indexes) <= 2
                    if len(parent_indexes) == 2:#rec
                        parent1 = self.arg.__getitem__(parent_indexes.pop())
                        parent2 = self.arg.__getitem__(parent_indexes.pop())
                        assert parent1.index != parent2.index
                        assert parent1.left_child.index == parent1.right_child.index
                        child = parent1.left_child
                        leftparent = child.left_parent
                        rightparent = child.right_parent
                        if self.need_to_visit.__contains__(child.index):
                            valid = self.check_material(child, child, leftparent, rightparent)
                            if not valid:
                                break
                        else:
                            self.active_nodes.discard(child.index)
                            assert leftparent.first_segment != None
                            assert rightparent.first_segment != None
                            assert not self.partial_floatings.__contains__(child.index)
                            if leftparent.left_parent == None: # future floating
                                assert self.need_to_visit.__contains__(leftparent.index)
                                self.floats[leftparent.index] = leftparent.index
                                self.active_links += leftparent.num_links()
                            else:
                                self.active_nodes[leftparent.index] = leftparent.index
                            if rightparent.left_parent == None:# future floating
                                assert self.need_to_visit.__contains__(rightparent.index)
                                self.floats[rightparent.index] = rightparent.index
                                self.active_links += rightparent.num_links()
                            else:
                                self.active_nodes[rightparent.index] = rightparent.index
                    else: # coal
                        parent = self.arg.__getitem__(parent_indexes.pop())
                        assert len(parent_indexes) == 0
                        leftchild = parent.left_child
                        rightchild = parent.right_child
                        assert leftchild.index != rightchild.index
                        if self.need_to_visit.__contains__(leftchild.index) or\
                            self.need_to_visit.__contains__(rightchild.index):
                            valid = self.check_material(leftchild, rightchild, parent, parent)
                            if not valid:
                                break
                        else:
                            assert not self.partial_floatings.__contains__(leftchild.index)
                            assert not self.partial_floatings.__contains__(rightchild.index)
                            self.active_nodes.discard(leftchild.index)
                            self.active_nodes.discard(rightchild.index)
                            if parent.left_parent != None:
                                assert parent.first_segment != None
                                self.active_nodes[parent.index] = parent.index
                            elif parent.first_segment != None:
                                #it is a future floating
                                assert self.need_to_visit.__contains__(parent.index)
                                self.floats[parent.index] = parent.index
                                self.active_links += parent.num_links()
                            else:# might have been added to check future floating
                                self.need_to_visit.discard(parent.index)
                    lower_time = upper_time
            else: #passed gmrca
                assert len(self.floats) >1
                assert self.active_nodes.is_empty()
                new_event = self.new_event_time(lower_time, passed_gmrca=True)
                new_time = new_event[1]; newevent_type = new_event[0]
                valid = self.apply_new_event(newevent_type, new_time)
                lower_time = new_time
                if not valid:
                    break
        print("valid is", valid)
        if valid:
            assert self.floats.is_empty()
            assert self.active_links == 0
            # assert self.active_nodes.is_empty()
            assert len(self.partial_floatings) == 0
            assert self.need_to_visit.is_empty()
            #----
            new_log_lk = self.arg.log_likelihood(self.theta, self.data)
            new_log_prior, new_roots, new_coals = self.arg.log_prior(self.n,
                                            self.seq_length, self.rho, self.Ne, False, True, True)
            #--- reverse transition
            self.transition_prob.kuhner_num_nodes(self.arg.__len__() - len(new_roots), False)
            self.Metropolis_Hastings(new_log_lk, new_log_prior, trans_prob = False, kuhner =True)
            if self.accept:
                print("accepted")
                self.arg.roots = new_roots
                self.arg.coal = new_coals
        self.floats.clear()
        self.need_to_visit.clear()
        self.active_nodes.clear()
        self.active_links = 0
        self.partial_floatings = collections.defaultdict(list)
        self.higher_times = collections.defaultdict(set)#time: (index)
        self.transition_prob.log_prob_forward = 0
        self.transition_prob.log_prob_reverse = 0
        self.arg.nextname = max(self.arg.nodes) + 1
        self.arg.get_available_names()

    def write_summary(self, row = [], write = False):
        '''
        write the ARG summary to a pd.df
        [lk, prior, numAncRec, numNonancRec, branchLength, setup]
        for setup: iteration, thining, burnin, n, seq_length, m,
            Ne, theta (mu), rho (r), acceptance rate, time took]
        '''
        if not write:
            self.summary.loc[0 if math.isnan(self.summary.index.max())\
                else self.summary.index.max() + 1] = row
        else:
            self.summary.to_hdf(self.outpath+ "/summary.h5", key = "df")

    def run_transition(self, w = [1, 1, 1, 0, 1, 0]):
        '''
        choose a transition move proportional to the given weights (w)
        Note that order is important:
        orders: "spr", "remove", "add", "at"(adjust_times), "ab" (breakpoint), kuhner
        NOTE: if rem and add weights are not equal, we must include
                that in the transition probabilities
        '''
        if len(self.arg.rec) == 0:
            w[1] = 0#remove rec
            w[4] = 0# adjust breakpoint
        ind = random.choices([i for i in range(len(w))], w)[0]
        if ind == 0:
            print("------spr")
            self.spr()
        elif ind == 1:
            print("-----remove")
            self.remove_recombination()
        elif ind == 2:
            print("---------add")
            self.add_recombination()
        elif ind == 3:
            self.adjust_times()
        elif ind == 4:
            print("---------adjust breakpoint")
            self.adjust_breakpoint()
        elif ind == 5:#kuhner
            original_ARG = copy.deepcopy(self.arg)
            self.kuhner()
            if not self.accept:
                self.arg = original_ARG
                print("rejected")
        self.detail_acceptance[ind][0] += 1
        if self.accept:
            self.detail_acceptance[ind][1] += 1

    def run(self, iteration = 20, thin = 1, burn = 0, verify = True):
        it = 0
        accepted = 0
        t0 = time.time()
        #---test #1 for no division by zero
        self.detail_acceptance = {i:[1, 0] for i in range(6)}
        #----test
        # for it in tqdm(range(iteration)):
        while it < iteration:
            print("iteration ~~~~~", it)
            self.run_transition()
            if self.accept:
                accepted += 1
                self.accept = False
            if it > burn and it % thin == 0:
                self.write_summary([self.log_lk, self.log_prior,
                                    self.log_lk + self.log_prior,
                                self.arg.num_ancestral_recomb,
                                self.arg.num_nonancestral_recomb,
                                self.arg.branch_length, -1])
                #dump arg
            if verify:
                self.arg.verify()
            it += 1
        if iteration > 15:
            self.summary.setup[0:17] = [iteration, thin, burn, self.n, self.seq_length,
                                       self.m, self.Ne, self.theta, self.rho,
                                        round(accepted/iteration, 2), round((time.time() - t0), 2),
                                        round(self.detail_acceptance[0][1]/self.detail_acceptance[0][0], 2),
                                        round(self.detail_acceptance[1][1]/self.detail_acceptance[1][0], 2),
                                        round(self.detail_acceptance[2][1]/self.detail_acceptance[2][0], 2),
                                        round(self.detail_acceptance[3][1]/self.detail_acceptance[3][0], 2),
                                        round(self.detail_acceptance[4][1]/self.detail_acceptance[4][0], 2),
                                        round(self.detail_acceptance[5][1]/self.detail_acceptance[5][0], 2)]
            self.write_summary(write = True)
        print("detail acceptance", self.detail_acceptance)

    def print_state(self):
        print("self.arg.coal", self.arg.coal)
        print("self.arg.rec", self.arg.rec)
        print("self.arg.roots", self.arg.roots)
        print("node", "time", "left", "right", "l_chi", "r_chi", "l_par", "r_par",
              "l_bp", "snps", "fir_seg_sam",
              sep="\t")
        for j in self.arg.nodes:
            node = self.arg.__getitem__(j)
            if node.left_parent is not None or node.left_child is not None:

                s = node.first_segment
                if s is None:
                    print(j, "%.5f" % node.time, "root", "root",
                              node.left_child.index,
                              node.right_child.index,
                              node.left_parent,node.right_parent,
                              node.breakpoint,
                              node.snps ,None, sep="\t")
                while s is not None:
                    l = s.left
                    r = s.right
                    if node.left_child is None:
                        print(j, "%.5f" % node.time, l,r, "Leaf", "Leaf",
                              node.left_parent.index,node.right_parent.index,
                              node.breakpoint,
                              node.snps,  s.samples,  sep="\t")#
                    elif  node.left_parent is None:
                        print(j, "%.5f" % node.time, l, r,
                              node.left_child.index,
                              node.right_child.index, "Root", "Root",
                              node.breakpoint,
                              node.snps ,s.samples, sep="\t")
                    else:
                        print( j, "%.5f" % node.time, l, r,
                             node.left_child.index, node.right_child.index,
                              node.left_parent.index, node.right_parent.index,
                              node.breakpoint,
                              node.snps, s.samples, sep="\t")
                    s = s.next

if __name__ == "__main__":
    pass
    mcmc = MCMC()
    mcmc.run(1000, 1, 0)



