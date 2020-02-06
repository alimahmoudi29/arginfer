''' This module is responsible for ARG classes'''
import math
from sortedcontainers import SortedSet
import msprime
import bintrees
import treeSequence
import random
import pickle
import numpy as np
import collections
# import compress_pickle
# import zipfile
import sys
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
        u = self
        assert u.left_parent is not None
        assert u.left_parent.index == u.right_parent.index
        p = u.left_parent
        v = p.left_child
        if v.index == u.index:
            v = p.right_child
        assert v.index is not u.index
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
        self.nextname = 1 # next node index
        self.roots = bintrees.AVLTree()# root indexes
        self.rec = bintrees.AVLTree() # arg rec parents nodes
        self.coal = bintrees.AVLTree() # arg CA parent node

    def __iter__(self):
        '''iterate over nodes in the arg'''
        return list(self.nodes)

    def __len__(self):
        '''number of nodes'''
        return len(self.nodes)

    def __getitem__(self, name):
        '''returns node by key: item'''
        return self.nodes[name]

    def __setitem__(self, name, node):
        '''adds a node to the ARG'''
        node.name = name
        self.add(node)

    def __contains__(self, name):
        '''if ARG contains node key '''
        return name in self.nodes

    def equal(self, other):
        '''if self is equal with other (structural equality)'''
        for node in self.nodes.values():
            if node.name not in other:
                return False
            if not node.equal(other[node.name]):
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

    def new_name(self):
        '''returns a new name for a node'''
        name = self.nextname
        self.nextname += 1
        return name

    def add(self, node):
        ''' add a ready node to the ARG:
        '''
        self.nodes[node.index] = node
        return node

    def rename(self, oldname, newname):
        '''renames a node in the ARG'''
        node = self.nodes[oldname]
        node.name = newname
        del self.nodes[oldname]
        self.nodes[newname] = node

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
        print("Detach()",node.index, "sib", sib.index)
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
        assert v.left_parent == None or t < v.left_parent.time
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
                  NAM = True, new_roots = False):
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
        new_coal =bintrees.AVLTree()
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
                number_of_links -= gap
                number_of_lineages += 1
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

#====== verification

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
                self.log_prob_forward += math.log(lambd) - \
                                         (lambd * (new_time- lower_bound))
            else: #uniform
                self.log_prob_forward += math.log(1/(upper_bound -lower_bound))
        else:
            if reattach_root:
                self.log_prob_reverse += math.log(lambd)- (lambd * (new_time- lower_bound))
            else:
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

class MCMC(object):

    def __init__(self, data = {}):
        self.data = data #a dict, key: snp_position- values: seqs with derived allele
        self.arg = ARG()

        self.theta = 1e-8 #mu
        self.rho = 1e-8 #r
        self.n = 5# sample size
        self.Ne = 5000
        self.seq_length = 10 # sequence length
        self.m = 0 # number of snps
        #------- To start get msprime output as initial
        self.get_initial_arg()
        self.transition_prob = TransProb()
        self.floatings = bintrees.AVLTree()#key: time, value: node index, d in the document
        self.floatings_to_ckeck = bintrees.AVLTree()#key index, value: index; this is for checking new roots
        self.new_names = bintrees.AVLTree()# index:index, of the new names (new parent indexes)
        self.arg.nextname = max(self.arg.nodes) + 1
        self.log_lk = self.arg.log_likelihood(self.theta, self.data)
        self.log_prior = self.arg.log_prior(self.n, self.seq_length, self.rho, self.Ne, False)
        self.lambd = 1/(2*self.Ne) # lambd in expovariate
        self.NAM_recParent = bintrees.AVLTree() # rec parent with seg = None
        self.NAM_coalParent = bintrees.AVLTree() # coal parent with seg = None
        self.coal_to_cleanup = bintrees.AVLTree()# ca parent with atleast a child.seg = None

    def get_initial_arg(self):
        '''
        TODO: build an ARG for the given data.
        '''
        # To start I use the msprime output
        recombination_rate = 1e-8
        Ne = 5000
        sample_size = 20
        length = 5e4
        ts_full = msprime.simulate(sample_size = sample_size, Ne = Ne,
                                   length = length, mutation_rate = 1e-8,
                                   recombination_rate = recombination_rate,
                                   random_seed = 20, record_full_arg = True)
        tsarg = treeSequence.TreeSeq(ts_full)
        tsarg.ts_to_argnode()
        self.arg = tsarg.arg
        self.data = treeSequence.get_arg_genotype(ts_full)
        self.m = len(self.data)
        self.seq_length = length
        self.log_lk = 0
        self.log_prior = 0
        self.Ne = 5000
        self.n = sample_size
        # --------

    def Metropolis_Hastings(self, new_log_lk, new_log_prior):
        ratio = new_log_lk + new_log_prior + self.transition_prob.log_prob_reverse - \
                (self.log_lk + self.log_prior + self.transition_prob.log_prob_forward)
        print("self.transition_prob.spr_log_reverse", self.transition_prob.log_prob_reverse)
        print("self.transition_prob.spr_log_forward", self.transition_prob.log_prob_forward)
        print("ratio", ratio)
        print("new_log_lk", new_log_lk, "new_log_prior", new_log_prior)
        print("self.log_lk", self.log_lk,"self.log_prior", self.log_prior)
        if math.log(random.random()) <= ratio: # accept
            self.log_lk = new_log_lk
            self.log_prior = new_log_prior
            return True
        else: #reject
            return False

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
        else:
            for node in self.arg.nodes.values():
                if node.time > detach_time:
                    if node.first_segment != None:
                        reattach_nodes[node.index] = node.index
                    elif not self.new_names.__contains__(node.index):
                        # that is a original node that will float in reverse
                        reattach_nodes[node.index] =  node.index
                elif node.left_parent != None and node.left_parent.time > detach_time:
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
        node = self.arg.nodes[node_index]
        if node.left_parent is None:
            if not backtrack:
                raise ValueError("The root node shouldn't be added "
                             "to node_to_check at the first place")
        elif node.left_parent == node.right_parent:
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
                        nodesToUpdateTimes[node.left_parent.left_parent.time] = node.left_parent.index
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
                            assert node.left_parent.left_parent == node.left_parent.right_parent
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
                    nodesToUpdateTimes[node.left_parent.left_parent.time] = node.left_parent.index
                self.NAM_coalParent[node.left_parent.index] = node.left_parent.index
                self.coal_to_cleanup[node.left_parent.index] = node.left_parent.index
            else: # s is not None or  sib_seg is None
                if sib_segs is None:
                    z = s.get_first_segment()
                else:# s is None
                    z = sib_segs.get_first_segment()
                if node.left_parent.left_parent is not None:
                    nodes_to_update[node.left_parent.index] = z
                    nodesToUpdateTimes[node.left_parent.left_parent.time] = node.left_parent.index
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
            nodes_to_update[node.left_parent.index] = lhs_tail
            nodesToUpdateTimes[node.left_parent.left_parent.time] = node.left_parent.index
            nodes_to_update[node.right_parent.index] = z
            nodesToUpdateTimes[node.right_parent.left_parent.time] = node.right_parent.index
            # TODO: Implement a way to check whether a recombination is removable, and
            # delete those recombination nodes which are not. Will have to be done once the
            # ARG has been constructed by checking which ones will violate snps
            # node = self.arg.alloc_node(self.arg.new_name(), time, lhs_tail.node, lhs_tail.node)
            # lhs_tail.node.left_parent = node
            if backtrack:
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
    def get_detach_SF(self, detach):
        '''get snps from detach
         that we need to check for incompatibility
          1. for a seg find all snps within.
          2. for each snp check if the mut has happened or will happen in future
          3. if in future add to detach_snps
          '''
        detach_snps = bintrees.AVLTree()
        seg = detach.first_segment
        while seg is not None:
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
        nodes_times= bintrees.AVLTree() #key: time, value, node_index
        nodes_to_update = {}
        for n in node:
            if n.left_parent is not None:
                nodes_to_update[n.index] = self.arg.copy_node_segments(n)
                nodes_times[n.left_parent.time] = n.index
        while nodes_to_update:
            min_time = nodes_times.min_key()
            next_index = nodes_times[min_time]
            nodes_to_update, nodes_times = self.update_ancestral_material(next_index,
                                                    nodes_to_update, nodes_times, backtrack)
            nodes_times.discard(min_time)

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
                if node.index == original_parent.left_child:
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
            while node.left_child != None:
                if not self.new_names.__contains__(node.left_child.index) and \
                    not self.new_names.__contains__(node.right_child.index):
                    original_child = node
                    break
                elif self.new_names.__contains__(node.left_child.index) and \
                    self.new_names.__contains__(node.right_child.index):
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
            print("on top self.floatings", self.floatings)
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
                if all_reattachment_nodes.is_empty(): # the grand root
                    assert len(self.floatings) > 0
                    reattach = self.arg.nodes[self.floatings[self.floatings.min_key()]]
                else:
                    reattach = self.arg.nodes[random.choice(list(all_reattachment_nodes))]
                print("node", node.index, "rejoins to ", reattach.index)
                #---trans_prob for choose reattach
                self.transition_prob.spr_choose_reattach(len(all_reattachment_nodes))
                max_time = max(min_time, reattach.time)
                if node.index == detach.index and reattach.index == sib.index:
                    self.floatings.discard(old_merger_time) # this is for removing sib
                if reattach.left_parent is None:
                    new_merger_time = max_time + random.expovariate(self.lambd)
                    print("new time is from an exponential distribution + reattach.time:", reattach.time)
                    print("new_merger_time", new_merger_time)
                    self.transition_prob.spr_reattach_time(new_merger_time, max_time, 0, True, True, self.lambd)
                else:
                    new_merger_time = random.uniform(max_time, reattach.left_parent.time)
                    self.transition_prob.spr_reattach_time(new_merger_time, max_time,
                                                   reattach.left_parent.time, False)
                    print("new_merger_time", new_merger_time)
                #-- reattach
                self.new_names = self.arg.reattach(node, reattach, new_merger_time, self.new_names)
                #---- update
                self.update_all_ancestral_material([node])
                #---
                self.floatings.discard(reattach.time) #if any

    def spr_revert_original_ARG(self, detach, sib, old_merger_time):
        '''revert to the original ARG'''
        new_sib = detach.sibling()
        self.arg.detach(detach, new_sib)
        new_names = self.arg.reattach(detach, sib, old_merger_time, self.new_names)
        # do we need to check the grand parent too? No
        # update materials
        self.update_all_ancestral_material([detach, new_sib], True)

    def clean_up(self, coal_to_cleanup):
        '''clean up the Accepted ARG. the ancestral material and snps
        has already set up. This method is only for cleaning the ARG from NAM lineages
        NAM is No ancestral Material nodes that are not a root.
        order the nodes by time and then clean up'''
        def reconnect(child, node, parent):
            '''from child--> node--> parent: TO child ---> parent '''
            if parent.left_child.index == node.index:
                parent.left_child = child
            if parent.right_child.index == node.index:
                parent.right_child = child
            if child.left_parent.index == node.index:
                child.left_parent = parent
            if child.right_parent.index == node.index:
                child.right_parent = parent

        while coal_to_cleanup:
            node = self.arg.nodes[coal_to_cleanup.pop(coal_to_cleanup.min_key())]
            if node.left_child == None and node.right_child == None:
                if node.left_parent is not None:
                    assert node.left_parent.index == node.right_parent.index
                    if node.left_parent.left_child.index == node.index:
                        node.left_parent.left_child = None
                    if node.left_parent.right_child.index == node.index:
                        node.left_parent.right_child = None
            elif node.left_child != None and node.right_child is None:
                if node.left_parent is not None:
                    reconnect(node.left_child, node, node.left_parent)
                    if  node.left_parent.index != node.right_parent.index:
                        reconnect(node.left_child, node, node.right_parent)
            elif node.right_child != None and node.left_child is None:
                if node.left_parent is not None:
                    reconnect(node.right_child, node, node.left_parent)
                    if  node.left_parent.index != node.right_parent.index:
                        reconnect(node.right_child, node, node.right_parent)
            else: # both not None
                if node.left_child.first_segment == None and\
                        node.right_child.first_segment is None:
                    if node.left_parent is not None:
                        assert node.first_segment is not None
                        assert node.left_parent.index == node.right_parent.index
                        node.left_child.left_parent = None
                        node.left_child.right_parent = None
                        node.right_child.left_parent = None
                        node.right_child.right_parent = None
                        if node.left_parent.left_child.index == node.index:
                            node.left_parent.left_child = None
                        if node.left_parent.right_child.index == node.index:
                            node.left_parent.right_child = None
                elif node.left_child.first_segment != None and node.right_child.first_segment is None:
                    assert node.left_parent is not None
                    reconnect(node.left_child, node, node.left_parent)
                    if node.left_parent.index != node.right_parent.index:
                        reconnect(node.left_child, node, node.right_parent)
                elif node.right_child.first_segment != None and node.left_child.first_segment is None:
                    assert node.left_parent is not None
                    reconnect(node.right_child, node, node.left_parent)
                    if node.left_parent.index != node.right_parent.index:
                        reconnect(node.right_child, node, node.right_parent)
                else: # non None
                    raise ValueError("both children have seg, so this shouldnt be in coal_to_clean")
            del self.arg.nodes[node.index]

        # nodes = [k for k in self.arg.nodes]
        # while nodes:
        #     node = self.arg.nodes[nodes.pop(0)]
        #     if node.first_segment is None and node.left_parent is not None:
        #         assert node.left_parent.index == node.right_parent.index
        #         sib = node.sibling()
        #         if node.left_parent.left_parent is None:
        #             sib.left_parent = None
        #             sib.right_parent = None
        #         elif node.left_parent.left_parent.index == node.left_parent.right_parent:
        #             # remove node.left_parent
        #             grandparent = node.left_parent.left_parent
        #             if grandparent.left_child.index == node.left_parent:
        #                 grandparent.left_child = sib
        #             if grandparent.righ_child.index == node.left_parent:
        #                 grandparent.right_child = sib
        #             sib.left_parent = grandparent
        #             sib.right_parent = grandparent
        #         del self.arg.nodes[node.left_parent.index]
        #         nodes.remove(node.left_parent.index)
        #         node.left_parent = None
        #         node.right_parent = None

    # def update_arg_coal(self):
    #     '''
    #     update arg.coal + return the CA parents that need to be cleaned (removed)
    #     TODO: a better way to find coal_to_clean + we can reset self.coal in prior
    #     '''
    #     coal_to_cleanup = bintrees.AVLTree()# CA parent that need to be cleaned
    #     for ind in self.arg.coal:
    #         assert self.arg.nodes[ind].left_child.index != self.arg.nodes[ind].right_child.index
    #         if self.arg.nodes[ind].left_child.first_segment == None or \
    #             self.arg.nodes[ind].right_child.first_segment == None:
    #             coal_to_cleanup[ind] = ind
    #     self.arg.coal = self.arg.coal.difference(coal_to_cleanup)
    #     return coal_to_cleanup

    def spr_validity_check(self, node,  clean_nodes, detach_snps, detach, completed_snps):
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
                original_parent, original_child, valid, second_child = self.is_new_root(node)
                # if we have original parent and original child
                # the floating will be from node.time and rejoins to original.parent.time
                # reverse: choose original parent, choose time on
                # original_parent's second_child.time to original parent.left_parent.time
                # then add what to clean_up? node.left_parent, since we need to check incompatibility
                # without considering incompatibility then original_parent.
                #second child is the second child of the original_parent. we need to find its original child
                # and original parent (if any) for the time prob calculation
                if valid and original_parent != None and original_child != None:
                    all_reattachment_nodes = self.spr_reattachment_nodes(node.time, False)
                    all_reattachment_nodes.discard(original_parent.index)
                    all_reattachment_nodes.discard(node.index)
                    # ---- reverse of choosin g a lineage to rejoin
                    self.transition_prob.spr_choose_reattach(len(all_reattachment_nodes), False)
                    #----- reverse prob time
                    if node.index != detach.index:# already done for detach
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
                            if sc_original_parent is None:
                                self.transition_prob.spr_reattach_time(original_parent.time,
                                        max(node.time, second_child.time), 0 , True, False)
                            else:
                                self.transition_prob.spr_reattach_time(original_parent.time,
                                        max(node.time, second_child.time), sc_original_parent.time
                                                                       , False, False)
                # add nodeleft parent ot cleanup
                clean_nodes[node.left_parent.time].add(node.left_parent.index)
        return valid, clean_nodes, detach_snps, completed_snps

    def all_validity_check(self, clean_nodes, detach_snps, detach):
        '''do spr_validity_check()for all the needed nodes'''
        valid = True # move is valid
        completed_snps = bintrees.AVLTree() # those of detach_snps that completed already
        while valid and clean_nodes:
            # get the min_time one
            nodes = clean_nodes.pop(min(clean_nodes))
            assert 0 < len(nodes) <= 2
            if len(nodes) == 2:# two rec parents
                nodes = [self.arg.nodes[nodes.pop()], self.arg.nodes[nodes.pop()]]
                assert nodes[0].left_child.index == nodes[0].right_child.index
                assert nodes[1].left_child.index == nodes[1].right_child.index
                assert nodes[0].left_child.index == nodes[1].left_child.index
                if nodes[0].first_segment is None or nodes[1].first_segment is None:
                    valid = False # cancels a rec
                    break
            else:
                assert len(nodes) == 1
                nodes = [self.arg.nodes[nodes.pop()]]
            while nodes:
                node = nodes.pop(0)
                valid, clean_nodes, detach_snps, completed_snps = \
                    self.spr_validity_check(node, clean_nodes, detach_snps,
                                            detach, completed_snps)
                if not valid:
                    break
        return valid

    def spr(self):
        '''perform an SPR move on the ARG'''
        # Choose a random coalescence node, and one of its children to detach.
        # Record the current sibling and merger time in case move is rejected.
        # TODO: We need a better way to sample a uniform choice from an AVL tree, or
        # a better container.
        self.floatings = bintrees.AVLTree()
        self.floatings_to_ckeck = bintrees.AVLTree()#key index, value: index; this is for checking new roots
        self.new_names = bintrees.AVLTree()
        detach = self.arg.nodes[random.choice(list(self.arg.coal))]
        if random.random() < 0.5:
            detach = detach.left_child
        else:
            detach = detach.right_child
        self.floatings[detach.time] = detach.index
        self.floatings_to_ckeck[detach.index] = detach.index
        self.new_names[detach.left_parent.index] = detach.left_parent.index
        #---- forward trans prob of choosing detach
        self.transition_prob.spr_choose_detach(len(self.arg.coal))
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
            print("detach_snps", detach_snps)
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
            new_log_lk = self.arg.log_likelihood(self.theta, self.data)
            new_log_prior, new_roots, new_coals = \
                self.arg.log_prior(self.n,self.seq_length, self.rho,
                                   self.Ne, True, True)
            #--- reverse prob choose detach
            self.transition_prob.spr_choose_detach(len(new_coals), False)
            mh_accept = self.Metropolis_Hastings(new_log_lk, new_log_prior)
            print("mh_accept ", mh_accept)
            #if acceptt
            if mh_accept: # clean up
                # the ancestral material already set up. we just need to
                # remove the NAM nodes.
                self.clean_up(self.coal_to_cleanup)
                self.check_root_parents(new_roots)
                self.arg.roots = new_roots # update roots
                self.arg.coal = new_coals
            else: # rejected: retrieve the original- backtrack--> no changes on arg.coal
                self.spr_revert_original_ARG(detach, sib, old_merger_time)
        else:
            self.spr_revert_original_ARG(detach, sib, old_merger_time)
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

    def rem_revert_original_ARG(self, remParent, otherParent,
                                child, remGrandparent, remPsib, old_child_bp, invisible):
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
        remove a recombination event
        1. randomly choose a rec parent (remParent) and calc the forward prob
        2. if remParent.left_parent!= remParent.right_parent: reject, else:
            detach(remNode), detach(otherParent),
        3. update ancestral material
        4. reattach the floatings (if any)
        5. check validity, compatibility and reverse prob
        6. if not valid, revert the move
        '''
        assert not self.arg.rec.is_empty()
        #1. choose a rec parent
        remParent = self.arg.nodes[random.choice(list(self.arg.rec.keys()))]
        assert remParent.left_child == remParent.right_child
        #-- forward transition prob
        self.transition_prob.rem_choose_remParent(len(self.arg.rec))
        if remParent.left_parent.index != remParent.right_parent.index:
            valid = False
        else:
            #--- detach R:
            remPsib = remParent.sibling()
            remGrandparent = self.arg.nodes[remParent.left_parent.index]
            #-----
            child = remParent.left_child
            old_child_bp = child.breakpoint
            otherParent = child.left_parent
            if otherParent.index == remParent.index:
                otherParent = child.right_parent
            if remGrandparent.left_parent is None:
                self.floatings[remPsib.left_parent.time] = remPsib.index
                self.floatings_to_ckeck[remPsib.index] = remPsib.index
            if remParent.left_parent.index != otherParent.left_parent.index:
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
            remGrandparent = self.arg.nodes.pop(remParent.left_parent.index)

            #--- update ancestral material
            if invisible: #otherParent == remPsib
                self.update_all_ancestral_material([child])
            else:#
                self.update_all_ancestral_material([child, remPsib])
            #--- reattach all the floatings (if any) NOTE: the *args are fake:
            print("floatinf is ", self.floatings)
            self.spr_reattach_floatings(remParent, otherParent, remParent.time)
            # is there any canceled rec?
            if self.NAM_recParent:
                print("self.NAM_recParebt", self.NAM_recParent)
                valid = False
            else:
                #--- check validity, compatibility and mutations, reverse prob
                remParent_snps = self.get_detach_SF(remParent)
                print("remParent_snps", remParent_snps)
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
                mh_accept = self.Metropolis_Hastings(new_log_lk, new_log_prior)
                print("mh_accept ", mh_accept)
                if mh_accept:
                    #update coal, and rec
                    self.clean_up(self.coal_to_cleanup)
                    self.check_root_parents(new_roots)
                    self.arg.roots = new_roots # update roots
                    self.arg.coal = new_coals
                    self.arg.rec.discard(remParent.index)
                    self.arg.rec.discard(otherParent.index)
                else:
                    self.rem_revert_original_ARG(remParent, otherParent, child,
                                            remGrandparent, remPsib, old_child_bp, invisible)
            else:
                self.rem_revert_original_ARG(remParent, otherParent, child,
                                            remGrandparent, remPsib, old_child_bp, invisible)
        self.empty_containers()

    # ============
    # ADD recombination

    def check_root_parents(self, new_roots):
        '''after mh acceptance, make sure all the new roots dont have any parent
        This is not taken care of in clean_up(), because coal_to_clean does
        not contain nodes without seg where both children have segments (roots)
        '''
        for ind in new_roots:
            root = self.arg.nodes[ind]
            if root.left_parent is not None:
                assert self.arg.nodes.__contains__(root.left_parent.index)
                root.left_parent = None
                root.right_parent = None
            else:
                root.right_parent = None


    def add_revert_original_ARG(self, child, followParent, detachParent, oldbr):
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
        if sib.index == followParent.index:#invisible rec
            self.update_all_ancestral_material([child], True)
        else:
            self.update_all_ancestral_material([child, sib], True)

    def add_choose_child(self):
        '''choose a node to put a recombination on it'''
        ind = random.choice(list(self.arg.nodes.keys()))
        if not self.arg.roots.__contains__(ind):
            return self.arg.nodes[ind]
        else:
            return self.add_choose_child()

    def split_node(self, child, k, t):
        '''split a node (child) to two parent node from k at time t
        and add the parents to the arg
        '''
        s = self.arg.copy_node_segments(child) #child segments
        y = self.find_break_seg(s, k)
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
        child = self.add_choose_child()
        assert child.first_segment != None and child.left_parent != None
        self.transition_prob.add_choose_node(len(self.arg.nodes) - len(self.arg.coal))#1
        head = child.first_segment
        tail = child.get_tail()
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
        print("child", child.index, "followparent", followParent.index, "detach", detachParent.index)
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
            print("detachParent_snps", detach_snps)
            clean_nodes = collections.defaultdict(set) #key: time, value: nodes
            clean_nodes[child.left_parent.time].add(child.left_parent.index)
            clean_nodes[child.right_parent.time].add(child.right_parent.index)
            valid = self.all_validity_check(clean_nodes, detach_snps, child)#thirs *arg fake
        if valid:
            #-- calc prior and likelihood and then M-H
            new_log_lk = self.arg.log_likelihood(self.theta, self.data)
            new_log_prior, new_roots, new_coals = self.arg.log_prior(self.n,
                                            self.seq_length, self.rho, self.Ne, True, True)
            #--- reverse prob--
            self.transition_prob.rem_choose_remParent(len(self.arg.rec), False)
            mh_accept = self.Metropolis_Hastings(new_log_lk, new_log_prior)
            print("mh_accept ", mh_accept)
            if mh_accept:
                #update coal, and rec
                self.clean_up(self.coal_to_cleanup)
                self.check_root_parents(new_roots)
                self.arg.roots = new_roots # update roots
                self.arg.coal = new_coals
            else:
                self.add_revert_original_ARG(child, followParent,
                                             detachParent, oldbreakpoint)
        else:
            self.add_revert_original_ARG(child, followParent,
                                             detachParent, oldbreakpoint)
        self.empty_containers()

    def print_state(self):
        print("self.arg.coal", self.arg.coal)
        print("self.arg.rec", self.arg.rec)
        print("self.arg.roots", self.arg.roots)
        print("node", "time", "left", "right", "l_chi", "r_chi", "l_par", "r_par",
              "l_bp", "snps", "fir_seg_sam",
              sep="\t")
        for j in self.arg.nodes:
            node = self.arg.nodes[j]
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
                        # print(node.time, node.index, int(l), int(r), node.left_child.index, node.right_child.index)
                        print( j, "%.5f" % node.time, l, r,
                             node.left_child.index, node.right_child.index,
                              node.left_parent.index, node.right_parent.index,
                              node.breakpoint,
                              node.snps, s.samples, sep="\t")
                    s = s.next

if __name__ == "__main__":
    pass
    mcmc = MCMC()
    mcmc.spr()
    mcmc.remove_recombination()
    mcmc.add_recombination()
    mcmc.print_state()
