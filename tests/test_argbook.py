import unittest
import math
import collections
import msprime
import os
from arginfer import argbook, treeSequence, mcmc as MC
import bintrees


class TestSegment(unittest.TestCase):

    def test_equal_samples(self):
        #equal
        s1 = argbook.Segment(); s1.samples.update( {k:k for k in [0, 1, 2]} )
        s1.left = 0; s1.right = 10
        s2 = argbook.Segment(); s2.samples.update( {k:k for k in [0, 1, 2]} )
        s2.left = 0; s2.right = 10
        self.assertTrue(s1.equal_samples(s2), True)
        #------ not equal
        s3 = argbook.Segment(); s3.samples.update( {k:k for k in [0, 1]} )
        s3.left = 1; s3.right = 10
        self.assertFalse(s1.equal_samples(s3), False)
        #----- cases with repeat
        # s4 = argbook.Segment(); s4.samples = AVLTree().update({k:k for k in [0, 1,1,2]})
        # s4.left = 1; s4.right = 10
        # self.assertFalse(s1.equal_samples(s4), False)

    def test_contains(self):
        s1 = argbook.Segment(); s1.samples.update( {k:k for k in [0, 1, 2]} )
        s1.left = 0; s1.right = 10
        self.assertFalse(s1.contains(10), False)
        self.assertTrue(s1.contains(0), True)
        self.assertTrue(s1.contains(7), True)

    def test_is_mrca(self):
        sample_size = 5
        s1 = argbook.Segment(); s1.samples.update( {k:k for k in [0, 1, 2, 3 , 4]} )
        s1.left = 0; s1.right = 10
        self.assertTrue(s1.is_mrca(sample_size), True)
        self.assertEqual(len(set(s1.samples)), len(s1.samples))
        self.assertTrue(len(s1.samples) <= sample_size)

    def test_union_samples(self):
        s1 = argbook.Segment(); s1.samples.update( {k:k for k in [0, 1, 2]} )
        s1.left = 0; s1.right = 10
        s2 = argbook.Segment(); s2.samples.update( {k:k for k in [3, 4]} )
        s2.left = 0; s2.right = 10
        self.assertEqual(sorted(s1.union_samples(s2)), [0, 1, 2, 3, 4])
        self.assertEqual(sorted(s1.samples),[0, 1, 2])

    def test_equal(self):
        s1 = argbook.Segment(); s1.samples =[0, 1, 2]
        s1.left = 0; s1.right = 10; s1.node = argbook.Node( 2 )
        s2 = argbook.Segment(); s2.samples =[0, 1, 2]
        s2.left = 0; s2.right = 10; s2.node = argbook.Node( 2 )
        self.assertTrue(s1.equal(s2))
        #----
        s3 = argbook.Segment(); s3.samples.update( {k:k for k in [0, 1, 2]} )
        s3.left = 0; s3.right = 10; s3.node = argbook.Node( 1 )
        self.assertFalse(s3.equal(s1))
        #----- samples
        #----
        # s4 = argbook.Segment(); s4.samples.update({k:k for k in [0, 1,2, 2]})
        # s4.left = 0; s4.right = 10; s4.node = argbook.Node(1)
        # self.assertFalse(s3.equal(s4))

    def test_defrag_segment_chain(self):
        s1 = argbook.Segment(); s2 = argbook.Segment()
        s1.samples.update({k:k for k in [0, 1,2]})
        s1.left = 10; s1.right = 20; s1.node = argbook.Node( 2 )
        s2.samples.update({k:k for k in [0, 1,2]})
        s2.left = 0; s2.right = 10; s2.node = argbook.Node( 2 )
        s1.prev = s2; s2.next = s1
        s1.defrag_segment_chain()
        self.assertEqual(s2.right, 20)
        self.assertEqual(s2.left, 0)
        self.assertEqual(sorted(s2.samples), [0, 1,2])

    def test_get_first_segment(self):
        s1 = argbook.Segment(); s2 = argbook.Segment()
        s1.samples.update({k:k for k in [0, 1,2]})
        s1.left = 10; s1.right = 20; s1.node = argbook.Node( 2 )
        s2.samples.update({k:k for k in [0, 1]})
        s2.left =0; s2.right = 4; s2.node = argbook.Node( 2 )
        s1.prev = s2; s2.next = s1
        self.assertEqual(s1.get_first_segment().left, 0)
        self.assertEqual(s1.get_first_segment().right, 4)
        self.assertEqual(sorted(s1.get_first_segment().samples), [0,1])

class TestNode(unittest.TestCase):

    def test_contains(self):
        node = argbook.Node( 2 )
        s1 = argbook.Segment(); s2 = argbook.Segment()
        s1.left = 10; s1.right = 20; s1.node = node
        s2.left = 0; s2.right = 10; s2.node = node
        s1.prev = s2; s2.next = s1
        node.first_segment = s2
        self.assertFalse(node.contains(20))
        self.assertTrue(node.contains(10))
        self.assertTrue(node.contains(0))
        self.assertEqual(node.first_segment.prev, None)
        #---- testx_segment here
        self.assertEqual(node.x_segment(5).left, 0)
        self.assertEqual(node.x_segment(10).left, 10)
        self.assertRaises(ValueError, node.x_segment,20)
        self.assertRaises(ValueError, node.x_segment,25)

    def test_num_links(self):
        node = argbook.Node( 2 )
        s1 = argbook.Segment(); s2 = argbook.Segment()
        s1.left = 10; s1.right = 20; s1.node = node
        s2.left = 0; s2.right = 5; s2.node = node
        s1.prev = s2; s2.next = s1
        node.first_segment = s2
        self.assertEqual(node.num_links(), 19)
        self.assertNotEqual(node.num_links(), 20)

    def test_equal(self):
        node1 = argbook.Node( 2 )
        s1 = argbook.Segment(); s2 = argbook.Segment()
        s1.samples.update({k:k for k in [0, 1,2]})
        s1.left = 10; s1.right = 20; s1.node = node1
        s2.samples.update({k:k for k in [0, 1]})
        s2.left = 0; s2.right = 5; s2.node = node1
        s1.prev = s2; s2.next = s1
        node1.first_segment = s2
        node2 = argbook.Node( 2 )
        s3 = argbook.Segment(); s4 = argbook.Segment()
        s3.samples.update({k:k for k in [0, 1,2]})
        s3.left = 10; s3.right = 20; s3.node = node2
        s4.samples.update({k:k for k in [0, 1]})
        s4.left = 0; s4.right = 5; s4.node = node2
        s3.prev = s4; s4.next = s3
        node2.first_segment = s4
        self.assertTrue(node2.equal(node1))
        self.assertTrue(node1.equal(node2))
        #----- not equal
        s5 = argbook.Segment()
        s5.samples.update({k:k for k in [0, 1]})
        s5.left = 21; s5.right = 25; s5.node = node2
        s3.next = s5; s5.prev = s3
        self.assertFalse(node2.equal(node1))
        self.assertFalse(node1.equal(node2))
        #---- raise
        node3 = None
        self.assertRaises(ValueError, node2.equal, node3)
        #-------snps

        node4 = argbook.Node( 2 )
        s3 = argbook.Segment(); s4 = argbook.Segment()
        s3.samples.update({k:k for k in [0, 1,2]})
        s3.left = 10; s3.right = 20; s3.node = node4
        s4.samples.update({k:k for k in [0, 1]})
        s4.left = 0; s4.right = 5; s4.node = node4
        s3.prev = s4; s4.next = s3
        node4.first_segment = s4
        node4.snps.update({12:12, 23:23})
        self.assertFalse(node1.equal(node4))
        node1.snps.update({23:23, 12:12})
        self.assertTrue(node1.equal(node4))

    def test_tree_node_age_and_also_upward_path(self):
        #-------- ts_full from msprime:
        recombination_rate=1e-8
        Ne= 5000
        sample_size = 5
        length = 6e2
        ts_full = msprime.simulate(sample_size = sample_size, Ne = Ne, length = length, mutation_rate = 1e-8,
                            recombination_rate = recombination_rate, random_seed = 20, record_full_arg = True)
        # print(ts_full.tables.edges)
        # print(ts_full.tables.nodes.time)
        tsarg= treeSequence.TreeSeq( ts_full )
        tsarg.ts_to_argnode()
        argnode = tsarg.arg
        #-------------- the age of node 2 at position 10 ---- easy
        self.assertEqual(argnode[2].tree_node_age(10), ts_full.tables.nodes[6].time)
        # now node 6 undergo recombination, then for x=10, it should give us time[13] -time[6]
        self.assertEqual(argnode[6].tree_node_age(10),
                         ts_full.tables.nodes[13].time - ts_full.tables.nodes[6].time )
        #there is a back rec, which is great for this test x= 10 in node 9 age=time[14] - time[9]
        self.assertEqual(argnode[9].tree_node_age(10),
                         ts_full.tables.nodes[14].time - ts_full.tables.nodes[9].time)
        # edges values
        self.assertEqual(argnode[9].tree_node_age(0),
                         ts_full.tables.nodes[14].time - ts_full.tables.nodes[9].time)
        self.assertEqual(argnode[9].tree_node_age(447),
                         ts_full.tables.nodes[14].time - ts_full.tables.nodes[9].time)

    def test_sibling(self):
        '''already tested in test_ts_to_argnode()
        #TODO: test extreme cases
        '''

    def test_push_snp_down(self):
        '''TODO'''

class TestTreeSeq(unittest.TestCase):

    def test_ts_to_argnode(self):
        #-------- ts_full from msprime:
        recombination_rate=1e-8
        Ne= 5000
        sample_size = 5
        length = 6e2
        ts_full = msprime.simulate(sample_size = sample_size, Ne = Ne, length = length, mutation_rate = 1e-8,
                            recombination_rate = recombination_rate, random_seed = 20, record_full_arg = True)

        tsarg = treeSequence.TreeSeq( ts_full )
        tsarg.ts_to_argnode()
        argnode = tsarg.arg
        ##----- ts.edges dict
        edges_dict= collections.defaultdict(list)
        child_edges_dict= collections.defaultdict(list)
        for edge in ts_full.tables.edges:
            edges_dict[edge.parent].append(edge)
            child_edges_dict[edge.child].append(edge)
        # number of nodes
        self.assertEqual(len(edges_dict) + ts_full.sample_size, argnode.__len__())
        while edges_dict:
            parent = next(iter(edges_dict))
            if ts_full.tables.nodes[parent].flags == msprime.NODE_IS_RE_EVENT:
                child = edges_dict[parent][0].child
                parent2 = parent + 1
                #time
                self.assertEqual(ts_full.tables.nodes[parent].time, argnode[parent].time)
                self.assertEqual(ts_full.tables.nodes[parent2].time, argnode[parent2].time)
                # find breakpoint
                if edges_dict[parent][-1].right == edges_dict[parent2][0].left:
                    l_break = math.ceil(edges_dict[parent][-1].right)
                    r_break = None
                else:
                    r_break = math.ceil(edges_dict[parent2][0].left)
                    l_break = math.ceil(edges_dict[parent][-1].right)
                #breakpoints
                self.assertEqual(argnode[child].breakpoint,  l_break)
                #assert left
                self.assertEqual(argnode[parent].first_segment.left,  math.ceil(edges_dict[parent][0].left))
                self.assertEqual(argnode[parent].get_tail().right,  math.ceil(edges_dict[parent][-1].right))
                #parent2
                self.assertEqual(argnode[parent2].first_segment.left,  math.ceil(edges_dict[parent2][0].left))
                self.assertEqual(argnode[parent2].get_tail().right,  math.ceil(edges_dict[parent2][-1].right))
                #child
                self.assertEqual(argnode[child].first_segment.left,  math.ceil(edges_dict[parent][0].left))
                self.assertEqual(argnode[child].get_tail().right,  math.ceil(edges_dict[parent2][-1].right))
                # parent or child
                self.assertEqual(argnode[child].left_parent.index, parent)
                self.assertEqual(argnode[child].right_parent.index, parent2)
                self.assertEqual(argnode[parent].left_child.index, argnode[parent2].left_child.index)
                self.assertEqual(argnode[parent].right_child.index, argnode[parent2].right_child.index)
                del edges_dict[parent]
                del edges_dict[parent2]
            else: # CA
                child0 = edges_dict[parent][0].child
                child1 = edges_dict[parent][-1].child
                assert child0 != child1
                #time
                self.assertEqual(ts_full.tables.nodes[parent].time, argnode[parent].time)
                #child0
                self.assertEqual(argnode[child0].first_segment.left,  math.ceil(child_edges_dict[child0][0].left))
                self.assertEqual(argnode[child0].get_tail().right,  math.ceil(child_edges_dict[child0][-1].right))
                #child1
                self.assertEqual(argnode[child1].first_segment.left,  math.ceil(child_edges_dict[child1][0].left))
                self.assertEqual(argnode[child1].get_tail().right,  math.ceil(child_edges_dict[child1][-1].right))
                # left_parent
                self.assertEqual(argnode[child0].left_parent.index, parent)
                self.assertEqual(argnode[child0].right_parent.index, parent)
                self.assertEqual(argnode[child1].left_parent.index, parent)
                self.assertEqual(argnode[child1].right_parent.index, parent)
                #sibling
                self.assertEqual(argnode[child0].sibling().index, child1)
                self.assertEqual(argnode[child1].sibling().index, child0)
                #-----
                self.assertEqual(argnode[parent].left_child.index, child0)
                self.assertEqual(argnode[parent].right_child.index, child1)
                del edges_dict[parent]

    def test_verify_mutation_is_on_the_lowest_possible_node(self):
        recombination_rate=1e-8
        Ne= 5000
        sample_size = 5
        length = 6e5
        ts_full = msprime.simulate(sample_size = sample_size, Ne = Ne, length = length, mutation_rate = 1e-8,
                            recombination_rate = recombination_rate, random_seed = 20, record_full_arg = True)

        tsarg = treeSequence.TreeSeq( ts_full )
        tsarg.ts_to_argnode()
        argnode = tsarg.arg
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
                # assert node samples contain all the derived for snp x.
                assert sorted(node_samples) == sorted(data[x])

        data = treeSequence.get_arg_genotype( ts_full )
        # print(data)
        for node in argnode.nodes.values():
            verify_mutation_node(node, data)

class TestARG(unittest.TestCase):

    def test_arg_leaves(self):
        recombination_rate=1e-8
        Ne= 5000
        sample_size = 5
        length = 6e2
        ts_full = msprime.simulate(sample_size = sample_size, Ne = Ne, length = length, mutation_rate = 1e-8,
                            recombination_rate = recombination_rate, random_seed = 20, record_full_arg = True)
        tsarg = treeSequence.TreeSeq( ts_full )
        tsarg.ts_to_argnode()
        arg = tsarg.arg
        manual_leaves = [i for i in  range(sample_size)]
        leaves = list(arg.leaves(arg.__getitem__(arg.roots.max_key())))
        get_leaves =[]
        for node in leaves:
            get_leaves.append(node.index)
        self.assertTrue(sorted(get_leaves) == sorted(manual_leaves), True)


    def test_log_likelihood(self):
        recombination_rate=1e-8
        Ne= 5000
        sample_size = 5
        length = 6e2
        ts_full = msprime.simulate(sample_size = sample_size, Ne = Ne, length = length, mutation_rate = 1e-8,
                            recombination_rate = recombination_rate, random_seed = 20, record_full_arg = True)

        tsarg= treeSequence.TreeSeq( ts_full )
        tsarg.ts_to_argnode()
        argnode = tsarg.arg
        data = treeSequence.get_arg_genotype( ts_full )

        # put some mutations on some nodes
        argnode[6].snps.__setitem__(101, 101)
        argnode[6].snps.__setitem__(10, 10)
        argnode[3].snps.__setitem__(20, 20)
        argnode[9].snps.__setitem__(448, 448)
        a = bintrees.AVLTree()
        a.update({2:2, 4:4})
        data[10] = a
        a = bintrees.AVLTree()
        a.update({2:2, 4:4})
        data[101] = a
        a = bintrees.AVLTree()
        a.update({3:3})
        data[20] = a
        a = bintrees.AVLTree()
        a.update({2:2, 4:4, 3:3})
        data[448] = a
        #print
        nodes_with_mutation = []
        for node in argnode.nodes.values():
            if node.snps:
                nodes_with_mutation.append(node)
                # print("node", node.index, node.snps)

        total_material = (ts_full.tables.nodes.time[5] * 600) + (ts_full.tables.nodes.time[5] * 600) + \
                         (ts_full.tables.nodes.time[6] * 600) + (ts_full.tables.nodes.time[6] * 600) + \
                        ((ts_full.tables.nodes.time[7] - ts_full.tables.nodes.time[6]) * 600) + \
                        ((ts_full.tables.nodes.time[9] - ts_full.tables.nodes.time[8]) * 554) + \
                         ((ts_full.tables.nodes.time[9] - ts_full.tables.nodes.time[3]) * 600) + \
                         ((ts_full.tables.nodes.time[10] - ts_full.tables.nodes.time[9]) * 600) + \
                         ((ts_full.tables.nodes.time[12] - ts_full.tables.nodes.time[10]) * 448) + \
                         ((ts_full.tables.nodes.time[12] - ts_full.tables.nodes.time[11]) * 152) + \
                         ((ts_full.tables.nodes.time[13] - ts_full.tables.nodes.time[5]) * 600) +  \
                         ((ts_full.tables.nodes.time[13] - ts_full.tables.nodes.time[7]) * 46) + \
                         ((ts_full.tables.nodes.time[14] - ts_full.tables.nodes.time[13]) * 600) + \
                         ((ts_full.tables.nodes.time[14] - ts_full.tables.nodes.time[12]) * 600)
        number_of_mutations = 6
        m = 6 # number of snps
        mu = 0.1
        true_log_likelihood = (number_of_mutations * math.log(total_material * mu) -
                            total_material * mu)
        true_log_likelihood += math.log((ts_full.tables.nodes.time[14] -
                                         ts_full.tables.nodes.time[3]) / total_material)#x=20 , node=3
        true_log_likelihood += math.log((ts_full.tables.nodes.time[14] -
                                         ts_full.tables.nodes.time[5]) / total_material)#x=111 , node=5
        true_log_likelihood += math.log((ts_full.tables.nodes.time[14] -
                                         ts_full.tables.nodes.time[5]) / total_material)#x=558 , node=5
        true_log_likelihood += math.log((ts_full.tables.nodes.time[13] -
                                         ts_full.tables.nodes.time[6]) / total_material)#x=10 , node=6
        true_log_likelihood += math.log((ts_full.tables.nodes.time[9] -
                                         ts_full.tables.nodes.time[6]) / total_material)#x=101 , node=6
        true_log_likelihood += math.log((ts_full.tables.nodes.time[14] -
                                         ts_full.tables.nodes.time[9]) / total_material)#x=448 , node=9
        #----- log likelihood function
        self.assertTrue(math.isclose(
                            true_log_likelihood, argnode.log_likelihood(mu, data)))

    def test_log_prior(self):
        recombination_rate=1e-8
        Ne= 5000
        sample_size = 5
        length = 6e2
        ts_full = msprime.simulate(sample_size = sample_size, Ne = Ne, length = length, mutation_rate = 1e-8,
                            recombination_rate = recombination_rate, random_seed = 20, record_full_arg = True)
        tsarg = treeSequence.TreeSeq( ts_full )
        tsarg.ts_to_argnode()
        argnode = tsarg.arg
        r = 0.1
        k = 5# number_of_lineages
        num_link = 5 * (length - 1)
        rate = (k * (k - 1) / (2*2*Ne)) + (num_link * r)
        #ca node =5
        true_log_prior = 0
        true_log_prior  -= rate * (ts_full.tables.nodes.time[5] - 0) + math.log(2*Ne)
        num_link -= 599
        k = 4
        # ca, node =6
        rate = (k * (k - 1) / (2*2*Ne)) + (num_link * r)
        true_log_prior  -= rate * (ts_full.tables.nodes.time[6] - ts_full.tables.nodes.time[5])+\
                                 math.log(2*Ne)
        num_link -= 599
        k = 3
        #rec nodes 7, 8
        rate = (k * (k - 1) / (2*2*Ne)) + (num_link * r)
        gap = 1
        true_log_prior  -= rate * (ts_full.tables.nodes.time[7] - ts_full.tables.nodes.time[6])
        true_log_prior += math.log(r )
        num_link -= 1
        k = 4
        # CA , node = 9
        rate = (k * (k - 1) / (2*2*Ne)) + (num_link * r)
        true_log_prior  -= rate * (ts_full.tables.nodes.time[9] - ts_full.tables.nodes.time[8]) +\
                                     math.log(2*Ne)
        num_link -= 553
        k = 3
        #Rec , nodes = 10, 11
        rate = (k * (k - 1) / (2*2*Ne)) + (num_link * r)
        gap = 1
        true_log_prior  -= rate * (ts_full.tables.nodes.time[10] - ts_full.tables.nodes.time[9])
        true_log_prior += math.log(r )
        num_link -= 1
        k = 4
        # CA, node= 12
        rate = (k * (k - 1) / (2*2*Ne)) + (num_link * r)
        true_log_prior  -= rate * (ts_full.tables.nodes.time[12] - ts_full.tables.nodes.time[10]) +\
                                                                             math.log(2*Ne)
        num_link += 1
        k = 3
        # CA , node = 13
        rate = (k * (k - 1) / (2*2*Ne)) + (num_link * r)
        true_log_prior  -= rate * (ts_full.tables.nodes.time[13] - ts_full.tables.nodes.time[12])+\
                                                                                 math.log(2*Ne)
        num_link -= 45
        k = 2
        # CA, node 14
        rate = (k * (k - 1) / (2*2*Ne)) + (num_link * r)
        true_log_prior  -= rate * (ts_full.tables.nodes.time[14] - ts_full.tables.nodes.time[13])+\
                                                                         math.log(2*Ne)
        num_link -= (599 + 599)
        k = 1
        #----- compare
        self.assertTrue(math.isclose(
                            true_log_prior, argnode.log_prior(sample_size, length, r, Ne)))

    def test_total_branch_length(self):
        recombination_rate=1e-8
        Ne= 5000
        sample_size = 5
        length = 6e2
        ts_full = msprime.simulate(sample_size = sample_size, Ne = Ne, length = length, mutation_rate = 1e-8,
                            recombination_rate = recombination_rate, random_seed = 20, record_full_arg = True)
        tsarg = treeSequence.TreeSeq( ts_full )
        tsarg.ts_to_argnode()
        argnode = tsarg.arg
        true_total_branch_length =\
            (ts_full.tables.nodes.time[5] * 600) + (ts_full.tables.nodes.time[5] * 600) + \
                         (ts_full.tables.nodes.time[6] * 600) + (ts_full.tables.nodes.time[6] * 600) + \
                        ((ts_full.tables.nodes.time[7] - ts_full.tables.nodes.time[6]) * 600) + \
                        ((ts_full.tables.nodes.time[9] - ts_full.tables.nodes.time[8]) * 554) + \
                         ((ts_full.tables.nodes.time[9] - ts_full.tables.nodes.time[3]) * 600) + \
                         ((ts_full.tables.nodes.time[10] - ts_full.tables.nodes.time[9]) * 600) + \
                         ((ts_full.tables.nodes.time[12] - ts_full.tables.nodes.time[10]) * 448) + \
                         ((ts_full.tables.nodes.time[12] - ts_full.tables.nodes.time[11]) * 152) + \
                         ((ts_full.tables.nodes.time[13] - ts_full.tables.nodes.time[5]) * 600) +  \
                         ((ts_full.tables.nodes.time[13] - ts_full.tables.nodes.time[7]) * 46) + \
                         ((ts_full.tables.nodes.time[14] - ts_full.tables.nodes.time[13]) * 600) + \
                         ((ts_full.tables.nodes.time[14] - ts_full.tables.nodes.time[12]) * 600)
        self.assertTrue(math.isclose(true_total_branch_length, argnode.total_branch_length()))
        #------ also verify other methdods of arg
        self.assertTrue(argnode.__contains__(14))

    def test_arg_coal_and_rec_nodes(self):
        recombination_rate=1e-8
        Ne= 5000
        sample_size = 5
        length = 6e2
        ts_full = msprime.simulate(sample_size = sample_size, Ne = Ne, length = length, mutation_rate = 1e-8,
                            recombination_rate = recombination_rate, random_seed = 20, record_full_arg = True)
        tsarg = treeSequence.TreeSeq( ts_full )
        tsarg.ts_to_argnode()
        argnode = tsarg.arg
        self.assertEqual(set([key for key in argnode.rec.keys()]), set([7, 8, 10, 11]))
        self.assertEqual(set([key for key in argnode.coal.keys()]), set([5, 6, 9, 13, 12, 14]))
        self.assertEqual(argnode.coal.__len__(), 6)

    def test_dump_and_load(self):
        recombination_rate=1e-8
        Ne= 5000
        sample_size = 25
        length = 6e2
        ts_full = msprime.simulate(sample_size = sample_size, Ne = Ne,
                                   length = length, mutation_rate = 1e-8,
                                    recombination_rate = recombination_rate,
                                   random_seed = 20, record_full_arg = True)
        tsarg = treeSequence.TreeSeq( ts_full )
        tsarg.ts_to_argnode()
        argnode = tsarg.arg
        argnode.dump(path = os.getcwd(),
                     file_name = 'pickle_out.arg')
        loaded_arg = argbook.ARG().load( path =os.getcwd() + '/pickle_out.arg' )
        for key in loaded_arg.nodes:
            self.assertEqual(argnode[key].index, loaded_arg[key].index)
            self.assertEqual(sorted(argnode[key].snps), sorted(loaded_arg[key].snps))
            if loaded_arg[key].left_parent is not None:
                self.assertEqual(argnode[key].left_parent.index, loaded_arg[key].left_parent.index)
                self.assertEqual(argnode[key].right_parent.index, loaded_arg[key].right_parent.index)
                self.assertEqual(argnode[key].first_segment.left, loaded_arg[key].first_segment.left)
                self.assertEqual(argnode[key].first_segment.right, loaded_arg[key].first_segment.right)
                self.assertEqual(sorted(argnode[key].first_segment.samples),
                                 sorted(loaded_arg[key].first_segment.samples))
                tail_argnode = argnode[key].get_tail()
                tail_loaded_arg = loaded_arg[key].get_tail()
                self.assertEqual(tail_argnode.left, tail_loaded_arg.left)
                self.assertEqual(tail_argnode.right, tail_loaded_arg.right)
                self.assertEqual(sorted(tail_argnode.samples), sorted(tail_loaded_arg.samples))
            else:
                self.assertEqual(argnode[key].left_parent, loaded_arg[key].left_parent)
                self.assertEqual(argnode[key].right_parent, loaded_arg[key].right_parent)
                self.assertEqual(argnode[key].first_segment, None)
                self.assertEqual(loaded_arg[key].first_segment, None)
            if loaded_arg[key].left_child is not None:
                self.assertEqual(argnode[key].left_child.index, loaded_arg[key].left_child.index)
                self.assertEqual(argnode[key].right_child.index, loaded_arg[key].right_child.index)
            else:
                self.assertEqual(argnode[key].left_child, loaded_arg[key].left_child)
                self.assertEqual(argnode[key].right_child, loaded_arg[key].right_child)
            self.assertEqual(argnode[key].breakpoint, loaded_arg[key].breakpoint)

    def test_arg_copy_and_equal(self):
        recombination_rate=1e-8
        Ne= 5000
        sample_size = 5
        length = 6e2
        ts_full = msprime.simulate(sample_size = sample_size, Ne = Ne, length = length, mutation_rate = 1e-8,
                            recombination_rate = recombination_rate, random_seed = 20, record_full_arg = True)

        tsarg = treeSequence.TreeSeq( ts_full )
        tsarg.ts_to_argnode()
        arg = tsarg.arg
        arg_copy= arg.copy()
        self.assertTrue(arg.equal(arg_copy), True)
        self.assertFalse(arg==arg_copy, False)

    def test_arg_total_tmrca(self):
        recombination_rate=1e-8
        Ne= 5000
        sample_size = 5
        length = 6e2
        ts_full = msprime.simulate(sample_size = sample_size, Ne = Ne, length = length, mutation_rate = 1e-8,
                            recombination_rate = recombination_rate, random_seed = 20, record_full_arg = True)
        tsarg = treeSequence.TreeSeq( ts_full )
        tsarg.ts_to_argnode()
        arg = tsarg.arg
        tot_tmrca= arg.total_tmrca(length)
        # print(tot_tmrca)

    def test_arg_allele_age(self):
        recombination_rate=1e-8
        Ne= 5000
        sample_size = 5
        length = 6e4
        ts_full = msprime.simulate(sample_size = sample_size, Ne = Ne,
                                   length = length, mutation_rate = 1e-8,
                            recombination_rate = recombination_rate,
                                   random_seed = 20, record_full_arg = True)
        tsarg = treeSequence.TreeSeq( ts_full )
        tsarg.ts_to_argnode()
        arg = tsarg.arg
        allele_age= arg.allele_age()

    def test_arg_breakpoints(self):
        recombination_rate=1e-8
        Ne= 5000
        sample_size = 5
        length = 6e4
        ts_full = msprime.simulate(sample_size = sample_size, Ne = Ne,
                                   length = length, mutation_rate = 1e-8,
                            recombination_rate = recombination_rate,
                                   random_seed = 20, record_full_arg = True)
        tsarg = treeSequence.TreeSeq( ts_full )
        tsarg.ts_to_argnode()
        arg = tsarg.arg
        all_recomb_events = arg.breakpoints()
        ancestral_recomb_events = arg.breakpoints(only_ancRec= True)


class TestMCMC(unittest.TestCase):

    def test_detach_update(self):
        '''detach a node and then update ancestral material
        RE: (2)b=5--> (3, 4),   t= 1.5
        CA: (3, 4)-->   6,      t= 2.5
        CA: (0, 1)-->   5,      t= 3.5
        CA: (5, 6)-->   7,      t= 4.5
        '''
        arg = argbook.ARG()
        arg.nodes[0] = argbook.Node( 0 ); arg.nodes[0].first_segment = argbook.Segment()
        arg.nodes[1] = argbook.Node( 1 ); arg.nodes[1].first_segment = argbook.Segment()
        arg.nodes[2] = argbook.Node( 2 ); arg.nodes[2].first_segment = argbook.Segment()
        arg.nodes[0].time = 0 ; arg.nodes[1].time = 0 ; arg.nodes[2].time = 0
        #--------- n = 3 , m=0 , seq length =10
        arg.nodes[0].first_segment.left = 0; arg.nodes[0].first_segment.right = 10
        arg.nodes[1].first_segment.left = 0; arg.nodes[1].first_segment.right = 10
        arg.nodes[2].first_segment.left = 0; arg.nodes[2].first_segment.right = 10
        arg.nodes[0].first_segment.samples.__setitem__(0, 0)
        arg.nodes[1].first_segment.samples.__setitem__(1, 1)
        arg.nodes[2].first_segment.samples.__setitem__(2, 2)
        #----rec on 2 at b= 5
        arg.nodes[3] = argbook.Node( 3 ); arg.nodes[3].first_segment = argbook.Segment()
        arg.nodes[3].first_segment.left = 0; arg.nodes[3].first_segment.right = 5
        arg.nodes[3].left_child = arg.nodes[2]; arg.nodes[3].right_child = arg.nodes[2]
        arg.nodes[3].first_segment.samples.__setitem__(2, 2)
        arg.nodes[4] = argbook.Node( 4 ); arg.nodes[4].first_segment = argbook.Segment()
        arg.nodes[4].first_segment.left = 5; arg.nodes[4].first_segment.right = 10
        arg.nodes[4].left_child = arg.nodes[2]; arg.nodes[4].right_child = arg.nodes[2]
        arg.nodes[4].first_segment.samples.__setitem__(2, 2)
        arg.nodes[2].left_parent = arg.nodes[3]; arg.nodes[2].right_parent = arg.nodes[4]
        arg.nodes[2].breakpoint = 5; arg.nodes[3].time = 1.5 ; arg.nodes[4].time = 1.5
        #------ CA (3,4 ) ---6
        arg.nodes[6] = argbook.Node( 6 ); arg.nodes[6].first_segment = argbook.Segment()
        arg.nodes[6].first_segment.left = 0; arg.nodes[6].first_segment.right = 10
        arg.nodes[6].left_child = arg.nodes[3]; arg.nodes[6].right_child = arg.nodes[4]
        arg.nodes[6].first_segment.samples.__setitem__(2, 2)
        arg.nodes[6].time = 2.5
        arg.nodes[3].left_parent = arg.nodes[6]; arg.nodes[3].right_parent = arg.nodes[6]
        arg.nodes[4].left_parent = arg.nodes[6]; arg.nodes[4].right_parent = arg.nodes[6]
        #-------- CA (0,1 ) ---> 5
        arg.nodes[5] = argbook.Node( 5 ); arg.nodes[5].first_segment = argbook.Segment()
        arg.nodes[5].first_segment.left = 0; arg.nodes[5].first_segment.right = 10
        arg.nodes[5].left_child = arg.nodes[0]; arg.nodes[5].right_child = arg.nodes[1]
        arg.nodes[5].first_segment.samples.update({0:0, 1:1})
        arg.nodes[5].time = 3.5
        arg.nodes[0].left_parent = arg.nodes[5]; arg.nodes[0].right_parent = arg.nodes[5]
        arg.nodes[1].left_parent = arg.nodes[5]; arg.nodes[1].right_parent = arg.nodes[5]

        #--- CA (5, 6) ---7
        arg.nodes[7] = argbook.Node( 7 ); arg.nodes[7].first_segment = argbook.Segment()
        arg.nodes[7].left_child = arg.nodes[5]; arg.nodes[7].right_child = arg.nodes[6]
        arg.nodes[7].time = 4.5
        arg.nodes[5].left_parent = arg.nodes[7]; arg.nodes[5].right_parent = arg.nodes[7]
        arg.nodes[6].left_parent = arg.nodes[7]; arg.nodes[6].right_parent = arg.nodes[7]
        #-------
        mcmc = MC.MCMC(ts_full='test')
        mcmc.arg = arg
        mcmc.arg.coal.update({5:5, 6:6 , 7:7})
        mcmc.arg.rec.update({3:3, 4:4 })
        mcmc.data = {}
        mcmc.n = 3
        mcmc.m = 0
        mcmc.seq_length = 10
        # mcmc.print_state()
        # -- - - - -
        detach = mcmc.arg.nodes[6]
        old_merger_time = detach.left_parent.time
        sib = mcmc.arg.nodes[6].sibling()
        mcmc.arg.detach(detach, sib)

        #--- parent is root --> add both to floating
        mcmc.floatings[detach.time] = detach.index
        mcmc.floatings[detach.left_parent.time] = sib.index
        assert sib.left_parent is None
        if sib.left_parent is not None:
            mcmc.update_all_ancestral_material(sib)
        mcmc.spr_reattach_floatings(detach, sib, old_merger_time)
        # print("coal", mcmc.arg.coal)
        # mcmc.print_state()
        mcmc.spr()

if __name__=="__main__":
    unittest.main()


