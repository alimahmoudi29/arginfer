'''run ARGweaver on real data'''
import os
import subprocess
import sys
import subprocess
import logging
import math
import re
import gzip
import csv
import os.path
import pandas as pd
import numpy as np
import time
import msprime
import shutil
f_dir = os.path.dirname(os.getcwd())+"/ARGinfer"
sys.path.append(f_dir)
# print(sys.path)
import treeSequence

import plot

'''
python3 aw_sample.py --sites file.sites \
    -d  /Users/amahmoudi/Dropbox/PhDMelbourne/Paper/ARGweaver/argweaver-master/bin \
    -x arg-sample --iterations 3 --sample_step 2 \
    --full_prefix /Users/amahmoudi/Ali/phd/github_projects/mcmc/test/alaki
     -Ne 5000 -mu 1e-8 -rho 1e-8
'''

class CyclicalARGError(Exception):
    """
    Exception raised when ARG Weaver generates a cyclical ARG. This is a bug in
    ARGWeaver, so there's nothing we can do about it other than catch the
    error and abort the conversion.
    See https://github.com/mdrasmus/argweaver/issues/19
    """

# def ARGweaver_smc_to_ts_txts(smc2bin_executable, prefix, nodes_fh, edges_fh):
#     """
#     convert the ARGweaver smc representation to tree sequence text format
#     """
#     logging.debug(
#         "== Converting the ARGweaver smc output file '{}' to .arg format using '{}' ==".format(
#             prefix + ".smc.gz", smc2bin_executable))
#     subprocess.call([smc2bin_executable, prefix + ".smc.gz", prefix + ".arg"])
#     with open(prefix + ".arg", "r+") as arg_fh:
#         return ARGweaver_arg_to_ts_txts(arg_fh, nodes_fh, edges_fh)
#
# def ARGweaver_arg_to_ts_txts(ARGweaver_arg_filehandle, nodes_fh, edges_fh):
#     """
#     convert the ARGweaver arg representation to tree sequence tables
#     We need to split ARGweaver records that extend over the whole genome into sections
#     that cover just that coalescence point.
#     returns the mapping of ARGweaver node names to TS node names
#     """
#     logging.debug("== Converting .arg output to tree seq ==")
#     ARG_nodes={} #cr[X] = child1:[left,right], child2:[left,right],... : serves as intermediate ARG storage
#     ARG_node_times={} #node_name => time
#     node_names={} #map of ARGweaver names -> numbers
#     tips = set()
#     root_node = None
#
#     #first row gives start and end
#     ARGweaver_arg_filehandle.seek(0)
#     firstline = next(ARGweaver_arg_filehandle)
#     m = re.match(r'^start=(\d+)\s+end=(\d+)\s*$', firstline)
#     if m:
#         start=float(m.group(1))
#         end=float(m.group(2))
#     else:
#         raise ValueError("Could not find start and end positions in .arg file")
#
#     for line_num, fields in enumerate(csv.DictReader(ARGweaver_arg_filehandle, delimiter='\t')):
#         assert (fields['name'] not in ARG_node_times), \
#                 "duplicate node names identified: line {}".format(line_num)
#         #HACK: make sure that parent nodes are strictly older than children.
#         #This assumes that parents always have a higher node number
#         ARG_node_times[fields['name']] = float(fields['age'])
#         #we save info about nodes when looking at their children, so we
#         # should save info into parent nodes
#         if fields['parents'] == '':
#             assert(root_node == None)
#             root_node = fields['name']
#             #don't need to record anything here, as we will grab details of the
#             # root when looking at children
#         else:
#             if fields['event']=='recomb':
#                 #each recombination event has multiple parents
#                 for second_parent, parent in enumerate(fields['parents'].split(",")):
#                     if parent not in ARG_nodes:
#                         ARG_nodes[parent]={}
#                     ARG_nodes[parent][fields['name']]=[
#                         (float(fields['pos']) if second_parent else start),
#                         (end if second_parent else float(fields['pos']))]
#             else:
#                 #these should all have one parent
#                 if fields['parents'] not in ARG_nodes:
#                     ARG_nodes[fields['parents']]={}
#                 ARG_nodes[fields['parents']][fields['name']]=[start,end]
#
#                 if fields['event']=='gene':
#                     #we should trust the labels from
#                     node_names[fields['name']] = int(fields['name'])
#                     tips.add(fields['name'])
#     #now relabel the internal nodes
#     for key in ARG_nodes:
#         node_names[key]=len(node_names)
#     print("ARG_nodesARG_nodesARG_nodesARG_nodes\n", ARG_nodes)
#     print("ARG_node_timesARG_node_timesARG_node_times\n",
#           ARG_node_times)
#     print("node_namesnode_names\n", node_names)
#     #recursive hack to make times strictly decreasing, using depth-first topological
#     # sorting algorithm
#     def set_child_times(node_name, node_order, temporary_marks=set()):
#         if node_name in ARG_nodes:
#             if node_name in temporary_marks:
#                 raise CyclicalARGError(
#                     "ARG has a cycle in it, around node {}. This should not be possible."
#                     "Aborting this conversion!".format(node_name))
#             if node_name not in node_order:
#                 temporary_marks.add(node_name)
#                 for child_name in ARG_nodes[node_name]:
#                     set_child_times(child_name, node_order, temporary_marks)
#                 node_order.append(node_name)
#                 temporary_marks.remove(node_name)
#
#     node_order = [] #contains the internal nodes, such that parent is always after child
#     set_child_times(root_node, node_order)
#
#     max_epsilon = len(node_order)
#     for epsilon, nm in enumerate(node_order):
#         ARG_node_times[nm] += 0.001 * (epsilon+1) / max_epsilon
#
#     print("id\tis_sample\ttime", file=nodes_fh)
#     for node_name in sorted(node_names, key=node_names.get): #sort by id
#         print("{id}\t{is_sample}\t{time}".format(
#             id=node_names[node_name],
#             is_sample=int(node_name in tips),
#             time=ARG_node_times[node_name]),
#             file=nodes_fh)
#
#     print("left\tright\tparent\tchild", file=edges_fh)
#     for node_name in sorted(ARG_node_times, key=ARG_node_times.get): #sort by time
#         # look at the break points for all the child sequences, and break up
#         # into that number of records
#         try:
#             children = ARG_nodes[node_name]
#             assert all([ARG_node_times[child] < ARG_node_times[node_name] for child in children])
#             breaks = set()
#             for leftright in children.values():
#                 breaks.update(leftright)
#             breaks = sorted(breaks)
#             for i in range(1,len(breaks)):
#                 leftbreak = breaks[i-1]
#                 rightbreak = breaks[i]
#                 #The read_text function allows `child` to be a comma-separated list of children
#                 children_str = ",".join(map(str, sorted([
#                     node_names[cnode] for cnode, cspan in children.items()
#                         if cspan[0]<rightbreak and cspan[1]>leftbreak])))
#                 print("{left}\t{right}\t{parent}\t{children}".format(
#                     left=leftbreak, right=rightbreak, parent=node_names[node_name],
#                     children=children_str), file=edges_fh)
#         except KeyError:
#             #these should all be the tips
#             assert node_name in tips, (
#                 "The node {} is not a parent of any other node, but is not a tip "
#                 "either".format(node_name))
#     nodes_fh.flush()
#     nodes_fh.seek(0)
#     edges_fh.flush()
#     edges_fh.seek(0)
#     return node_names
#


def main(args):
    import os
    import subprocess
    if not os.path.exists(args.full_prefix):
            os.makedirs(args.full_prefix)
    else:
        shutil.rmtree(args.full_prefix)
        os.mkdir(args.full_prefix)
    output= args.full_prefix
    args.full_prefix = args.full_prefix +"/"+"out"
    pre_time=time.time()
    cmd = [os.path.join(args.ARGweaver_executable_dir,
                        args.ARGweaver_sample_executable),
        '--sites', args.sites,
        '--popsize', str(args.effective_population_size),
        '--recombrate', str(args.recombination_rate),
        '--mutrate', str(args.mutation_rate),
        '--overwrite',
        '--randseed', str(int(args.random_seed)),
        '--infsites',
        '--ntimes', str(20),
        '--compress-seq', str(1),
        '--iters', str(args.iterations),
        '--sample-step', str(args.sample_step),
        '--output', args.full_prefix]#the prefix for the output
    # assert os.stat(aw_in.name).st_size > 0, "Initial .sites file is empty"
    logging.debug("running '{}'".format(" ".join(cmd)))
    subprocess.call(cmd)
    # smc = args.full_prefix + "." + str(args.iterations) + ".smc.gz"
    # assert os.path.isfile(smc),  "No output file names {}".format(smc)
    # arg_nex=smc.replace(".smc.gz", ".ts_nex")
    # with open(smc.replace(".smc.gz", ".TSnodes"), "w+") as nodes, \
    #     open(smc.replace(".smc.gz", ".TSedges"), "w+") as edges, \
    #     open(arg_nex, "w+") as ts_nex:
    #     ARGweaver_smc_to_ts_txts(
    #         os.path.join(args.ARGweaver_executable_dir, args.ARGweaver_smc2arg_executable),
    #         smc.replace(".smc.gz", ""),
    #         nodes, edges)
    #
    #
    #
    #
    #
    end_time = time.time()
    np.save(args.full_prefix+"_time",[end_time-pre_time])
    if args.plot:
        p = plot.Trace(output, argweaver= True)
        p.argweaver_trace()

if __name__=='__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(prog="AW",
            description='run argweaver on real data')
    # parser.add_argument('--sites', type=argparse.FileType('r', encoding='UTF-8'), default=None,
    #                                         help='--sites file')
    parser.add_argument('--sites', type=str, default=None, help='--sites file')
    parser.add_argument('--ARGweaver_executable_dir', '-d',
        default= "/Users/amahmoudi/Dropbox/PhDMelbourne/Paper/ARGweaver/argweaver-master/bin/",
                        #os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','argweaver/bin/')
        help='the path to the directory containing the ARGweaver executables')
    parser.add_argument('--ARGweaver_sample_executable', '-x', default="arg-sample",
                        help='the name of the ARGweaver executable')
    parser.add_argument('--ARGweaver_smc2arg_executable', '-s', default="smc2arg",
                        help='the name of the ARGweaver executable')
    parser.add_argument('--sample_size', '-n', type=int, default=5,
                        help='the sample size if a .trees file is not given')
    parser.add_argument('--effective_population_size', '-Ne', type=float,
                        default=5000, help='the effective population '
                        'size if a .trees file is not given')
    parser.add_argument('--iterations', type=int, default=10,
                        help= 'number of mcmc iterations')
    parser.add_argument('--sample_step', type=int, default= 5,
                        help=' the MCMC sample steps for storing the outputs')
    parser.add_argument('--sequence_length', '-l', type=float, default=55000,
                        help='the sequence length if a .trees file is not given')
    parser.add_argument('--recombination_rate', '-rho', type=float, default=1e-8,
                        help='the recombination rate ')
    parser.add_argument('--mutation_rate', '-mu', type=float, default=1e-8,
                        help='the mutation rate ')
    parser.add_argument('--random_seed', '-seed', type=int, default=1234,
                        help='a random seed for msprime & AW simulation')
    parser.add_argument('--outputdir', nargs="?", default=None,
                        help='the directory in which to store the intermediate files.'
                        ' If None, files are saved under temporary names')
    parser.add_argument('--full_prefix', type=str, default="out",
                        help='The full prefix of the out path')
    parser.add_argument( '-p','--plot', help="plot the output", action="store_true")
    parser.add_argument('--verbosity', '-v', action='count', default=0)
    args = parser.parse_args()
    main(args)
