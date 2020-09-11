from mcmc import *
import argparse
from plots import *
import comparison.plot

'''
python3 main.py -I 10000 --thin 20 --burn 0 -n 5 -L 1e3  --Ne 5000 -r 1e-8 -mu 1e-8 \
        --tsfull /Users/amahmoudi/Ali/phd/github_projects/mcmc/test1/ts_sim/sim_r1/n5Ne5K_L1K_iter0.args \
        -O /Users/amahmoudi/Ali/phd/github_projects/mcmc/ARGinfer/output \
        --random-seed 5 -p -v --verify
'''

def add_arguments(parser):
    parser.add_argument('--tsfull', type=argparse.FileType('r', encoding='UTF-8'), default=None,
                                            help='an msprime .srgs file.'
                                                 ' If none, simulate one with defaults')
    parser.add_argument('--data', '-D', type = str,
                        default= None, help='the path to the data', required=False)
    parser.add_argument('--iteration','-I', type=int, default=20,
                        help= 'the number of mcmc iterations')
    parser.add_argument('--thin', type=int, default= 10, help=' thining steps')
    parser.add_argument('--burn', '-b', type=int, default= 0, help=' The burn-in')
    parser.add_argument('--sample_size', '-n', type=int, default= 5, help=' sample size')
    parser.add_argument('--seq_length','-L', type=float, default=1e4,help='sequence length')
    parser.add_argument('--Ne', type=int, default= 5000, help=' effective population size')
    parser.add_argument('--recombination_rate', '-r', type=float, default=1e-8,
                        help=' the recombination rate per site per generation ')
    parser.add_argument('--mutation_rate', '-mu', type=float, default=1e-8,
                        help='the mutation rate per site per generation')
    parser.add_argument('--outpath', '-O',type=str,
                        default=os.getcwd()+"/output", help='The output path')
    parser.add_argument( '-p','--plot', help="plot the output", action="store_true")
    parser.add_argument("--random-seed", "-s", type = int, default=1)
    parser.add_argument(
            "-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument( "--verify", help="verify the output ARG", action="store_true")

def run_mcmc(args):
    data = args.data
    iteration = args.iteration
    thin = args.thin
    burn = args.burn
    n = args.sample_size
    seq_length = args.seq_length
    mu = args.mutation_rate
    r= args.recombination_rate
    Ne= args.Ne
    outpath = args.outpath
    tsfull = None
    if args.tsfull !=None:
        try:
            tsfull = msprime.load(args.tsfull.name) #trees is a fh
        except AttributeError:
            tsfull = msprime.load(args.tsfull)
    # random.seed(args.random_seed)
    # np.random.seed(args.random_seed+1)
    mcmc = MCMC(tsfull, n, Ne, seq_length, mu, r, data, outpath, args.verbose)
    mcmc.run(iteration, thin, burn, args.verify)
    if args.plot:
        p= comparison.plot.Trace(outpath, name= "summary")
        p.arginfer_trace()
    if args.plot:
        p = plot_summary(outpath)
        p.plot()
    if args.verbose:
        mcmc.print_state()

def main():
    parser = argparse.ArgumentParser(prog="arginfer",
                                     description='sample from the ARG')
    add_arguments(parser)
    args = parser.parse_args()
    run_mcmc(args)

if __name__=='__main__':
    main()
