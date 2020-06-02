import numpy as np
np.random.seed(19680801)
data = np.random.randn(2, 100)
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import math

class Figure(object):
    """
    Superclass of figures . Each figure is a concrete subclass.
    """
    # name = None
    def __init__(self, outpath = os.getcwd() +"/output",
                 name = "summary", argweaver = False):
        self.outpath = outpath
        self.name= name
        if not argweaver:# arginfer
            datafile_name = self.outpath + "/{}.h5".format(self.name)
            self.data = pd.read_hdf(datafile_name, mode="r")
        else: # argweaver
            datafile_name = self.outpath+"/out.stats"
            self.data = self.argweaver_read(datafile_name)

    def save(self, figure_name=None, bbox_inches="tight"):
        if figure_name is None:
            figure_name = self.name
        print("Saving figure '{}'".format(figure_name))
        plt.savefig(self.outpath+"/{}.pdf".format(figure_name),
                    bbox_inches='tight', dpi=400)
        # plt.savefig("figures/{}.png".format(figure_name), bbox_inches='tight', dpi=400)
        plt.close()

    def load_true_values(self,filename = "true_values.npy"):
        data_filename = self.outpath + "/{}".format(filename)
        return np.load(data_filename)

    def argweaver_read(self, aw_out):
        '''takes the argweaver out stats file and read id as a pd.df'''
        aw_stat_df = pd.DataFrame(columns=["prior","likelihood","posterior",
                                         "total recomb","noncompats", "branch length"])
        with open(aw_out, "r") as f:
            for line in f:
                if line[0]=="r":
                    l= line.split("\t")
                    l.remove("resample")
                    l=[float(e) for e in l]
                    new_df= pd.DataFrame([l[1:]],
                                         columns = aw_stat_df.columns.values.tolist())
                    aw_stat_df= aw_stat_df.append(new_df, ignore_index = True)
        return aw_stat_df

class Trace(Figure):

    def arginfer_trace(self,  true_values= True):
        df = self.data
        truth =  self.load_true_values()
        true_branch_length = truth[3]
        true_anc_recomb= truth[4]
        true_nonanc_rec = truth[5]
        fig = plt.figure()
        fig.subplots_adjust(hspace = 0.35, wspace = 0.6)
        for i,  d in zip(range(4), ["posterior", "branch length",
                                    "ancestral recomb", "non ancestral recomb"]):
            fig.add_subplot(2, 2, i+1)
            df = self.data[d]
            plt.plot(df)
            plt.ticklabel_format(style='sci',scilimits=(0,0),axis='y')# (0,0) includes all
            if true_values:
                plt.axhline(y= truth[i+2], color="r", linestyle = "--", lw= 1)
            plt.ylabel(d)
            if i>1:
                plt.xlabel("Iteration")
        fig.suptitle("Iter = " +str(int(self.data.setup[0]/1000)) + "K "+", thin = "+\
            str(int(self.data.setup[1]))+ " "+", burn: "+ str(int(self.data.setup[2]))+\
            ", n= " + str(int(self.data.setup[3]))+", Ne = "+ str(int(self.data.setup[6]/1000))  +\
                     "K,\n L= "+ str(int(self.data.setup[4]/1000))+\
            "K, m= " + str(int(self.data.setup[5]))+ ", accept= "+ str(self.data.setup[9])+\
            ", CPU time = " + str(int(self.data.setup[10]/60))+ " min\n" +
                     "detail accept: ["+ str(self.data.setup[11]) +", " + str(self.data.setup[12])+ ", " +
                     str(self.data.setup[13]) + ", "+ str(self.data.setup[14])+ ", "
                     + str(self.data.setup[15])+ " ,"+ str(self.data.setup[16]) +\
                     " ,"+ str(self.data.setup[17])+"]")
        self.save(figure_name="arginfertrace" + time.strftime("%Y%m%d-%H%M%S"))
        plt.show()

    def argweaver_trace(self, true_values= True):
        df = self.data
        iterations= self.data.shape[0]
        cpu_time = np.load(self.outpath+"/out_time.npy")[0]
        if true_values:
            truth =  self.load_true_values()
            true_branch_length = truth[3]
            true_anc_recomb= truth[4]
            true_nonanc_rec = truth [5]

            wanted_true = [truth[2], true_branch_length,
                           true_anc_recomb+true_nonanc_rec]
        burn_in=10000
        sample_step=20
        # keep every sample_stepth after burn_in
        df = df.iloc[burn_in::sample_step, :]
        fig = plt.figure()
        fig.subplots_adjust(hspace = 0.35, wspace = 0.6)
        for i,  d in zip(range(3), ["posterior", "branch length",
                                    "total recomb"]):
            fig.add_subplot(2, 2, i+1)
            df = self.data[d]
            plt.plot(df)
            plt.ticklabel_format(style='sci',scilimits=(0,0),axis='y')# (0,0) includes all
            if true_values:
                if i != 0:# posterior is different formula in argweaver
                    plt.axhline(y= wanted_true[i], color="r", linestyle = "--", lw= 1)
                if i == 2:
                    plt.axhline(y= true_anc_recomb, color="g", linestyle = "--", lw= 1)
            plt.ylabel(d)
            if i>1:
                plt.xlabel("Iteration")
        fig.suptitle("Iter = " +str(int(iterations/1000)) + "K "+", thin = "+\
            str(int(sample_step))+ " "+", burn: "+ str(int(burn_in))+\
            ", CPU time = " + str(int(cpu_time/60))+ " min\n")

        self.save(figure_name="argweavertrace" + time.strftime("%Y%m%d-%H%M%S"))
        plt.show()

class Scatter(object):

    def __init__(self, truth_path, inferred_path,
                 columns =["likelihood","prior", "posterior", "branch length"], std= False):
        self.columns = columns
        self.truth_path = truth_path
        self.inferred_path = inferred_path
        if len(self.columns)/2 >=1:
            plot_dimension =[math.ceil(len(self.columns)/2),2]
        else:
            plot_dimension =[1, 1]
        self.fig = plt.figure(tight_layout=False)
        self.gs = gridspec.GridSpec(int(plot_dimension[0]), int(plot_dimension[1]))
        if not std:
            self.truth_data = pd.read_hdf(self.truth_path +  "/true_summary.h5", mode="r")
            self.inferred_data = pd.read_hdf(self.inferred_path + '/summary_all.h5', mode="r")
        else: # both inferred, note that instead of truth
            self.inferred_data = pd.read_hdf(self.inferred_path + '/summary_all.h5', mode="r")
            self.inferred2_data = pd.read_hdf(self.truth_path + "/summary_all.h5", mode="r")
        #row indexes with not None values
        self.not_None_rows = np.where(self.inferred_data['prior'].notnull())[0]

    def single_scatter(self, column_index, CI = False, argweaver= False):
        line_color= "red"
        point_color= "black"
        ecolor = "purple"
        elinewidth =1
        self.gs.update(wspace=0.3, hspace=0.7) # set the spacing between axes.
        if column_index <2:
            ax = self.fig.add_subplot(self.gs[0, column_index])
        elif column_index<4:
            ax = self.fig.add_subplot(self.gs[1, column_index-2])
        elif column_index <6:
            ax = self.fig.add_subplot(self.gs[2, column_index-4])
        else:
            ax = self.fig.add_subplot(self.gs[3, column_index-6])
        if argweaver and self.columns[column_index] == "ancestral recomb":
            col= 'total recomb'
        else:
            col = self.columns[column_index]
        if not CI:
            ax.errorbar(self.truth_data.loc[self.not_None_rows][self.columns[column_index]],
                            self.inferred_data.loc[self.not_None_rows][col],
                            color = point_color,linestyle='', fmt="o")
        else:
            ax.errorbar(self.truth_data.loc[self.not_None_rows][self.columns[column_index]],
                        self.inferred_data.loc[self.not_None_rows][col],
                        yerr= [self.inferred_data.loc[self.not_None_rows][col] -
                               self.inferred_data.loc[self.not_None_rows]['lower '+col],
                               self.inferred_data.loc[self.not_None_rows]['upper '+col] -
                               self.inferred_data.loc[self.not_None_rows][col]],
                                                            linestyle='', fmt="o",color= point_color,
                                                            ecolor= ecolor, elinewidth=elinewidth)
            # ax.set_aspect('equal',adjustable="box")
        minimum = np.min((ax.get_xlim(),ax.get_ylim()))
        maximum = np.max((ax.get_xlim(),ax.get_ylim()))
        ax.set_ylabel("Inferred ")
        ax.set_xlabel("True "+ self.columns[column_index])
        #scientific number
        ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        ax.ticklabel_format(style='sci',scilimits=(-2,2),axis='x')
        ax.plot([minimum, maximum], [minimum, maximum], ls="--",  color= line_color)

    def multi_scatter(self, CI = False, argweaver = False):
        for ind in range(len(self.columns)):
            self.single_scatter(column_index=ind, CI= CI, argweaver= argweaver)

        if not argweaver:
            self.fig.suptitle("ARGinfer")
            figure_name= "scatter"+"ARGinfer"
        else:
            self.fig.suptitle("ARGweaver")
            figure_name= "scatter"+"ARGweaver"
        if CI:
            figure_name = figure_name+"CI"
        plt.savefig(self.inferred_path+"/{}.pdf".format(figure_name),
                    bbox_inches='tight', dpi=400)
        plt.close()

    def single_std(self, column_index):
        self.gs.update(wspace=0.3, hspace=0.7) # set the spacing between axes.
        if column_index <2:
            ax = self.fig.add_subplot(self.gs[0, column_index])
        elif column_index<4:
            ax = self.fig.add_subplot(self.gs[1, column_index-2])
        elif column_index <6:
            ax = self.fig.add_subplot(self.gs[2, column_index-4])
        else:
            ax = self.fig.add_subplot(self.gs[3, column_index-6])
        if self.columns[column_index] == "ancestral recomb":
            col= 'total recomb'
        else:
            col = self.columns[column_index]
        ax.plot(range(self.inferred_data.loc[self.not_None_rows].shape[0]),
                      self.inferred_data.loc[self.not_None_rows]["std "+self.columns[column_index]],
                      color= "black", label ="ARGinfer")
        ax.plot(range(self.inferred2_data.loc[self.not_None_rows].shape[0]), # inferred 2 is argweaver
                      self.inferred2_data.loc[self.not_None_rows]["std "+col],
                      color= "red", label ="ARGweaver")
        ax.set_ylabel( self.columns[column_index])
        ax.set_xlabel('Data sets')
        #scientific number
        ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
        ax.ticklabel_format(style='sci',scilimits=(-2,2),axis='x')
        ax.legend( labelspacing=0,loc='upper right',frameon=False)

    def multi_std(self):
        for ind in range(len(self.columns)):
            self.single_std(column_index=ind)
        self.fig.suptitle("standard deviation of MCMC samples")
        figure_name= "std"

        plt.savefig(self.inferred_path+"/{}.pdf".format(figure_name),
                    bbox_inches='tight', dpi=400)
        plt.close()

def plot_tmrca(truth_path ='', argweaver_path='', arginfer_path='',
               inferred_filename='tmrca.h5'):
    '''a 1*2 plot for tmrca of arginfer and argweaver for same data set'''
    true_tmrca= np.load(truth_path)
    argweaver_tmrca= pd.read_hdf(argweaver_path+"/"+inferred_filename, mode="r")
    arginfer_tmrca= pd.read_hdf(arginfer_path+"/"+inferred_filename, mode="r")
    fig = plt.figure(tight_layout=False)
    gs = gridspec.GridSpec(1, 2)
    #arginfer
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(true_tmrca.size),
                  true_tmrca,
                  color= "black", label ="True")
    ax1.plot(range(arginfer_tmrca.shape[0]),
                  arginfer_tmrca["tmrca"],
                  color= "red", label ="Inferred")
    ax1.fill_between(range(arginfer_tmrca.shape[0]),
                    arginfer_tmrca["lower tmrca"],
                    arginfer_tmrca["upper tmrca"],
                    color='lightcoral', alpha=.2)
    # ax1.legend(labelspacing=0,loc='upper right',frameon=False)
    ax1.set_ylabel("TMRCA")
    ax1.set_xlabel('Genomic Position')
    ax1.set_title("ARGinfer")
    #scientific number
    ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
    #argweaver
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax2.plot(range(true_tmrca.size),
                  true_tmrca,
                  color= "black")
    ax2.plot(range(argweaver_tmrca.shape[0]),
                  argweaver_tmrca["tmrca"],
                  color= "red")
    ax2.fill_between(range(argweaver_tmrca.shape[0]),
                    argweaver_tmrca["lower tmrca"],
                    argweaver_tmrca["upper tmrca"],
                    color='lightcoral', alpha=.2)

    # ax2.legend( labelspacing=0,loc='upper right',frameon=False)
    ax2.set_title("ARGweaver")
    ax2.set_xlabel('Genomic Position')
    #scientific number
    ax2.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    ax2.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
    fig.legend(loc = 'best')#upper right
    figure_name= "tmrca_combined"
    plt.savefig(arginfer_path+"/{}.pdf".format(figure_name),
                bbox_inches='tight', dpi=400)
    plt.close()
    from scipy import stats
    pearson_coef1, p_value1 = stats.pearsonr(true_tmrca, arginfer_tmrca["tmrca"])
    pearson_coef2, p_value2 = stats.pearsonr(true_tmrca, argweaver_tmrca["tmrca"])
    print("arginfer pearson_coef", pearson_coef1, "p_value", p_value1)
    print("argweaver pearson_coef", pearson_coef2, "p_value", p_value2)

if __name__=='__main__':
    s= Scatter(truth_path= '/data/projects/punim0594/Ali/phd/mcmc_out/sim10L100K/second/sim_r0.5',
               inferred_path='/data/projects/punim0594/Ali/phd/mcmc_out/ARGinfer/n10L100K_r0.5',
               columns=["branch length", 'total recomb', "ancestral recomb", 'posterior'])
    s.multi_scatter(CI=True, argweaver= False)
    s= Scatter(truth_path= '/data/projects/punim0594/Ali/phd/mcmc_out/sim10L100K/second/sim_r0.5',
               inferred_path='/data/projects/punim0594/Ali/phd/mcmc_out/aw/r0.5/n10L100K',
               columns=["branch length", 'total recomb', "ancestral recomb"])
    s.multi_scatter(CI=True, argweaver= True)
    # s= Scatter(truth_path= '/data/projects/punim0594/Ali/phd/mcmc_out/aw/r2/n10L100K',
    #            inferred_path='/data/projects/punim0594/Ali/phd/mcmc_out/ARGinfer/n10L100K_r2',
    #            columns=["branch length", 'total recomb', "ancestral recomb", 'posterior'], std=True)
    # s.multi_std()
    # plot_tmrca(truth_path ='/data/projects/punim0594/Ali/phd/mcmc_out/sim10L100K/sim_r4/true_tmrca7.npy',
    #                arginfer_path='/data/projects/punim0594/Ali/phd/mcmc_out/ARGinfer/n10L100K_r4/out7',
    #                 argweaver_path = '/data/projects/punim0594/Ali/phd/mcmc_out/aw/r4/n10L100K/out7',
    #                inferred_filename='tmrca.h5')


