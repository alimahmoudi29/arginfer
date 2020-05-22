import numpy as np
np.random.seed(19680801)
data = np.random.randn(2, 100)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

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
        aw_stat_df=pd.DataFrame(columns=["prior","likelihood","posterior",
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
        true_nonanc_rec = truth [5]

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
