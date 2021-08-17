import os
import pandas as pd
import numpy as np
# matplotlib.use('Agg')  # NOQA
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import time

class Figure(object):
    """
    Superclass of figures . Each figure is a concrete subclass.
    """
    name = None
    def __init__(self, outpath = os.getcwd() +"/output"):
        self.outpath = outpath
        datafile_name = self.outpath + "/{}.h5".format(self.name)
        self.data = pd.read_hdf(datafile_name, mode="r")

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

class plot_summary(Figure):
    name = "summary"
    def plot(self,  true_values= False):
        df = self.data
        if true_values:
            truth =  self.load_true_values()
            true_anc_recomb= truth[3]
            true_nonanc_rec = truth [4]
            true_branch_length = truth[5]
        fig = plt.figure()
        fig.subplots_adjust(hspace = 0.35, wspace = 0.6)
        for i,  d in zip(range(9), ["likelihood", "prior", "posterior","branch length",
                                    "ancestral recomb", "non ancestral recomb", "mu", "r", "Ne"]):
            fig.add_subplot(3, 3, i+1)
            df = self.data[d]
            plt.plot(df)
            plt.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
            if true_values:
                plt.axhline(y= truth[i], color="r", linestyle = "--", lw= 1)
            plt.ylabel(d)
            if i>5:
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
        self.save(figure_name="summary" + time.strftime("%Y%m%d-%H%M%S"))
        plt.show()

class Trace(Figure):
    name = "summary"
    def arginfer_trace(self,  true_values= False):
        df = self.data
        if true_values:
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
            ", CPU time = " + str(int(self.data.setup[10]/60))+ " min\n")# +
                     # "detail accept: ["+ str(self.data.setup[11]) +", " + str(self.data.setup[12])+ ", " +
                     # str(self.data.setup[13]) + ", "+ str(self.data.setup[14])+ ", "
                     # + str(self.data.setup[15])+ " ,"+ str(self.data.setup[16]) +\
                     # " ,"+ str(self.data.setup[17])+"]")
        self.save(figure_name="arginfertrace" + time.strftime("%Y%m%d-%H%M%S"))
        plt.show()



