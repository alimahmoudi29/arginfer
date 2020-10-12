''' the final qc files are AFR_ref_clean3.ped and ancestral_allele_clean.txt in
/Users/amahmoudi/Ali/phd/github_projects/real_data/qced_data

Here we need to make them ready for the algorithms.
Also for some SNPs, the ancestral allele is not known,
so we need to replace the reference allele for those.

'''
import pandas as pd
import numpy as np
import bintrees
import math
import os


class prepareD(object):
    def __init__(self, genome_start_bp= 1800001, seq_length = 5e4, data_path='', haplotype_name="AFR_ref_clean3.ped",
                 ancAllele_name="ancestral_allele_clean3.txt",
                 wanted_snps_name="wanted_snp_ids.txt", output_postfix="ready"):
        '''

        :param genome_start_bp: the start basepair on the genome
        :param data_path:
        :param haplotype_name:
        :param ancAllele_name:
        :param wanted_snps_name:
        :param output_postfix:
        '''
        self.genome_start_bp =genome_start_bp
        self.data_path = data_path
        self.haplotype_name= haplotype_name
        self.ancAllele_name= ancAllele_name
        self.wanted_snps_name = wanted_snps_name
        self.output_postfix=output_postfix
        self.seq_length= seq_length

        #read data
        self.haplotypes =  pd.read_csv(general_path + '/' +self.haplotype_name,
                          sep="\t", delimiter=' ',header=None)
        self.ancestral_allele = pd.read_csv(general_path + '/'+ self.ancAllele_name,
                               sep="\t",delimiter=' ', header=None)
        self.wanted_snps = pd.read_csv(general_path + '/'+ self.wanted_snps_name,
                               sep="\t",delimiter=' ', header=None)


    def QC_haplotype(self):
        # for some reason the last column is None, so lets drop it
        # if self.haplotypes[self.haplotypes.shape[1]-1][0]=="NaN":
        self.haplotypes.drop([self.haplotypes.shape[1]-1], axis=1, inplace=True)
        # else:
        #     print(self.haplotypes[self.haplotypes.shape[1]-1])
        #drop the haplotype Id and ...
        self.haplotypes.drop(list(range(6)), axis=1, inplace=True)
        self.haplotypes.columns = range(self.haplotypes.shape[1])

    def QC_ancAllele(self):
        #------ for some reason SNP 114 does not match the rest so,
        row = 0
        need_to_drop=False
        for i in range(self.ancestral_allele.shape[0]):
            '''check if the SNPs ID in ancestral correspond to the 
            SNPs we wanted'''
            if self.ancestral_allele.loc[i, 0]!= self.wanted_snps.loc[i,0]:
                need_to_drop=True
                break
            row+=1
        if need_to_drop:
            self.ancestral_allele.drop(axis=1,index=row, inplace=True)
            self.ancestral_allele.reset_index(drop=True, inplace=True)
        #validate
        for i in range(self.ancestral_allele.shape[0]):
            '''check if the SNPs ID in ancestral correspond to the 
            SNPs we wanted'''
            assert self.ancestral_allele.loc[i, 0]== self.wanted_snps.loc[i,0]
        self.ancestral_allele.reset_index(drop=True, inplace=True)

    def QC_notknown_ancAllele(self, policy="omit"):
        '''SNPs that the ancestral allele is not given, substitute the ref allele as anestral, or
        delete them from the data
        :param policy: either "omit" them or "sub_ref" subsitute the ref allele
        '''
        if policy =="omit":
            not_known_ancestrals = np.where(self.ancestral_allele[4]==".")[0]
            self.ancestral_allele = self.ancestral_allele.drop(not_known_ancestrals, axis = 0)#rows
            self.ancestral_allele.reset_index(inplace=True, drop=True)
            self.haplotypes = self.haplotypes.drop(not_known_ancestrals, axis=1)
            self.haplotypes.columns = range(self.haplotypes.shape[1])
        if policy=="sub_ref":
            for i in range(self.ancestral_allele.shape[0]):
                if self.ancestral_allele.loc[i, 4]==".":
                    self.ancestral_allele.loc[i, 4]= self.ancestral_allele.loc[i, 2]
    def QC_not_match_ancestral_allele(self):
        '''
        for some sites the ancestral allele does not match eighter
        the ref or the alternative, we check those here and
        give the reference value to them
        '''
        for ind in range(self.ancestral_allele.shape[0]):
            if self.ancestral_allele.loc[ind, 4] != self.ancestral_allele.loc[ind, 2] or \
                self.ancestral_allele.loc[ind, 4] != self.ancestral_allele.loc[ind, 3]:
                self.ancestral_allele.loc[ind, 4] =self.ancestral_allele.loc[ind, 2]

    def write_clean_data(self):
        #---------- writing file
        print("haplotype dimension:", self.haplotypes.shape)
        print("ancestral dimension:", self.ancestral_allele.shape)
        print("wanted snp dimension:", self.wanted_snps.shape)
        # for snp in range(self.haplotypes.shape[1]):
        #     assert
        test = np.where(self.ancestral_allele[1]==46356+1800001)[0]
        assert self.haplotypes.shape[1]==self.ancestral_allele.shape[0]
        assert self.wanted_snps.shape[0] ==self.ancestral_allele.shape[0]
        out_path = self.data_path + "/cleaned_data"
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        self.haplotypes.to_csv(out_path+'/haplotype_'+ self.output_postfix+'.txt',
                               header=None, index=None, sep='\t',  mode='w')
        modified_snp_pos = self.ancestral_allele[1]- self.genome_start_bp
        np.savetxt(out_path+'/SNP_pos_'+self.output_postfix+'.txt',
                   modified_snp_pos , delimiter='')
        np.savetxt(out_path+'/ancestral_allele_'+self.output_postfix+'.txt',
                   self.ancestral_allele[4].tolist(),
                   delimiter='',  fmt='%s')
    def make_argweaver_sites_file(self):
        with open(self.data_path + "/cleaned_data"+"/argweaver.sites", "w+") as aw_in:
            print("\t".join(["NAMES"]+[str(x) for x in range(self.haplotypes.shape[0])]), file=aw_in)
            print("\t".join(["REGION", "chr", "1", str(int(self.seq_length))]), file=aw_in)
            modified_snp_pos = self.ancestral_allele[1]- self.genome_start_bp
            for snp in range(self.haplotypes.shape[1]):
                print(modified_snp_pos[snp], "".join(self.haplotypes[snp]), sep="\t", file=aw_in)

            # aw_in.flush()
            # aw_in.seek(0)

if __name__=='__main__':
    # general_path = "/Users/amahmoudi/Ali/phd/github_projects/real_data/50Kdata"
    general_path = "/Users/amahmoudi/Ali/phd/github_projects/real_data/n10L50Kdata"
    p= prepareD(genome_start_bp= 1800001, seq_length= 5e4,data_path=general_path,
                haplotype_name="AFR_ref_clean50K3.ped",
                 ancAllele_name="ancestral_allele_clean50K3.txt",
                 wanted_snps_name="wanted_snp_ids50K.txt", output_postfix="ready")
    p.QC_haplotype()
    p.QC_ancAllele()
    p.QC_not_match_ancestral_allele()
    p.write_clean_data()
    p.make_argweaver_sites_file()

