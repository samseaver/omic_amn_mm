# %% markdown
# # Install conda on your Colab environment
# %% markdown
# Ignore this first cell if you are running the notebook in a local environment.
#
# One can still run it locally but it will have no effect.
# %% codecell
# Run this cell first - it will install a conda distribution (mamba)
# on your Drive then restart the kernel automatically
# (don't worry about the crashing/restarting kernel messages)
# It HAS to be runned FIRST everytime you use the notebook in colab

import os
import sys

# %% markdown
# # Set up your Colab or local environment
# # Then import libraries
# %% markdown
# Run this cell in both cases of use (local or Colab)
# %% codecell
DIRECTORY = './'
font = 'arial'

# printing the working directory files. One can check you see the same folders and files as in the git webpage.
print(os.listdir(DIRECTORY))

from Library.Build_Dataset import *
# %% markdown
# # Generate Training Sets with FBA simulation or experimental data file

# What you can change
seed = 10
np.random.seed(seed=seed)  # seed for random number generator

spc = 'athaliana'
cobraname = 'athaliana_plastidial_thylakoid_052324_duplicated'
# cobraname = 'sbicolor_plastidial_model_duplicated'
mediumname = 'plastidial_model_duplicated_restricted_media_noATP'


if 'atha' in spc:
    time_stamp = 'ZT9'
    other_colm = 'TSU' # 'TSU', 'C24'
    treatments = ['Control', 'Cold']
    # athaliana_thylakoid_C24_ZT9_Vbf_maxCtrl
    Vbfname = f"athaliana_thylakoid_{other_colm}_{time_stamp}_Vbf_maxCtrl.csv"
else:
    time_stamp = '21d'
    other_colm = 'Leaf'
    treatments = ['Control', 'FeLim', 'FeEX', 'ZnLim', 'ZnEx']
    Vbfname = f"Sorghum_{time_stamp}_Vbf_noOrganellar_maxCtrl.csv"
    Vbfname = f"Sorghum_thylakoid_{other_colm}_{time_stamp}_noOrganellar_Vbf_maxCtrl.csv"


mediumbound = 'UB' # Exact bound (EB) or upper bound (UB)
method = 'Vbf' #'FBA' # FBA, pFBA or EXP, Vbf, Vbf_Wt
reduce = False # Set at True if you want to reduce the model

size = len(treatments)
measure = []
# rfl = ['rxn00018_d0', 'rxn00018_d0_f', 'rxn00018_d0_r']
rfl = []
verbose = True
# End of What you can change

# Run cobra
Vbffile    = DIRECTORY+'Dataset_input/'+Vbfname
cobrafile  = DIRECTORY+'Dataset_input/'+cobraname
mediumfile = DIRECTORY+'Dataset_input/'+mediumname
parameter  = TrainingSet(cobraname=cobrafile,
                        mediumname=mediumfile, mediumbound=mediumbound,
                        method=method,objective=[],
                        measure=measure, Vbfname=Vbffile,
                        restrictedFittingList = rfl, treatments=treatments, verbose=verbose)
# Note: Leaving objective and mesaure as empty lists sets the default
# objective reaction of the SBML model as the objective reaction
# and the measure (Y) as this objective reaction.
parameter.get(sample_size=size, treatments=treatments, verbose=verbose)

# np.savetxt("Result/AfterGetTempPout.tsv", parameter.Pout, delimiter='\t')
# Saving file
trainingfile  = DIRECTORY+'Dataset_model/'+mediumname+'_'+parameter.mediumbound+'_'+str(size)+'_'+spc+'_'+other_colm+'_'+time_stamp+'_thylakoid'
parameter.save(trainingfile, reduce=reduce, verbose=verbose)
# np.savetxt("Result/AfterSaveTempPout.tsv", parameter.Pout, delimiter='\t')
# Verifying
parameter = TrainingSet()
parameter.load(trainingfile)
parameter.printout()
# np.savetxt("Result/AfterLoadTempPout.tsv", parameter.Pout, delimiter='\t')

# trainingfile  = DIRECTORY+'Dataset_model/e_coli_core_UB_1000'
# parameter.load(trainingfile)
# parameter.printout()
# %% markdown
# Using iML1515, alongside an experimental file that is guiding the generation of the training set (instead of the usual 'mediumname' we have a 'expname' file which contains all experimental media compositions, in order to obtain a training sets of all biologically relevant flux distributions according to these compositions. Note that we reduce the model in this next cell.
# %% codecell
# Generate training set with E coli iML1515 with FBA simulation
# constrained by experimental file: metabolites in medium are not drawn at
# random but are the same than in the provided training experimental file
# This cell may take several hours to execute! Avoid running this in Colab
