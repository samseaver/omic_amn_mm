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
cobraname = 'Athaliana_Thylakoid_Reconstruction_ComplexFix_070224_duplicated_noP'
mediumname = 'plant_autotrophic_media_restricted'

spc = 'Poplar'
cobraname = 'ptrich_4.1_plastid_Thylakoid_Reconstruction_ComplexFix_070224_noADP_duplicated_noP'
# cobraname = 'sbicolor_plastidial_model_duplicated'
mediumname = 'plastidial_model_duplicated_restricted_media_noATP_noADP_noP'

spc = 'Sorghum'
cobraname = 'sbicolor_3.1.1_plastid_Thylakoid_Reconstruction_ComplexFix_070224_noADP_duplicated_noP'
# cobraname = 'sbicolor_plastidial_model_duplicated'
mediumname = 'plastidial_model_duplicated_restricted_media_noATP_noADP_noP'


if 'atha' in spc:
    time_stamp = 'all' #'ZT9'
    other_colm = 'TSU' # 'TSU', 'C24'
    if time_stamp == 'all':
        treatments = ["Control_ZT1", "Control_ZT5", "Control_ZT9", "Control_ZT13", "Control_ZT17",
                      "Control_ZT21", "Freeze_ZT1", "Freeze_ZT5", "Freeze_ZT9", "Freeze_ZT13",
                      "Freeze_ZT17", "Freeze_ZT21"]
    else:
        treatments = ['Control', 'Freeze']
    # athaliana_thylakoid_C24_ZT9_Vbf_maxCtrl
                # athaliana_complexFix_C24_all_noADP_Vbf_maxCtrl_mixedRelab
    Vbfname = f"athaliana_complexFix_{other_colm}_{time_stamp}_noADP_Vbf_maxCtrl_mixedRelab.csv"
else:
    time_stamp = '21d'
    other_colm = 'Leaf'
    treatments = ['Control', 'FeLim', 'FeEX', 'ZnLim', 'ZnEx']
    # Sorghum_thylakoid_Leaf_21d_Vbf_maxCtrl
    # Vbfname = f"Sorghum_{time_stamp}_Vbf_noOrganellar_maxCtrl.csv"
    # Vbfname = f"Sorghum_thylakoid_{other_colm}_{time_stamp}_Vbf_maxCtrl.csv"
    Vbfname = f"{spc}_complexFix_{other_colm}_{time_stamp}_noADP_Vbf_maxCtrl_mixedRelab.csv"

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
trainingfile  = DIRECTORY+'Dataset_model/'+mediumname+'_'+parameter.mediumbound+'_'+str(size)+'_'+spc+'_'+other_colm+'_'+time_stamp+'_complexFix'
parameter.save(trainingfile, reduce=reduce, verbose=verbose)
# np.savetxt("Result/AfterSaveTempPout.tsv", parameter.Pout, delimiter='\t')
# Verifying
parameter = TrainingSet()
print("All Saved .. now loading")
parameter.load(trainingfile)
print("printing ... ")
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
