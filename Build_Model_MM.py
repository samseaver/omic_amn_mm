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
import numpy as np
from time import time
# RunningInCOLAB  = 'google.colab' in str(get_ipython())
#
# if RunningInCOLAB:
#     !pip install -q condacolab
#     import condacolab
#     condacolab.install()
# %% markdown
# # Set up your Colab or local environment
# # Then import libraries
# %% markdown
# Run this cell in both cases of use (local or Colab)
# %% codecell
# conda install keras
# %% codecell
# import os
# import sys
# RunningInCOLAB  = 'google.colab' in str(get_ipython())
# print(RunningInCOLAB)
# if RunningInCOLAB:
#
#     # Check everything is fine with conda in Colab
#     import condacolab
#     condacolab.check()
#
#     # Mount your drive environment in the colab runtime
#     from google.colab import drive
#     drive.mount('/content/drive',force_remount=True)
#
#     # Change this variable to your path on Google Drive to which the repo has been cloned
#     # If you followed the colab notebook 'repo_cloning.ipynb', nothing to change here
#     repo_path_in_drive = '/content/drive/My Drive/Github/amn_release/'
#     # Change directory to your repo cloned in your drive
#     DIRECTORY = repo_path_in_drive
#     os.chdir(repo_path_in_drive)
#     # Copy the environment given in the environment_amn_light.yml
#     !mamba env update -n base -f environment_amn_light.yml
#
#     # This is one of the few Colab-compatible font
#     font = 'Liberation Sans'
#
# else:
#
#     # In this case the local root of the repo is our working directory
DIRECTORY = './'
font = 'arial'

# printing the working directory files. One can check you see the same folders and files as in the git webpage.
print(os.listdir(DIRECTORY))

# from pathlib import Path
# project_root = str(Path(__file__).resolve())
# sys.path.append(project_root)
# print(project_root)
from Library.Build_Model import *

# We declare this function here and not in the
# function-storing python file to modify it easily
# as it can change the printouts of the methods
def printout(V, Stats, model, id='all'):
    # printing Stats
    print("R2 = %.2f (+/- %.2f) Constraint = %.2f (+/- %.2f)" % \
          (Stats.train_objective[0], Stats.train_objective[1],
           Stats.train_loss[0], Stats.train_loss[1]))
    Vout = tf.convert_to_tensor(np.float32(model.Y))
    Loss_norm, dLoss = Loss_Vout(V, model.Pout, Vout)
    print('Loss Targets', np.mean(Loss_norm))
    Loss_norm, dLoss = Loss_SV(V, model.S)
    print('Loss SV', np.mean(Loss_norm))
    Vin = tf.convert_to_tensor(np.float32(model.X))
    Pin = tf.convert_to_tensor(np.float32(model.Pin))
    Vlb = tf.convert_to_tensor(np.float32(model.LB))
    if Vin.shape[1] == model.S.shape[1]: # special case
        Vin  = tf.linalg.matmul(Vin, tf.transpose(Pin), b_is_sparse=True)
    Loss_norm, dLoss = Loss_Vin(V, model.Pin, Vin, model.mediumbound)
    print('Loss Vin bound', np.mean(Loss_norm))
    Loss_norm, dLoss = Loss_Vpos(V, Vlb, model)
    print('Loss V positive', np.mean(Loss_norm))
    # print(V)

    processResults(model.reactions, model.treatments, V, Vin, Pin, model.Pout, id)

def processResults(reactions, treatments, V, Vin, Pin, Pout, id):
    import pandas
    temp_df = pandas.DataFrame(data=V, columns=reactions, index=treatments)
    temp_df.to_csv(f"Result/{id}_V_rxn.tsv", sep='\t')

    # temp_df = pandas.DataFrame(data=Vin, columns=reactions)
    # temp_df.to_csv("Result/tempVin.tsv", sep='\t')
    #
    # temp_df = pandas.DataFrame(data=Pin, columns=reactions)
    # temp_df.to_csv("Result/tempPin.tsv", sep='\t')
    #
    # temp_df = pandas.DataFrame(data=Pout, columns=reactions)
    # temp_df.to_csv("Result/tempPout.tsv", sep='\t')

    np.savetxt(f"Result/{id}_V.tsv", V, delimiter='\t')
    np.savetxt(f"Result/{id}_X.tsv", Vin, delimiter='\t')
    np.savetxt(f"Result/{id}_Pin.tsv", Pin, delimiter='\t')
    np.savetxt(f"Result/{id}_Pout.tsv", Pout, delimiter='\t')


# %% markdown
# # Mechanistic Models
#
# # Examples with non-trainable mechanistic models, using FBA simulated training sets
# %% markdown
# In both LP and QP solver, one can change the `trainname` suffix (EB or UB) to use exact or upper bounds as inputs.
# %% markdown
# ## LP solver
# %% codecell
# Run Mechanistic model (no training) QP (quadratic program) or LP (linear program)
# using E. coli core simulation training sets and EB (or UB) bounds
LP = False
if LP:
    # What you can change
    seed = 10
    np.random.seed(seed=seed)
    trainname = 'e_coli_core_EB' # the training set file name
    trainname = 'athaliana_plastidial_model_duplicated_UB_5' # the training set file name

    size = 10 # number of runs must be lower than the number of element in trainname
    timestep = int(1.0e4) # LP 1.0e4 QP 1.0e5
    learn_rate = 0.3 # LP 0.3 QP 1.0
    decay_rate = 0.33 # only in QP, UB 0.333 EB 0.9
    objective = ['bio1_biomass'] # ['BIOMASS_Ecoli_core_w_GAM']
    # End of What you can change

    # Create model and run GD for X and Y randomly drawn from trainingfile
    trainingfile = DIRECTORY+'Dataset_model/'+trainname
    model = Neural_Model(trainingfile = trainingfile,
                  objective=objective,
                  model_type = 'MM_LP',
                  timestep = timestep,
                  learn_rate = learn_rate,
                  decay_rate = decay_rate,
                  verbose = True)

    # Select a random subset of the training set (of specified size)
    # With LP we also have to change b_ext and b_int accordingly
    ID = np.random.choice(model.X.shape[0], size, replace=False)
    model.X, model.Y = model.X[ID,:], model.Y[ID,:]
    if model.mediumbound == 'UB':
        model.b_ext = model.b_ext[ID,:]
    if model.mediumbound == 'EB':
        model.b_int = model.b_int[ID,:]

    # Prints a summary of the model before running
    model.printout()

    # Runs the appropriate method
    Ypred, Stats = MM_LP(model, verbose=True)

    # Printing results
    printout(Ypred, Stats, model)

# What you can change
seed = 10

np.random.seed(seed=seed)
trainname = 'e_coli_core_EB' # the training set file name
trainname = 'athaliana_plastidial_model_duplicated_UB_5' # the training set file name
trainname = 'athaliana_plastidial_model_duplicated_restricted_media_limNH3_8_UB_5'
# trainname = 'sandbox_model_restritected_media_UB_5'
#trainname = 'sorghum_plastidial_model_duplicated_restricted_media_rubisco_UB_5_sorghum_10min'
#trainname = 'sorghum_plastidial_model_duplicated_restricted_media_rubisco_UB_5_sorghum_wt'
# trainname = 'sandbox_model_restritected_media_UB_6_athaliana'
trainname = 'sorghum_plastidial_model_duplicated_restricted_media_rubisco_UB_5_sorghum_relab_newModel'
trainname = 'sorghum_plastidial_model_duplicated_restricted_media_noATPADP_UB_5_sorghum_relab_newModel'
trainname = 'sorghum_plastidial_model_duplicated_restricted_media_rubisco_UB_5_relab_obj'
trainname = 'sorghum_plastidial_model_duplicated_restricted_media_rubisco_UB_5_sorghuminitKapp'
trainname = 'sorghum_plastidial_model_duplicated_restricted_media_rubisco_UB_5_21d_noOrganellar'
trainname = 'plastidial_model_duplicated_restricted_media_UB_5_21d_noOrganellar_minBio'

loss_outfile="Result/"+trainname+"_loss"
targets_outfile= "Result/"+trainname+"_targets"
size = 6 # number of runs must be lower than the number of element in trainname
timestep = int(1.0e5) # LP 1.0e4 QP 1.0e5
timestep = int(1.7e6) # 3.5e6
learn_rate = 1 # LP 0.3 QP 1.0
decay_rate = .33 # only in QP, UB 0.333 EB 0.9


use_objective = False
objective = [use_objective, 'bio1_biomass'] #[] # ['bio1_biomass'] # ['BIOMASS_Ecoli_core_w_GAM']

biomass_max = 200.0
# End of What you can change


sTime = time.time()
# Create model and run GD for X and Y randomly drawn from trainingfile
trainingfile = DIRECTORY+'Dataset_model/'+trainname
model = Neural_Model(trainingfile = trainingfile,
              objective=objective,
              model_type = 'MM_QP',
              timestep = timestep,
              learn_rate = learn_rate,
              decay_rate = decay_rate,
              biomass_max = biomass_max,
              verbose=True)

model.printout()

# Select a random subset of the training set (of specified size)
if size < model.X.shape[0]:
    ID = np.random.choice(model.X.shape[0], size, replace=False)
    # print(abc)
    # model.X, model.Y= model.X[ID:ID+1,:], model.Y[ID:ID+1,:]
    model.X, model.Y, model.LB= model.X[ID,:], model.Y[ID,:], model.LB[ID,:]
    if len(objective):
        model.objY = model.objY[ID,:]


id = trainname.split(f'{model.mediumbound}_{len(model.treatments)}')[1]
np.savetxt(f"Y_{id}.csv", model.Y, delimiter=',')
# print(abc)

# Prints a summary of the model before running
model.printout()

# Runs the appropriate method
if model.model_type == 'MM_QP':
    Ypred, Stats = MM_QP(model, loss_outfile=loss_outfile, targets_outfile=targets_outfile, verbose=True)

# Printing results
printout(Ypred, Stats, model, id)
print('Done after ', (time.time()- sTime))
os.replace("Result/sandbox_model_restritected_media_UB_5_loss",\
 f"Result/sandbox_model_restritected_media_UB_5_loss_{id}")

print(abc)
for i in range(len(model.treatments)):
    id = model.treatments[i]
    sTime = time.time()
    # Create model and run GD for X and Y randomly drawn from trainingfile
    trainingfile = DIRECTORY+'Dataset_model/'+trainname
    model = Neural_Model(trainingfile = trainingfile,
                  objective=objective,
                  model_type = 'MM_QP',
                  timestep = timestep,
                  learn_rate = learn_rate,
                  decay_rate = decay_rate,
                  biomass_max = biomass_max,
                  verbose=True)

    model.printout()

    # Select a random subset of the training set (of specified size)
    # ID = np.random.choice(model.X.shape[0], size, replace=False)
    ID = [i]
    # print(abc)
    # model.X, model.Y= model.X[ID:ID+1,:], model.Y[ID:ID+1,:]
    model.X, model.Y, model.LB= model.X[ID,:], model.Y[ID,:], model.LB[ID,:]
    if len(objective):
        model.objY = model.objY[ID,:]

    # print(model.Y)
    np.savetxt(f"Y_{id}.csv", model.Y, delimiter=',')
    # print(abc)

    # Prints a summary of the model before running
    model.printout()

    # Runs the appropriate method
    if model.model_type == 'MM_QP':
        Ypred, Stats = MM_QP(model, loss_outfile=loss_outfile, targets_outfile=targets_outfile, verbose=True)

    # Printing results
    printout(Ypred, Stats, model, id)
    print('Done after ', (time.time()- sTime))
    os.replace("Result/sandbox_model_restritected_media_UB_5_loss",\
     f"Result/sandbox_model_restritected_media_UB_5_loss_{id}")
