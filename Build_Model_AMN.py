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


DIRECTORY = './'
font = 'arial'

# printing the working directory files. One can check you see the same folders and files as in the git webpage.
print(os.listdir(DIRECTORY))

from Library.Build_Model import *

# We declare this function here and not in the
# function-storing python file to modify it easily
# as it can change the printouts of the methods
def printout(filename, Stats, model, time):
    # printing Stats
    print('Stats for %s CPU-time %.4f' % (filename, time))
    print('R2 = %.4f (+/- %.4f) Constraint = %.4f (+/- %.4f)' % \
          (Stats.train_objective[0], Stats.train_objective[1],
           Stats.train_loss[0], Stats.train_loss[1]))
    print('Q2 = %.4f (+/- %.4f) Constraint = %.4f (+/- %.4f)' % \
          (Stats.test_objective[0], Stats.test_objective[1],
           Stats.test_loss[0], Stats.test_loss[1]))


# What you can change
seed = 10
ratio_test = 0.1 # part of the training set removed for test
np.random.seed(seed=seed)
trainname = 'e_coli_core_UB_100' # can change EB by UB
timestep = 4
# End of What you can change

other  = False
if other:
    # Create model 90% for training 10% for testing
    trainingfile = DIRECTORY+'Dataset_model/'+trainname
    model = Neural_Model(trainingfile = trainingfile,
                         objective=['BIOMASS_Ecoli_core_w_GAM'],
                         model_type='AMN_QP',
                         timestep = timestep, learn_rate=0.01,
                         scaler=True,
                         n_hidden = 1, hidden_dim = 50,
                         epochs=50, xfold=5,
                         verbose=True)
    ID = np.random.choice(model.X.shape[0],
                          size=int(model.X.shape[0]*ratio_test), replace=False)
    Xtest,  Ytest  = model.X[ID,:], model.Y[ID,:]
    Xtrain, Ytrain = np.delete(model.X, ID, axis=0), np.delete(model.Y, ID, axis=0)
    model.printout()

    # Train and evaluate
    reservoirname = trainname+'_'+model.model_type
    reservoirfile = DIRECTORY+'Reservoir/'+reservoirname
    start_time = time.time()
    model.X, model.Y = Xtrain, Ytrain
    reservoir, pred, stats, _ = train_evaluate_model(model, verbose=False)
    delta_time = time.time() - start_time

    # Printing cross-validation results
    printout(reservoirname, stats, model, delta_time)

    # Save, reload and run idependent test set
    reservoir.save(reservoirfile)
    reservoir.load(reservoirfile)
    reservoir.printout()
    if len(Xtest) > 0:
        start_time = time.time()
        reservoir.X, reservoir.Y = Xtest, Ytest
        X, Y = model_input(reservoir, verbose=False)
        pred, stats = evaluate_model(reservoir.model, X, Y, reservoir, verbose=False)
        delta_time = time.time() - start_time
        printout('Test set', stats, model, delta_time)


    # What you can change
    seed = 10
    ratio_test = 0.1 # part of the training set removed for test
    np.random.seed(seed=seed)
    trainname = 'e_coli_core_UB_100' # can change EB by UB
    timestep = 4
    # End of What you can change

    # Create model 90% for training 10% for testing
    trainingfile = DIRECTORY+'Dataset_model/'+trainname
    model = Neural_Model(trainingfile = trainingfile,
                         objective=['BIOMASS_Ecoli_core_w_GAM'],
                         model_type='AMN_LP',
                         timestep = timestep, learn_rate=1.0e-6,
                         scaler=True,
                         n_hidden = 1, hidden_dim = 50,
                         epochs= 50, xfold=5,
                         verbose=True)
    ID = np.random.choice(model.X.shape[0],
                          size=int(model.X.shape[0]*ratio_test),
                          replace=False)
    # LP keeps track of boundary fluxes in b_int and b_ext
    # and these are different in EB or UB modes
    Xtest,  Ytest  = model.X[ID,:], model.Y[ID,:]
    btest = model.b_ext[ID,:] if model.mediumbound == 'UB' else model.b_int[ID,:]
    bint  = model.b_int if model.mediumbound == 'UB' else model.b_ext
    Xtrain, Ytrain = np.delete(model.X, ID, axis=0), np.delete(model.Y, ID, axis=0)
    btrain = np.delete(model.b_ext, ID, axis=0) if model.mediumbound == 'UB' \
    else np.delete(model.b_int, ID, axis=0)
    model.printout()

    # Train and evaluate
    reservoirname = trainname+'_'+model.model_type
    reservoirfile = DIRECTORY+'Reservoir/'+reservoirname
    start_time = time.time()
    model.X, model.Y, model.b_ext, model.b_int = Xtrain, Ytrain, btrain, bint
    reservoir, pred, stats, _ = train_evaluate_model(model, verbose=False)
    delta_time = time.time() - start_time

    # Printing cross-validation results
    printout(reservoirname, stats, model, delta_time)

    # Save, reload and run idependent test set
    reservoir.save(reservoirfile)
    reservoir.load(reservoirfile)
    # Issue for loading AMN-LP models: takes about 15 minutes (when QP takes a few)
    reservoir.printout()
    if len(Xtest) > 0:
        start_time = time.time()
        reservoir.X, reservoir.Y = Xtest, Ytest
        reservoir.b_ext, reservoir.b_int =  btest, bint
        X, Y = model_input(reservoir,verbose=False)
        pred, stats = evaluate_model(reservoir.model, X, Y, reservoir, verbose=False)
        delta_time = time.time() - start_time
        printout('Test set', stats, model, delta_time)

    # %% markdown
    # ### AMN with QP solver on iML1515 FBA simulated training set
    # %% codecell
    # Create, train and evaluate AMN_QP models with FBA simulated training set for iML1515
    # with EB or UB with a mechanistic layer
    # This cell takes several hours to execute

    # What you can change
    seed = 1
    ratio_test = 0.1 # part of the training set removed for test
    np.random.seed(seed=seed)
    trainname = 'iML1515_UB' # can change EB by UB
    timestep = 4
    # End of What you can change

    # Create model
    trainingfile = DIRECTORY+'Dataset_model/'+trainname
    model = Neural_Model(trainingfile = trainingfile,
                  objective=['BIOMASS_Ec_iML1515_core_75p37M'],
                  model_type = 'AMN_QP',
                  scaler = True,
                  timestep = timestep, learn_rate=0.01,
                  n_hidden = 1, hidden_dim = 500,
                  epochs = 25, xfold = 5,
                  verbose=True)
    ID = np.random.choice(model.X.shape[0],
                          size=int(model.X.shape[0]*ratio_test), replace=False)
    Xtest,  Ytest  = model.X[ID,:], model.Y[ID,:]
    Xtrain, Ytrain = np.delete(model.X, ID, axis=0), np.delete(model.Y, ID, axis=0)
    model.printout()

    # Train and evaluate
    reservoirname = trainname+'_'+model.model_type
    reservoirfile = DIRECTORY+'Reservoir/'+reservoirname
    start_time = time.time()
    model.X, model.Y = Xtrain, Ytrain
    reservoir, pred, stats, _ = train_evaluate_model(model, verbose=2)
    delta_time = time.time() - start_time

    # Printing cross-validation results
    printout(reservoirname, stats, model, delta_time)

    # Save, reload and run idependent test set
    reservoir.save(reservoirfile)
    reservoir.load(reservoirfile)
    reservoir.printout()
    if len(Xtest) > 0:
        start_time = time.time()
        reservoir.X, reservoir.Y = Xtest, Ytest
        X, Y = model_input(reservoir, verbose=False)
        reservoir.model.b_ext = btest
        pred, stats = evaluate_model(reservoir.model, X, Y, reservoir, verbose=False)
        delta_time = time.time() - start_time
        printout('Test set', stats, model, delta_time)

    # %% markdown
    # ### AMN with QP solver on iJN1463 FBA simulated training set
    # %% codecell
    # Create, train and evaluate AMN_QP models with FBA simulated training set for P. putida iJN1463
    # with UB with a mechanistic layer
    # This cell takes several hours to execute

    # What you can change
    seed = 1
    ratio_test = 0.1 # part of the training set removed for test
    np.random.seed(seed=seed)
    trainname = 'IJN1463_10_UB' # can change EB by UB
    timestep = 4
    # End of What you can change

    # Create model
    trainingfile = DIRECTORY+'Dataset_model/'+trainname
    model = Neural_Model(trainingfile = trainingfile,
                  objective=['BIOMASS_KT2440_WT3'],
                  model_type = 'AMN_QP',
                  scaler = True,
                  timestep = timestep, learn_rate=0.01,
                  n_hidden = 1, hidden_dim = 500, batch_size=100,
                  epochs = 500, xfold = 5,
                  verbose=True)
    ID = np.random.choice(model.X.shape[0],
                          size=int(model.X.shape[0]*ratio_test), replace=False)
    Xtest,  Ytest  = model.X[ID,:], model.Y[ID,:]
    Xtrain, Ytrain = np.delete(model.X, ID, axis=0), np.delete(model.Y, ID, axis=0)
    model.printout()

    # Train and evaluate
    reservoirname = trainname+'_'+model.model_type
    reservoirfile = DIRECTORY+'Reservoir/'+reservoirname
    start_time = time.time()
    model.X, model.Y = Xtrain, Ytrain
    reservoir, pred, stats, _ = train_evaluate_model(model, verbose=False)
    delta_time = time.time() - start_time

    # Printing cross-validation results
    printout(reservoirname, stats, model, delta_time)

    # Save, reload and run idependent test set
    reservoir.save(reservoirfile)
    reservoir.load(reservoirfile)
    reservoir.printout()
    if len(Xtest) > 0:
        start_time = time.time()
        reservoir.X, reservoir.Y = Xtest, Ytest
        X, Y = model_input(reservoir, verbose=False)
        pred, stats = evaluate_model(reservoir.model, X, Y, reservoir, verbose=False)
        delta_time = time.time() - start_time
        printout('Test set', stats, model, delta_time)
    # %% markdown
    # ### AMN with LP solver on iML1515 FBA simulated training set
    # %% codecell
    # Create, train and evaluate AMN_LP models with FBA simulated training set for iML1515
    # with UB with a mechanistic layer
    # This cell takes several hours to execute

    # What you can change
    seed = 1
    ratio_test = 0.1 # part of the training set removed for test
    np.random.seed(seed=seed)
    trainname = 'iML1515_UB'
    timestep = 4
    # End of What you can change

    # Create model
    trainingfile = DIRECTORY+'Dataset_model/'+trainname
    model = Neural_Model(trainingfile = trainingfile,
                  objective=['BIOMASS_Ec_iML1515_core_75p37M'],
                  model_type = 'AMN_LP',
                  scaler = True,
                  timestep = timestep, learn_rate=1.0e-6,
                  n_hidden = 1, hidden_dim = 250,
                  epochs = 25, xfold = 5,
                  verbose=True)

    ID = np.random.choice(model.X.shape[0],
                          size=int(model.X.shape[0]*ratio_test),
                          replace=False)
    Xtest,  Ytest  = model.X[ID,:], model.Y[ID,:]
    btest = model.b_ext[ID,:]
    bint = model.b_int
    Xtrain, Ytrain = np.delete(model.X, ID, axis=0), np.delete(model.Y, ID, axis=0)
    btrain = np.delete(model.b_ext, ID, axis=0)
    model.printout()

    # Train and evaluate
    reservoirname = trainname+'_'+model.model_type
    reservoirfile = DIRECTORY+'Reservoir/'+reservoirname
    start_time = time.time()
    model.X, model.Y, model.b_ext, model.b_int = Xtrain, Ytrain, btrain, bint
    reservoir, pred, stats, _ = train_evaluate_model(model, verbose=2)
    delta_time = time.time() - start_time

    # Printing cross-validation results
    printout(reservoirname, stats, model, delta_time)

    # Save, reload and run idependent test set
    reservoir.save(reservoirfile)
    # reservoir.load(reservoirfile)
    # Issue for loading AMN-LP models: takes about 15 minutes (when QP takes a few)
    reservoir.printout()
    if len(Xtest) > 0:
        start_time = time.time()
        reservoir.X, reservoir.Y = Xtest, Ytest
        reservoir.b_ext, reservoir.b_int =  btest, bint
        X, Y = model_input(reservoir,verbose=False)
        pred, stats = evaluate_model(reservoir.model, X, Y, reservoir, verbose=False)
        delta_time = time.time() - start_time
        printout('Test set', stats, model, delta_time)

# %% markdown
# ### AMN-Wt on *E. coli* core FBA simulated training set
# %% codecell
# Create, train and evaluate AMN_Wt models with FBA simulated training set for E. coli core
# with UB (not working with M1 chips)

# What you can change
seed = 10
ratio_test = 0.1 # part of the training set removed for test
np.random.seed(seed=seed)
trainname = 'e_coli_core_UB' # can change EB by UB
trainname = 'sandbox_model_restritected_media_UB_6_athaliana_wt'
trainname = 'sandbox_model_restritected_media_UB_5_sorghum_wt'
timestep = 4
batch_size = 5
constraint_loss = True
# End of What you can change

# Create model 90% for training 10% for testing
trainingfile = DIRECTORY+'Dataset_model/'+trainname
model = Neural_Model(trainingfile = trainingfile,
                     objective=[], #['BIOMASS_Ecoli_core_w_GAM']
                     model_type='AMN_Wt',
                     timestep = timestep,
                     n_hidden = 1, hidden_dim = 100,
                     scaler=True,
                     train_rate=1e-2,
                     epochs=2000,
                     batch_size = batch_size,
                     xfold=1,
                     verbose=True)
if constraint_loss:
    model.Y = model.X.copy()
print("starting Y shapes ", model.Y.shape)

# if ratio_test > 0:
ID = np.random.choice(model.X.shape[0],
                      size=int(model.X.shape[0]*ratio_test), replace=False)


Xtest,  Ytest  = model.X[ID,:], model.Y[ID,:]
Xtrain, Ytrain = np.delete(model.X, ID, axis=0), np.delete(model.Y, ID, axis=0)

print("Here's ", Xtest, " shape ", Xtrain.shape)
model.printout()

# Train and evaluate
reservoirname = trainname+'_'+model.model_type
reservoirfile = DIRECTORY+'Reservoir/'+reservoirname
start_time = time.time()
model.X, model.Y = Xtrain, Ytrain
reservoir, pred, stats, _ = train_evaluate_model(model, verbose=True)
print(abc)
# np.savetxt("Result/final_PredEval.csv", pred, delimiter=',')

delta_time = time.time() - start_time

# Printing cross-validation results
printout(reservoirname, stats, model, delta_time)

# Save, reload and run idependent test set
#reservoir.save(reservoirfile)
#reservoir.load(reservoirfile)
reservoir.printout()
# print(abc)

if len(Xtest) > 0:
    start_time = time.time()
    reservoir.X, reservoir.Y = Xtest, Ytest
    X, Y = model_input(reservoir, verbose=False)
    pred, stats = evaluate_model(reservoir.model, X, Y, reservoir, verbose=False)
    delta_time = time.time() - start_time
    printout('Test set', stats, model, delta_time)

print(abc)
# %% markdown
# ### AMN-Wt on iML1515 FBA simulated training set
# %% codecell
# Create, train and evaluate AMN_QP models with FBA simulated training set for iML1515
# with EB or UB with a mechanistic layer
# This cell takes several hours to execute

# What you can change
seed = 1
ratio_test = 0.1 # part of the training set removed for test
np.random.seed(seed=seed)
trainname = 'iML1515_UB'
timestep = 4
# End of What you can change

# Create model
trainingfile = DIRECTORY+'Dataset_model/'+trainname
model = Neural_Model(trainingfile = trainingfile,
              objective=['BIOMASS_Ec_iML1515_core_75p37M'],
              model_type = 'AMN_Wt',
              scaler = True,
              timestep = timestep,
              n_hidden = 1, hidden_dim = 500,
              train_rate=1e-2,
              epochs = 100, xfold = 5,
              verbose=True)

ID = np.random.choice(model.X.shape[0],
                      size=int(model.X.shape[0]*ratio_test),
                      replace=False)
Xtest,  Ytest  = model.X[ID,:], model.Y[ID,:]
Xtrain, Ytrain = np.delete(model.X, ID, axis=0), np.delete(model.Y, ID, axis=0)
model.printout()

# Train and evaluate
reservoirname = trainname+'_'+model.model_type
reservoirfile = DIRECTORY+'Reservoir/'+reservoirname
start_time = time.time()
model.X, model.Y = Xtrain, Ytrain
reservoir, pred, stats, _ = train_evaluate_model(model, verbose=2)
delta_time = time.time() - start_time

# Printing cross-validation results
printout(reservoirname, stats, model, delta_time)

# Save, reload and run idependent test set
#reservoir.save(reservoirfile)
#reservoir.load(reservoirfile)
reservoir.printout()
if len(Xtest) > 0:
    start_time = time.time()
    reservoir.X, reservoir.Y = Xtest, Ytest
    X, Y = model_input(reservoir,verbose=False)
    pred, stats = evaluate_model(reservoir.model, X, Y, reservoir, verbose=False)
    delta_time = time.time() - start_time
    printout('Test set', stats, model, delta_time)
# %% markdown
# ### AMN-QP with experimental training set
# %% codecell
# Create, train and evaluate AMN_QP models on experimental training set with UB
# Repeat the process with different seeds
# This cell takes several hours to execute

Maxloop, Q2, PRED = 3, [], []

for Nloop in range(Maxloop):
    # What you can change
    seed = Nloop+1
    np.random.seed(seed=seed)
    trainname = 'iML1515_EXP_UB'
    timestep = 4
    # End of What you can change

    # Create model 100% for training 0% for testing
    trainingfile = DIRECTORY+'Dataset_model/'+trainname
    model = Neural_Model(trainingfile = trainingfile,
              objective=['BIOMASS_Ec_iML1515_core_75p37M'],
              model_type = 'AMN_QP',
              scaler = True,
              timestep = timestep, learn_rate=0.001,
              n_hidden = 1, hidden_dim = 500,
              #train_rate = 1.0e-2,
              epochs = 1000, xfold = 10,
              verbose=True)

    # Train and evaluate
    reservoirname = trainname +'_'+model.model_type
    reservoirfile = DIRECTORY+'Reservoir/'+reservoirname
    start_time = time.time()
    reservoir, pred, stats, _ = train_evaluate_model(model, verbose=True)
    delta_time = time.time() - start_time

    # Printing cross-validation results
    printout(reservoirname, stats, model, delta_time)
    r2 = r2_score(model.Y, pred[:,0], multioutput='variance_weighted')
    print('Iter', Nloop, 'Collated Q2', r2)
    Q2.append(r2)
    PRED.append(pred[:,0])

# Save in Result folder
Q2, PRED = np.asarray(Q2), np.asarray(PRED)
print('Averaged Q2 = %.4f (+/- %.4f)' % (np.mean(Q2), np.std(Q2)))
filename = DIRECTORY+'Result/'+reservoirname+'_Q2.csv'
np.savetxt(filename, Q2, delimiter=',')
filename = DIRECTORY+'Result/'+reservoirname+'_PRED.csv'
np.savetxt(filename, PRED, delimiter=',')

# %% markdown
# ### AMN-LP with experimental training set
# %% codecell
# Create, train and evaluate AMN_QP models on experimental training set with UB
# Repeat the process with different seeds
# This cell takes several hours to execute

Maxloop, Q2, PRED = 3, [], []

for Nloop in range(Maxloop):
    # What you can change
    seed = Nloop+1
    np.random.seed(seed=seed)
    trainname = 'iML1515_EXP_UB'
    timestep = 4
    # End of What you can change

    # Create model 100% for training 0% for testing
    trainingfile = DIRECTORY+'Dataset_model/'+trainname
    model = Neural_Model(trainingfile = trainingfile,
              objective=['BIOMASS_Ec_iML1515_core_75p37M'],
              model_type = 'AMN_LP',
              scaler = True,
              timestep = timestep, learn_rate=0.001,
              n_hidden = 1, hidden_dim = 500,
              #train_rate = 1.0e-2,
              epochs = 1000, xfold = 10,
              verbose=True)

    # Train and evaluate
    reservoirname = trainname +'_'+model.model_type
    reservoirfile = DIRECTORY+'Reservoir/'+reservoirname
    start_time = time.time()
    reservoir, pred, stats, _ = train_evaluate_model(model, verbose=2)
    delta_time = time.time() - start_time

    # Printing cross-validation results
    printout(reservoirname, stats, model, delta_time)
    r2 = r2_score(model.Y, pred[:,0], multioutput='variance_weighted')
    print('Iter', Nloop, 'Collated Q2', r2)
    Q2.append(r2)
    PRED.append(pred[:,0])

# Save in Result folder
Q2, PRED = np.asarray(Q2), np.asarray(PRED)
print('Averaged Q2 = %.4f (+/- %.4f)' % (np.mean(Q2), np.std(Q2)))
filename = DIRECTORY+'Result/'+reservoirname+'_Q2.csv'
np.savetxt(filename, Q2, delimiter=',')
filename = DIRECTORY+'Result/'+reservoirname+'_PRED.csv'
np.savetxt(filename, PRED, delimiter=',')

# %% markdown
# ### AMN-Wt with experimental training set
# %% codecell
# Create, train and evaluate AMN_Wt models on experimental training set with UB
# Repeat the process with different seeds
# This cell takes several hours to execute

Maxloop, Q2, PRED = 3, [], []

for Nloop in range(Maxloop):
    # What you can change
    seed = Nloop+1
    np.random.seed(seed=seed)
    trainname = 'iML1515_EXP_UB'
    timestep = 4
    # End of What you can change

    # Create model 100% for training 0% for testing
    trainingfile = DIRECTORY+'Dataset_model/'+trainname
    model = Neural_Model(trainingfile = trainingfile,
              objective=['BIOMASS_Ec_iML1515_core_75p37M'],
              model_type = 'AMN_Wt',
              scaler = True,
              timestep = timestep,
              n_hidden = 1, hidden_dim = 500,
              #train_rate = 1.0e-2,
              epochs = 1000, xfold = 10,
              verbose=True)

    # Train and evaluate
    reservoirname = trainname +'_'+model.model_type
    reservoirfile = DIRECTORY+'Reservoir/'+reservoirname
    start_time = time.time()
    reservoir, pred, stats, _ = train_evaluate_model(model, verbose=True)
    delta_time = time.time() - start_time

    # Printing cross-validation results
    printout(reservoirname, stats, model, delta_time)
    r2 = r2_score(model.Y, pred[:,0], multioutput='variance_weighted')
    print('Iter', Nloop, 'Collated Q2', r2)
    Q2.append(r2)
    PRED.append(pred[:,0])

# Save in Result folder
Q2, PRED = np.asarray(Q2), np.asarray(PRED)
print('Averaged Q2 = %.4f (+/- %.4f)' % (np.mean(Q2), np.std(Q2)))
filename = DIRECTORY+'Result/'+reservoirname+'_Q2.csv'
np.savetxt(filename, Q2, delimiter=',')
filename = DIRECTORY+'Result/'+reservoirname+'_PRED.csv'
np.savetxt(filename, PRED, delimiter=',')
