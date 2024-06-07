import warnings
warnings.simplefilter(action='ignore', category=Warning)
import os
import sys

# In this case the local root of the repo is our working directory
DIRECTORY = './'
font = 'arial'

# printing the working directory files. One can check you see the same folders and files as in the git webpage.
print(os.listdir(DIRECTORY))

from Library.Duplicate_Model import *

# %% markdown
# # Duplicate two-sided reactions in a SBML model
# %% markdown
# A requirement for neural computations with metabolic networks using AMNs is the positivity of all fluxes.
#
# This notebook shows the steps to transforms a SBML model into an *AMN-compatible* SBML model where all exchange reactions are possible in both ways, and reversible internal reactions are duplicated for each way. Added to that, the reactions encoded as backward are recoded in the other way, the forward way. In this transformed model, we also add a suffix "i", for inflowing, and "o" for outflowing reactions. It is further described what we consider "inflow" and "outflow".
# %% markdown
# ## Download and inspect the model
# %% markdown
# Run this cell for E. coli core model
# %% codecell
model_path = "Dataset_input/e_coli_core.xml"
url = "http://bigg.ucsd.edu/static/models/e_coli_core.xml"
response = requests.get(url)
open(model_path, "wb").write(response.content)
model = cobra.io.read_sbml_model(model_path)
# %% markdown
# Run this cell for iML1515 model
# %% codecell
model_path = "Dataset_input/iML1515.xml"
url = "http://bigg.ucsd.edu/static/models/iML1515.xml"
response = requests.get(url)
open(model_path, "wb").write(response.content)
model = cobra.io.read_sbml_model(model_path)

model_path = "Dataset_input/athaliana_plastidial_model.xml"
model_path = "Dataset_input/sbicolor_plastidial_model.xml"
model_path = "Dataset_input/Plastid_Sandbox_model.xml"
model_path = "Dataset_input/athaliana_plastidial_thylakoid_051024.xml"
model_path = "Dataset_input/athaliana_plastidial_thylakoid_052324.xml"
# model_path = "Dataset_input/sbicolor_plastidial_model_noOrganellar.xml"
# model_path = "/Users/sea/Projects/QPSI_project/QPSI_Modeling/data/metabolic_models/plastidial_models/ortho_nov29_models/sbicolor_plastidial_model.xml"
model = cobra.io.read_sbml_model(model_path)
# %% markdown
# First, we can get some basic informations on the model.
#
# model.boundary gets all reactions that introduce or remove matter in the system (inflows, outflows)
#
# model.medium lists all reactions of model.boundary that are reversible (irreversible are outflows/sinks)
#
# model.objective is the expression to be optimized by default
# %% codecell
print(model.boundary)
print(model.medium)
print(model.objective)
solution = model.optimize()
print(solution)
# print(abc)
# %% markdown
# ## Checking the objective of the model
# Optional step for exploring how the biomass is encoded. In some models, several biomass reactions are available and one has to make sure using the right one.
# %% codecell
for reac in model.reactions:
     if "biomass" in reac.id or "BIOMASS" in reac.id:
        print(reac)
# %% markdown
# ## Screen outflowing and inflowing reactions
# %% markdown
# For each reaction that has different compartments in reactants and products (we call it "transfer reactions"), we annotate the reaction with a suffix "i" for inflowing (None --> e --> p --> c --> m) and "o" for outflowing (m --> c --> p --> e --> None). When the compartment-changing of metabolites is balanced, or not present, we use different suffix: "for" as in forward, designating the default way of the reaction (positive flux) and "rev" as in reverse, designating the opposite way (negative flux). We reverse the products and reactants so that the same reactions happen, ensuring that we have a positive flux for all reactions.
#
# To do so, we first define a dictionary for mapping which (reactant compartment, product compartment) pair is matching which suffix: "io_dict".
#
# Some reactions are problematic because they are showing both inflow and outflow simultaneously. To tackle this, we ignore the small molecules listed in "unsignificant_mols".
#
# For each reaction, we count the number of "inflowing" and "outflowing" pairs, and the way the reaction happens (forward, backward, reversible or other).
# %% codecell
# "i" for inflowing (None --> e --> c --> d) and "o" for outflowing (d --> c --> e --> None)
io_dict = {"_i": [(None, "e0"), (None, "c0"), ("e0","c0"), ("c0", "d0"), ("e0", "d0")],
           "_o": [("c0", None), ("e0", None), ("d0", "e0"), ("d0", "c0"), ("c0", "e0")]}

unsignificant_mols = ["h_p", "h_c", "pi_c", "pi_p", "adp_c", "h2o_c", "atp_c"]

# Will print a dictionary counting the reactions in reversible, forward, backward
reac_id_to_io_count_and_way = screen_out_in(model, io_dict, unsignificant_mols)

# To uncomment in order to see the structure of the screening dictionary
# print(reac_id_to_io_count_and_way)
# %% markdown
# ## Duplicate reactions
# %% markdown
# Here we make a copy of the model,  named 'new_model', then perform the duplication of appropriate reactions.
#
# We duplicate all exchange reactions (excepted sink reactions) and reversible internal reactions (not unidirectional ones). We get the suffix '_i' for compartment changing reaction from the exterior to the cytoplasm, and '_o' for the other way. We also use the suffix "_f" and "_r" for forward and reverse duplicated reactions that do not change compartment, or show equal compartment exchanges.
# %% codecell
new_model = duplicate_model(model, reac_id_to_io_count_and_way)



# # ## Lower bounds check-up
# # Here we simply check which reactions have a non-zero lower bound
# for reac in new_model.reactions:
#     if reac.lower_bound != 0:
#         print('reaction with non-zero lower bound:', reac.id, reac.bounds)
# for el in new_model.medium:
#     if new_model.reactions.get_by_id(el).lower_bound != 0:
#         print('medium reaction with non-zero lower bound:',el)



# The new model with duplicated exchange reactions is badly handled by COBRA for the medium object generation: all inflowing reactions are inside the medium object.
# There is a problem to be corrected: all upper bounds are set at 1000 by default. If we change this upper bound to 0 when we duplicate reactions, we get an empty medium object.
# The solution is to correct the medium of the duplicated model, all non-default medium exchange reactions put at 1e-300, so they are at a value very close to 0 but still appear in the medium object.
# %% codecell
default_med = model.medium
new_med = new_model.medium
correct_med =  correct_duplicated_med(default_med, new_med)
new_model.medium = correct_med
print(new_model.medium)
# %% markdown
# ## Medium check-up (default model V.S. duplicated-reaction model)
# %% markdown
# Here we compare the results with randomized medium objects for both models, reporting the absolute difference between the two.
# %% codecell
for i in range(10):
    # print('_'*50)
    s, new_s = change_medium(model, new_model, i*3)
    if s != None and new_s != None:
        print(s, new_s, "diff = ", abs(s-new_s))
    elif s != None:
        print("infeasible duplicated medium")
    elif new_s != None:
        print("infeasible default medium")
    elif s == None and new_s == None:
        print("Both medium are impossible")
# %% markdown
# ## Saving the duplicated-reactions model
# %% codecell
new_model.repair() # rebuild indices and pointers in the model if necessary
# %% codecell
new_name = model_path[:-4] + "_duplicated" + model_path[-4:]

## restrict media
# remove_med = ["EX_cpd00067_e0_i", "EX_cpd00007_e0_i", "EX_cpd00009_e0_i", "EX_cpd00008_e0_i", "EX_cpd00013_e0_o", "EX_cpd00048_e0_o", "EX_cpd11632_e0_o", "EX_cpd00011_e0_o", "EX_cpd00001_e0_o", "EX_cpd00002_e0_o"]
remove_med = ["EX_cpd00067_e0_i", "EX_cpd00007_e0_i", "EX_cpd00008_e0_i", "EX_cpd11632_e0_o", "EX_cpd00011_e0_o", "EX_cpd00001_e0_o", "EX_cpd00002_e0_o", 'EX_cpd00005_e0_o', 'EX_cpd00006_e0_i']
# EX_cpd00009_e0_i
solution = new_model.optimize()
print(solution)
for med in remove_med:
    print(med)
    new_model.remove_reactions(med)
    solution = new_model.optimize()
    print(solution)
    print("After restricting media, model has ", len(new_model.reactions), " reactions")

# print("After restricting media, model has ", len(new_model.reactions), " reactions")

cobra.io.write_sbml_model(new_model, new_name)
solution = model.optimize()
print(solution)

solution = new_model.optimize()
print(solution)
print("Duplicated model's location: " + new_name)
# print("new model media ", new_model.medium)
