import warnings
warnings.simplefilter(action='ignore', category=Warning)
import os
import sys

from modelseedpy import FlexibleBiomassPkg
from cobra.flux_analysis import flux_variability_analysis as fva

from cobrakbase.core.kbase_object_factory import KBaseObjectFactory
KBOF = KBaseObjectFactory()

model_path = "Dataset_input/sbicolor_3.1.1_plastid_Thylakoid_Reconstruction_061124.json"
model_path = "/Users/sea/Projects/QPSI_project/Enzyme_Abundance/data/metabolic_models/plastidial_models/ortho_jun20_models/sbicolor_3.1.1_plastid_Thylakoid_Reconstruction_ComplexFix_070224_noADP.json"
media_path = "Dataset_input/PlantPlastidialAutotrophicMedia_noATP.json"

# model_path = "/Users/sea/Projects/QPSI_project/Enzyme_Abundance/data/metabolic_models/fullmodels_media/ortho_jun20_models/Athaliana_Thylakoid_Reconstruction_ComplexFix_070224.json"
# media_path = "/Users/sea/Projects/QPSI_project/Enzyme_Abundance/data/metabolic_models/fullmodels_media/PlantAutotrophicMedia.json"


model = KBOF.build_object_from_file(model_path, "KBaseFBA.FBAModel")
co_media = KBOF.build_object_from_file(media_path, "KBaseBiochem.Media")
model.medium = co_media

# In this case the local root of the repo is our working directory
DIRECTORY = './'
font = 'arial'

# printing the working directory files. One can check you see the same folders and files as in the git webpage.
print(os.listdir(DIRECTORY))

from Library.Duplicate_Model import *


# model_path = "Dataset_input/athaliana_plastidial_thylakoid_051024.xml"
# model = cobra.io.read_sbml_model(model_path)


# %% codecell
print(model.boundary)
print(model.medium)
print(model.objective)
solution = model.optimize()
print(solution)

from cobra.io import write_sbml_model
write_sbml_model(model, model_path.replace('json', 'xml'))
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
# Here we make a copy of the model,  named 'dup_co_model', then perform the duplication of appropriate reactions.
#
# We duplicate all exchange reactions (excepted sink reactions) and reversible internal reactions (not unidirectional ones). We get the suffix '_i' for compartment changing reaction from the exterior to the cytoplasm, and '_o' for the other way. We also use the suffix "_f" and "_r" for forward and reverse duplicated reactions that do not change compartment, or show equal compartment exchanges.
# %% codecell
dup_co_model = duplicate_model(model, reac_id_to_io_count_and_way)



# # ## Lower bounds check-up
# # Here we simply check which reactions have a non-zero lower bound
# for reac in dup_co_model.reactions:
#     if reac.lower_bound != 0:
#         print('reaction with non-zero lower bound:', reac.id, reac.bounds)
# for el in dup_co_model.medium:
#     if dup_co_model.reactions.get_by_id(el).lower_bound != 0:
#         print('medium reaction with non-zero lower bound:',el)



# The new model with duplicated exchange reactions is badly handled by COBRA for the medium object generation: all inflowing reactions are inside the medium object.
# There is a problem to be corrected: all upper bounds are set at 1000 by default. If we change this upper bound to 0 when we duplicate reactions, we get an empty medium object.
# The solution is to correct the medium of the duplicated model, all non-default medium exchange reactions put at 1e-300, so they are at a value very close to 0 but still appear in the medium object.
# %% codecell
default_med = model.medium
new_med = dup_co_model.medium
correct_med =  correct_duplicated_med(default_med, new_med)
dup_co_model.medium = correct_med
print(dup_co_model.medium)
# %% markdown
# ## Medium check-up (default model V.S. duplicated-reaction model)
# %% markdown
# Here we compare the results with randomized medium objects for both models, reporting the absolute difference between the two.
# %% codecell
for i in range(10):
    # print('_'*50)
    s, new_s = change_medium(model, dup_co_model, i*3)
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
dup_co_model.repair() # rebuild indices and pointers in the model if necessary
# %% codecell
new_name = model_path[:-5] + "_duplicated" + model_path[-5:]
new_name = new_name.replace('json', 'xml')

## restrict media
# remove_med = ["EX_cpd00067_e0_i", "EX_cpd00007_e0_i", "EX_cpd00009_e0_i", "EX_cpd00008_e0_i", "EX_cpd00013_e0_o", "EX_cpd00048_e0_o", "EX_cpd11632_e0_o", "EX_cpd00011_e0_o", "EX_cpd00001_e0_o", "EX_cpd00002_e0_o"]
remove_med = ["EX_cpd00067_e0_i", "EX_cpd00007_e0_i", "EX_cpd00008_e0_i", "EX_cpd00008_e0_o", "EX_cpd11632_e0_o", "EX_cpd00011_e0_o", "EX_cpd00001_e0_o", "EX_cpd00002_e0_o", 'EX_cpd00005_e0_o', 'EX_cpd00006_e0_i']
# EX_cpd00009_e0_i
solution = dup_co_model.optimize()
print(solution)
for med in remove_med:
    print(med)
    dup_co_model.remove_reactions(med)
    solution = dup_co_model.optimize()
    print(solution)
    print("After restricting media, model has ", len(dup_co_model.reactions), " reactions")

# print("After restricting media, model has ", len(dup_co_model.reactions), " reactions")

cobra.io.write_sbml_model(dup_co_model, new_name)
solution = model.optimize()
print(solution)

solution = dup_co_model.optimize()
print(solution)
print("Duplicated model's location: " + new_name)

# print(abc)
print("Running triple FVA ...")
dup_co_model.reactions.get_by_id("bio1_biomass").lower_bound=0.5
fva_rxns_explore = list(dup_co_model.reactions)

flux_dict = dict()
fva_dict = dict()
for reaction in dup_co_model.reactions:
	if('bio1' in reaction.id):
		continue

	reaction.objective_coefficient=1.0

	flux_df = dup_co_model.optimize().fluxes
	flux_dict[reaction.id]=flux_df

	fva_df = fva(dup_co_model, fva_rxns_explore, processes=1, fraction_of_optimum=0.8)
	fva_dict[reaction.id]=fva_df

	reaction.objective_coefficient=0.0

	#if(reaction.id == dup_co_model.reactions[10].id):
	#	break

ofh = open('Dataset_input/FVA_Output.tsv','w')
ofh.write("\t".join(["reaction"]+list(fva_dict.keys())+["avg"])+'\n')
for reaction in dup_co_model.reactions:
	if('bio1' in reaction.id or reaction.id == 'protein_flex'):
		continue

	max_sum=0.0
	count=0.0
	max_list=list()
	for max_flux_rxn in fva_dict:
		if(max_flux_rxn == reaction.id):
			continue

		rxn_fva = fva_dict[max_flux_rxn].loc[fva_dict[max_flux_rxn].index == reaction.id]
		max = rxn_fva.iloc[0]['minimum']

		max_list.append("{:.6f}".format(max))
		max_sum+=max
		count+=1.0

	mean_max = "0.0"
	if(count>0):
		mean_max = "{:.6f}".format(max_sum/count)
	ofh.write("\t".join([reaction.id]+max_list+[mean_max])+'\n')

# print("new model media ", dup_co_model.medium)
