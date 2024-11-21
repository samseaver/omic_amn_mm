#!/usr/bin/env python
import pandas as pa
import urllib3 # must have version >2
print(urllib3.__version__)

import sys, os
from pathlib import Path
project_root = str(Path(__file__).resolve()).split('src')[0]
sys.path.append(project_root)

# from https://github.com/ModelSEED/ModelSEEDDatabase/blob/master/Libs/Python/BiochemPy/Compounds.py
ModelSEEDDB_path = '/Users/seaver/Projects/ModelSEEDDatabase/Libs/Python/'
ModelSEEDDB_path = '/Users/sea/packages/ModelSEED_KBase/ModelSEEDDatabase/Libs/Python/'
# ModelSEEDDB_path = '/Users/selalaoui/packages/ModelSEED_KBase/ModelSEEDDatabase/Libs/Python/'
ModelSEEDDB_path = '/Users/selalaoui/packages/ModelSEEDDatabase/Libs/Python/'
ModelSEEDDB_path = '/Users/selalaoui/packages/ModelSEEDDatabase/Libs/Python/'
sys.path.append(ModelSEEDDB_path)
from BiochemPy import Compounds
from cobra.io import read_sbml_model
import cobra as co

from urllib.request import urlopen
import json
# Load data from PlantSEED_Roles.json
# PS_url  = "https://raw.githubusercontent.com/ModelSEED/PlantSEED/"
# PS_tag  = "8cf60046e4af68912f7a7d3eeff16880a07f56bd"
# PS_json = "/Data/PlantSEED_v3/PlantSEED_Roles.json"
#
# print("Loading PS Roles ...")
# # try:
# data = json.load(urlopen(PS_url+PS_tag+PS_json))
# except URLError:
roles_file = "/Users/selalaoui/Projects/QPSI_project/Enzyme_Abundance_all/data/metabolic_models/PlantSEED_Roles.json"
data = None
with open(roles_file, 'r') as f:
    data = json.loads(f.read())

# http = urllib3.PoolManager()
# cpd_url = 'https://raw.githubusercontent.com/ModelSEED/ModelSEEDDatabase/dev/Biochemistry/compound_'
# cpd_db = http.request('GET',cpd_url+'00.json').json()

cpds_file = "/Users/selalaoui/Projects/QPSI_project/Enzyme_Abundance_all/data/metabolic_models/compound_00.json"
data = None
with open(cpds_file, 'r') as f:
    cpd_db = json.loads(f.read())
cpd_db= {cpd['id']:cpd for cpd in cpd_db}


class Calculate_Carbon_Flux:
    def __init__(self):
        self.subsys_dict = dict()
        # self.readRoles()

        self.compound_helper = Compounds()

    # Parse data in PlantSEED_Roles.json and generate the
    # dictionary of pathways to reactions
    def readRoles(self):
        for item in data:
            reactions = set(item["reactions"])
            for subsys in item["subsystems"]:
                subsys = subsys.replace("_", " ")
                if subsys in self.subsys_dict:
                    new_set = self.subsys_dict[subsys]
                    new_set.update(reactions)
                    self.subsys_dict[subsys] = new_set
                else:
                    self.subsys_dict[subsys] = reactions

    def med_bio_flux(self, flux_df, co_model, molecule='C', verbose=False):
        if 'rxn_ID' in flux_df.columns: flux_df = flux_df.set_index('rxn_ID')
        flux_io_df = flux_df.filter(regex=("EX_*|bio*"), axis = 0)
        print(flux_io_df)

        for rxn_id in list(flux_io_df.index):
            if verbose: print("----------------------- processing ", rxn_id)
            rxn = None
            if rxn_id in co_model.reactions:
                rxn = co_model.reactions.get_by_id(rxn_id)
            else:
                cpd = rxn_id.split('_')[1]
                coeff = 1 if '_i' in rxn_id else -1
            # print(rxn)
            t_carbons = 0
            if rxn:
                for cpd, coeff in rxn.metabolites.items():
                    formula = cpd_db[cpd.id.split('_')[0]]['formula'] if not cpd.formula else cpd.formula
                    # if verbose: print(cpd, coeff, cpd.formula)
                    if molecule.lower() in formula.lower():
                        t_carbons += (self.compound_helper.parseFormula(formula)[molecule] * coeff)
            else:
                print("no rxn for ", rxn_id)


            # if 'bio' in rxn_id:
            #     print(t_carbons)
            #     print(abc)

            flux_io_df.loc[rxn_id] = flux_io_df.loc[rxn_id]*t_carbons #*coeff
            if verbose: print("---------------------------------- ")

        # df.loc['total'] = df.sum(numeric_only=True, axis=0)
        print("Carbon flux sum uptake - sum release:")
        print(flux_io_df)
        print(flux_io_df.sum(axis=0))

        if verbose:
            print(flux_io_df)
            # compute biomass carbon flow:
            # bio_id = "bio1_biomass"
            obj_id = str(co_model.objective.expression)
            obj_id = obj_id.split()[0].split('*')[1]
            print("Biomass carbon flux: ", flux_io_df.loc[obj_id])

    def model_rxns_flux(self, flux_df, co_model, molecule='C'):
        flux_df.loc['total_in']  =  0
        flux_df.loc['total_out'] =  0

        for rxn_id in list(flux_df.index):
            print(rxn_id)
            if ('EX' in rxn_id) or ('bio' in rxn_id) or ('rxn' not in rxn_id):
                # print("Skipping ", rxn_id)
                continue
            t_carbons = 0
            rxn = co_model.reactions.get_by_id(rxn_id)
            for rct, coeff in rxn.metabolites.items():
                formula = cpd_db[rct.id.split('_')[0]]['formula'] if not rct.formula else rct.formula
                if molecule.lower() in formula.lower():
                    t_carbons += self.compound_helper.parseFormula(formula)[molecule] * coeff

            flux_df.loc[rxn_id] = flux_df.loc[rxn_id]*t_carbons


        # df.loc['total'] = df.sum(numeric_only=True, axis=0)
        print(flux_df.shape)
        filter = flux_df.index.str.contains("EX_*|bio*")
        flux_df = flux_df[~filter]
        print(flux_df.shape)

        print("Carbon flux sum uptake - sum release:")
        print(flux_df.sum(axis=0))

        # print(flux_df.loc['total_in'])
        # print("Carbon flux sum uptake - sum release:")
        # print(flux_df.sum(axis=0))

    def metabolites_balance(self, flux_df, co_model):
        # Compute metabolite fluxes
        stoichio_df = co.util.array.create_stoichiometric_matrix(co_model, array_type="DataFrame")
        stoichio_df.sort_index(axis=1, inplace=True)
        flux_df = flux_df.set_index('rxn_ID')
        flux_df.sort_index(inplace=True)
        print("stoichio_df ", stoichio_df.shape)
        print("flux_df ", flux_df.shape)
        cpd_flux = stoichio_df.dot(flux_df)
        cpd_flux.to_csv("cpd_flux_Vbf_ReLU.tsv", sep='\t')

    def metabolite_molecule(self, co_model, molecule):
        met_mol_dict = {}
        for cpd in co_model.metabolites:
            formula = cpd_db[cpd.id.split('_')[0]]['formula'] if not cpd.formula else cpd.formula
            try:
                met_mol_dict[cpd.id] = self.compound_helper.parseFormula(formula)[molecule]
            except KeyError: 
                met_mol_dict[cpd.id] = 0

        return met_mol_dict


if __name__ == '__main__':
    ccf_obj = Calculate_Carbon_Flux()

    # http = urllib3.PoolManager()
    # cpd_url = 'https://raw.githubusercontent.com/ModelSEED/ModelSEEDDatabase/dev/Biochemistry/compound_'
    #
    # cpd_db = http.request('GET',cpd_url+'00.json').json()
    # cpd_db= {cpd['id']:cpd for cpd in cpd_db}
    #
    # for cpd in cpd_db:
    #     print(cpd['id'],cpd['formula'],compound_helper.parseFormula(cpd['formula']))

    ## Project files
    spc_name = 'sbicolor'
    metModel_path = os.path.join(project_root, "data", "metabolic_models", "plastidial_models",
                        "ortho_nov29_models", f"{spc_name}_plastidial_model.xml")
    metModel_path = "/Users/sea/Projects/AMN/omic_amn/Dataset_input/sbicolor_plastidial_model_duplicated.xml"
    metModel_thylakoid_path = "/Users/sea/Projects/AMN/omic_amn_mm/Dataset_input/athaliana_plastidial_thylakoid_051024_duplicated.xml"

    co_model = read_sbml_model(metModel_thylakoid_path)

    #
    # # Compute metabolite fluxes
    # ccf_obj.metabolites_balance(flux_df, co_model)
    #
    # ## Compute carbon flux
    # ccf_obj.med_bio_flux(flux_df, co_model)
    # ccf_obj.model_rxns_flux(flux_df, co_model)

    ## Generate FBA results
    solution = co_model.optimize()
    print(solution)
    # pint(abc)
    solution = solution.fluxes.to_frame()
    solution.to_csv("FBA.csv")
    solution[solution.abs() < 10**-6] = 0
    solution = solution.rename_axis('rxn_ID').reset_index()

    co_model = read_sbml_model(metModel_path)
    ## Read ML flux results
    fluxes_path = "/Users/sea/Projects/AMN/omic_amn_mm/Result/21d_sbicolor_thylakoid_V_rxn_fixed.tsv"
    fluxes_path = "/Users/sea/Projects/AMN/omic_amn_mm/Result/ZT9_thylakoid_V_rxn_fixed.tsv"
    # ZT9_thylakoid_V_rxn
    sep = '\t' if 'tsv' in fluxes_path else ','
    flux_df = pa.read_csv(fluxes_path, sep=sep)
    flux_df.set_index('rxn_ID', inplace=True)

    ## Append FBA fluxes to ML fluxes DF
    flux_df = flux_df.merge(solution, how='left', on='rxn_ID')
    flux_df.set_index('rxn_ID', inplace=True)

    # Compute metabolite fluxes
    ccf_obj.metabolites_balance(flux_df, co_model)

    ## Compute carbon flux
    ccf_obj.med_bio_flux(flux_df, co_model)
    ccf_obj.model_rxns_flux(flux_df, co_model)
