import pandas as pa
import numpy as np
import matplotlib.pyplot as plt

import plotly.express as px

from cobra.io import read_sbml_model

# fluxes_file = "/Users/sea/Projects/QPSI_project/QPSI_Modeling/data/FVA_results/draft_fva_results_flexible_biomass_110323.tsv"
# fluxes_file = "/Users/sea/Projects/QPSI_project/QPSI_Modeling/data/FVA_results/draft_fva_results_flexible_biomass_0.8_maxmax_110623.tsv"

# spc, day, tissue = 'Athaliana', 'ZT1', 'Leaf'
# spc, day, tissue = 'Poplar', '21d', 'Root'


## FVA fluxes
fluxes_file = "/Users/sea/Projects/QPSI_project/Enzyme_Abundance/data/FVA_results/draft_fva_results_dup_fb_fva0.8_110823.tsv"
## FVA thylakoid fluxes
fluxes_file = "/Users/sea/Projects/QPSI_project/Enzyme_Abundance/data/FVA_results/draft_fva_results_dup_fb_fva0.8_051324.tsv"
fluxes_file = "/Users/sea/Projects/AMN/omic_amn_mm/Dataset_input/FVA_Output.tsv"

spc, day, grpr3 = 'athaliana', 'ZT9', 'C24' #'C24' or 'TSU'
# spc, day, grpr3 = 'Sorghum', '21d', 'Leaf'
if 'atha' in spc:
    ## new thylakoid fluxes
    scores_file = f"/Users/sea/Projects/QPSI_project/Enzyme_Abundance/integration_results/secMetResults/{spc}_objective_abundance_Control.tsv"
    ctrl_trmt = 'Control'
    treatments = ['Cold']
    value_col = 'value'
    other_colm = 'genotype'
else:
    msr = 'tmm'
    avogadro = 6.02214076e+23
    ## Relative abundance
    relab_scores_file = f"/Users/sea/Projects/QPSI_project/Enzyme_Abundance/integration_results/reaction_scores_binding_Dec6/{spc}_relab_rxn_scores_{msr}.csv"
    ## Objective abundance
    scores_file = f"/Users/sea/Projects/QPSI_project/Enzyme_Abundance/integration_results/reaction_scores_binding_Dec6/{spc}_objective_abundance_Control.csv"
    treatments = ['FeLim', 'FeEX', 'ZnLim', 'ZnEx']
    ctrl_trmt = 'Control'
    other_colm = 'tissue'



# scores_file = "/Users/sea/Projects/QPSI_project/QPSI_Modeling/integration_results/secMetResults/Atha_objective_abundance_ZT1.csv"
# # scores_file = f"/Users/sea/Projects/QPSI_project/QPSI_Modeling/integration_results/reaction_scores_binding_Dec6/{spc}_tmm_scores.csv"

def get_rxn_ID(row):
    # print(row['rxn_ID'])

    if any(y in row['rxn_ID'] for y in ['_f', '_r', '_i', '_o']):
        # print(row['rxn_ID'].rsplit("_", 1)[0])
        id_only = row['rxn_ID'].rsplit("_", 1)[0]
    else:
        id_only = row['rxn_ID']

    # if id_only in ['rxn08173_y0', 'rxn20595_y0', 'rxn20632_y0']:
    #     return id_only.replace('y0', 'd0')
    # else:
    return id_only


def load_fluxes(fluxes_file, all_rxn=True, verbose=False):
    if all_rxn:
        fva_flux_df = pa.read_csv(fluxes_file, sep='\t')
        # remove reactions added by the
        fva_flux_df = fva_flux_df[fva_flux_df.columns.drop(
                                            list(fva_flux_df.filter(regex="FLEX_*|protein*|avg"))
                                        )]

        fva_flux_df['mean_flux'] = np.abs(fva_flux_df[fva_flux_df.columns[1:]]).mean(axis=1)
        # fva_flux_df['diffs'] = fva_flux_df['mean_flux'] - fva_flux_df['avg']
        print(fva_flux_df.describe())
        # print(abc)

        # print()
        if verbose:
            print(fva_flux_df.columns)
            print(fva_flux_df.head())
        # print(abc)

        fva_flux_df = fva_flux_df[['reaction', 'mean_flux']]
        fva_flux_df.rename(columns={'reaction':'rxn_ID'}, inplace=True)
        print(fva_flux_df.head())
        # print(abc)

    else:
        fva_flux_df = pa.read_csv(fluxes_file, sep='\t', names=["rxn_ID", "reaction", "flux", "max", "min"])#, skiprows=[0, 1])

        # shift values for reactions without equations
        no_rxn = ['rxn_cpd00002_c0', 'rxn_cpd00008_c0', 'rxn_cpd00103_c0']
        fva_flux_df[fva_flux_df['rxn_ID'].isin(no_rxn)]['min'] = \
                            fva_flux_df[fva_flux_df['rxn_ID'].isin(no_rxn)]['max']
        fva_flux_df[fva_flux_df['rxn_ID'].isin(no_rxn)]['max'] = \
                            fva_flux_df[fva_flux_df['rxn_ID'].isin(no_rxn)]['flux']
        fva_flux_df[fva_flux_df['rxn_ID'].isin(no_rxn)]['flux'] = \
                            fva_flux_df[fva_flux_df['rxn_ID'].isin(no_rxn)]['reaction']

        # remove empty rows
        fva_flux_df = fva_flux_df.dropna(subset=["flux", "max", "min"])#, how='all')

        # fix exchange reactions IDs
        fva_flux_df['rxn_ID'] = fva_flux_df['rxn_ID'].apply(lambda x : x.rsplit("cpd", 1)[0] if 'EX_' in x else x)

        # Compute flus mean value
        fva_flux_df['mean_flux'] = (fva_flux_df['min']+fva_flux_df['max'])/2

        if verbose:
            print(fva_flux_df.shape)
            print(fva_flux_df.head())

    return fva_flux_df

def load_scores(ctrl_trmt= 'Control', value_col='value', verbose=False):
    sep = '\t' if '.tsv' in scores_file else ','
    scores_df = pa.read_csv(scores_file, sep = sep)
    # relab_scores_df = pa.read_csv(relab_scores_file)
    relab_scores_df = pa.read_csv(scores_file, sep = sep)
    if verbose: print(scores_df.head())


    if value_col not in  scores_df: scores_df[value_col] = grpr3
    if value_col not in  relab_scores_df: relab_scores_df[value_col] = grpr3

    print(scores_df.head())
    if 'Timestamp' in scores_df:
        scores_df.rename({"Timestamp": 'time_stamp', "Treatment": "treatment"}, inplace=True, axis=1)
        relab_scores_df.rename({"Timestamp": 'time_stamp', "Treatment": "treatment"}, inplace=True, axis=1)



    # Use score DF for K_app computation only
    scores_df[value_col] = scores_df[value_col].astype('float')
    # Use control treatment only
    control = scores_df[(scores_df['treatment'] == ctrl_trmt) &
                            (scores_df[other_colm] == grpr3)]
    # drop all columns except score and IDs
    control = control[[value_col, 'rxn_ID']]
    control = control.groupby('rxn_ID').mean()
    if verbose: print(control.head())

    # # Convert units using avogadro number
    # relab_scores_df[value_col] = relab_scores_df[value_col].astype('float') / avogadro
    # Keep one set of scores: other_colm, time stamp
    relab_scores_df = relab_scores_df[(relab_scores_df[other_colm] == grpr3) &
                                    (relab_scores_df['time_stamp'] == day)]
    # Keep only important columns
    relab_scores_df = relab_scores_df[[value_col, 'treatment', 'rxn_ID']]

    print(relab_scores_df.head())
    print(control.head())
    return relab_scores_df, control

def generate_all_df(treatments, value_col='value', trmt_column='treatment', verbose=False):
    trmt_column = 'treatment'

    # Reaction score for non-duplicated model
    scores_df, control = load_scores(ctrl_trmt= ctrl_trmt)
    # scores_df.to_csv('temp_scores_df_1.csv')

    # Fluxes for duplicated model
    fluxes_dup = load_fluxes(fluxes_file)
    # --> Get matching reaction ID with non-duplicated model
    fluxes_dup['rxn_ID_only'] = fluxes_dup.apply(lambda row: get_rxn_ID(row), axis=1)
    # print(fluxes_dup)
    # fluxes_dup.to_csv('temp_fluxes.csv')

    # replace reaction ID in scores DF by the duplicated scores and suplicate entries for
    # reversible reactions keeping the same RES value for _f and _r reactions
    scr_cols = scores_df.columns
    how = 'left' # 'left' if QPSI otherwise 'right'
    scores_df = scores_df.merge(fluxes_dup[['rxn_ID_only', 'rxn_ID']], left_on='rxn_ID',
                                right_on='rxn_ID_only', how=how, suffixes=('_l', ''))

    # scores_df.to_csv('temp_scores_df2.csv')
    scores_df = scores_df[scr_cols]
    # scores_df.to_csv('temp_scores_df3.csv')


    # Compute K_app
    print(control.head())
    if verbose:
        print("Control DF: \n", control.head())
    all_control = control.merge(fluxes_dup, left_on='rxn_ID', right_on='rxn_ID_only',
                                    how=how, suffixes=('_l', ''))
    print("all_control: ", all_control.head())

    if verbose:
        print('Control after merging reversible reactions: ', all_control.shape)
    all_control = all_control[['rxn_ID', value_col, 'mean_flux']]
    all_control['kapp'] = np.abs(all_control['mean_flux'])/all_control[value_col]
    print("all_control Kapp: ", all_control.head())
    # all_control.loc[all_control.rxn_ID=='rxn00018_d0', 'kapp'] = 1.0


    # all_control['v_Control_obj'] = all_control[value_col] * all_control['kapp']
    if verbose:
        print("----- all_control selected reactions: ")
        print(all_control[all_control['rxn_ID'].isin(['rxn27927_d0_f', 'rxn27927_d0_r', 'rxn00018_d0'])])
    # print(abc)

    # all_control.drop(columns=[value_col], axis=1, inplace=True)
    all_control.rename(columns={value_col:'ctrl_score'}, inplace=True)
    if verbose:
        print("----- K_app describe: ")
        print(all_control['kapp'].describe())

    # Compute V_bv for every treatment
    if verbose:  print("Treatments: ", treatments)
    scores_df['rxn_ID'] = scores_df['rxn_ID'].astype(str)
    all_control['rxn_ID'] = all_control['rxn_ID'].astype(str)
    treatments = [ctrl_trmt] + treatments
    for trmt in treatments:
        if verbose:  print("   --> Processing: ", trmt)
        temp = scores_df[scores_df[trmt_column] == trmt][[value_col, 'rxn_ID']].copy()
        # print(abc)
        # print("merging temp ", temp.shape, ' to all_control ', all_control.shape)
        # print("rxn_ID diffs: ", set(temp['rxn_ID'].unique())-set(all_control['rxn_ID'].unique()))
        # print(set(temp['rxn_ID'].unique()))
        # print(set(all_control['rxn_ID'].unique()))
        all_control = all_control.merge(temp, on='rxn_ID', how='left')

        if verbose:
            print(all_control.columns)
            print("After mergin temp: ", all_control.shape)
            print("computing V_bf ...")
        all_control['v_'+trmt] = all_control[value_col] * all_control['kapp']

        if verbose:
            print("renaming RES column ...")
        all_control.rename(columns={value_col:trmt+'_score'}, inplace=True)
        # print(abc)

    print(all_control[all_control['rxn_ID'].isin(['rxn00018_d0'])]) #'rxn27927_d0_f', 'rxn27927_d0_r',

    # plot V_bf to compare
    ls = ['v_'+x for x in treatments]
    ls = ['rxn_ID'] + ls
    all_control[ls].set_index('rxn_ID').plot()
    # plt.show()

    all_control.replace([np.inf, -np.inf], 0, inplace=True)
    all_control = all_control.fillna(0)

    # Write to file
    # if 'tissue' in all_control.columns:
    #     name = [spc, 'thylakoid', tissue, day, 'Vbf']
    # else:
    name = [spc, 'thylakoid', grpr3, day, 'Vbf']

    if 'relab' in scores_file:
        name = name + ['relab']

    all_control[ls].to_csv("/Users/sea/Projects/AMN/omic_amn_mm/Dataset_input/"+"_".join(name)+"_maxCtrl.csv", index=False)

    all_control.to_csv("/Users/sea/Projects/AMN/omic_amn_mm/Dataset_input/"+"_".join(name+['kapp'])+"_maxCtrl.csv", index=False)
    return all_control[ls]

def compare_predictions():
    predictions_file = "/Users/sea/Projects/AMN/amn_release-main/Result/tempV_poplar_leaf_2d.tsv"
    vbf_file = "/Users/sea/Projects/AMN/amn_release-main/Result/tempY_poplar_leaf_2d.csv"

    V_df = pa.read_csv(predictions_file, sep='\t')
    # print(V_df.head())
    V_df.set_index('treatment', inplace=True)
    V_df = V_df.T

    print(V_df.head())

    Y_df = pa.read_csv(vbf_file)
    Y_df.set_index('treatment', inplace=True)
    Y_df = Y_df.T
    print(Y_df.head())

def all_kapp(value_col):
    tmm_scores_file = "/Users/sea/Projects/QPSI_project/QPSI_Modeling/integration_results/reaction_scores_binding_Dec6/Poplar_tmm_scores.csv"
    scores_df = pa.read_csv(tmm_scores_file)
    scores_df.rename(columns={'value':'score'}, inplace=True)
    value_col = 'score'

    scores_df = scores_df[(scores_df['treatment'] == 'Control') & (scores_df[other_colm] == 'Leaf')]
    scores_df.loc[scores_df[value_col] == 0, value_col] = 0.0001


    fluxes_dup = load_fluxes(fluxes_file)
    # fluxes_dup.loc[fluxes_dup.mean_flux == 0, 'mean_flux'] = 0.0001
    fluxes_dup['rxn_ID_only'] = fluxes_dup.apply(lambda row: get_rxn_ID(row), axis=1)
    print(fluxes_dup.head())

    scr_cols = [other_colm, 'treatment', 'time_stamp', 'rxn_ID', value_col, 'mean_flux']
    scores_df = scores_df.merge(fluxes_dup, left_on='rxn_ID',
                                right_on='rxn_ID_only', how='left', suffixes=('_l', ''))

    scores_df = scores_df[scr_cols]
    scores_df['kapp'] = np.abs(scores_df['mean_flux'])/scores_df[value_col]

    fig = px.scatter(scores_df, x='rxn_ID', y="kapp", color='time_stamp', symbol='time_stamp')
    # fig.show()
    groups = scores_df.groupby(['time_stamp'])
    # computes group-wise mean/std,
    # then auto broadcasts to size of group chunk
    min_score = groups[value_col].transform("min")
    max_score = groups[value_col].transform("max")
    min_flux = groups['mean_flux'].transform("min")
    max_flux = groups['mean_flux'].transform("max")

    scores_df['norm_'+value_col] = (scores_df[value_col] - min_score) / (max_score - min_score)
    scores_df['norm_mean_flux'] = (scores_df['mean_flux'] - min_flux) / (max_flux - min_flux)
    fig = px.scatter(scores_df, x='norm_score', y="norm_mean_flux", facet_row='time_stamp',
    width=600, height=1100)
    # fig.show()
    # , size="counts", title=title, facet_col='treatment', category_orders={"ds": dss, 'subsystems':sys_class_ls}, symbol_sequence=['circle', 'circle'],
    # color_discrete_map=color_map, facet_col_spacing=0.03, height=1100, width=1350, facet_row=other_colm


    scores_df_long = scores_df.melt(id_vars=['time_stamp', 'rxn_ID'], value_vars=[value_col, 'mean_flux', 'kapp']) #'value', 'mean_flux', 'kapp'
    print(scores_df_long.head())
    # print(abc)

    # fig = px.scatter(scores_df_long, x='rxn_ID', y="value", color='variable', symbol='variable',
    # facet_row='time_stamp')
    fig = px.box(scores_df_long, x='time_stamp', y="value", color='variable', width=800, height=800)
    # , size="counts", title=title, facet_col='treatment', category_orders={"ds": dss, 'subsystems':sys_class_ls}, symbol_sequence=['circle', 'circle'],
    # color_discrete_map=color_map, facet_col_spacing=0.03, height=1100, width=1350, facet_row=other_colm
    fig.show()

    print(scores_df.groupby(['time_stamp'])['kapp'].describe())


if __name__ == '__main__':


    # all_kapp(value_col)

    compute = True
    if compute:
        all_control = generate_all_df(treatments, value_col='value', trmt_column='treatment')
        print(all_control.tail(20))
        print(all_control.describe())
        print(abc)
        all_control= all_control.set_index('rxn_ID')
        print(list(all_control.index)[:5])
        if 'rxn00001_d0_f' in all_control.index:
            print("------"*3)
            print(all_control.at['rxn00001_d0_f', 'v_FeLim'])


    compare = False
    if compare:
        compare_predictions()

        co_model = read_sbml_model("/Users/sea/Projects/AMN/omic_amn/Dataset_input/sbicolor_plastidial_model_duplicated.xml")
        co2 = "EX_cpd00011_e0_i"
        vals = {"FeEx":328.0062545, "Control/ZnLim":606.3606508, "ZnEx":220.5894569, "FeLim":72.83864248}
        for trmt, val in vals.items():
            co2_rxn = co_model.reactions.get_by_id(co2)
            co2_rxn.lower_bound = 0
            co2_rxn.upper_bound = val/10

            solution = co_model.optimize()
            print(trmt, " with CO2 set to ", co2_rxn.upper_bound, " -> ",solution, " (", solution.fluxes['EX_cpd00011_e0_i'],")")
