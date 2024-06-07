import pandas as pa
import os
import numpy as np

import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import plotly.io as pio
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt


from Calculate_Carbon_Flux import Calculate_Carbon_Flux
from cobra.io import read_sbml_model


def set_boundaries(co_model, vbf_df, colm_name='Vbf'):
    found = 0

    for rxn in co_model.reactions:
        if (rxn.id in vbf_df.index):
            found+=1
            if vbf_df.loc[rxn.id, colm_name] == 0:
                continue

            rxn.upper_bound = vbf_df.loc[rxn.id, colm_name]


    # co_model.reactions.get_by_id("bio1_biomass").lower_bound = 1
    return co_model

def get_rxn_ID(row):
    if any(y in row['rxn_ID'] for y in ['_f', '_r', '_i', '_o']):
        # print(row['rxn_ID'].rsplit("_", 1)[0])
        id_only = row['rxn_ID'].rsplit("_", 1)[0]
    else:
        id_only = row['rxn_ID']

    return id_only

def consolidate(grp_obj, value_cols=[]):
    grp_obj[value_cols] = 2*grp_obj[value_cols] - grp_obj[value_cols].sum() #+grp_obj
    return grp_obj

def process_predictions(pred_df):
    pred_df['rxn_ID_only'] = pred_df.apply(lambda row: get_rxn_ID(row), axis=1)

    value_cols = list(pred_df.columns)
    value_cols.remove('rxn_ID')
    value_cols.remove('rxn_ID_only')

    # Consilidate the fluxes of reversible reactions:
    # if V_r > V_f:
    #     V_r = V_r - V_f
    #     V_f = 0
    # else:
    #     V_f = V_f - V_r
    #     V_r = 0
    pred_df = pred_df.groupby('rxn_ID_only', as_index=False).apply(lambda grp: consolidate(grp, value_cols))

    pred_df[value_cols] = pred_df[value_cols].clip(lower=0)
    return  pred_df

def plot_results(all_df, msrs, spc, time_stamp, tissue=''):
    # fig = px.line(all_df,
    #                 x='rxn_ID',
    #                 y='FBA',
    #                 color="treatment",
    #                 title='FBA',
    #                 height=1000,
    #                 width=1000)
    # fig.show()
    #
    # fig = px.line(all_df,
    #                 x='rxn_ID',
    #                 y='Pred',
    #                 color="treatment",
    #                 title='Pred',
    #                 height=1000,
    #                 width=1000)
    # fig.show()
    nbin = 100
    for msr in msrs:
        all_df.sort_values(msr, inplace=True)
        msr_df = all_df.copy()
        if msr in ['Vbf']:
            msr_df = msr_df[msr_df[msr] <= 20]

        if msr in ['Pred', 'FBA']:
            msr_df = msr_df[msr_df[msr] <= 0.15]

        if msr in ['FBA']:
            msr_df = msr_df[msr_df[msr] <= 0.06]

        fig = px.histogram(msr_df,
                            x=msr,
                            color="treatment",
                            title = f"{msr} -- {spc} {tissue} {time_stamp} histogram",
                            marginal="rug",
                            nbins=nbin,
                            height=900,
                            width=900
                            )
        # Overlay both histograms
        fig.update_layout(barmode='overlay')
        # Reduce opacity to see both histograms
        fig.update_traces(opacity=0.75)
        fig.show()



## Original code minus payee logic and unnecessary index check.
def per_Reaction_diffs(df, clmns):
    diffs = pa.DataFrame(columns=clmns)
    ln = len(df.index)**2

    for r1 in df.index:
        for r2 in df.index:
            ln -= 1
            if ln % 10000 == 0 : print(ln, ' ...')

            if r1 == r2: continue

            diffs.loc[r1+'_'+r2, clmns] = np.abs(df[clmns].loc[r2] - \
                                                 df[clmns].loc[r1])

    return diffs # for combine portion of split-apply-combine

def calculate_c_flux(flux_df, co_model):
    ccf_obj = Calculate_Carbon_Flux()
    print(flux_df.head())
    # print(abc)

    # # Compute metabolite fluxes
    # ccf_obj.metabolites_balance(flux_df, co_model)

    ## Compute carbon flux
    ccf_obj.med_bio_flux(flux_df, co_model)
    ccf_obj.model_rxns_flux(flux_df, co_model)

def plot_hist_diffs(diff_df, msrs, treatments, spc, tissue, time_stamp):
    nbin = 100
    diff_df = diff_df.reset_index()
    msrs = ['Vbf']
    for msr in msrs:
        clmns = [msr+'_'+trmt for trmt in treatments]
        msr_df = diff_df[['rxn_ID']+clmns]
        msr_df.rename(columns={clm:clm.split('_')[1] for clm in clmns}, inplace=True)
        msr_df = msr_df.melt(id_vars=['rxn_ID'], \
                        value_vars=treatments, var_name='treatment', value_name=msr)
        msr_df = msr_df[msr_df[msr]<=10]
        fig = px.histogram(msr_df,
                            x=msr,
                            marginal="rug",
                            color="treatment",
                            title = f"{msr} -- {spc} {tissue} {time_stamp} diffs histogram",
                            nbins=nbin,
                            height=900,
                            width=900)
        # Overlay both histograms
        fig.update_layout(barmode='overlay')
        # Reduce opacity to see both histograms
        fig.update_traces(opacity=0.75)
        fig.show()
        # sleep(10)
        # print(abc)

# https://medium.com/@reinapeh/creating-a-complex-radar-chart-with-python-31c5cc4b3c5c
def plot_radar_chart(data, id_col, metrics, title):
    N = len(metrics)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    theta = np.concatenate([theta, [theta[0]]])

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': 'polar'})

    ax.set_title(title, y=1.15, fontsize=20)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(90)
    ax.spines['polar'].set_zorder(1)
    ax.spines['polar'].set_color('lightgrey')

    color_palette = ['#339F00', '#0500FF', '#9CDADB', '#FF00DE', '#FF9900', '#FFFFFF']

    for idx, (i, row) in enumerate(data.iterrows()):
        values = row[metrics].values.flatten().tolist()
        values = values + [values[0]]
        ax.plot(theta, values, linewidth=1.75, linestyle='solid', label=row[id_col], marker='o', markersize=1, color=color_palette[idx % len(color_palette)])
        ax.fill(theta, values, alpha=0.50, color=color_palette[idx % len(color_palette)])

    plt.yticks([0, 0.005, 0.01, 0.015, 0.02], ["0", "5", "10", "15", "20"], color="black", size=12)
    plt.xticks(theta, metrics + [metrics[0]], color='black', size=12)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.show()
    return fig

if __name__ == '__main__':
    treatments = ["Control", "FeLim", "FeEX", "ZnLim", "ZnEx"]
    msrs = ['Pred', 'FBA', 'Vbf', 'RES']
    control_id = "Control"

    spc = 'Sorghum'
    projCols = ['tissue', 'treatment', 'time_stamp']
    tissue = 'Leaf'
    time_stamp = '21d'

    treatments = ["control", "cold"]
    projCols = ['Treatment', 'Timestamp']
    spc = 'athaliana'
    time_stamp = 'ZT9'
    tissue = ''
    genotype = 'C24'

    model_path = "/Users/sea/Projects/AMN/omic_amn_mm/Dataset_input/athaliana_plastidial_thylakoid_052324_duplicated.xml"

    prediction_file = f"/Users/sea/Projects/AMN/omic_amn_mm/Result/{time_stamp}_sbicolor_thylakoid_V_rxn_fixed.tsv"
    # prediction_file = f"/Users/sea/Projects/AMN/omic_amn_mm/Result/{time_stamp}_thylakoid_V_rxn_fixed.tsv"
    prediction_file = f"/Users/sea/Projects/AMN/omic_amn_mm/Result/athaliana_{genotype}_{time_stamp}_thylakoid_V_rxn.tsv"

    kapp_vbf_path = f"/Users/sea/Projects/AMN/omic_amn_mm/Dataset_input/{spc}_thylakoid_{time_stamp}_Vbf_kapp_maxCtrl.csv"

    scores_path = "/Users/sea/Projects/QPSI_project/Enzyme_Abundance/integration_results/reaction_scores_binding_Dec6/Sorghum_objective_abundance_Control.csv"
    scores_path = f"/Users/sea/Projects/QPSI_project/Enzyme_Abundance/integration_results/secMetResults/athaliana_objective_abundance_control_{genotype}.tsv"

    sig_reactions = f"/Users/sea/Projects/QPSI_project/Enzyme_Abundance/src/util/sig_reactions_{genotype}.tsv"



    # if os.path.exists(prediction_file.replace(".tsv", "_fba_Vbf_RES.csv")) and False:
    #     pred_df = pa.read_csv(prediction_file.replace(".tsv", "_fba_Vbf_RES.csv"))
    #     ## Generate plots: histograms and line plots
    #     plot_results(pred_df, msrs, spc, time_stamp, tissue)
    #
    # # print(abc)
    #
    # if os.path.exists(prediction_file.replace(".tsv", "_diffs.csv")):
    #     diffs = pa.read_csv(prediction_file.replace(".tsv", "_diffs.csv"))
    #     plot_hist_diffs(diffs, msrs, treatments, spc, tissue, time_stamp)
    #
    # # print(abc)

    ## Read biologically feasible fluxes
    print('------- Vbf -- Kapp')
    vbf_df = pa.read_csv(kapp_vbf_path)
    clms = ['rxn_ID', 'kapp']+['v_'+trmt for trmt in treatments]
    # vbf_df = vbf_df.set_index('rxn_ID')
    vbf_df.rename({'v_'+trmt:trmt for trmt in treatments}, inplace=True, axis=1)
    vbf_df = vbf_df.melt(id_vars=['rxn_ID', 'kapp'], \
                    value_vars=treatments, var_name='treatment', value_name='Vbf')
    print(vbf_df.head())

    ## Comppute FBA using Vbf constraints
    print('------- FBA')
    vbf_df = vbf_df.set_index('rxn_ID')

    fba_fluxes_df = pa.DataFrame()
    for treatment in treatments:
        co_model = read_sbml_model(model_path)
        co_model = set_boundaries(co_model, vbf_df[vbf_df['treatment'] == treatment], 'Vbf')

        solution = co_model.optimize()
        print(treatment, solution)
        solution = solution.fluxes.to_frame()
        solution[solution.abs() < 10**-6] = 0
        solution = solution.rename_axis('rxn_ID')
        solution.rename(columns={'fluxes':treatment}, inplace=True)

        fba_fluxes_df = fba_fluxes_df.join(solution, how='outer')

    fba_fluxes_df = fba_fluxes_df.reset_index()
    vbf_df = vbf_df.reset_index()
    print(fba_fluxes_df.head())

    ## Read fluxes predictions and compute net fluxes for reversible reactions
    print('------- Pred')
    sep = "\t" if 'tsv' in prediction_file else ','
    pred_df = pa.read_csv(prediction_file, sep = sep)
    pred_df = process_predictions(pred_df)
    pred_df = pred_df[['rxn_ID']+treatments]
    print(pred_df.head())

    ## Compute carbon flux
    fluxes_only = pa.merge(pred_df, fba_fluxes_df, how="left", on='rxn_ID')
    co_model = read_sbml_model(model_path)
    # calculate_c_flux(fluxes_only, co_model)
    # print(abc)


    ## Read scores
    print('------- Scores')
    sep = '\t' if 'tsv' in scores_path else ','
    scores_df = pa.read_csv(scores_path, sep = sep)

    scores_colmns = projCols+['value', 'subsystems', 'rxn_ID', 'rxn_dist_quantile']
    if 'atha' not in spc:
        time_stamp = '0'+time_stamp if time_stamp not in ['14d', '21d'] else time_stamp


    scores_df = scores_df[scores_colmns]
    if 'tissue' in scores_df:
        scores_df = scores_df[(scores_df['tissue'] == tissue) & \
                                (scores_df['time_stamp'] == time_stamp)]
        scores_df = scores_df.drop(columns=['tissue', 'time_stamp'])
    else:
        scores_df = scores_df[(scores_df['Timestamp'] == time_stamp)]
        scores_df = scores_df.drop(columns=['Timestamp'])
    scores_df.rename({'value': 'RES', 'Treatment': 'treatment'}, inplace=True, axis=1)
    # scores_df
    print(scores_df.head())


    ## merge all dataframes to consolidate data
    #    create the treatment columns in the FBA and pred fluxes dataframes
    fba_fluxes_df = fba_fluxes_df.melt(id_vars=['rxn_ID'], value_vars=treatments, \
                    var_name='treatment', value_name='FBA')
    pred_df = pred_df.melt(id_vars=['rxn_ID'], value_vars=treatments, \
                    var_name='treatment', value_name='Pred')
    #     merge DFs
    merge_on = ['rxn_ID', 'treatment']
    pred_df = pa.merge(pred_df, fba_fluxes_df, how="left", on=merge_on)
    pred_df = pa.merge(pred_df, vbf_df, how="left", on=merge_on)
    pred_df = pa.merge(pred_df, scores_df, how="left", on=merge_on)
    #     write to file
    pred_df.to_csv(prediction_file.replace(".tsv", "_fba_Vbf_RES.csv"))

    # # rxn_ID	treatment	Pred	FBA	Vbf	RES
    # for trmt in treatments:
    #     trmt_df = pred_df[pred_df['treatment'] == trmt].copy()
    #     # trmt_df = trmt_df[['rxn_ID']+msrs]
    #     # metrics = trmt_df.columns[1:].tolist()
    #     plot_radar_chart(trmt_df[['rxn_ID']+msrs], 'rxn_ID', msrs, "title")
    #     print(abc)

    # ## Generate plots: histograms and line plots
    # plot_results(pred_df, msrs, spc, time_stamp, tissue)

    ## Radar chart for all reactions:



    # 'FBA', 'Pred', 'Vbf', 'RES', 'rxn_dist_quantile'
    pred_df = pred_df[(pred_df['Vbf']>0)]# & (pred_df['Pred']<=0.04)]
    ind = ['rxn_ID', 'subsystems', 'kapp']
    col = ['treatment']
    val = ['Pred', 'FBA', 'Vbf', 'RES']
    wide_pred_df = pred_df.pivot(index=ind, columns=col, values=msrs)

    wide_pred_df.columns = wide_pred_df.columns.map('{0[0]}_{0[1]}'.format)
    wide_pred_df = wide_pred_df.reset_index()
    wide_pred_df.to_csv(prediction_file.replace(".tsv", "_fba_Vbf_RES_wide.csv"), index=False)
    # print(abc)
    # print(pred_df.head())


    # wide_pred_df.set_index('rxn_ID', inplace=True)
    print(wide_pred_df.head())
    print(wide_pred_df.shape)
    print(wide_pred_df.columns)

    # rxn_ID	treatment	Pred	FBA	Vbf	RES
    reactions=['rxn02914_d0', 'rxn02914_d0_f', 'rxn02914_d0_r', 'rxn01975_d0', 'rxn01975_d0_f',
    'rxn01975_d0_r', 'rxn00179_d0', 'rxn00179_d0_f', 'rxn00179_d0_r', 'rxn00737_d0',
    'rxn00737_d0_f', 'rxn00737_d0_r', 'rxn27259_d0', 'rxn27259_d0_f', 'rxn27259_d0_r',
    'rxn02834_d0', 'rxn02834_d0_f', 'rxn02834_d0_r', 'rxn00076_d0', 'rxn00076_d0_f',
    'rxn00076_d0_r', 'rxn24330_d0', 'rxn24330_d0_f', 'rxn24330_d0_r', 'rxn01069_d0',
    'rxn01069_d0_f', 'rxn01069_d0_r', 'rxn00313_d0', 'rxn00313_d0_f', 'rxn00313_d0_r',
    'rxn00148_d0', 'rxn00148_d0_f', 'rxn00148_d0_r', 'rxn14012_d0', 'rxn14012_d0_f',
    'rxn14012_d0_r', 'rxn02476_d0', 'rxn02476_d0_f', 'rxn02476_d0_r', 'rxn03891_d0',
    'rxn03891_d0_f', 'rxn03891_d0_r', 'rxn31768_d0', 'rxn31768_d0_f', 'rxn31768_d0_r',
    'rxn02835_d0', 'rxn02835_d0_f', 'rxn02835_d0_r', 'rxn03892_d0', 'rxn03892_d0_f',
    'rxn03892_d0_r', 'rxn00834_d0', 'rxn00834_d0_f', 'rxn00834_d0_r', 'rxn03062_d0',
    'rxn03062_d0_f', 'rxn03062_d0_r', 'rxn01644_d0', 'rxn01644_d0_f', 'rxn01644_d0_r',
    'rxn19253_d0', 'rxn19253_d0_f', 'rxn19253_d0_r', 'rxn00781_d0', 'rxn00781_d0_f',
    'rxn00781_d0_r', 'rxn00338_d0', 'rxn00338_d0_f', 'rxn00338_d0_r', 'rxn00495_c0',
    'rxn00495_c0_f', 'rxn00495_c0_r', 'rxn02373_d0', 'rxn02373_d0_f', 'rxn02373_d0_r',
    'rxn19071_d0', 'rxn19071_d0_f', 'rxn19071_d0_r', 'rxn02929_d0', 'rxn02929_d0_f',
    'rxn02929_d0_r', 'rxn02895_d0', 'rxn02895_d0_f', 'rxn02895_d0_r', 'rxn01362_c0',
    'rxn01362_c0_f', 'rxn01362_c0_r', 'rxn02938_d0', 'rxn02938_d0_f', 'rxn02938_d0_r',
    'rxn00802_d0', 'rxn00802_d0_f', 'rxn00802_d0_r', 'rxn05287_d0', 'rxn05287_d0_f',
    'rxn05287_d0_r', 'rxn29919_d0', 'rxn29919_d0_f', 'rxn29919_d0_r', 'rxn00097_d0',
    'rxn00097_d0_f', 'rxn00097_d0_r', 'rxn37610_d0', 'rxn37610_d0_f', 'rxn37610_d0_r',
    'rxn00790_d0', 'rxn00790_d0_f', 'rxn00790_d0_r', 'rxn02928_d0', 'rxn02928_d0_f',
    'rxn02928_d0_r', 'rxn00710_c0', 'rxn00710_c0_f', 'rxn00710_c0_r', 'rxn00527_d0',
    'rxn00527_d0_f', 'rxn00527_d0_r']

    reactions = ['rxn00148_d0', 'rxn00527_d0', 'rxn00781_d0', 'rxn01975_d0', 'rxn02476_d0',
    'rxn02914_d0', 'rxn29919_d0', 'rxn00148_d0_f', 'rxn00527_d0_f', 'rxn00781_d0_f',
    'rxn01975_d0_f', 'rxn02476_d0_f', 'rxn02914_d0_f', 'rxn29919_d0_f', 'rxn00148_d0_r',
    'rxn00527_d0_r', 'rxn00781_d0_r', 'rxn01975_d0_r', 'rxn02476_d0_r', 'rxn02914_d0_r',
    'rxn29919_d0_r']

    reactions = ['rxn00148_d0', 'rxn00148_d0_f', 'rxn00148_d0_r', 'rxn00179_d0', 'rxn00179_d0_f',
     'rxn00179_d0_r', 'rxn00313_d0', 'rxn00313_d0_f', 'rxn00313_d0_r', 'rxn00527_d0',
     'rxn00527_d0_f', 'rxn00527_d0_r', 'rxn00737_d0', 'rxn00737_d0_f', 'rxn00737_d0_r',
     'rxn00781_d0', 'rxn00781_d0_f', 'rxn00781_d0_r', 'rxn00802_d0', 'rxn00802_d0_f',
     'rxn00802_d0_r', 'rxn01069_d0', 'rxn01069_d0_f', 'rxn01069_d0_r', 'rxn01256_d0',
     'rxn01256_d0_f', 'rxn01256_d0_r', 'rxn01644_d0', 'rxn01644_d0_f', 'rxn01644_d0_r',
     'rxn01975_d0', 'rxn01975_d0_f', 'rxn01975_d0_r', 'rxn02373_d0', 'rxn02373_d0_f',
     'rxn02373_d0_r', 'rxn02476_d0', 'rxn02476_d0_f', 'rxn02476_d0_r', 'rxn02834_d0',
     'rxn02834_d0_f', 'rxn02834_d0_r', 'rxn02835_d0', 'rxn02835_d0_f', 'rxn02835_d0_r',
     'rxn02914_d0', 'rxn02914_d0_f', 'rxn02914_d0_r', 'rxn02928_d0', 'rxn02928_d0_f',
     'rxn02928_d0_r', 'rxn02929_d0', 'rxn02929_d0_f', 'rxn02929_d0_r', 'rxn03062_d0',
     'rxn03062_d0_f', 'rxn03062_d0_r', 'rxn19253_d0', 'rxn19253_d0_f', 'rxn19253_d0_r',
     'rxn29919_d0', 'rxn29919_d0_f', 'rxn29919_d0_r']

    reactions = ['rxn00148_d0', 'rxn00148_d0_f', 'rxn00148_d0_r', 'rxn00179_d0', 'rxn00179_d0_f', 'rxn00179_d0_r', 'rxn00257_c0', 'rxn00257_c0_f', 'rxn00257_c0_r', 'rxn00313_d0', 'rxn00313_d0_f', 'rxn00313_d0_r', 'rxn00337_d0', 'rxn00337_d0_f', 'rxn00337_d0_r', 'rxn00527_d0', 'rxn00527_d0_f', 'rxn00527_d0_r', 'rxn01069_d0', 'rxn01069_d0_f', 'rxn01069_d0_r', 'rxn01256_d0', 'rxn01256_d0_f', 'rxn01256_d0_r', 'rxn01975_d0', 'rxn01975_d0_f', 'rxn01975_d0_r', 'rxn02373_d0', 'rxn02373_d0_f', 'rxn02373_d0_r', 'rxn02834_d0', 'rxn02834_d0_f', 'rxn02834_d0_r', 'rxn02835_d0', 'rxn02835_d0_f', 'rxn02835_d0_r', 'rxn02914_d0', 'rxn02914_d0_f', 'rxn02914_d0_r', 'rxn02928_d0', 'rxn02928_d0_f', 'rxn02928_d0_r', 'rxn02929_d0', 'rxn02929_d0_f', 'rxn02929_d0_r', 'rxn03062_d0', 'rxn03062_d0_f', 'rxn03062_d0_r', 'rxn19253_d0', 'rxn19253_d0_f', 'rxn19253_d0_r', 'rxn27497_d0', 'rxn27497_d0_f', 'rxn27497_d0_r', 'rxn29919_d0', 'rxn29919_d0_f', 'rxn29919_d0_r']
    sig_df = pa.read_csv(sig_reactions, sep='\t')
    print(sig_df.shape)
    sig_df = sig_df[(sig_df['Timestamp']=='ZT09') & (sig_df['class'].isin(['Central Carbon', 'Amino acids']))]
    sig_df = sig_df[['rxn_ID', 'role']]
    print(sig_df.shape)
    for sufx in ['_f', '_r']:
        temp = sig_df.copy()
        temp['rxn_ID'] = temp['rxn_ID']+sufx
        sig_df = pa.concat([sig_df, temp], ignore_index=True)
        print(sig_df.shape)

    print(sig_df.shape)

    # sig_df = sig_df[sig_df['rxn_ID'].isin(reactions)]
    sig_df = sig_df.drop_duplicates()
    print(sig_df.head())
    print(sig_df.shape)
    noin = ['Histidinol-phosphate aminotransferase (EC 2.6.1.9)', 'Tyrosine aminotransferase (EC 2.6.1.5)', 'Gamma-glutamyl phosphate reductase (EC 1.2.1.41)']
    sig_df = sig_df[~(sig_df['role'].isin(noin))]
    # sig_df = sig_df.groupby('rxn_ID').agg({'role': lambda x: ', '.join(str(x))}).reset_index()
    print(sig_df.shape)
    # print(abc)
    print("beefore filter", wide_pred_df.shape)
    reactions = list(sig_df['rxn_ID'].unique())
    wide_pred_df = wide_pred_df[wide_pred_df['rxn_ID'].isin(reactions)]
    print("after filter", wide_pred_df.shape)
    # print(abc)

    print(set(wide_pred_df['rxn_ID'].unique())-set(sig_df['rxn_ID'].unique()))
    # print(abc)
    wide_pred_df = pa.merge(wide_pred_df, sig_df, how="left", on='rxn_ID')
    print(wide_pred_df.head())
    # print(abc)
    # wide_pred_df = wide_pred_df[wide_pred_df['rxn_ID'].isin(reactions)]

    wide_pred_df = wide_pred_df.sort_values(by=['Pred_control'])

    colmn =  'role' #'role'
    df1 = wide_pred_df[[colmn, 'Pred_cold']]
    df1.rename(columns={'Pred_cold': 'Pred'}, inplace=True)
    df1 = df1[df1['Pred'] > 0]
    df1['Pred'] = np.log(df1['Pred'])

    df2 = wide_pred_df[[colmn, 'Pred_control']]
    df2.rename(columns={'Pred_control': 'Pred'}, inplace=True)
    df2 = df2.sort_values(by=['Pred'])
    df2 = df2[df2['Pred'] > 0]
    df2['Pred'] = np.log(df2['Pred'])

    rcts = set(df2[colmn].unique()).intersection(set(df1[colmn].unique()))
    df1 = df1[df1[colmn].isin(rcts)]
    df2 = df2[df2[colmn].isin(rcts)]


    df1['trmt'] = 'cold'
    df2['trmt'] = 'control'
    df = pa.concat([df1,df2], axis=0)

    # df.dropna(how='any', inplace=True)
    # df = df.sort_values(by=['Pred'])
    # df = df.sort_values(by=['Pred'])

    fig = px.line_polar(df, r='Pred', color='trmt', theta=colmn, line_close=True,
                        labels={'trmt': ''})
    fig.update_traces(fill='toself')
    fig.update_layout(
        font=dict(
            family="Arial",
            size=16
        )
    )
    fig.show()
    print(abc)



    # fig = go.Figure()
    # metrics = wide_t.columns[1:].tolist()
    # for msr in msrs:
    #     for trmt in treatments:
    #         print(msr+'_'+trmt+" " , wide_t[wide_t['index'] == msr+'_'+trmt][metrics])
    #         vals = list(wide_t[wide_t['index'] == msr+'_'+trmt][metrics].values)
    #         print(vals)
    #         # print(abc)
    #         fig.add_trace(go.Scatterpolar(
    #               r=vals,
    #               theta=metrics,
    #               fill='toself',
    #               name=msr+'_'+trmt
    #         ))
    #         fig.show()
    #     fig.show()
    #
    #     fig.update_layout(
    #       polar=dict(
    #         radialaxis=dict(
    #           visible=True,
    #           range=[0, 0.05]
    #         )),
    #       showlegend=False
    #     )
    #
    #     fig.show()
    #     print(abc)

    for msr in msrs:
        trmt_df = wide_pred_df[['rxn_ID']+[msr+'_'+trmt for trmt in treatments]].copy()
        trmt_df = trmt_df.sort_values(by=[msr+'_'+treatments[0]])
        print(trmt_df.head())
        print(abc)

        trmt_df = trmt_df.T.reset_index()
        print(trmt_df)
        print(trmt_df.columns)
        # trmt_df = wide_t[wide_t['index'].isin([msr+'_'+trmt for trmt in treatments])].copy()
        # trmt_df = trmt_df[['rxn_ID']+msrs]
        metrics = trmt_df.columns[1:].tolist()
        plot_radar_chart(trmt_df, 'index', metrics, title = msr)
        print(abc)

    print(abc)

    clmns = list(wide_pred_df.columns)
    for elem in ind:
        if elem in clmns: clmns.remove(elem)
    print(clmns)

    diffs = pa.DataFrame(columns=clmns)
    ln = len(wide_pred_df.index)
    index_list = list(wide_pred_df.index)
    for r1 in index_list:
        ln -= 1
        if ln % 100 == 0 :
            print(ln, ' ...')

        diffs_in = wide_pred_df[clmns].copy()

        diffs_in = diffs_in.reset_index()
        diffs_in['rxn_ID'] = diffs_in['rxn_ID'] + '_' + r1
        diffs_in[clmns] = np.abs(diffs_in[clmns] - wide_pred_df[clmns].loc[r1].values)

        diffs = pa.concat([diffs, diffs_in], ignore_index=True)
        wide_pred_df = wide_pred_df.drop(r1)

    diffs.set_index('rxn_ID', inplace=True)
    diffs.to_csv(prediction_file.replace(".tsv", "_diffs.csv"))

    plot_hist_diffs(diffs, msrs, treatments)
    # fba_fluxes_df.to_csv("/Users/sea/Projects/AMN/omic_amn/Result/noBiomassMax_constraints_3.5_noEXcpd00727_fixedRev/fba_results_nonZero.csv")
