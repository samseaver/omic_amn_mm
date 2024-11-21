import pandas as pa
import os
import numpy as np
import json

import plotly.express as px
import seaborn as sns
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import plotly.io as pio
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
pio.templates.default = "plotly_white" #"none"


from plotly.subplots import make_subplots
from Calculate_Carbon_Flux import Calculate_Carbon_Flux
from cobra.io import read_sbml_model


from dash import Dash, dcc, html, Input, Output
import dash_cytoscape as cyto

avogadro = 6.02214076e+23
spc = 'Sorghum'
genotype = 'Leaf'
time_stamp = '21d'
sv = 'sv15/'
model_path = "/Users/selalaoui/Projects/QPSI_project/Enzyme_Abundance_all/data/metabolic_models/plastidial_models/ortho_jun20_models/sbicolor_3.1.1_plastid_Thylakoid_Reconstruction_ComplexFix_070224_noADP_duplicated_noP.xml"

ccf_obj = Calculate_Carbon_Flux()


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

def read_predictions(prediction_file):
    sep = "\t" if 'tsv' in prediction_file else ','
    pred_df = pa.read_csv(prediction_file, sep = sep)

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

    pred_df = pred_df[['rxn_ID']+treatments]
    print(pred_df[pred_df['rxn_ID']=='bio1_biomass'])

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
    ## Compute metabolite fluxes
    # ccf_obj.metabolites_balance(flux_df, co_model)

    ## Compute carbon flux
    ccf_obj.med_bio_flux(flux_df, co_model)
    # print(abc)
    # ccf_obj.model_rxns_flux(flux_df, co_model)

def plot_hist_diffs(diff_df, msrs, treatments, spc, tissue, time_stamp):
    nbin = 100
    diff_df = diff_df.reset_index()
    # msrs = ['Vbf']
    for msr in msrs:
        clmns = [msr+'_'+trmt for trmt in treatments]
        msr_df = diff_df[['rxn_ID']+clmns]
        msr_df.rename(columns={clm:clm.split('_')[1] for clm in clmns}, inplace=True)
        msr_df = msr_df.melt(id_vars=['rxn_ID'], \
                        value_vars=treatments, var_name='treatment', value_name=msr)
        # msr_df = msr_df[msr_df[msr]<=10]
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
        print(msr_df.head())
        generate_CDFs(msr_df, treatments, variables=[msr])
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

def generate_CDFs(data, treatments, time_stamp, variables=[]):
    data.dropna(how='all', inplace=True)
    print(data.head())
    # print(abc)
    if not variables: variables = ['kapp', 'Vbf', 'RES', 'mean_flux', 'Pred']

    # defining the libraries
    import numpy as np
    import matplotlib.pyplot as plt

    # for trmt in treatments:
    #     plt.title(f'CDF for Sorghum {trmt} - day 21')
    #     trmt_data = data[data['treatment'] == trmt]
    #     for var in variables:
    #         print(f"---*--- processing {var} ---*---")
    #         var_data = trmt_data[var].dropna()
    #         # Calculate the cumulative proportion of the data that falls below each value
    #         cumulative = np.linspace(0, 1, len(var_data))
    #
    #         # Sort the data in ascending order
    #         sorted_data = np.sort(var_data)
    #
    #         # Calculate the cumulative proportion of the sorted data
    #         cumulative_data = np.cumsum(sorted_data) / np.sum(sorted_data)
    #         # Plot the CDF
    #         plt.plot(sorted_data, cumulative_data, label=f"{var}")
    #
    #     plt.legend()
    #     plt.show()


    if 'RES_relab' in data: variables = variables+['RES_relab']
    variables = ['Pred', 'Vbf', 'RES']
    for var in variables:
        print(f"---*--- processing {var} ---*---")
        plt.title(f'{var} CDF for Sorghum - {time_stamp}')
        trmt_data = data[['rxn_ID', 'treatment', var]]

        ind = ['rxn_ID']
        col = ['treatment']
        val = [var]
        trmt_data = trmt_data.pivot(index=ind, columns=col, values=val)
        trmt_data.columns = trmt_data.columns.map('{0[0]}_{0[1]}'.format)
        trmt_data = trmt_data.reset_index()
        # print(trmt_data.head(10))

        for trmt in treatments:
            var_data = trmt_data[var+'_'+trmt].dropna()
            if 'relab' in var:
                var_data = np.log(var_data)
            # Calculate the cumulative proportion of the data that falls below each value
            cumulative = np.linspace(0, 1, len(var_data))
            # Sort the data in ascending order
            sorted_data = np.sort(var_data)
            # Calculate the cumulative proportion of the sorted data
            cumulative_data = np.cumsum(sorted_data) / np.sum(sorted_data)
            # Plot the CDF
            plt.plot(sorted_data, cumulative_data, label=f"{trmt}")

        plt.legend()
        if var in ['Pred', 'FBA', 'Vbf', 'RES']:
            fname = f"/Users/selalaoui/Projects/AMN/omic_amn_mm/Result/pred_fluxes_figures/{sv}{var}_CDF_{spc}_{time_stamp}"
            plt.savefig(fname, dpi=800)#, bbox_inches='tight', dpi=400)
        # plt.show()


    return True

def read_scores(scores_path, spc, tissue, time_stamp, projCols, relab=False):
    sep = '\t' if 'tsv' in scores_path else ','
    scores_df = pa.read_csv(scores_path, sep = sep)
    clm_name = 'RES'
    if relab: clm_name = clm_name+'_relab'

    scores_colmns = projCols+['value', 'subsystems', 'rxn_ID', 'rxn_score_I_dist', 'rxn_dist_quantile']
    # if 'atha' not in spc:
    #     time_stamp = '0'+time_stamp if time_stamp not in ['14d', '21d'] else time_stamp


    scores_df = scores_df[scores_colmns]
    if 'tissue' in scores_df:
        scores_df = scores_df[(scores_df['tissue'] == tissue) & \
                                (scores_df['time_stamp'] == time_stamp)]
        scores_df = scores_df.drop(columns=['tissue', 'time_stamp'])
    else:
        scores_df = scores_df[(scores_df['Timestamp'] == time_stamp)]
        scores_df = scores_df.drop(columns=['Timestamp'])
    scores_df.rename({'value': clm_name, 'Treatment': 'treatment'}, inplace=True, axis=1)

    # # Convert units using avogadro number

    if relab:
        # # Convert units using avogadro number
        scores_df[clm_name] = scores_df[clm_name].astype('float') / avogadro

    print(scores_df.head())
    return scores_df

def generate_FBA(vbf_df, model_path, treatments):
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
    return fba_fluxes_df

def read_vbf(kapp_vbf_path, treatments):
    vbf_df = pa.read_csv(kapp_vbf_path)
    clms = ['rxn_ID', 'kapp', 'mean_flux']+['v_'+trmt for trmt in treatments]
    # vbf_df = vbf_df.set_index('rxn_ID')
    if 'atha' in spc:
        temp_trmts = ["Control", "Cold"]
    else:
        temp_trmts = treatments
    print(vbf_df.head())
    vbf_df.rename({'v_'+trmt:trmt for trmt in temp_trmts}, inplace=True, axis=1)
    print(vbf_df.head())
    vbf_df = vbf_df.melt(id_vars=['rxn_ID', 'kapp', 'mean_flux'], \
                    value_vars=treatments, var_name='treatment', value_name='Vbf')

    return vbf_df

def generate_radarPlot(sig_reactions, wide_pred_df):
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

    reactions = ['rxn00148_d0', 'rxn00148_d0_f', 'rxn00148_d0_r', 'rxn00179_d0', 'rxn00179_d0_f',
    'rxn00179_d0_r', 'rxn00257_c0', 'rxn00257_c0_f', 'rxn00257_c0_r', 'rxn00313_d0',
    'rxn00313_d0_f', 'rxn00313_d0_r', 'rxn00337_d0', 'rxn00337_d0_f', 'rxn00337_d0_r',
    'rxn00527_d0', 'rxn00527_d0_f', 'rxn00527_d0_r', 'rxn01069_d0', 'rxn01069_d0_f',
    'rxn01069_d0_r', 'rxn01256_d0', 'rxn01256_d0_f', 'rxn01256_d0_r', 'rxn01975_d0',
    'rxn01975_d0_f', 'rxn01975_d0_r', 'rxn02373_d0', 'rxn02373_d0_f', 'rxn02373_d0_r',
    'rxn02834_d0', 'rxn02834_d0_f', 'rxn02834_d0_r', 'rxn02835_d0', 'rxn02835_d0_f',
    'rxn02835_d0_r', 'rxn02914_d0', 'rxn02914_d0_f', 'rxn02914_d0_r', 'rxn02928_d0',
    'rxn02928_d0_f', 'rxn02928_d0_r', 'rxn02929_d0', 'rxn02929_d0_f', 'rxn02929_d0_r',
    'rxn03062_d0', 'rxn03062_d0_f', 'rxn03062_d0_r', 'rxn19253_d0', 'rxn19253_d0_f',
    'rxn19253_d0_r', 'rxn27497_d0', 'rxn27497_d0_f', 'rxn27497_d0_r', 'rxn29919_d0',
    'rxn29919_d0_f', 'rxn29919_d0_r']

    reactions = ['EX_cpd00007_e0_o', 'EX_cpd00009_e0_o', 'EX_cpd00011_e0_i',
                 'EX_cpd00013_e0_i', 'EX_cpd00067_e0_o', 'EX_cpd11632_e0_i', 'rxn00069_d0_f',
                 'rxn00097_d0_r', 'rxn00102_d0_r', 'rxn00121_d0', 'rxn00122_d0',
                 'rxn00161_d0_r', 'rxn00187_d0', 'rxn00248_d0_f', 'rxn00249_d0_f',
                 'rxn00257_c0_f', 'rxn00330_c0', 'rxn00337_d0_f', 'rxn00391_d0_f',
                 'rxn00392_d0_f', 'rxn00533_d0', 'rxn00747_d0_f', 'rxn00781_d0_r',
                 'rxn00782_d0_r', 'rxn00799_c0_f', 'rxn01100_d0_f', 'rxn01362_c0_r',
                 'rxn01975_d0_r', 'rxn02938_d0_r', 'rxn05319_d0_i', 'rxn05467_c0_i',
                 'rxn05467_d0_i', 'rxn05468_c0_o', 'rxn05468_d0_o', 'rxn08173_y0',
                 'rxn08217_c0_o', 'rxn08730_c0_o', 'rxn08730_d0_o', 'rxn09121_c0_o',
                 'rxn09736_c0_i', 'rxn09736_d0_i', 'rxn09839_d0_o', 'rxn10929_d0_i',
                 'rxn10967_d0_i', 'rxn13365_d0_i', 'rxn15298_d0_f', 'rxn15493_d0_f',
                 'rxn15494_d0_f', 'rxn19828_d0_r', 'rxn24508_d0_r', 'rxn26754_d0',
                 'rxn29240_d0_f', 'rxn29733_d0_i', 'rxn31154_d0_i', 'EX_cpd00001_e0_i',
                 'rxn05319_c0_i', 'rxn05465_d0_f', 'rxn15280_d0_f', 'rxn26477_d0_r',
                 'rxn00770_d0_r', 'rxn00777_d0_f', 'rxn00974_d0_r', 'rxn02380_d0_f',
                 'rxn11700_d0_f', 'rxn17196_d0', 'rxn_cpd00103_c0_i', 'rxn00001_d0_f',
                 'rxn00799_c0_r', 'rxn00899_d0_r', 'rxn01207_d0_r', 'rxn08766_d0_f',
                 'rxn11013_d0_o', 'rxn17744_d0_r', 'rxn17803_d0_r']

    # dup_rxns = []
    # for rxn in reactions:
    #     dup_rxns += [rxn, rxn+"_f", rxn+"_r"]
    # reactions = dup_rxns

    sig_df = pa.read_csv(sig_reactions, sep='\t')
    print(sig_df.shape)
    sig_df = sig_df[(sig_df['day']==time_stamp) & (sig_df['class'].isin(['Central Carbon', 'Amino acids']))]
    print(sig_df.columns)
    sig_df = sig_df[['rxn_ID', 'role']]
    print(sig_df.shape)
    sig_df_dup = pa.DataFrame()
    for sufx in ['_f', '_r']:
        temp = sig_df.copy()
        temp['rxn_ID'] = temp['rxn_ID']+sufx
        sig_df_dup = pa.concat([sig_df_dup, temp], ignore_index=True)
        print(sig_df_dup.shape)
    sig_df_dup = pa.concat([sig_df, sig_df_dup], ignore_index=True)
    sig_df = sig_df_dup
    print(sig_df.shape)

    sig_df = sig_df[sig_df['rxn_ID'].isin(reactions)]
    sig_df = sig_df.drop_duplicates()
    print("sig_df \n", sig_df.head())
    print(sig_df.shape)
    noin = ['Histidinol-phosphate aminotransferase (EC 2.6.1.9)', 'Tyrosine aminotransferase (EC 2.6.1.5)', 'Gamma-glutamyl phosphate reductase (EC 1.2.1.41)']
    sig_df = sig_df[~(sig_df['role'].isin(noin))]
    # sig_df = sig_df.groupby('rxn_ID').agg({'role': lambda x: ', '.join(str(x))}).reset_index()
    print(sig_df.shape)

    print("before filter", wide_pred_df.shape)
    print("before filter", wide_pred_df.head())
    print("there are ", len(reactions))
    print(set(wide_pred_df['rxn_ID'].unique()).intersection(set(reactions)))
    # print(abc)
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

    wide_pred_df = wide_pred_df.sort_values(by=['Pred_Control'])

    # colmn =  'role' #'role'
    wide_pred_df['role'] = wide_pred_df['role'].astype('str')
    wide_pred_df['rxn_ID'] = wide_pred_df['rxn_ID'].astype('str')
    wide_pred_df['role_rxn'] = wide_pred_df['role'] + "_" + wide_pred_df['rxn_ID']
    colmn = 'role_rxn'
    # colmn = 'role'
    clms = []
    cmn_role = []
    # $turquoise: 'rgba(85, 214, 190, 1)';
    # $glaucous: 'rgba(96, 113, 150, 1)';
    # $jasper: 'rgba(205, 83, 52, 1)';
    # $citron: 'rgba(225, 206, 122, 1)';
    # $orchid-pink: 'rgba(212, 175, 185, 1)';
    # color_map = {'Control': 'rgba(85, 214, 190, 1)', 'FeLim': 'rgba(96, 113, 150, 1)', 'FeEX': 'rgba(205, 83, 52, 1)', 'ZnLim': 'rgba(225, 206, 122, 1)', 'ZnEx': 'rgba(212, 175, 185, 1)'}
    color_map = {'Control': 'rgba(56, 29, 42, 1)', 'FeEX': 'rgba(62, 105, 144, 1)', 'FeLim': 'rgba(170, 189, 140, 1)', 'ZnEx': 'rgba(206, 83, 116, 1)', 'ZnLim': 'rgba(243, 155, 109, 1)'}
    color_map = {'Control': 'rgba(171, 146, 191, 1)', 'FeEX': 'rgba(29, 138, 153, 1)', 'FeLim': 'rgba(252, 208, 161, 1)', 'ZnEx': 'rgba(149, 155, 177, 1)', 'ZnLim': 'rgba(93, 211, 158, 1)'}
    # rgba(184, 142, 143, 1)
    # 'rgba(93, 211, 158, 1)'
    for trmt in treatments:
        df = pa.DataFrame()
        for tmt in [trmt, 'Control']:
            df1 = wide_pred_df[[colmn, 'Pred_'+tmt]]
            df1.rename(columns={'Pred_'+tmt: 'Pred'}, inplace=True)
            df1 = df1[df1['Pred'] > 0]
            df1['trmt'] = tmt
            # df1['Pred'] = np.log(df1['Pred'])
            df1 = df1.sort_values(by=['Pred'])

            if not cmn_role:
                cmn_role = set(df1[colmn].unique())
            else:
                cmn_role = set(df1[colmn].unique()).intersection(cmn_role)

            df = pa.concat([df, df1], axis=0)
        df = df[df[colmn].isin(cmn_role)]

        df = df.sort_values(by=[colmn])
        # df = df.sort_values(by=['Pred', 'trmt'])
        title = f'Significant reactions for {spc} {genotype} - {time_stamp} {trmt}'
        fig = px.bar(df, y='Pred', color='trmt', x=colmn, color_discrete_map=color_map,
                            labels={'trmt': ''}, title=title, barmode="group")
        # fig.update_traces(fill='toself')
        fig.update_layout(
            font=dict(
                family="Arial",
                size=12
            )
        )
        # fname = f"/Users/selalaoui/Projects/AMN/omic_amn_mm/Result/pred_fluxes_figures/sig_rxn_radar_{spc}_{genotype}_{time_stamp}_{trmt}.png"
        # pio.write_image(fig, fname, scale=6, width=1000, height=1000)
        # fig.show()
    # print(abc)
    cmn_role = []
    df = pa.DataFrame()
    for trmt in treatments:
        df1 = wide_pred_df[[colmn, 'Pred_'+trmt]]
        df1.rename(columns={'Pred_'+trmt: 'Pred'}, inplace=True)
        df1 = df1[df1['Pred'] > 0]
        df1['trmt'] = trmt
        # df1['Pred'] = np.log(df1['Pred'])
        df1 = df1.sort_values(by=['Pred'])

        if not cmn_role:
            cmn_role = set(df1[colmn].unique())
        else:
            cmn_role = set(df1[colmn].unique()).intersection(cmn_role)

        df = pa.concat([df, df1], axis=0)
    df = df[df[colmn].isin(cmn_role)]

    df = df.sort_values(by=[colmn])
    # df = df.sort_values(by=['Pred', 'trmt'])
    title = f'Significant reactions for {spc} {genotype} - {time_stamp}'
    # fig = px.line_polar(df, r='Pred', color='trmt', theta=colmn, line_close=True,
    #                     labels={'trmt': ''}, title=title)
    # fig = px.line_polar(df, r='Pred', color='trmt', theta=colmn, line_close=True,
    #                     labels={'trmt': ''}, title=title)
    # fig.update_traces(fill='toself')

    fig = px.bar(df, y='Pred', color='trmt', x=colmn,
                        labels={'trmt': ''}, title=title, barmode="group", color_discrete_map=color_map)
    fig.update_layout(
        font=dict(
            family="Arial",
            size=12
        )
    )
    fname = f"/Users/selalaoui/Projects/AMN/omic_amn_mm/Result/pred_fluxes_figures/sig_rxn_radar_{spc}_{genotype}_{time_stamp}_all.png"
    pio.write_image(fig, fname, scale=6, width=1000, height=1000)
    fig.show()



if __name__ == '__main__':
    calculate_C = True
    relab = False

    treatments = ["Control", "FeLim", "FeEX", "ZnLim", "ZnEx"]
    treatments = ["Control", "FeEX", "FeLim", "ZnEx", "ZnLim"]
    msrs = ['Pred', 'FBA', 'Vbf', 'RES']
    control_id = "Control"

    
    projCols = ['tissue', 'treatment', 'time_stamp']
    tissue = 'Leaf'

    # treatments = ["control", "cold"]
    # projCols = ['Treatment', 'Timestamp']
    # spc = 'athaliana'
    # time_stamp = 'ZT9'
    # tissue = ''
    # genotype = 'C24'

    more = "_noADP_noP"
    # more = "_noBioLimit_newError"
    # more = "_decay5" # SV_penalty = 10
    # more = "_dr5_lr5" # SV_penalty = 10
    # more = "_noBioLimit" # SV_penalty = 10
    prediction_file = f"/Users/selalaoui/Projects/AMN/omic_amn_mm/Result/{sv}{spc}_{genotype}_{time_stamp}_complexFix{more}_V_rxn.tsv"

    # kapp_vbf_path = f"/Users/selalaoui/Projects/AMN/omic_amn_mm/Dataset_input/{spc}_complexFix_{genotype}_{time_stamp}_Vbf_kapp_maxCtrl_mixRelab.csv"
    kapp_vbf_path = f"/Users/selalaoui/Projects/AMN/omic_amn_mm/Dataset_input/{spc}_complexFix_{genotype}_{time_stamp}_noADP_Vbf_kapp_maxCtrl_mixedRelab.csv"
    # Sorghum_complexFix_Leaf_21d_Vbf_kapp_maxCtrl_mixedRelab_noADP

    scores_path = f"/Users/selalaoui/Projects/QPSI_project/Enzyme_Abundance_all/integration_results/reaction_scores_binding_Jul2/plastidial_model/{spc}_objective_abundance_Control.tsv"
    # scores_path = f"/Users/selalaoui/Projects/QPSI_project/Enzyme_Abundance_all/integration_results/secMetResults/athaliana_objective_abundance_control_{genotype}.tsv"

    sig_reactions = f"/Users/selalaoui/Projects/QPSI_project/Enzyme_Abundance_all/src/util/sig_reactions_{spc.lower()}_{genotype.lower()}.tsv"

    s_matrix_path = f"/Users/selalaoui/Projects/AMN/omic_amn_mm/Result/s_matrix/s_matrix_plastid_noP_T_new.csv"
    saveTo = f"/Users/selalaoui/Projects/AMN/omic_amn_mm/Result/{sv}"
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
    # print(abc)

    ## Read biologically feasible fluxes
    print('------- Vbf -- Kapp')
    vbf_df = read_vbf(kapp_vbf_path, treatments)
    print(vbf_df.head())

    ## Comppute FBA using Vbf constraints
    print('------- FBA')
    fba_fluxes_df = generate_FBA(vbf_df, model_path, treatments)
    print(fba_fluxes_df.head())

    ## Read fluxes predictions and compute net fluxes for reversible reactions
    print('------- Pred')
    pred_df = read_predictions(prediction_file)
    print(pred_df.head())
    # print(abc)

    print('------- Processing S matrix')
    sMatrix = False
    if sMatrix:
        co_model = read_sbml_model(model_path)
        met_c_dict = ccf_obj.metabolite_molecule(co_model, 'C')

        color_map = {'Control': 'rgba(64, 61, 88, 1)', 'FeEX': 'rgba(0, 175, 181, 1)', 'FeLim': 'rgba(170, 189, 140, 1)',
        'ZnEx': 'rgba(206, 83, 116, 1)', 'ZnLim': 'rgba(243, 155, 109, 1)'}
        color_map_sns = {'Control': (64/255, 61/255, 88/255, 1), 
                        'FeEX': (0/255, 175/255, 181/255, 1), 
                        'FeLim': (170/255, 189/255, 140/255, 1),
                        'ZnEx': (206/255, 83/255, 116/255, 1), 
                        'ZnLim': (243/255, 155/255, 109/255, 1)}

        fig = make_subplots(rows=1, cols=len(treatments),
                        # specs=specs, 
                        # subplot_titles=titles, 
                        # shared_xaxes=True,
                        shared_yaxes=True,
                        vertical_spacing=0.02,
                        horizontal_spacing=0.02,)
        s_matrix_df = pa.read_csv(s_matrix_path)
        s_matrix_df.rename(columns={'Unnamed: 0': 'rxn_ID'}, inplace=True)
        s_matrix_df = s_matrix_df.set_index('rxn_ID')
        s_matrix_df.sort_index(inplace=True)
        col = 0
        # Set the seaborn style for better aesthetics
        sns.set(style="whitegrid")

        # Create the plot
        plt.figure(figsize=(10, 6))

        for trmt in treatments:
            print(trmt, "-*-"*20)
            col += 1
            trmt_pred = pred_df[['rxn_ID', trmt]]
            trmt_pred = trmt_pred.set_index('rxn_ID')
            trmt_pred.sort_index(inplace=True)
            # print(s_matrix_df.head())
            # print(abc)

            trmt_matrix = s_matrix_df.apply(lambda x : np.asarray(x) * np.asarray(trmt_pred[trmt]))
            trmt_matrix = trmt_matrix.fillna(0)
            


            # trmt_matrix = trmt_matrix.set_index('rxn_ID').transpose()
            # trmt_matrix = trmt_matrix.reset_index()
            # trmt_matrix.rename(columns={'index': 'rxn_ID'}, inplace=True)
            # print(trmt_matrix.head())
            

            # Transpose the DataFrame
            trmt_matrix = trmt_matrix.T
            trmt_matrix['sum'] = trmt_matrix.sum(axis=1)
            trmt_matrix['sum_c'] = trmt_matrix.index.map(met_c_dict) * trmt_matrix['sum']
            

            print(trmt, np.abs(trmt_matrix['sum']).describe(percentiles = [.5, .6, .73, .75, .78, .8, .83, .86, .95, .96, .97, .98, .99]))

            trmt_matrix.to_csv(f"{saveTo}{spc}_{time_stamp}_{trmt}_s_matrix.tsv", sep='\t')
            # print(abc)
            # # Plot KDE for each treatment DataFrame 
            # # sns.kdeplot(trmt_matrix['sum'], label=f'Treatment {trmt}', 
            # #              fill=True, color=color_map_sns[trmt])
            # sns.ecdfplot(trmt_matrix['sum'], label=f'Treatment {trmt}', 
            #             linewidth=1.5, color=color_map_sns[trmt]) #, linestyle='-.'
            # # sns.ecdfplot(trmt_matrix['sum_c'], label=f'Carbon {trmt}', 
            # #             linewidth=1.5, linestyle='-.', color=color_map_sns[trmt])

            # trmt_matrix = trmt_matrix[np.abs(trmt_matrix['sum']) > 5]
            # trmt_matrix = trmt_matrix.sort_values(by='sum')
            # metabolites = list(trmt_matrix.index.unique())
            # find_reactions(metabolites, model_path, trmt_pred[trmt], trmt_matrix, trmt)

            # # print(trmt_matrix['sum'].describe())
            # fig.add_trace(
            #             go.Scatter(x=trmt_matrix.index, 
            #                     y=trmt_matrix["sum"], 
            #                     name = f"{trmt} - Predicted Flux", 
            #                     # mode = 'markers',
            #                     marker = dict(
            #                         color=color_map[trmt],  # Marker color
            #                         size=7,       # Marker size
            #                         symbol='circle' # Marker shape
            #                     ),
            #                     line=dict(
            #                         color=color_map[trmt],  # Line color
            #                         width=2       # Line width
            #                     ), 
            #                     showlegend=True
            #                 ), row=1, col=col#, secondary_y=False#, showlegend=True  
            #         )
            # # trmt_matrix = trmt_matrix[['cpd00810_c0', 'cpd00070_d0', 'cpd00242_d0', 'cpd00247_c0', 'cpd00103_c0', 'cpd00011_d0']]
            # # trmt_matrix = trmt_matrix[(np.abs(trmt_matrix['cpd00810_c0'])>0) | (np.abs(trmt_matrix['cpd00070_d0'])>0) | (np.abs(trmt_matrix['cpd00242_d0'])>0)]
            # # trmt_matrix.to_csv(f"{saveTo}{spc}_{time_stamp}{trmt}_s_matrix_subset.tsv", sep='\t')
        
        # # fig.update_yaxes(range=[-105, 105], secondary_y=True)
        # # fig.update_yaxes(range=[-0.2, 7.5], secondary_y=False)
        # fig.update_layout(height=400, width=1600, font=dict(family='Times New Roman'), showlegend=True)
        # # plot_path = f"export_cBalance_{trmt}.png"
        # # pio.write_image(fig, plot_path, scale=5, width=380, height=500)
        # fig.show()

        # # Add titles and labels
        # plt.title('PDF of Net Flux/Carbon flux by Treatment', fontsize=16)
        # plt.xlabel('Net Flux/Carbon flux', fontsize=12)
        # plt.ylabel('Density', fontsize=12)
        # # Add a legend to differentiate treatments
        # plt.legend(title='Treatment')
        # # Show the plot
        # plt.show()


    ## Compute carbon flux
    if calculate_C:
        print('------- Calculating Carbon flux and metabolite balance...')
        fluxes_only = pa.merge(pred_df, fba_fluxes_df, how="left", on='rxn_ID', suffixes=('_pred', '_fba'))
        co_model = read_sbml_model(model_path)
        calculate_c_flux(fluxes_only, co_model)
        # print(abc)


    ## Read scores
    print('------- Scores')
    scores_df = read_scores(scores_path, spc, tissue, time_stamp, projCols)
    print(scores_df.head())
    if relab:
        scores_path = f"/Users/selalaoui/Projects/QPSI_project/Enzyme_Abundance_all/integration_results/reaction_scores_binding_Dec6/{spc}_relab_rxn_scores_tmm.csv"
        relab_scores_df = read_scores(scores_path, spc, tissue, time_stamp, projCols, relab=True)
        print(relab_scores_df.head())
        print(relab_scores_df.describe())
        scores_df = pa.merge(scores_df, relab_scores_df[['rxn_ID', 'treatment', 'RES_relab']], how="left", on=['rxn_ID', 'treatment'])
        # print(scores_df.head())
        # print(abc)




    ## merge all dataframes to consolidate data
    #    create the treatment columns in the FBA and pred fluxes dataframes
    fba_fluxes_df = fba_fluxes_df.melt(id_vars=['rxn_ID'], value_vars=treatments, \
                    var_name='treatment', value_name='FBA')
    pred_df = pred_df.melt(id_vars=['rxn_ID'], value_vars=treatments, \
                    var_name='treatment', value_name='Pred')
    #     merge DFs
    print(pred_df[pred_df['rxn_ID'] == 'bio1_biomass'])

    merge_on = ['rxn_ID', 'treatment']
    pred_df['rxn_ID_only'] = pred_df.apply(lambda row: get_rxn_ID(row), axis=1)
    print(pred_df.head())
    pred_df = pa.merge(pred_df, scores_df, how="left", left_on=['rxn_ID_only', 'treatment'], right_on=merge_on, suffixes=(None, "_x"))
    pred_df.drop(columns=['rxn_ID_only', "rxn_ID_x"], inplace=True)

    pred_df = pa.merge(pred_df, fba_fluxes_df, how="left", on=merge_on)
    pred_df = pa.merge(pred_df, vbf_df, how="left", on=merge_on)
    #     write to file
    print(pred_df.head())
    print(prediction_file.replace(".tsv", "_fba_Vbf_RES.tsv"))
    pred_df.to_csv(prediction_file.replace(".tsv", "_fba_Vbf_RES.tsv"), sep="\t", index=False)
    print(abc)
    generate_CDFs(pred_df, treatments, time_stamp)
    # print(abc)


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
    # pred_df = pred_df[(pred_df['Vbf']>0)]# & (pred_df['Pred']<=0.04)]
    ind = ['rxn_ID', 'subsystems', 'kapp']
    col = ['treatment']
    val = ['Pred', 'FBA', 'Vbf', 'RES']
    wide_pred_df = pred_df.pivot(index=ind, columns=col, values=msrs)

    wide_pred_df.columns = wide_pred_df.columns.map('{0[0]}_{0[1]}'.format)
    wide_pred_df = wide_pred_df.reset_index()
    wide_pred_df.to_csv(prediction_file.replace(".tsv", "_fba_Vbf_RES_wide.tsv"), sep="\t", index=False)
    # print(abc)
    # print(pred_df.head())


    # wide_pred_df.set_index('rxn_ID', inplace=True)
    print(wide_pred_df.head())
    print(wide_pred_df.shape)
    print(wide_pred_df.columns)

    # Generate radar plot
    if False:
        generate_radarPlot(sig_reactions, wide_pred_df)
        print(abc)

        for msr in msrs:
            trmt_df = wide_pred_df[['rxn_ID']+[msr+'_'+trmt for trmt in treatments]].copy()
            trmt_df = trmt_df.sort_values(by=[msr+'_'+treatments[0]])
            print(trmt_df.head())
            # print(abc)

            trmt_df = trmt_df.T.reset_index()
            print(trmt_df)
            print(trmt_df.columns)
            # trmt_df = wide_t[wide_t['index'].isin([msr+'_'+trmt for trmt in treatments])].copy()
            # trmt_df = trmt_df[['rxn_ID']+msrs]
            metrics = trmt_df.columns[1:].tolist()
            plot_radar_chart(trmt_df, 'index', metrics, title = msr)



    print(abc)
    clmns = list(wide_pred_df.columns)
    for elem in ind:
        if elem in clmns: clmns.remove(elem)
    print(clmns)

    diffs = pa.DataFrame(columns=clmns)
    wide_pred_df = wide_pred_df.set_index('rxn_ID')
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

    plot_hist_diffs(diffs, msrs, treatments, spc, tissue, time_stamp)
    # fba_fluxes_df.to_csv("/Users/selalaoui/Projects/AMN/omic_amn/Result/noBiomassMax_constraints_3.5_noEXcpd00727_fixedRev/fba_results_nonZero.csv")
