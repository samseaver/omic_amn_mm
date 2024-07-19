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

avogadro = 6.02214076e+23

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
    ccf_obj = Calculate_Carbon_Flux()

    ## Compute metabolite fluxes
    ccf_obj.metabolites_balance(flux_df, co_model)

    ## Compute carbon flux
    ccf_obj.med_bio_flux(flux_df, co_model)
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
    variables = ['Pred']
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
            fname = f"/Users/sea/Projects/AMN/omic_amn_mm/Result/{var}_CDF_Sorghum_{time_stamp}"
            plt.savefig(fname, dpi=800)#, bbox_inches='tight', dpi=400)
        plt.show()


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

    reactions = ['rxn00069_d0' 'rxn00102_d0' 'rxn00121_d0' 'rxn00148_d0' 'rxn00151_d0'
                 'rxn00154_c0' 'rxn00154_d0' 'rxn00161_d0' 'rxn00175_d0' 'rxn00179_d0'
                 'rxn00187_d0' 'rxn00191_d0' 'rxn00192_d0' 'rxn00211_d0' 'rxn00248_d0'
                 'rxn00257_c0' 'rxn00260_d0' 'rxn00272_d0' 'rxn00275_d0' 'rxn00300_d0'
                 'rxn00330_c0' 'rxn00333_d0' 'rxn00337_d0' 'rxn00361_d0' 'rxn00416_d0'
                 'rxn00493_d0' 'rxn00495_c0' 'rxn00499_c0' 'rxn00527_d0' 'rxn00533_d0'
                 'rxn00720_d0' 'rxn00737_d0' 'rxn00770_d0' 'rxn00781_d0' 'rxn00790_d0'
                 'rxn00830_d0' 'rxn00899_d0' 'rxn00947_d0' 'rxn00974_d0' 'rxn01000_d0'
                 'rxn01007_d0' 'rxn01069_d0' 'rxn01101_d0' 'rxn01102_d0' 'rxn01111_d0'
                 'rxn01200_d0' 'rxn01207_d0' 'rxn01257_d0' 'rxn01258_d0' 'rxn01332_d0'
                 'rxn01334_d0' 'rxn01476_d0' 'rxn01643_d0' 'rxn01827_d0' 'rxn01975_d0'
                 'rxn02303_d0' 'rxn02373_d0' 'rxn02507_d0' 'rxn02895_d0' 'rxn03084_d0'
                 'rxn03891_d0' 'rxn03892_d0' 'rxn03909_d0' 'rxn03983_d0' 'rxn04954_c0'
                 'rxn05040_d0' 'rxn05287_d0' 'rxn05324_d0' 'rxn05325_d0' 'rxn05326_d0'
                 'rxn05337_d0' 'rxn05338_d0' 'rxn05340_d0' 'rxn05342_d0' 'rxn05343_d0'
                 'rxn05345_d0' 'rxn05346_d0' 'rxn05348_d0' 'rxn05350_d0' 'rxn05736_d0'
                 'rxn07579_d0' 'rxn08392_d0' 'rxn08436_d0' 'rxn09444_d0' 'rxn09449_d0'
                 'rxn09453_d0' 'rxn11700_d0' 'rxn12282_d0' 'rxn13975_d0' 'rxn13995_d0'
                 'rxn14004_d0' 'rxn14012_d0' 'rxn14059_d0' 'rxn14102_d0' 'rxn14156_d0'
                 'rxn14183_d0' 'rxn14205_d0' 'rxn14248_d0' 'rxn14278_d0' 'rxn14308_d0'
                 'rxn14347_d0' 'rxn15069_d0' 'rxn15116_d0' 'rxn15271_d0' 'rxn15280_d0'
                 'rxn15435_d0' 'rxn15493_d0' 'rxn16426_d0' 'rxn16427_d0' 'rxn16428_d0'
                 'rxn17469_d0' 'rxn17744_d0' 'rxn17803_d0' 'rxn19071_d0' 'rxn19240_d0'
                 'rxn19241_d0' 'rxn19242_d0' 'rxn19246_d0' 'rxn19253_d0' 'rxn19343_d0'
                 'rxn19345_d0' 'rxn19828_d0' 'rxn19846_d0' 'rxn20632_y0' 'rxn24310_d0'
                 'rxn24330_d0' 'rxn25637_d0' 'rxn25641_d0' 'rxn25647_d0' 'rxn25649_d0'
                 'rxn25661_d0' 'rxn25716_d0' 'rxn25718_d0' 'rxn25743_d0' 'rxn25747_d0'
                 'rxn25750_d0' 'rxn25762_d0' 'rxn27029_d0' 'rxn27030_d0' 'rxn27069_d0'
                 'rxn27070_d0' 'rxn27071_d0' 'rxn27072_d0' 'rxn27073_d0' 'rxn27266_d0'
                 'rxn27497_d0' 'rxn27927_d0' 'rxn28487_d0' 'rxn31768_d0' 'rxn37324_d0'
                 'rxn37610_d0' 'rxn37651_d0' 'rxn37767_d0' 'rxn37768_d0' 'rxn44318_d0']

    dup_rxns = []
    for rxn in reactions:
        dup_rxns += [rxn, rxn+"_f", rxn+"_r"]
    reactions = dup_rxns

    sig_df = pa.read_csv(sig_reactions, sep='\t')
    print(sig_df.shape)
    sig_df = sig_df[(sig_df['day']==time_stamp) & (sig_df['class'].isin(['Central Carbon', 'Amino acids']))]
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

    wide_pred_df = wide_pred_df.sort_values(by=['Pred_Control'])

    colmn =  'role' #'role'
    clms = []
    cmn_role = []
    for trmt in treatments:
        df = pa.DataFrame()
        for tmt in [trmt, 'Control']:
            df1 = wide_pred_df[[colmn, 'Pred_'+tmt]]
            df1.rename(columns={'Pred_'+tmt: 'Pred'}, inplace=True)
            df1 = df1[df1['Pred'] > 0]
            df1['trmt'] = tmt
            df1['Pred'] = np.log(df1['Pred'])
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
        fig = px.line_polar(df, r='Pred', color='trmt', theta=colmn, line_close=True,
                            labels={'trmt': ''}, title=title)
        fig.update_traces(fill='toself')
        fig.update_layout(
            font=dict(
                family="Arial",
                size=16
            )
        )
        fname = f"/Users/sea/Projects/AMN/omic_amn_mm/Result/sig_rxn_radar_{spc}_{genotype}_{time_stamp}_{trmt}.png"
        pio.write_image(fig, fname, scale=6, width=1000, height=1000)
        fig.show()
    # print(abc)
    cmn_role = []
    df = pa.DataFrame()
    for trmt in treatments:
        df1 = wide_pred_df[[colmn, 'Pred_'+trmt]]
        df1.rename(columns={'Pred_'+trmt: 'Pred'}, inplace=True)
        df1 = df1[df1['Pred'] > 0]
        df1['trmt'] = trmt
        df1['Pred'] = np.log(df1['Pred'])
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
    fig = px.line_polar(df, r='Pred', color='trmt', theta=colmn, line_close=True,
                        labels={'trmt': ''}, title=title)
    fig.update_traces(fill='toself')
    fig.update_layout(
        font=dict(
            family="Arial",
            size=16
        )
    )
    fname = f"/Users/sea/Projects/AMN/omic_amn_mm/Result/sig_rxn_radar_{spc}_{genotype}_{time_stamp}_all.png"
    pio.write_image(fig, fname, scale=6, width=1000, height=1000)
    fig.show()

if __name__ == '__main__':
    calculate_C = True
    relab = False

    treatments = ["Control", "FeLim", "FeEX", "ZnLim", "ZnEx"]
    msrs = ['Pred', 'FBA', 'Vbf', 'RES']
    control_id = "Control"

    spc = 'Sorghum'
    projCols = ['tissue', 'treatment', 'time_stamp']
    tissue = 'Leaf'
    genotype = 'Leaf'
    time_stamp = '21d'

    # treatments = ["control", "cold"]
    # projCols = ['Treatment', 'Timestamp']
    # spc = 'athaliana'
    # time_stamp = 'ZT9'
    # tissue = ''
    # genotype = 'C24'

    model_path = "/Users/sea/Projects/QPSI_project/Enzyme_Abundance/data/metabolic_models/plastidial_models/ortho_jun20_models/sbicolor_3.1.1_plastid_Thylakoid_Reconstruction_ComplexFix_070224_noADP_duplicated.xml"

    more = "_noBioLimit_noADP"
    # more = "_noBioLimit_newError"
    # more = "_decay5" # SV_penalty = 10
    # more = "_dr5_lr5" # SV_penalty = 10
    # more = "_noBioLimit" # SV_penalty = 10
    prediction_file = f"/Users/sea/Projects/AMN/omic_amn_mm/Result/{spc}_{genotype}_{time_stamp}_complexFix{more}_V_rxn.tsv"

    # kapp_vbf_path = f"/Users/sea/Projects/AMN/omic_amn_mm/Dataset_input/{spc}_complexFix_{genotype}_{time_stamp}_Vbf_kapp_maxCtrl_mixRelab.csv"
    kapp_vbf_path = f"/Users/sea/Projects/AMN/omic_amn_mm/Dataset_input/{spc}_complexFix_{genotype}_{time_stamp}_Vbf_kapp_maxCtrl_mixedRelab_noADP.csv"
    # Sorghum_complexFix_Leaf_21d_Vbf_kapp_maxCtrl_mixedRelab_noADP

    scores_path = f"/Users/sea/Projects/QPSI_project/Enzyme_Abundance/integration_results/reaction_scores_binding_Jul2/plastidial_model/{spc}_objective_abundance_Control.tsv"
    # scores_path = f"/Users/sea/Projects/QPSI_project/Enzyme_Abundance/integration_results/secMetResults/athaliana_objective_abundance_control_{genotype}.tsv"

    sig_reactions = f"/Users/sea/Projects/QPSI_project/Enzyme_Abundance/src/util/sig_reactions_{genotype}.tsv"

    s_matrix_path = f"/Users/sea/Projects/AMN/omic_amn_mm/s_matrix_plastid_T.csv"

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
    s_matrix_df = pa.read_csv(s_matrix_path)
    s_matrix_df.rename(columns={'Unnamed: 0': 'rxn_ID'}, inplace=True)
    s_matrix_df = s_matrix_df.set_index('rxn_ID')
    s_matrix_df.sort_index(inplace=True)
    for trmt in treatments:
        trmt_pred = pred_df[['rxn_ID', trmt]]
        trmt_pred = trmt_pred.set_index('rxn_ID')
        trmt_pred.sort_index(inplace=True)

        trmt_matrix = s_matrix_df.apply(lambda x : np.asarray(x) * np.asarray(trmt_pred[trmt]))
        trmt_matrix = trmt_matrix.fillna(0)

        trmt_matrix.to_csv(f"{trmt}_s_matrix.tsv", sep='\t')
        trmt_matrix = trmt_matrix[['cpd00810_c0', 'cpd00070_d0', 'cpd00242_d0', 'cpd00247_c0', 'cpd00103_c0', 'cpd00011_d0']]
        trmt_matrix = trmt_matrix[(np.abs(trmt_matrix['cpd00810_c0'])>0) | (np.abs(trmt_matrix['cpd00070_d0'])>0) | (np.abs(trmt_matrix['cpd00242_d0'])>0)]
        trmt_matrix.to_csv(f"{trmt}_s_matrix_subset.tsv", sep='\t')


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
        scores_path = f"/Users/sea/Projects/QPSI_project/Enzyme_Abundance/integration_results/reaction_scores_binding_Dec6/{spc}_relab_rxn_scores_tmm.csv"
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
    pred_df.to_csv(prediction_file.replace(".tsv", "_fba_Vbf_RES.tsv"), sep="\t", index=False)
    # print(abc)
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
    if True:
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
    # fba_fluxes_df.to_csv("/Users/sea/Projects/AMN/omic_amn/Result/noBiomassMax_constraints_3.5_noEXcpd00727_fixedRev/fba_results_nonZero.csv")
