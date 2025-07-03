#!/usr/bin/env python3
"""
Stats and plots for paper

Copyright 2025, J.S. Gómez-Cañón
Licensed under GNU AFFERO GENERAL PUBLIC LICENSE
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from statannotations.Annotator import Annotator
import statsmodels.api as sm
from statsmodels.formula.api import ols
from krippendorff import alpha
from scipy.stats import pearsonr, ttest_rel
import pdb

save_plots = 'pdf' # or 'svg'
###########################
# 1) Data exploration
# load full data
print('-'*50)
df = pd.read_csv('annotations/AllData.csv')
ratings_per_rater = df.groupby('raterID')['stimulusNumber'].nunique()
avg_ratings_per_rater = ratings_per_rater.mean()
min_ratings_per_rater = ratings_per_rater.min()
max_ratings_per_rater = ratings_per_rater.max()

ratings_per_stimulus = df.groupby('stimulusNumber')['raterID'].nunique()
avg_ratings_per_stimulus = ratings_per_stimulus.mean()
min_ratings_per_stimulus = ratings_per_stimulus.min()
max_ratings_per_stimulus = ratings_per_stimulus.max()

print(f"Data were collected across six different rating studies from a total of {df['raterID'].nunique()} unique participants, "
    f"with each participant rating an average of {avg_ratings_per_rater:.1f} stimuli "
    f"(range {min_ratings_per_rater}-{max_ratings_per_rater}).")

print(f"The average number of ratings per stimulus was {avg_ratings_per_stimulus:.1f} "
    f"(range {min_ratings_per_stimulus}-{max_ratings_per_stimulus}).")
sex = [df[df['speakerNumber'] == _].speakerSex.values[0] for _ in df['speakerNumber'].unique()]
print('Sex from speakers:\n', Counter(sex))

print(df[['valence', 'arousal', 'dominance']].describe())

# calculate summary data from AllData.csv
affective_cols = ['valence', 'arousal', 'dominance']
agg_df = df.groupby('stimulusNumber')[affective_cols].agg(['mean', 'std']).reset_index()
agg_df.columns = ['stimulusNumber'] + [f"{dim}_{stat}" for dim, stat in agg_df.columns[1:]]
meta_cols = ['stimulusNumber', 'speakerNumber', 'affectNumber', 'affectName', 'speakerSex', 'variant']
meta_df = df[meta_cols].drop_duplicates(subset='stimulusNumber')
sum_df = pd.merge(meta_df, agg_df, on='stimulusNumber')
sum_df = sum_df.rename(columns={'arousal_mean': 'arousal',
                                'valence_mean': 'valence',
                                'dominance_mean': 'dominance'})

# load summary data
# sum_df = pd.read_csv('annotations/SummaryData.csv')
# sum_df = sum_df.rename(columns={'Stimulus Number': 'stimulusNumber',
#                         'Speaker Number': 'speakerNumber',
#                         'Affect Number': 'affectNumber',
#                         'Affect Name': 'affectName',
#                         'Sex': 'speakerSex',
#                         'Variant': 'variant',
#                         'meanValence': 'valence',
#                         'meanArousal': 'arousal',
#                         'meanDominance': 'dominance'})

# sum_df['new_id'] = sum_df.apply(lambda row: f"speaker_{row['speakerNumber']}-emo_{row['affectName']}", axis=1)

###########################
# 2) Reliability
# print('-'*50)
# import statsmodels.formula.api as smf

# for var in df.variant.unique():
#     this_df = df[df.variant == var]
#     print('*'*30)
#     print(f'Variant {var} has {this_df.shape[0]} unique annotations!')
#     df_pivot = this_df.pivot_table(index="raterID", columns="stimulusNumber", values=["valence", "arousal", "dominance"])
#     for mod in ['arousal', 'valence', 'dominance']:
#         # krippendorff

#         alp = alpha(reliability_data=df_pivot[mod].values, level_of_measurement='interval')
#         print(f'Krippendorff alpha for {mod}: {alp:.3f}')

#         # # icc
#         df_clean = this_df[['stimulusNumber', 'raterID', mod]].dropna()
#         model = smf.mixedlm(f"{mod} ~ 1", data=df_clean, groups="stimulusNumber", re_formula="~1")
#         result = model.fit()
#         var_subject = result.cov_re.iloc[0, 0]
#         var_resid = result.scale

#         icc = var_subject / (var_subject + var_resid)
#         print(f'Mixed-effects ICC for {mod}: {icc:.3f}')

###########################
# 3) Stats
print('-'*50)

# Create row for comparisons
df['new_id'] = df.apply(lambda row: f"speaker_{row['speakerNumber']}-emo_{row['affectName']}", axis=1)

# Separate variants
speech_df = df[df.variant == 'speech']
tegg_df = df[df.variant == 'tEGG']

# Keep only relevant columns for merging
speech_ratings = speech_df[['new_id', 'raterID', 'valence', 'arousal', 'dominance']].copy()
tegg_ratings = tegg_df[['new_id', 'raterID', 'valence', 'arousal', 'dominance']].copy()
speech_ratings = speech_ratings.rename(columns={'valence': 'valence_speech', 'arousal': 'arousal_speech', 'dominance': 'dominance_speech'})
tegg_ratings = tegg_ratings.rename(columns={'valence': 'valence_tegg', 'arousal': 'arousal_tegg', 'dominance': 'dominance_tegg'})

# Merge on new_id and raterID to keep same ratings per stimulus/rater
merged_df = pd.merge(speech_ratings, tegg_ratings, on=['new_id', 'raterID'])
merged_df = merged_df.dropna()
valence_corr, _ = pearsonr(merged_df['valence_speech'], merged_df['valence_tegg'])
arousal_corr, _ = pearsonr(merged_df['arousal_speech'], merged_df['arousal_tegg'])
dominance_corr, _ = pearsonr(merged_df['dominance_speech'], merged_df['dominance_tegg'])

# Paired t-tests
valence_ttest = ttest_rel(merged_df['valence_speech'], merged_df['valence_tegg'])
arousal_ttest = ttest_rel(merged_df['arousal_speech'], merged_df['arousal_tegg'])
dominance_ttest = ttest_rel(merged_df['dominance_speech'], merged_df['dominance_tegg'])

def compute_cohens_d_and_r2(x, y):
    # from https://en.wikipedia.org/wiki/Effect_size
    diff = x - y
    mean_diff = diff.mean()
    std_diff = diff.std(ddof=1)
    d = mean_diff / std_diff
    r = d / (d**2 + 4)**0.5
    r2 = r**2

    # print(f"Mean diff: {mean_diff:.3f}")
    # print(f"Cohen's d: {d:.3f}")
    # print(f"Variance explained (r²): {r2:.3f}")
    return d, r2

# Compute effect sizes
valence_d, val_r = compute_cohens_d_and_r2(merged_df['valence_speech'], merged_df['valence_tegg'])
arousal_d, aro_r = compute_cohens_d_and_r2(merged_df['arousal_speech'], merged_df['arousal_tegg'])
dominance_d, dom_r = compute_cohens_d_and_r2(merged_df['dominance_speech'], merged_df['dominance_tegg'])

# Bonferroni correction
alpha = 0.05
corrected_alpha = alpha / 3
df_n = len(merged_df)

# print results
print('*'*30)
print(f"Matched ratings: {df_n}")
print('T-tests on means on full ratings')
print(f"Pearson correlation for Valence: {valence_corr:.2f}") 
print(f"Pearson correlation for Arousal: {arousal_corr:.2f}") 
print(f"Pearson correlation for Dominance: {dominance_corr:.2f}")  

print(f"\nPaired-sample t-tests (df = {df_n - 1}), Bonferroni-corrected α = {corrected_alpha:.4f}")
print(f"Valence: t = {valence_ttest.statistic:.2f}, p = {valence_ttest.pvalue:.4f}, significant: {valence_ttest.pvalue < corrected_alpha}")
print(f"Arousal: t = {arousal_ttest.statistic:.2f}, p = {arousal_ttest.pvalue:.4f}, significant: {arousal_ttest.pvalue < corrected_alpha}")
print(f"Dominance: t = {dominance_ttest.statistic:.2f}, p = {dominance_ttest.pvalue:.4f}, significant: {dominance_ttest.pvalue < corrected_alpha}")

print('\nEffect sizes:')
print(f"Cohen's d (valence): {valence_d:.2f} with variance explained r²: {val_r:.2f}")
print(f"Cohen's d (arousal): {arousal_d:.2f} with variance explained r²: {aro_r:.2f}")
print(f"Cohen's d (dominance): {dominance_d:.2f} with variance explained r²: {dom_r:.2f}")

# # Stimulus level comparison
# # This reduces the data to one comparison per stimulus, which reflects the real sample size and avoids pseudo-replication.
# speech_agg = speech_df.groupby('new_id', as_index=False)[['valence', 'arousal', 'dominance']].mean() 
# tegg_agg = tegg_df.groupby('new_id', as_index=False)[['valence', 'arousal', 'dominance']].mean() 
# merged_df = pd.merge(speech_agg, tegg_agg, on='new_id', suffixes=('_speech', '_tegg')) 
# merged_df = merged_df.dropna(subset=['valence_speech', 'valence_tegg', 'arousal_speech', 'arousal_tegg', 'dominance_speech', 'dominance_tegg'])
                            
# valence_corr, _ = pearsonr(merged_df['valence_speech'], merged_df['valence_tegg']) 
# arousal_corr, _ = pearsonr(merged_df['arousal_speech'], merged_df['arousal_tegg']) 
# dominance_corr, _ = pearsonr(merged_df['dominance_speech'], merged_df['dominance_tegg'])  

# # Compute effect sizes
# valence_d, val_r = compute_cohens_d_and_r2(merged_df['valence_speech'], merged_df['valence_tegg'])
# arousal_d, aro_r = compute_cohens_d_and_r2(merged_df['arousal_speech'], merged_df['arousal_tegg'])
# dominance_d, dom_r = compute_cohens_d_and_r2(merged_df['dominance_speech'], merged_df['dominance_tegg'])

# # print('T-tests on means per stimulus level')
# # print(f"Pearson correlation for Valence: {valence_corr:.2f}") 
# # print(f"Pearson correlation for Arousal: {arousal_corr:.2f}") 
# # print(f"Pearson correlation for Dominance: {dominance_corr:.2f}")  
# # Perform paired-sample t-tests 
# valence_ttest = ttest_rel(merged_df['valence_speech'], merged_df['valence_tegg']) 
# arousal_ttest = ttest_rel(merged_df['arousal_speech'], merged_df['arousal_tegg']) 
# dominance_ttest = ttest_rel(merged_df['dominance_speech'], merged_df['dominance_tegg']) 

# # Adjust for multiple comparisons (Bonferroni correction) 
# alpha = 0.05 
# corrected_alpha = alpha / 3 
# df_n = len(merged_df)  
# # print(f"\nPaired-sample t-tests (df = {df_n - 1}), Bonferroni-corrected α = {corrected_alpha:.4f}") 
# # print(f"Valence: t = {valence_ttest.statistic:.2f}, p = {valence_ttest.pvalue:.4f}, significant: {valence_ttest.pvalue < corrected_alpha}") 
# # print(f"Arousal: t = {arousal_ttest.statistic:.2f}, p = {arousal_ttest.pvalue:.4f}, significant: {arousal_ttest.pvalue < corrected_alpha}") 
# # print(f"Dominance: t = {dominance_ttest.statistic:.2f}, p = {dominance_ttest.pvalue:.4f}, significant: {dominance_ttest.pvalue < corrected_alpha}")
# print('*'*30)
# print(f"Matched ratings: {df_n}")
# print('T-tests on means per stimulus')
# print(f"""
# For valence, ratings in the speech condition were higher than in the tEGG condition (M_diff = {merged_df['valence_speech'].mean() - merged_df['valence_tegg'].mean():.2f}, 
# t({len(merged_df)-1}) = {valence_ttest.statistic:.2f}, p = {valence_ttest.pvalue:.3f}), 
# with a Cohen’s d of {valence_d:.2f} and r² = {val_r:.2f}, indicating a small effect.

# For arousal, ratings in the speech condition were higher than in the tEGG condition (M_diff = {merged_df['arousal_speech'].mean() - merged_df['arousal_tegg'].mean():.2f}, 
# t({len(merged_df)-1}) = {arousal_ttest.statistic:.2f}, p = {arousal_ttest.pvalue:.3f}), 
# yielding a Cohen’s d of {arousal_d:.2f} and r² = {aro_r:.2f}, indicating a small effect.

# For dominance, ratings in the speech condition were higher than in the tEGG condition (M_diff = {merged_df['dominance_speech'].mean() - merged_df['dominance_tegg'].mean():.2f}, 
# t({len(merged_df)-1}) = {dominance_ttest.statistic:.2f}, p = {dominance_ttest.pvalue:.3f}), 
# with a Cohen’s d of {dominance_d:.2f} and r² = {dom_r:.2f}, indicating a small effect.
# """)

# del df

###########################
# 4) Plots: on summarized data for comparison
print('-'*50)
df = sum_df.copy()
sns.set_style("whitegrid")

# # List of numerical columns to analyze
metrics = ["arousal", "valence", "dominance"]

# general data
def plot_and_save_grouped_violinplots(df, group_by, metrics, output_filename):
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 5 * len(metrics)))

    for i, metric in enumerate(metrics):
        this_df = df.dropna(subset=["affectName", "speakerSex", metric])
        sns.violinplot(x=group_by, y=metric, data=this_df, ax=axes[i], inner="box")
        # sns.stripplot(x="affectName", y=metric, hue=group_by, data=this_df, ax=axes[i], 
        #               dodge=True, jitter=True, alpha=0.5, size=3, palette='dark:black', marker="o")
        axes[i].set_title(f"{metric} by {group_by}")
        axes[i].set_xticks(range(len(axes[i].get_xticklabels())))
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
        axes[i].set_ylim(-1, 1)
    
    plt.tight_layout()
    plt.savefig(output_filename, format="pdf")
    plt.close()  # Close the figure to free memory


plot_and_save_grouped_violinplots(df, "affectName", metrics, "figs/01_affect_name_violin.pdf")


def plot_and_save_double_grouped_violinplots_with_stats(df, metrics, output_filename, variable):
    fig, axes = plt.subplots(len(metrics), 1, figsize=(5.5, 1.5 * len(metrics)), sharex=True)

    # Standardize categorical values
    df["sex"] = df["speakerSex"].str.strip().str.lower()
    df["variant"] = df["variant"].str.strip().str.lower()
    sex_values = set(df["speakerSex"].unique())
    affect_values = set(df["affectName"].unique())

    handles, labels = None, None

    affect_order = (
        df.groupby("affectName")["arousal"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    for i, metric in enumerate(metrics):
        ax = axes[i]
        show_legend = i == 2

        sns.violinplot(
            x="affectName", y=metric, hue=variable, data=df, ax=ax, 
            split=True, inner=None, legend=show_legend, linewidth=0.5,
            order=affect_order
        )
        # Scatter plot to show individual points
        # sns.stripplot(x="affectName", y=metric, hue=variable, data=df, ax=ax, 
        #               dodge=True, jitter=True, alpha=0.5, size=3, marker="o", legend=False, palette="dark")

        if variable == 'speakerSex':
            pairs = [
                ((affect, "male"), (affect, "female"))
                for affect in affect_values
                if {"male", "female"}.issubset(set(df[df["affectName"] == affect]["speakerSex"].str.lower().unique()))
            ]

        elif variable == "variant":
            pairs = [
                ((affect, "speech"), (affect, "tegg"))
                for affect in affect_values
                if {"speech", "tegg"}.issubset(set(df[df["affectName"] == affect]["variant"].str.lower().unique()))
            ]
       
        # Add statistical significance annotations
        if pairs:
            annotator = Annotator(ax, pairs, data=df, x="affectName", y=metric, hue=variable)
            annotator.configure(test="t-test_ind", 
                                text_format="star", 
                                loc="outside", 
                                comparisons_correction="holm", 
                                fontsize=9,
                                line_width=0.5)
            annotator.apply_and_annotate()

        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend_.remove()

        ax.set_ylim(-1, 1)
        ax.set_yticks(np.arange(-1, 1.1, 0.5))
        ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_xlabel("") 

        short_labels = ['enrag.', 'fright.', 'thrill.', 'joyful', 
                        'annoy.', 'confid.', 'happy', 'worried', 
                        'satis.', 'contem.', 'mellow', 'serene', 
                        'miser.', 'tranq.', 'dark', 'depre.']
        ax.set_xticklabels(short_labels, rotation=45, fontsize=9)

    plt.tight_layout(rect=[0, 0.05, 1, 1]) 
    # plt.tight_layout(rect=[0, 0, 1, 0.90])

    if handles and labels:
        label_map = {
            "male": "Male",
            "female": "Female",
            "speech": "speech",
            "tegg": "tEGG",
        }
        new_labels = [label_map.get(label.lower(), label) for label in labels]
        labels = ['enrag.', 'fright.', 'thrill.', 'joyful', 
        'annoy.', 'confid.', 'happy', 'worried', 
        'satis.', 'contem.', 'mellow', 'serene', 
        'miser.', 'tranq.', 'dark', 'depre.']
        fig.legend(
            handles, new_labels,
            title={"speakerSex": "Sex", "variant": "Modality"}.get(variable),
            loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.03)
        )
        # fig.legend(
        #     handles, new_labels,
        #     title={"speakerSex": "Sex", "variant": "Modality"}.get(variable),
        #     loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1)
        # )

    plt.savefig(output_filename, format="pdf", bbox_inches='tight', pad_inches=0.01)
    plt.close()


# Save violin plots with statistical significance and scatter points
plot_and_save_double_grouped_violinplots_with_stats(df, metrics, "figs/02_sex_affect_name_violin_stats.pdf", "speakerSex")
plot_and_save_double_grouped_violinplots_with_stats(df, metrics, "figs/02_variant_affect_name_violin.pdf", "variant")

# correlations
corr_df = df[["valence", "arousal", "dominance"]]
from scipy.stats import pearsonr
r_val, p_val = pearsonr(corr_df['arousal'].dropna(), corr_df['dominance'].dropna())
print(f"r = {r_val:.2f}, p = {p_val:.3g}")
corr_matrix = corr_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap: Valence, Arousal, and Dominance")
plt.savefig("figs/03_correlation_heatmap.pdf", format="pdf")
plt.close()

# distribution analysis
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, metric in enumerate(["valence", "arousal", "dominance"]):
    sns.kdeplot(df[metric], ax=axes[i], fill=True, color="blue")
    axes[i].set_title(f"Distribution of {metric}")

plt.tight_layout()
plt.savefig("figs/04_distribution_analysis.pdf", format="pdf") 
plt.close()

# affect name and dimensions
sns.pairplot(df, vars=["valence", "arousal", "dominance"], hue="affectName", palette="coolwarm")
plt.suptitle("Affect Categories in Valence-Arousal-Dominance Space", y=1.02)
plt.savefig("figs/05_affect_name_pairplot.pdf", format="pdf") 
plt.close()


# scatter plot
order = ['enraged', 'frightened', 'thrilled', 'joyful', 
        'annoyed', 'confident', 'happy', 'worried', 
        'satisfied', 'contemptuous', 'mellow', 'serene', 
        'miserable', 'tranquil', 'dark', 'depressed']
fig, ax = plt.subplots(figsize=(5, 3.2))
sns.scatterplot(data=df, x="valence", y="arousal", hue="affectName", style="variant", palette="viridis", ax=ax,
                hue_order=order, s=20)
ax.axvline(0, color='gray', linestyle='--')
ax.axhline(0, color='gray', linestyle='--')
# handles, labels = ax.get_legend_handles_labels()
# new_labels = [label[:5]+'.' if label in df['affectName'].unique() else label for label in labels]
# ax.legend(handles, new_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
# plt.savefig("figs/06_valence_arousal_quadrants.pdf", format="pdf", bbox_inches='tight', pad_inches=0.01)
# plt.close()


handles, labels = ax.get_legend_handles_labels()
affect_names = df['affectName'].unique().tolist()
affect_handles = []
affect_labels = []

modality_handles = []
modality_labels = []

labels = ['affectName', 'enraged', 'frightened', 'thrilled', 'joyful', 
        'annoyed', 'confident', 'happy', 'worried', 
        'satisf', 'contempt.', 'mellow', 'serene', 
        'miserable', 'tranquil', 'dark', 'depressed',
        'variant', 'speech', 'tEGG']

for handle, label in zip(handles, labels):
    if any(name.startswith(label.replace('.', '')) for name in affect_names):
        affect_handles.append(handle)
        affect_labels.append(label) 
    else:
        if label != 'affectName' and label != 'variant':
            modality_handles.append(handle)
            modality_labels.append(label)


from matplotlib.lines import Line2D
affect_title = Line2D([0], [0], linestyle='none', label='\u0332'.join('Affect') + '\u0332')
modality_title = Line2D([0], [0], linestyle='none', label='\u0332'.join('Modality') + '\u0332')
# affect_title = Line2D([0], [0], linestyle='none', label=r'$\bf{Affect}$')
# modality_title = Line2D([0], [0], linestyle='none', label=r'$\bf{Modality}$')

# # Build combined legend
# all_handles = [affect_title] + affect_handles + [modality_title] + modality_handles
# all_labels = ['Affect'] + affect_labels + ['Modality'] + modality_labels

# Combine everything into a single list
all_handles = [affect_title] + affect_handles + [modality_title] + modality_handles
all_labels = ['\u0332'+ '\u0332'.join('Affect')] + affect_labels + \
    ['\u0332' + '\u0332'.join('Modality')] + modality_labels

# Create the combined legend
ax.legend(
    all_handles,
    all_labels,
    loc='upper left',  # Align legend's upper left corner
    bbox_to_anchor=(1.0, 1.025),  # Place it just outside the top-right of the axes
    fontsize=8,
    frameon=True,
    labelspacing=0.35
)

plt.savefig("figs/06_valence_arousal_quadrants.pdf", format="pdf", bbox_inches='tight', pad_inches=0.01)
# plt.savefig("figs/06_valence_arousal_quadrants.svg", format="svg", bbox_inches='tight', pad_inches=0.01)
plt.close()