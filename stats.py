#!/usr/bin/env python3
"""
Stats and plots for paper

Copyright 2025, J.S. Gómez-Cañón
Licensed under GNU AFFERO GENERAL PUBLIC LICENSE
"""

import pandas as pd
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
# for var in df.variant.unique():
#     this_df = df[df.variant == var]
#     print('*'*30)
#     print(f'Variant {var} has {this_df.shape[0]} unique annotations!')
#     df_pivot = this_df.pivot_table(index="raterID", columns="stimulusNumber", values=["valence", "arousal", "dominance"])
#     for mod in ['arousal', 'valence', 'dominance']:
#         # krippendorff
#         alp = alpha(reliability_data=df_pivot[mod].values, level_of_measurement='interval')
#         print(f'Krippendorff alpha for {mod}: {alp:.02f}')

#         # icc
#         df_clean = this_df[['stimulusNumber', 'raterID', mod]].dropna()
#         model = ols(f'{mod} ~ C(stimulusNumber)', data=df_clean).fit()
#         anova_table = sm.stats.anova_lm(model, typ=2)
#         ms_between = anova_table['sum_sq']['C(stimulusNumber)'] / anova_table['df']['C(stimulusNumber)']
#         ms_within = anova_table['sum_sq']['Residual'] / anova_table['df']['Residual']
#         n_raters = df_clean.groupby('stimulusNumber')['raterID'].nunique().mean()

#         # Compute ICC(1,1)
#         icc = (ms_between - ms_within) / (ms_between + (n_raters - 1) * ms_within)
#         print(f'Estimated ICC(1,1) for {mod} ratings: {icc:.2f}')

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

# Stimulus level comparison
# This reduces the data to one comparison per stimulus, which reflects the real sample size and avoids pseudo-replication.
speech_agg = speech_df.groupby('new_id', as_index=False)[['valence', 'arousal', 'dominance']].mean() 
tegg_agg = tegg_df.groupby('new_id', as_index=False)[['valence', 'arousal', 'dominance']].mean() 
merged_df = pd.merge(speech_agg, tegg_agg, on='new_id', suffixes=('_speech', '_tegg')) 
merged_df = merged_df.dropna(subset=['valence_speech', 'valence_tegg', 'arousal_speech', 'arousal_tegg', 'dominance_speech', 'dominance_tegg'])
                            
valence_corr, _ = pearsonr(merged_df['valence_speech'], merged_df['valence_tegg']) 
arousal_corr, _ = pearsonr(merged_df['arousal_speech'], merged_df['arousal_tegg']) 
dominance_corr, _ = pearsonr(merged_df['dominance_speech'], merged_df['dominance_tegg'])  

print('*'*30)
print('T-tests on means per stimulus level')
print(f"Pearson correlation for Valence: {valence_corr:.2f}") 
print(f"Pearson correlation for Arousal: {arousal_corr:.2f}") 
print(f"Pearson correlation for Dominance: {dominance_corr:.2f}")  
# Perform paired-sample t-tests 
valence_ttest = ttest_rel(merged_df['valence_speech'], merged_df['valence_tegg']) 
arousal_ttest = ttest_rel(merged_df['arousal_speech'], merged_df['arousal_tegg']) 
dominance_ttest = ttest_rel(merged_df['dominance_speech'], merged_df['dominance_tegg']) 

# Adjust for multiple comparisons (Bonferroni correction) 
alpha = 0.05 
corrected_alpha = alpha / 3 
df_n = len(merged_df)  
print(f"\nPaired-sample t-tests (df = {df_n - 1}), Bonferroni-corrected α = {corrected_alpha:.4f}") 
print(f"Valence: t = {valence_ttest.statistic:.2f}, p = {valence_ttest.pvalue:.4f}, significant: {valence_ttest.pvalue < corrected_alpha}") 
print(f"Arousal: t = {arousal_ttest.statistic:.2f}, p = {arousal_ttest.pvalue:.4f}, significant: {arousal_ttest.pvalue < corrected_alpha}") 
print(f"Dominance: t = {dominance_ttest.statistic:.2f}, p = {dominance_ttest.pvalue:.4f}, significant: {dominance_ttest.pvalue < corrected_alpha}")
del df

###########################
# 4) Plots: on summarized data for comparison
print('-'*50)
df = sum_df.copy()
sns.set_style("whitegrid")

# # List of numerical columns to analyze
metrics = ["valence", "arousal", "dominance"]

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
    plt.savefig(output_filename, format="pdf")  # Save as PDF
    plt.close()  # Close the figure to free memory

# Save plots by Affect Name
plot_and_save_grouped_violinplots(df, "affectName", metrics, "figs/01_affect_name_violin.pdf")


def plot_and_save_double_grouped_violinplots_with_stats(df, metrics, output_filename, variable):
    fig, axes = plt.subplots(len(metrics), 1, figsize=(5.5, 2 * len(metrics)), sharex=True)

    # Standardize categorical values
    df["sex"] = df["speakerSex"].str.strip().str.lower()
    df["variant"] = df["variant"].str.strip().str.lower()
    sex_values = set(df["speakerSex"].unique())
    affect_values = set(df["affectName"].unique())

    handles, labels = None, None

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Only allow legend on the first plot so we can extract it
        show_legend = i == 2

        sns.violinplot(
            x="affectName", y=metric, hue=variable, data=df, ax=ax, 
            split=True, inner=None, legend=show_legend, width=1.3
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
            annotator.configure(test="t-test_ind", text_format="star", loc="outside")
            annotator.apply_and_annotate()

        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend_.remove()

        ax.set_ylim(-1, 1)
        ax.set_xlabel("") 

        # Delay setting xticklabels until ticks are defined
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # <- add space below

    if handles and labels:
        label_map = {
            "male": "Male",
            "female": "Female",
            "speech": "Speech",
            "tegg": "tEGG",
        }
        new_labels = [label_map.get(label.lower(), label) for label in labels]
        fig.legend(
            handles, new_labels,
            title={"speakerSex": "Sex", "variant": "Modality"}.get(variable),
            loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.02)
        )

    plt.savefig(output_filename, format="pdf")
    plt.close()


def plot_and_save_double_grouped_violinplots_with_stats_old(df, metrics, output_filename, variable):
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 1 * len(metrics)), sharex=True)

    # Standardize categorical values to avoid formatting mismatches
    df["sex"] = df["speakerSex"].str.strip().str.lower()
    df["variant"] = df["variant"].str.strip().str.lower()

    # Unique values in columns
    sex_values = set(df["speakerSex"].unique())
    affect_values = set(df["affectName"].unique())

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Violin plot with correct `hue`
        sns.violinplot(x="affectName", y=metric, hue=variable, data=df, ax=ax, 
                       split=True, inner="box", legend=False)

        # # Scatter plot to show individual points
        # sns.stripplot(x="affectName", y=metric, hue=variable, data=df, ax=ax, 
        #               dodge=True, jitter=True, alpha=0.5, size=3, marker="o", legend=False, palette="dark")

        # ax.set_title(f"{metric} by Affect Name and {variable}")
        # ax.set_xticks(range(len(ax.get_xticklabels())))
        
        ax.set_ylim(-1, 1)

        # # Generate valid pairs for annotation
        # if variable == 'Sex':
        #     # Check if both "male" and "female" exist for each Affect Name
        #     pairs = [(("male", affect), ("female", affect)) 
        #                 for affect in affect_values 
        #                 if {"male", "female"}.issubset(set(df[df["Affect Name"] == affect]["Sex"].unique()))]

        # elif variable == "Variant":
        #     # Check if both "speech" and "tegg" exist for each Affect Name
        #     pairs = [(("speech", affect), ("tEGG", affect)) 
        #                 for affect in affect_values 
        #                 if {"speech", "tEGG"}.issubset(set(df[df["Affect Name"] == affect]["Variant"].unique()))]

        # # Add statistical significance annotations
        # if pairs:
        #     pdb.set_trace()
        #     annotator = Annotator(ax, pairs, data=df, x="Affect Name", y=metric, hue=variable)
        #     annotator.configure(test="t-test_ind", text_format="star", loc="outside")
        #     annotator.apply_and_annotate()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend(title=variable, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_filename, format="pdf")  # Save as PDF
    plt.close()  # Close the figure to free memory

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
plt.savefig("figs/03_correlation_heatmap.pdf", format="pdf")  # Save as PDF
plt.close()

# distribution analysis
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, metric in enumerate(["valence", "arousal", "dominance"]):
    sns.kdeplot(df[metric], ax=axes[i], fill=True, color="blue")
    axes[i].set_title(f"Distribution of {metric}")

plt.tight_layout()
plt.savefig("figs/04_distribution_analysis.pdf", format="pdf")  # Save as PDF
plt.close()

# affect name and dimensions
sns.pairplot(df, vars=["valence", "arousal", "dominance"], hue="affectName", palette="coolwarm")
plt.suptitle("Affect Categories in Valence-Arousal-Dominance Space", y=1.02)
plt.savefig("figs/05_affect_name_pairplot.pdf", format="pdf")  # Save as PDF
plt.close()


# scatter plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=df, x="valence", y="arousal", hue="affectName", style="variant", palette="viridis", ax=ax)

ax.axvline(0, color='gray', linestyle='--')
ax.axhline(0, color='gray', linestyle='--')

ax.set_title("Valence vs. Arousal with Affect Name Labels")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig("figs/06_valence_arousal_quadrants.pdf", format="pdf", bbox_inches='tight')
plt.close()

