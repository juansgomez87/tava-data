#!/usr/bin/env python3
"""
Classification using logistic regression
Regression using LASSO for TAVA dataset

Copyright 2025, J.S. Gómez-Cañón
Licensed under GNU AFFERO GENERAL PUBLIC LICENSE
"""


import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from collections import Counter
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import ttest_rel


import os
import glob


import warnings
# warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

import pdb

class TavaClassifier():
    def __init__(self, df, feat_cols, audio, cv, pca_flag, plot_flag):
        self.seed = 1987
        self.audio = audio
        self.df = df[df.Variant == audio].copy()
        self.feature_columns = feat_cols
        self.cv = cv
        self.plot_flag = plot_flag

        if pca_flag:
            self.pca_reduction()
        else:
            self.X = self.df[self.feature_columns].values
            scaler = StandardScaler()
            self.X = scaler.fit_transform(self.X)

        self.y = self.df['Affect Name'].values

        # Calculate class weights
        self.class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.y),
            y=self.y
        )
        self.class_weight_dict = dict(zip(np.unique(self.y), self.class_weights))

        self.model = LogisticRegression(
            max_iter=2000,
            solver='lbfgs',
            class_weight=self.class_weight_dict
        )

        self.run_classification()
        
    def pca_reduction(self):
        print('PCA:')
        print(f'Number of features {len(self.feature_columns)}')
        print(f'Number of data instances {self.df.shape[0]}')
        self.X = self.df[self.feature_columns].values
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
    
        pca = PCA(n_components=0.95) 
        X_pca = pca.fit_transform(self.X)
        n_comp = X_pca.shape[1]
        print('-'*50)
        print(f'PCA reduced from {len(self.feature_columns)} to {n_comp} to keep 95% of variance!')

        self.feature_columns = [f'pca_{i}' for i in range(X_pca.shape[1])]
        df_pca = pd.DataFrame(X_pca, columns=self.feature_columns, index=self.df.index)
        self.df = pd.concat([self.df, df_pca], axis=1)
        self.X = self.df[self.feature_columns].values


    def run_classification(self):
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.seed)
        
        y_true_all, y_pred_all, y_score_all = [], [], []
        
        for train_idx, test_idx in skf.split(self.X, self.y):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            y_score = self.model.predict_proba(X_test)
            
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            y_score_all.extend(y_score)
        
        # Convert lists to numpy arrays
        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        y_score_all = np.array(y_score_all)
        
        # self.evaluate(y_true_all, y_pred_all, y_score_all)
        self.plot_stratified_splits()
        self.plot_roc_auc(y_true_all, y_score_all)
        self.compare_with_chance(y_true_all, y_pred_all)
        self.feature_importance()
        
    def plot_stratified_splits(self):
        labels, counts = np.unique(self.y, return_counts=True)
        plt.figure(figsize=(8, 4))
        sns.barplot(x=labels, y=counts, color='blue', alpha=0.7)
        plt.xticks(rotation=45)
        plt.xlabel('Classes')
        plt.ylabel('Count')
        plt.title('Class Distribution Before Splitting')
        plt.tight_layout()
        if self.plot_flag:
            plt.show()
        
    def plot_roc_auc(self, y_true, y_score):
        y_bin = label_binarize(y_true, classes=np.unique(y_true))
        n_classes = y_bin.shape[1]
        plt.figure(figsize=(8, 6))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        if self.plot_flag:
            plt.show()
        

    def compare_with_chance(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=self.model.classes_)
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.round(cm, 1)

        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model.classes_)
        disp.plot(
            ax=ax,
            xticks_rotation=90,
            cmap='Blues',  
            include_values=True,     
            colorbar=False,
        )

        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        for i in range(len(self.model.classes_)):
            for j in range(len(self.model.classes_)):
                text = disp.text_[i, j]
                text.set_fontsize(4)
                if i == j:
                    text.set_weight('bold')  # bold diagonal
                    # text.set_color('black') 
        # for text in disp.text_.ravel():
        #     text.set_fontsize(4)

        ax.tick_params(axis='both', which='major', labelsize=8)
    
        if self.audio == 'speech':
            ax.set_xticklabels([])
            ax.set_xlabel("")

        plt.tight_layout()
        plt.savefig(f"figs/conf_mat_{self.audio}.pdf", format='pdf', bbox_inches='tight', pad_inches=0.01)

        if self.plot_flag:
            plt.show()
        # Majority class baseline
        majority_class = Counter(y_true).most_common(1)[0][0]
        y_chance_pred = np.full_like(y_true, majority_class)
        model_acc = accuracy_score(y_true, y_pred)
        chance_acc = accuracy_score(y_true, y_chance_pred)

        print(f"Model Accuracy: {model_acc:.3f}")
        print(f"Chance Baseline Accuracy: {chance_acc:.3f}\n")

        print("Model Classification Report:")
        print(classification_report(y_true, y_pred))

        # Perform McNemar’s test
        contingency_table = np.array([
            [(y_true == y_pred).sum(), (y_true != y_pred).sum()],
            [(y_true == y_chance_pred).sum(), (y_true != y_chance_pred).sum()]
        ])
        result = mcnemar(contingency_table, exact=True)
        print(f"\nMcNemar’s test p-value: {result.pvalue:.7f}")

        if result.pvalue < 0.05:
            print("The classifier is significantly better than chance! 🎉")
        else:
            print("The classifier is NOT significantly better than chance. 🤔")
            
    def feature_importance(self, top_n=10):
        if not hasattr(self.model, "coef_"):
            print("Model is not trained yet.")
            return

        coefs = self.model.coef_

        if coefs.shape[0] == 1:
            # Binary classification
            importance = coefs[0]
            coef_df = pd.DataFrame({
                "feature": self.feature_columns,
                "coefficient": importance,
                "abs_coef": np.abs(importance)
            }).sort_values("abs_coef", ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(data=coef_df.head(top_n), x="coefficient", y="feature", palette="coolwarm")
            plt.title("Top Feature Importances (Binary Logistic Regression)")
            plt.tight_layout()
            if self.plot_flag:
                plt.show()
        else:
            # Multiclass classification in subplots
            n_classes = coefs.shape[0]
            fig, axes = plt.subplots(nrows=n_classes, figsize=(10, 2 * n_classes), sharex=True)

            if n_classes == 1:
                axes = [axes]

            for i, class_label in enumerate(self.model.classes_):
                importance = coefs[i]
                coef_df = pd.DataFrame({
                    "feature": self.feature_columns,
                    "coefficient": importance,
                    "abs_coef": np.abs(importance)
                }).sort_values("abs_coef", ascending=False)

                sns.barplot(
                    data=coef_df.head(top_n),
                    x="coefficient",
                    y="feature",
                    ax=axes[i],
                    # palette="coolwarm",
                )
                axes[i].set_title(f"Feature Importances for class '{class_label}'")

            plt.tight_layout()
            if self.plot_flag:
                plt.show()


class TavaRegressor():
    def __init__(self, df, feat_cols, audio, cv, pca_flag, plot_flag):
        self.seed = 1987
        init_shape = df[df.Variant == audio].shape[0]
        df = df.dropna()
        self.df = df[df.Variant == audio].copy()
        print(f'Reducing from {init_shape} to {self.df.shape[0]} annotated data instances!')
        self.feature_columns = feat_cols
        self.plot_flag = plot_flag

        if pca_flag:
            self.pca_reduction()
        else:
            self.X = self.df[self.feature_columns].values
            scaler = StandardScaler()
            self.X = scaler.fit_transform(self.X)

        self.val = self.df['meanValence'].values 
        self.aro = self.df['meanArousal'].values 
        self.dom = self.df['meanDominance'].values
        self.cv = cv

        self.hyperparams()
        self.plot_distribution()
        self.run_regression()

    def hyperparams(self):
        param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}

        grid_search = GridSearchCV(Lasso(max_iter=2000), param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(self.X, self.val)
        
        self.alpha = grid_search.best_params_['alpha']
        print(f"Best alpha: {self.alpha}")


    def plot_distribution(self):
        """
        Plots the distribution of Valence, Arousal, and Dominance.
        """
        plt.figure(figsize=(15, 5))

        # Plot for Valence
        plt.subplot(1, 3, 1)
        sns.histplot(self.val, kde=True, color='blue', bins=30)
        plt.title('Valence Distribution')
        plt.xlabel('Valence')
        plt.ylabel('Frequency')

        # Plot for Arousal
        plt.subplot(1, 3, 2)
        sns.histplot(self.aro, kde=True, color='green', bins=30)
        plt.title('Arousal Distribution')
        plt.xlabel('Arousal')
        plt.ylabel('Frequency')

        # Plot for Dominance
        plt.subplot(1, 3, 3)
        sns.histplot(self.dom, kde=True, color='orange', bins=30)
        plt.title('Dominance Distribution')
        plt.xlabel('Dominance')
        plt.ylabel('Frequency')

        plt.tight_layout()
        if self.plot_flag:
            plt.show()
 
    def pca_reduction(self):
        print('PCA:')
        print(f'Number of features {len(self.feature_columns)}')
        print(f'Number of data instances {self.df.shape[0]}')
        self.X = self.df[self.feature_columns].values
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
    
        pca = PCA(n_components=0.95) 
        X_pca = pca.fit_transform(self.X)
        n_comp = X_pca.shape[1]
        print('-'*50)
        print(f'PCA reduced from {len(self.feature_columns)} to {n_comp} to keep 95% of variance!')

        self.feature_columns = [f'pca_{i}' for i in range(X_pca.shape[1])]
        df_pca = pd.DataFrame(X_pca, columns=self.feature_columns, index=self.df.index)

        self.df = pd.concat([self.df, df_pca], axis=1)


    def run_regression(self):
        print("\nRunning linear regressions...\n")
        self.regression_results = {}

        for label, target in zip(['Valence', 'Arousal', 'Dominance'], [self.val, self.aro, self.dom]):
            print(f"→ Regressing {label}")
            model = Lasso(alpha=self.alpha, max_iter=2000)
            y_pred = cross_val_predict(model, self.X, target, cv=self.cv)

            metrics, chance, t_stat, p_value = self.evaluate_regression(target, y_pred)
            self.regression_results[label.lower()] = metrics

            print(f"  Predicted MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
            print(f"  Chance MSE: {chance['mse']:.4f}, MAE: {chance['mae']:.4f}, R²: {chance['r2']:.4f}")
            print(f"  T-statistic: {t_stat}")
            print(f"  P-value: {p_value}")
            
            if p_value < 0.05:
                print("The Lasso model is statistically significantly better than the baseline model.\n")
            else:
                print("No significant difference between the Lasso model and the baseline model.\n")


    def evaluate_regression(self, y_true, y_pred):
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
        }
        baseline = np.full_like(y_true, np.mean(y_true))
        chance = {
            'mse': mean_squared_error(y_true, baseline),
            'mae': mean_absolute_error(y_true, baseline),
            'r2': r2_score(y_true, baseline),
        }
        residuals_lasso = y_true - y_pred
        residuals_baseline = y_true - baseline
        
        # Perform paired t-test on residuals
        t_stat, p_value = ttest_rel(residuals_lasso, residuals_baseline)
        return metrics, chance, t_stat, p_value


def assemble_dataset():
    anno = pd.read_csv(f'annotations/SummaryData.csv', index_col=0)
    base_path = os.path.abspath('.')
    all_csv = glob.glob(os.path.join(base_path, 'feats', '*', '*.csv'))
    feats = pd.concat([pd.read_csv(_) for _ in all_csv]).reset_index(drop=True)
    return feats, anno


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify tava dataset")
    parser.add_argument('-audio', 
                        type=str,
                        choices=['speech', 'tEGG'],
                        default='speech',
                        help='Select audio to train predictors.')
    parser.add_argument('-cv', 
                        type=int,
                        choices=[5, 10],
                        default=5,
                        help='Select a number of cross-validation splits')
    parser.add_argument('-pca', 
                        dest='pca',
                        action='store_true',
                        default=False,
                        help='Select to do PCA before prediction.')
    parser.add_argument('-plot', 
                        dest='plot',
                        action='store_true',
                        default=False,
                        help='Select to show plots.')
    args = parser.parse_args()

    feats, anno = assemble_dataset()
    feat_cols = [_ for _ in feats.columns if _ not in ['file', 'start', 'end']]

    anno['audio'] = anno.apply(
        lambda row: (
            f"audio/recordings/s{int(row['Speaker Number']):02d}_{row['Affect Name']}_audio.wav"
            if row['Variant'] == 'speech'
            else f"audio/tegg/s{int(row['Speaker Number']):02d}_{row['Affect Name']}_tEGG.wav"
        ),
        axis=1
    )
    anno['feats'] = (
        anno['audio']
        .str.replace('audio/', 'feats/', regex=False)
        .str.replace('.wav', '.csv', regex=False)
    )
    anno['basename'] = anno['feats'].apply(os.path.basename)
    feats['basename'] = feats['file'].apply(os.path.basename).str.replace('.wav', '.csv', regex=False)

    df = pd.merge(anno, feats, on='basename', how='inner')

    # classification
    print('-'*50)
    print('CLASSIFICATION')
    clf = TavaClassifier(df=df, feat_cols=feat_cols, audio=args.audio, cv=args.cv, pca_flag=args.pca, plot_flag=args.plot)

    # regression
    print('-'*50)
    print('REGRESSION')
    reg = TavaRegressor(df=df, feat_cols=feat_cols, audio=args.audio, cv=args.cv, pca_flag=args.pca, plot_flag=args.plot)