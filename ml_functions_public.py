import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn. model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score, ConfusionMatrixDisplay, confusion_matrix, classification_report
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint, uniform, loguniform   # handy distributions

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

colors = {
    "blue": "blue",
    "red": "red",
    "orange" : "orange",
    "green": "#009E73",
    "skyblue": "#56B4E9",
    "yellow": "#F0E442",
    "gray": "#999999",
    "black": "#111111"
}

class DecisionTree:
    def __init__(self, dataframe, model,
                 n_estimators=600, n_iter=80, early_stopping_rounds=100,
                 n_jobs=-1, random_state=42,
                 test_size=0.4, validation_size=0.176,
                 all_features=False, save=False,
                 fig_size=(8, 4)):

        self.dataframe = dataframe
        self.model = model
        self.n_estimators = n_estimators
        self.n_iter = n_iter
        self.early_stopping_rounds = early_stopping_rounds
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.test_size = test_size
        self.validation_size = validation_size
        self.all_features = all_features
        self.save = save
        self.fig_size = fig_size

    def plot_loss(self, evals_result, save=False):
        save = save or self.save
        # evals_result_ keys: 'train', 'valid'
        keys = list(evals_result.keys())  # e.g. ['validation_0','validation_1']
        train_key, valid_key = keys[0], keys[1]
        for metric in evals_result[train_key]:
            train_vals = evals_result[train_key][metric]
            valid_vals = evals_result[valid_key][metric]
            if metric == "logloss" or metric == "binary_logloss":
                name = "Log Loss"
                legend_name = "log loss"
            else:
                name = "AUC"
                legend_name = name
            plt.figure(figsize=self.fig_size)
            plt.plot(train_vals, label=f"Training {legend_name}", color="#224E94")
            plt.plot(valid_vals, label=f"Validation {legend_name}", color="#942287")
            plt.xlabel("Boosting Rounds")
            plt.ylabel(name)
            plt.xlim(xmin=0)
            plt.legend()
            plt.tight_layout()
            if save:
                if self.all_features:
                    plt.savefig(f"{self.model}_loss_curve_{metric}_all_features.jpeg", dpi=300, bbox_inches="tight")
                else:
                    plt.savefig(f"{self.model}_loss_curve_{metric}.jpeg", dpi=300, bbox_inches="tight")
            plt.show()

    def plot_roc(self, y_true, y_score, save=False):
        save = save or self.save
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=self.fig_size)
        plt.plot(fpr, tpr, lw=2, label=rf"ROC curve ($\mathrm{{AUC}} = {roc_auc:.4f}$)", color="#224E94")
        plt.plot([0, 1], [0, 1], lw=1, linestyle="--", color=colors["gray"])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        if save:
            if self.all_features:
                plt.savefig(f"{self.model}_roc_all_features.jpeg", dpi=300, bbox_inches="tight")
            else:
                plt.savefig(f"{self.model}_roc.jpeg", dpi=300, bbox_inches="tight")
        plt.show()

    @staticmethod
    def fetch_metrics(y_true, y_pred, y_score):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)

        return {'accuracy' : accuracy, 'precision' : precision, 'recall' : recall, 'f1' : f1, 'ROC(AUC)' : roc_auc, 'PR(AUC)': pr_auc}

    def plot_confusions(self, y_true, y_pred, save=False):

        J_index = self.compute_threshold(y_true, y_pred)
        y_pred = np.where(y_pred > J_index, 1, 0)

        save = save or self.save
        cm = confusion_matrix(y_true, y_pred)
        cmn = confusion_matrix(y_true, y_pred, normalize="true")

        plt.figure(figsize=(4,4))
        for mat, name in [(cm, "Raw"), (cmn, "Normalized")]:
            disp = ConfusionMatrixDisplay(mat, display_labels=["B (0)", "W (1)"])
            disp.plot(values_format=".2f" if name == "Normalized" else "d", colorbar=True, cmap="Blues")

            # Add a label to the colorbar
            cbar = disp.im_.colorbar
            if cbar:
                if name == "Normalized":
                    cbar.set_label("Normalised Samples")
                else:
                    cbar.set_label("Number of Samples")

            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")

            plt.tight_layout()
            if save:
                if self.all_features:
                    plt.savefig(f"{self.model}_confusion_{name}_all_features.jpeg", dpi=300, bbox_inches="tight")
                else:
                    plt.savefig(f"{self.model}_confusion_{name}.jpeg", dpi=300, bbox_inches="tight")

    def label(self, feature) -> str:
        axis_info = {
            "phi": {
                "unit": "rad",
                "label": r"$\phi$"
            },
            "tgl": {
                "unit": "–",  # dimensionless
                "label": r"$\tan(\pi/2-\theta_\text{polar})$"
            },
            "signed1Pt": {
                "unit": r"(GeV/$c$)$^{-1}$",
                "label": r"$q/p_{T}$"
            },
            "nClusters": {
                "unit": "count",
                "label": r"Number of Clusters"
            },
            "pDCA": {
                "unit": "cm",
                "label": r"$p_{\mathrm{DCA}}$"
            },
            "rAtAbsorberEnd": {
                "unit": "cm",
                "label": r"$r_{\mathrm{At\,Absorber\,End}}$"
            },
            "sign": {
                "unit": "±1",
                "label": r"Charge Sign"
            },
            "chi2": {
                "unit": "–",
                "label": r"$\chi^2$"
            },
            "chi2MatchMCHMID": {
                "unit": "–",
                "label": r"$\chi^2_{\mathrm{MCH-MID}}$"
            },
            "chi2MatchMCHMFT": {
                "unit": "–",
                "label": r"$\chi^2_{\mathrm{MCH-MFT}}$"
            },
            "trackTime": {
                "unit": "ns",
                "label": r"Track Time"
            },
            "eta": {
                "unit": "–",  # pseudorapidity is dimensionless
                "label": r"$\eta$"
            },
            "pt": {
                "unit": "GeV/$c$",
                "label": r"$p_{T}$"
            },
            "p": {
                "unit": "GeV/$c$",
                "label": r"$p$"
            }
        }
        label = axis_info[feature]["label"]
        return label

    def export_xgb_feature_importance(self, final_pipe, top_n=30, aggregate_cats=True, save=False):
        save = save or self.save

        # Get fitted steps
        pre = final_pipe.named_steps["pre"]
        xgb = final_pipe.named_steps["xgb"]

        # Ensure the preprocessor is fitted (pipeline.fit(...) must have run)
        if not hasattr(pre, "transformers_"):
            raise RuntimeError(
                "Preprocessor is not fitted. Call this AFTER final_pipe.fit(...)."
            )

        # Names after preprocessing (includes one-hot columns)
        feat_names = pre.get_feature_names_out()
        importances = xgb.feature_importances_.astype(float)

        raw_df = pd.DataFrame({"feature": feat_names, "importance": importances})
        raw_df.sort_values("importance", ascending=False, inplace=True)

        # raw_df.to_csv(os.path.join(outdir, "xgb_feature_importance_raw.csv"), index=False)

        def get_display_label(feat):
            # Extract base feature name
            base = feat.split("_", 1)[0]
            try:
                return self.label(base)
            except KeyError:
                return feat

        top_raw = raw_df.head(top_n).iloc[::-1]
        top_raw_labels = top_raw["feature"].map(get_display_label)

        # top_raw = raw_df.head(top_n).iloc[::-1]
        # plt.figure(figsize=(8, max(3, 0.35 * len(top_raw))))
        plt.figure(figsize=self.fig_size)
        plt.barh(top_raw_labels, top_raw["importance"], color="#224E94")
        plt.xlabel("Importance (" + getattr(xgb, "importance_type", "gain") + ")")
        plt.tight_layout()
        if save:
            if self.all_features:
                plt.savefig(f"{self.model}_feature_importance_raw_top_all_features.jpeg", dpi=300, bbox_inches="tight")
            else:
                plt.savefig(f"{self.model}_feature_importance_raw_top.jpeg", dpi=300, bbox_inches="tight")
        plt.show()

        if not aggregate_cats:
            return

        num_cols = pre.transformers_[0][2] if len(pre.transformers_) > 0 else []
        cat_cols = pre.transformers_[1][2] if len(pre.transformers_) > 1 else []

        cat_cols_set = set(cat_cols)

        def base_name(out_name: str) -> str:
            """
            If the left part before the first '_' is a known categorical column,
            use it as the base; otherwise keep the full name (numeric or already base).
            """
            left = out_name.split("_", 1)[0]
            return left if left in cat_cols_set else out_name

        agg_df = (
            raw_df.assign(base_feature=raw_df["feature"].map(base_name))
            .groupby("base_feature", as_index=False)["importance"].sum()
            .sort_values("importance", ascending=False)
        )

        # agg_df.to_csv(os.path.join(outdir, "xgb_feature_importance_aggregated.csv"), index=False)

        top_agg = agg_df.head(top_n).iloc[::-1]

        def get_agg_label(feat):
            try:
                return self.label(feat)
            except KeyError:
                return feat

        top_agg_labels = top_agg["base_feature"].apply(get_agg_label)

        # plt.figure(figsize=(8, max(3, 0.35 * len(top_agg))))
        plt.figure(figsize=self.fig_size)
        plt.barh(top_agg_labels, top_agg["importance"], color="#224E94")
        plt.xlabel("Summed importance across one-hot levels")
        plt.tight_layout()
        if save:
            if self.all_features:
                plt.savefig(f"{self.model}_feature_importance_aggregated_top_all_features.jpeg", dpi=300, bbox_inches="tight")
            else:
                plt.savefig(f"{self.model}_feature_importance_aggregated_top.jpeg", dpi=300, bbox_inches="tight")
        plt.show()

    def export_lgbm_feature_importance(self, final_pipe, top_n=30,
                                       importance="gain", aggregate_cats=True, save=False):
        save = save or self.save

        pre = final_pipe.named_steps["pre"]
        clf = final_pipe.named_steps["lgbm"]

        # Names after preprocessing (order matches the model’s input order)
        feat_names = getattr(pre, "get_feature_names_out", lambda: None)()
        if feat_names is None:
            # Fallback: synthesize names if needed
            n_feats = clf.n_features_in_
            feat_names = np.array([f"f{i}" for i in range(n_feats)])

        if importance == "gain":
            # Prefer Booster-level gain (more informative)
            try:
                # vals = clf.booster_.feature_importance(importance_type="gain")
                vals = clf.booster_.feature_importance(importance_type="gain").astype(float)
                vals = vals / (vals.sum() + 1e-12)  # normalize to sum to 1 (like XGB)
                # Sometimes booster returns default names; replace with ours for clarity
            except Exception:
                # Fallback to sklearn attribute (usually 'split')
                vals = clf.feature_importances_
        elif importance == "split":
            vals = clf.feature_importances_
        else:
            raise ValueError("importance must be 'gain' or 'split'")

        vals = np.asarray(vals, dtype=float)

        # length alignment sanity check
        if len(vals) != len(feat_names):
            # When LightGBM trained on arrays, order still matches pre-transform output.
            # If mismatch occurs, truncate/pad conservatively.
            m = min(len(vals), len(feat_names))
            vals = vals[:m]
            feat_names = feat_names[:m]

        # -------- RAW importances --------
        raw = pd.DataFrame({"feature": feat_names, "importance": vals}) \
            .sort_values("importance", ascending=False)

        # raw.to_csv(os.path.join(outdir, "lgbm_feature_importance_raw.csv"), index=False)

        def get_display_label(feat):
            base = feat.split("_", 1)[0]
            try:
                return self.label(base)
            except KeyError:
                return feat

        top = raw.head(top_n).iloc[::-1]
        top_labels = top["feature"].apply(get_display_label)

        # plt.figure(figsize=(8, max(3, 0.35 * len(top))))
        plt.figure(figsize=self.fig_size)
        plt.barh(top_labels, top["importance"], color="#224E94")
        plt.xlabel(f"Importance ({importance})")
        plt.tight_layout()
        if save:
            if self.all_features:
                plt.savefig(f"{self.model}_feature_importance_raw_top_all_features.jpeg", dpi=300, bbox_inches="tight")
            else:
                plt.savefig(f"{self.model}_feature_importance_raw_top.jpeg", dpi=300, bbox_inches="tight")
        plt.show()

        if not aggregate_cats:
            return

        # -------- aggregate one-hot back to original feature --------
        # Pull original numeric/categorical lists from the fitted ColumnTransformer
        num_cols = pre.transformers_[0][2] if len(pre.transformers_) > 0 else []
        cat_cols = pre.transformers_[1][2] if len(pre.transformers_) > 1 else []
        cat_set = set(cat_cols)

        def base_name(out_name: str) -> str:
            # If the part before the first '_' is one of our categorical columns,
            # treat that as the base feature name; otherwise keep the full name.
            left = out_name.split("_", 1)[0]
            return left if left in cat_set else out_name

        agg = (
            raw.assign(base_feature=raw["feature"].map(base_name))
            .groupby("base_feature", as_index=False)["importance"].sum()
            .sort_values("importance", ascending=False))
        # agg.to_csv(os.path.join(outdir, "lgbm_feature_importance_aggregated.csv"), index=False)

        topa = agg.head(top_n).iloc[::-1]

        def get_agg_label(feat):
            try:
                return self.label(feat)
            except KeyError:
                return feat

        topa_labels = topa["base_feature"].apply(get_agg_label)

        # plt.figure(figsize=(8, max(3, 0.35 * len(topa))))
        plt.figure(figsize=self.fig_size)
        plt.barh(topa_labels, topa["importance"], color="#224E94")
        plt.xlabel(f"Summed importance across one-hot levels ({importance})")
        plt.tight_layout()
        if save:
            if self.all_features:
                plt.savefig(f"{self.model}_feature_importance_aggregated_top_all_features.jpeg", dpi=300, bbox_inches="tight")
            else:
                plt.savefig(f"{self.model}_feature_importance_aggregated_top.jpeg", dpi=300, bbox_inches="tight")
        plt.show()

    def plot_score_hist(self, y_true, y_score, save=False):

        threshold = self.compute_threshold(y_true=y_true, y_score=y_score)

        save = save or self.save
        bins = np.linspace(0, 1, 51)
        plt.figure(figsize=self.fig_size)

        plt.hist(y_score[y_true == 1], bins=bins, density=True,
                 histtype="stepfilled",
                 color="#152F67", alpha=0.75,
                 edgecolor="blue", linewidth=1.0, label=r'$W$-boson Muons')
        plt.hist(y_score[y_true == 0], bins=bins, density=True,
                 histtype="stepfilled", facecolor="none", edgecolor="red", linewidth=1.0,
                 alpha=1.0, color="red", hatch='//', label='HF Muons')

        plt.axvline(threshold, linestyle="--", linewidth=1, color='black',
                    label=fr'Decision Threshold $ = {threshold:.2f}$')

        plt.xlabel(r"Predicted Probability $P(\mu_W)$")
        plt.ylabel(r"$(1/N)\, dN/dP(\mu_W)$")
        plt.xlim(0, 1)
        plt.legend()

        if save:
            if self.all_features:
                plt.savefig(f"{self.model}_score_hist_all_features.jpeg", dpi=300, bbox_inches="tight")
            else:
                plt.savefig(f"{self.model}_score_hist.jpeg", dpi=300, bbox_inches="tight")
        plt.show()

        counts, edges = np.histogram(y_score[y_true == 1], bins=bins, density=True)
        area = np.sum(counts * np.diff(edges))
        print(area)  # ~ 1.0

    def plot_pr(self, y_true, y_score, save=False):
        save = save or self.save
        p, r, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        plt.figure(figsize=self.fig_size)
        plt.plot(r, p, linewidth=2, label=rf"$\mathrm{{AP}} = {ap:.4f}$", color="#224E94")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim(xmin=0)
        plt.legend()
        if save:
            if self.all_features:
                plt.savefig(f"{self.model}_pr_all_features.jpeg", dpi=300, bbox_inches="tight")
            else:
                plt.savefig(f"{self.model}_pr.jpeg", dpi=300, bbox_inches="tight")
        plt.show()

    @staticmethod
    def compute_threshold(y_true, y_score):
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        J = tpr - fpr
        optimal_idx = np.argmax(J)
        optimal_threshold = thresholds[optimal_idx]
        return optimal_threshold

class ModelXGB(DecisionTree):

    def __init__(self, dataframe, model,
                 n_estimators=600, n_iter=80, early_stopping_rounds=100,
                 n_jobs=-1, random_state=42,
                 test_size=0.4, validation_size=0.176,
                 all_features=False, save=False):

        super().__init__(dataframe, model,
                         n_estimators=n_estimators,
                         n_iter=n_iter,
                         early_stopping_rounds=early_stopping_rounds,
                         n_jobs=n_jobs,
                         random_state=random_state,
                         test_size=test_size,
                         validation_size=validation_size,
                         all_features=all_features,
                         save=save)

        self.x, self.y = None, None
        self.y_test = None
        self.final_pipe_ = None
        self.eval_xgb_ = None
        self.probabilities_ = None
        self.predictions_ = None
        self.model = "XGB"
        self.best_params_ = None

    def model_xgb(self):
        y = self.dataframe["label"].astype(int).values
        x = self.dataframe.drop(columns=["label"])

        # Identify numeric and categorical feature columns
        num_cols = selector(dtype_include=np.number)(x)
        cat_cols = selector(dtype_exclude=np.number)(x)

        numeric_pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="median"))])
        categorical_pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent")),
                                           ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

        pre = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, num_cols),
                ("cat", categorical_pipe, cat_cols)
            ],
            remainder="drop",
            verbose_feature_names_out=False
        )

        x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, test_size=self.test_size,
                                                                    random_state=self.random_state, stratify=y)
        x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=self.validation_size,
                                                          random_state=self.random_state, stratify=y_train_all)

        pos = np.sum(y_train_all == 1)
        neg = np.sum(y_train_all == 0)
        if pos == 0:
            spw = 1.0
        else:
            spw = max(1.0, neg / pos)

        xgb = XGBClassifier(
            objective="binary:logistic",
            n_estimators=self.n_estimators,
            tree_method="hist",
            random_state=self.random_state,
            eval_metric=["logloss", "auc"],
            n_jobs=self.n_jobs,
            scale_pos_weight=spw,
        )

        pipe = Pipeline(steps=[("pre", pre), ("xgb", xgb)])

        # Define parameter distributions instead of fixed grids
        param_dist = {
            "xgb__max_depth": randint(2, 21),  # integers
            "xgb__learning_rate": loguniform(1e-3, 2e-1),  # log scale
            "xgb__min_child_weight": randint(1, 21),
            "xgb__subsample": uniform(0.5, 0.5),  # 0.5–1.0
            "xgb__colsample_bytree": uniform(0.5, 0.5),  # 0.5–1.0
            "xgb__reg_lambda": loguniform(1e-3, 1e2),  # L2
            "xgb__reg_alpha": loguniform(1e-5, 1e1),  # add L1
            "xgb__gamma": uniform(0.0, 5.0),
            "xgb__n_estimators": randint(200, 2000)  # num trees
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        rs = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=self.n_iter,  # number of random samples to try
            scoring="roc_auc",
            cv=cv,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1,
            refit=True
        )

        rs.fit(x_train_all, y_train_all)

        best_params = rs.best_params_
        best_model = rs.best_estimator_

        xgb_best_kwargs = {k.replace("xgb__", ""): v for k, v in best_params.items() if k.startswith("xgb__")}

        xgb_final = XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            random_state=self.random_state,
            eval_metric=["logloss", "auc"],
            early_stopping_rounds=self.early_stopping_rounds,
            n_jobs=self.n_jobs,
            scale_pos_weight=spw,
            importance_type="gain",
            use_label_encoder=False,
            **xgb_best_kwargs
        )

        final_pipe = Pipeline(steps=[("pre", pre), ("xgb", xgb_final)])

        # Pre-transform the eval sets
        xt_train = pre.fit_transform(x_train)  # fit on train
        xt_val = pre.transform(x_val)

        final_pipe.fit(
            x_train, y_train,
            xgb__eval_set=[(xt_train, y_train),
                           (xt_val, y_val)],
            xgb__verbose=False,
        )

        self.x, self.y = x, y
        self.y_test = y_test
        self.best_params_ = best_params
        self.final_pipe_ = final_pipe
        self.eval_xgb_ = final_pipe.named_steps["xgb"].evals_result()
        self.probabilities_ = final_pipe.predict_proba(x_test)[:, 1]
        self.predictions_ = (self.probabilities_ >= 0.5).astype(int)

        return self

    def plot_loss_auto(self, save=False):
        save = save or self.save
        evals_result = self.eval_xgb_
        return super().plot_loss(evals_result, save=save)

    def plot_roc_auto(self, save=False):
        save = save or self.save
        y_true = self.y_test
        y_score = self.probabilities_
        return super().plot_roc(y_true, y_score, save=save)

    def plot_confusions_auto(self, save=False):
        save = save or self.save
        y_true = self.y_test
        y_score = self.probabilities_
        return super().plot_confusions(y_true, y_score, save=save)

    def plot_feature_importance_auto(self, save=False):
        save = save or self.save
        final_pipe = self.final_pipe_
        return super().export_xgb_feature_importance(final_pipe, save=save)

    def plot_score_hist_auto(self, save=False):
        save = save or self.save
        y_true = self.y_test
        y_score = self.probabilities_
        return super().plot_score_hist(y_true, y_score, save=save)

    def plot_pr_auto(self, save=False):
        save = save or self.save
        y_true = self.y_test
        y_score = self.probabilities_
        return super().plot_pr(y_true, y_score, save=save)
    
    def fetch_metrics_auto(self):
        y_true = self.y_test
        y_pred = self.predictions_
        y_score = self.probabilities_
        return super().fetch_metrics(y_true, y_pred, y_score)

    def compute_threshold_auto(self):
        y_true = self.y_test
        y_score = self.probabilities_
        return super().compute_threshold(y_true, y_score)
    
class ModelLGBM(DecisionTree):

    def __init__(self, dataframe, model,
                 n_estimators=600, n_iter=80, early_stopping_rounds=100,
                 n_jobs=-1, random_state=42,
                 test_size=0.4, validation_size=0.176,
                 all_features=False, save=False):

        super().__init__(dataframe, model,
                         n_estimators=n_estimators,
                         n_iter=n_iter,
                         early_stopping_rounds=early_stopping_rounds,
                         n_jobs=n_jobs,
                         random_state=random_state,
                         test_size=test_size,
                         validation_size=validation_size,
                         all_features=all_features,
                         save=save)

        self.x, self.y = None, None
        self.y_test = None
        self.best_params_ = None
        self.final_pipe_ = None
        self.eval_lgbm_ = None
        self.probabilities_ = None
        self.predictions_ = None
        self.model = "LGBM"

    def model_lgbm(self):
        y = self.dataframe["label"].astype(int).values
        x = self.dataframe.drop(columns=["label"])

        # Identify numeric and categorical feature columns
        num_cols = selector(dtype_include=np.number)(x)
        cat_cols = selector(dtype_exclude=np.number)(x)

        numeric_pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="median"))])
        categorical_pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent")),
                                           ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

        pre = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, num_cols),
                ("cat", categorical_pipe, cat_cols)
            ],
            remainder="drop",
            verbose_feature_names_out=False
        )

        x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, test_size=self.test_size,
                                                                    random_state=self.random_state, stratify=y)
        x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=self.validation_size,
                                                          random_state=self.random_state, stratify=y_train_all)

        pos = np.sum(y_train_all == 1)
        neg = np.sum(y_train_all == 0)
        if pos == 0:
            spw = 1.0
        else:
            spw = max(1.0, neg / pos)

        lgbm = LGBMClassifier(
            objective="binary",  # binary classification
            n_estimators=self.n_estimators,  # higher than needed; early stopping in final fit
            boosting_type="gbdt",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            scale_pos_weight=spw,
            verbosity=-1,
        )

        pipe = Pipeline(steps=[("pre", pre), ("lgbm", lgbm)])

        param_dist = {
            "lgbm__num_leaves":          randint(15, 128),     # smaller -> less reuse of 1 var
            "lgbm__max_depth":           randint(3, 10),       # cap depth to avoid deep pT paths
            "lgbm__learning_rate":       loguniform(1e-3, 2e-1),

            # regularize leaf creation (rough analogue of XGB min_child_weight)
            "lgbm__min_data_in_leaf":    randint(50, 400),

            # row/feature subsampling -> forces other vars to be tried
            "lgbm__subsample":           uniform(0.6, 0.4),    # 0.6–1.0 (bagging_fraction)
            "lgbm__colsample_bytree":    uniform(0.6, 0.4),    # 0.6–1.0 (feature_fraction)
            "lgbm__feature_fraction_bynode": uniform(0.6, 0.4),# per-split feature subsample

            # split threshold (XGB gamma-ish): require actual gain before splitting
            "lgbm__min_gain_to_split":   uniform(0.0, 2.0),

            # L1/L2
            "lgbm__reg_lambda":          loguniform(1e-3, 1e2),
            "lgbm__reg_alpha":           loguniform(1e-5, 1e1),

            # total trees
            "lgbm__n_estimators":        randint(150, 600)
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        rs = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=self.n_iter,
            scoring="roc_auc",
            cv=cv,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1,
            refit=True,
        )

        rs.fit(x_train_all, y_train_all)

        best_params = rs.best_params_

        lgbm_best_kwargs = {k.replace("lgbm__", ""): v for k, v in best_params.items() if k.startswith("lgbm__")}

        lgbm_final = LGBMClassifier(
            objective="binary",
            # n_estimators=3000,            # large cap; early stopping will trim
            boosting_type="gbdt",
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            scale_pos_weight=spw,
            verbosity=-1,
            **lgbm_best_kwargs
        )

        final_pipe = Pipeline(steps=[("pre", pre), ("lgbm", lgbm_final)])

        pre.fit(x_train)
        xt_train = pre.transform(x_train)
        xt_val = pre.transform(x_val)

        final_pipe.fit(
            x_train, y_train,
            # LightGBM expects (X, y) pairs
            lgbm__eval_set=[(xt_train, y_train), (xt_val, y_val)],
            # Request both loss and AUC for curves
            lgbm__eval_metric=["binary_logloss", "auc"],
            # Early stopping is passed via fit-kwargs in the sklearn API
            lgbm__callbacks=[
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, first_metric_only=True),
                lgb.log_evaluation(period=0),  # silence training logs
            ]
        )

        self.x, self.y = x, y
        self.y_test = y_test
        self.best_params_ = best_params
        self.final_pipe_ = final_pipe
        self.eval_lgbm_ = final_pipe.named_steps["lgbm"].evals_result_
        self.probabilities_ = final_pipe.predict_proba(x_test)[:, 1]
        self.predictions_ = (self.probabilities_ >= 0.5).astype(int)

        return self

    def plot_loss_auto(self, save=False):
        save = save or self.save
        evals_result = self.eval_lgbm_
        return super().plot_loss(evals_result, save=save)

    def plot_roc_auto(self, save=False):
        save = save or self.save
        y_true = self.y_test
        y_score = self.probabilities_
        return super().plot_roc(y_true, y_score, save=save)

    def plot_confusions_auto(self, save=False):
        save = save or self.save
        y_true = self.y_test
        y_prob = self.probabilities_
        return super().plot_confusions(y_true, y_prob, save=save)

    def plot_feature_importance_auto(self, save=False):
        save = save or self.save
        final_pipe = self.final_pipe_
        return super().export_lgbm_feature_importance(final_pipe, save=save)

    def plot_score_hist_auto(self, save=False):
        save = save or self.save
        y_true = self.y_test
        y_score = self.probabilities_
        return super().plot_score_hist(y_true, y_score, save=save)

    def plot_pr_auto(self, save=False):
        save = save or self.save
        y_true = self.y_test
        y_score = self.probabilities_
        return super().plot_pr(y_true, y_score, save=save)
    
    def fetch_metrics_auto(self):
        y_true = self.y_test
        y_pred = self.predictions_
        y_score = self.probabilities_
        return super().fetch_metrics(y_true, y_pred, y_score)

    def compute_threshold_auto(self):
        y_true = self.y_test
        y_score = self.probabilities_
        return super().compute_threshold(y_true, y_score)

class ModelLR(DecisionTree):
    def __init__(self, dataframe, model,
                 n_estimators=600, n_iter=80, early_stopping_rounds=100,
                 n_jobs=-1, random_state=42,
                 test_size=0.4, validation_size=0.176,
                 all_features=False, save=False,
                 regularization=None, alpha=1.0,
                 target_col="y", feature_cols=None):

        super().__init__(dataframe, model,
                         n_estimators=n_estimators,
                         n_iter=n_iter,
                         early_stopping_rounds=early_stopping_rounds,
                         n_jobs=n_jobs,
                         random_state=random_state,
                         test_size=test_size,
                         validation_size=validation_size,
                         all_features=all_features,
                         save=save)

        self.regularization = regularization
        self.alpha = float(alpha)
        self.target_col = target_col
        self.feature_cols = feature_cols

        self.final_pipe_ = None
        self.metrics_ = None
        self.coef_ = None
        self.x_train_ = None
        self.x_test_ = None
        self.y_train_ = None
        self.y_test_ = None
        self.probabilities_ = None
        self.predictions_ = None

    def _build_preprocessor(self, x: pd.DataFrame) -> ColumnTransformer:
        # Split numeric vs categorical columns
        num_cols = x.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in x.columns if c not in num_cols]

        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        pre = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, num_cols),
                ("cat", categorical_pipe, cat_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        return pre

    def _make_estimator(self):
        if self.regularization == "ridge":
            return Ridge(alpha=self.alpha, random_state=self.random_state)
        elif self.regularization == "lasso":
            # Increase max_iter to help convergence for sparse/large designs
            return Lasso(alpha=self.alpha, random_state=self.random_state, max_iter=10000)
        elif self.regularization == "none":
            return LinearRegression()
        else:
            raise ValueError("regularization must be one of {'none','ridge','lasso'}.")

    def fit(self):
        # Select features/target
        df = self.dataframe.copy()
        if self.feature_cols is None:
            # Use all columns except target if not specified
            if self.target_col not in df.columns:
                raise KeyError(f"Target column '{self.target_col}' not found in dataframe.")
            x = df.drop(columns=[self.target_col])
        else:
            for c in self.feature_cols + [self.target_col]:
                if c not in df.columns:
                    raise KeyError(f"Column '{c}' not found in dataframe.")
            x = df[self.feature_cols]

        y = df[self.target_col].astype(float)

        # Build preprocessor
        pre = self._build_preprocessor(x)

        # Train/test split (no stratify for regression)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state
        )

        # Estimator & pipeline
        est = self._make_estimator()
        pipe = Pipeline([("pre", pre), ("lin", est)])

        # Fit
        pipe.fit(x_train, y_train)

        # Store splits & pipeline
        self.x_train_, self.x_test_ = x_train, x_test
        self.y_train_, self.y_test_ = y_train, y_test
        self.final_pipe_ = pipe

        self.probabilities_ = pipe.predict(x_test)
        self.predictions_ = (self.probabilities_ >= 0.5).astype(int)

        # Evaluate
        yhat_train = pipe.predict(x_train)
        yhat_test = pipe.predict(x_test)

        def rmse(y_true, y_pred):
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))

        self.metrics_ = {
            "r2_train": float(r2_score(y_train, yhat_train)),
            "r2_test": float(r2_score(y_test, yhat_test)),
            "mae_train": float(mean_absolute_error(y_train, yhat_train)),
            "mae_test": float(mean_absolute_error(y_test, yhat_test)),
            "rmse_train": rmse(y_train, yhat_train),
            "rmse_test": rmse(y_test, yhat_test),
        }

        # Coefficients aligned to post-preprocessing feature names
        feat_names = pipe.named_steps["pre"].get_feature_names_out()
        lin = pipe.named_steps["lin"]
        coefs = lin.coef_.ravel() if hasattr(lin, "coef_") else np.zeros_like(feat_names, dtype=float)

        self.coef_ = pd.DataFrame(
            {"feature": feat_names, "coef": coefs, "abs_coef": np.abs(coefs)}
        ).sort_values("abs_coef", ascending=False).reset_index(drop=True)

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.final_pipe_ is None:
            raise RuntimeError("Call fit() before predict().")
        return self.final_pipe_.predict(df)

    def score(self) -> dict:
        if self.metrics_ is None:
            raise RuntimeError("Call fit() before score().")
        return self.metrics_

    def summary(self) -> pd.DataFrame:
        if self.coef_ is None:
            raise RuntimeError("Call fit() before summary().")
        return self.coef_.copy()
    
    def plot_roc_auto(self, save=False):
        save = save or self.save
        y_true = self.y_test_
        y_score = self.probabilities_
        return super().plot_roc(y_true, y_score, save=save)

    def plot_score_hist_auto(self, save=False):
        save = save or self.save
        y_true = self.y_test_
        y_score = self.final_pipe_.predict(self.x_test_)
        return super().plot_score_hist(y_true, y_score, save=save)

    def plot_pr_auto(self, save=False):
        save = save or self.save
        y_true = self.y_test_
        y_score = self.probabilities_
        return super().plot_pr(y_true, y_score, save=save)

    def plot_coefficients(self, top_n: int = 20, save=False):
        save = save or self.save
        if self.coef_ is None:
            raise RuntimeError("Call fit() before plot_coefficients().")

        top = self.coef_.head(top_n).iloc[::-1].copy()

        # Try to map to pretty labels from the base class; fall back to raw label
        def display_label(feat: str) -> str:
            base = feat.split("_", 1)[0]
            try:
                return self.label(base)
            except Exception:
                return feat

        top["display"] = top["feature"].map(display_label)

        top_size = self.coef_.head(top_n).iloc[::-1].copy()

        plt.figure(figsize=self.fig_size)
        plt.barh(top["display"], top["coef"], color="#224E94")
        plt.xlabel("Coefficient")
        plt.tight_layout()

        if save:
            if self.all_features:
                plt.savefig(f"{self.model}_top_coefficients_all_features.jpeg", dpi=300, bbox_inches="tight")
            else:
                plt.savefig(f"{self.model}_top_coefficients.jpeg", dpi=300, bbox_inches="tight")

        plt.show()

    def plot_confusions_auto(self, save=False):
        save = save or self.save
        y_true = self.y_test_
        y_prob = self.probabilities_
        return super().plot_confusions(y_true, y_prob, save=save)
    
    def fetch_metrics_auto(self):
        y_true = self.y_test_
        y_pred = self.predictions_
        y_score = self.probabilities_
        return super().fetch_metrics(y_true, y_pred, y_score)

    def compute_threshold_auto(self):
        y_true = self.y_test_
        y_score = self.probabilities_
        return super().compute_threshold(y_true, y_score)


