#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd

def generate_eda_stress_test(n=1000, random_state=42):
    """
    Create a binary target plus 5 numeric and 5 categorical features with varying relationship strengths.

    Columns:
      target  -> Bernoulli(0.5)
      num_strong_pos   -> strong positive shift when target=1
      num_mod_neg      -> moderate negative shift when target=1
      num_weak_pos     -> small positive shift when target=1
      num_low_snr      -> positive shift but high variance (harder signal)
      num_noise        -> pure noise
      cat_strong       -> strong association (3 levels) with target
      cat_moderate     -> moderate association (2 levels) with target
      cat_weak         -> weak association (3 levels) with target
      cat_high_card    -> 8 levels; mild skew tied to target
      cat_rare_signal  -> rare level enriched when target=1

    Returns:
      pandas.DataFrame of shape (n, 11)
    """
    rng = np.random.default_rng(random_state)

    # Binary target
    target = rng.binomial(1, 0.5, size=n)

    # ----- Numeric features -----
    # Strong positive: large mean separation
    num_strong_pos = rng.normal(loc=np.where(target==1, 2.5, -2.5), scale=1.0)

    # Moderate negative: moderate separation opposite direction
    num_mod_neg = rng.normal(loc=np.where(target==1, -1.2, 1.2), scale=1.0)

    # Weak positive: small mean shift
    num_weak_pos = rng.normal(loc=np.where(target==1, 0.4, 0.0), scale=1.0)

    # Low SNR positive: mean shift exists but variance high
    num_low_snr = rng.normal(loc=np.where(target==1, 1.0, 0.0), scale=2.5)

    # Noise: no relationship
    num_noise = rng.normal(loc=0.0, scale=1.0, size=n)

    # ----- Categorical features -----
    # Strong (3 levels): markedly different distributions by target
    cat_strong = np.where(
        target==1,
        rng.choice(["A","B","C"], size=n, p=[0.65,0.25,0.10]),
        rng.choice(["A","B","C"], size=n, p=[0.20,0.50,0.30])
    )

    # Moderate (2 levels)
    cat_moderate = np.where(
        target==1,
        rng.choice(["X","Y"], size=n, p=[0.35,0.65]),
        rng.choice(["X","Y"], size=n, p=[0.60,0.40])
    )

    # Weak (3 levels)
    cat_weak = np.where(
        target==1,
        rng.choice(["U","V","W"], size=n, p=[0.37,0.33,0.30]),
        rng.choice(["U","V","W"], size=n, p=[0.34,0.33,0.33])
    )

    # High-cardinality (8 levels) with mild skew toward first 3 when target=1
    levels_hc = [f"L{i}" for i in range(1,9)]
    p_t1 = np.array([0.14,0.14,0.14,0.12,0.12,0.12,0.11,0.11])  # sums to 1
    p_t0 = np.array([0.11,0.11,0.11,0.13,0.13,0.13,0.14,0.14])  # sums to 1
    cat_high_card = np.where(
        target==1,
        rng.choice(levels_hc, size=n, p=p_t1),
        rng.choice(levels_hc, size=n, p=p_t0)
    )

    # Rare level enriched for target=1
    cat_rare_signal = np.where(
        target==1,
        rng.choice(["R1","R2","R3"], size=n, p=[0.08,0.87,0.05]),
        rng.choice(["R1","R2","R3"], size=n, p=[0.02,0.96,0.02])
    )

    df = pd.DataFrame({
        "target": target,
        "num_strong_pos": num_strong_pos,
        "num_mod_neg": num_mod_neg,
        "num_weak_pos": num_weak_pos,
        "num_low_snr": num_low_snr,
        "num_noise": num_noise,
        "cat_strong": cat_strong,
        "cat_moderate": cat_moderate,
        "cat_weak": cat_weak,
        "cat_high_card": cat_high_card,
        "cat_rare_signal": cat_rare_signal,
    })

    return df


# In[6]:


generate_eda_stress_test(n=1000, random_state=42)


# In[8]:


df = generate_eda_stress_test(n=1000, random_state=42)


# In[9]:


df.head()


# In[10]:


df.info


# In[11]:


df.isna().isna().sum()


# In[12]:


df.describe()


# In[74]:


import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, FunctionTransformer
from scipy.stats import kendalltau, pointbiserialr, mannwhitneyu, t, skew, kurtosis, anderson

def num_bin_trans(train_df
                  , feature_column
                  , target_column
                  , no_transform=True
                  , standardize=True
                  , minmax=True
                  , yeo_johnson=True
                  , winsor_5pct=True
                  , winsor_10pct=True
                  , ln=True
                  , log10=True
                  , root_2=True
                  , root_3=True
                  , n_splits=5
                  , random_state=42
                  , plot=True
                  , return_fold_data=False):
    
    """
    Cross-validated effect sizes and normality measures (Anderson–Darling for normality)
    for a numeric feature vs binary target.

    Effect sizes (per fold, with correct per-fold p-values):
      - Rank Biserial via Mann–Whitney U  (p from mannwhitneyu)
      - Kendall's Tau                     (p from kendalltau)
      - Point-biserial r                  (p from t-test on r, df = n_fold - 2)

    Normality (per fold):
      - Skewness, excess Kurtosis
      - Anderson–Darling statistic for normality (AD_stat) and approximate p-value (AD_pval)
        using Stephens (1974) approximation with finite-n correction A2* = A2 * (1 + 4/n - 25/n^2).
    """

    def _ad_normal_pvalue(a2, n):
        
        """ Approximate p-value for Anderson–Darling normality test """
        
        a2a = a2 * (1.0 + 4.0 / n - 25.0 / (n ** 2))
        if a2a < 0.2:
            p = 1 - np.exp(-13.436 + 101.14 * a2a - 223.73 * a2a ** 2)
        elif a2a < 0.34:
            p = 1 - np.exp(-8.318 + 42.796 * a2a - 59.938 * a2a ** 2)
        elif a2a < 0.6:
            p = np.exp(0.9177 - 4.279 * a2a - 1.38 * a2a ** 2)
        else:
            p = np.exp(1.2937 - 5.709 * a2a + 0.0186 * a2a ** 2)
        return float(np.clip(p, 0.0, 1.0))

    # Check Inputs
    if feature_column not in train_df.columns or target_column not in train_df.columns:
        raise ValueError("Column(s) not found in dataset")
    data = train_df[[feature_column, target_column]].dropna()
    if len(data) < 10:
        raise ValueError(f"Insufficient data: {len(data)} rows")
    X = data[feature_column].astype(float).values
    y = data[target_column].astype(int).values
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Target must be binary {0,1}")
    if np.min(np.bincount(y)) < n_splits:
        warnings.warn("Small class size may cause CV issues")

    # Run Transforms
    shift = max(0.0, -np.min(X) + 1e-6) if np.min(X) <= 0 else 0.0
    transforms = {
        'original': FunctionTransformer(validate=False),
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'yeo_johnson': PowerTransformer(method="yeo-johnson", standardize=False),
        'winsor_5pct': FunctionTransformer(lambda z: np.clip(z, np.quantile(z, 0.05), np.quantile(z, 0.95)),
                                           validate=False),
        'winsor_10pct': FunctionTransformer(lambda z: np.clip(z, np.quantile(z, 0.10), np.quantile(z, 0.90)),
                                            validate=False),
        'ln': FunctionTransformer(lambda z: np.log(np.maximum(z + shift, 1e-10)), validate=False),
        'log10': FunctionTransformer(lambda z: np.log10(np.maximum(z + shift, 1e-10)), validate=False),
        'sqrt': FunctionTransformer(lambda z: np.sign(z) * np.sqrt(np.abs(z)) if np.any(z < 0)
                                    else np.sqrt(np.maximum(z, 0)), validate=False),
        'cbrt': FunctionTransformer(lambda z: np.cbrt(z), validate=False)
    }
    selected = [name for enabled, name in zip(
        [no_transform, standardize, minmax, yeo_johnson, winsor_5pct, winsor_10pct, ln, log10, root_2, root_3],
        ['original', 'standard', 'minmax', 'yeo_johnson', 'winsor_5pct', 'winsor_10pct', 'ln', 'log10', 'sqrt', 'cbrt']
    ) if enabled]

    # Cross Validate
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results, transform_data, failed = [], {name: [] for name in selected}, set()

    for name in selected:
        fold_num = 0
        for tr_idx, va_idx in cv.split(X, y):
            fold_num += 1
            try:
                base_tf = transforms[name]
                tf = base_tf.__class__(**base_tf.get_params()) if hasattr(base_tf, 'get_params') else base_tf
                X_tr, X_val, y_val = X[tr_idx].reshape(-1, 1), X[va_idx].reshape(-1, 1), y[va_idx]
                X_t = tf.fit(X_tr).transform(X_val).ravel()

                if len(np.unique(X_t)) <= 1 or not np.all(np.isfinite(X_t)):
                    failed.add(name)
                    continue

                # Store Transformed Values for Plotting
                for i, v in enumerate(X_t):
                    transform_data[name].append({'value': v, 'target': y_val[i], 'fold': fold_num})

                # Stats Per-Fold
                out = {'Transform': name, 'Fold': fold_num, 'n_fold': len(y_val)}

                # Kendall's Tau
                try:
                    tau, p_tau = kendalltau(X_t, y_val)
                    out.update({'KendallTau': tau, 'KT_pval': p_tau})
                except Exception:
                    out.update({'KendallTau': np.nan, 'KT_pval': np.nan})

                # Rank Biserial derived from Mann–Whitney U
                try:
                    g0, g1 = X_t[y_val == 0], X_t[y_val == 1]
                    if len(g0) > 0 and len(g1) > 0:
                        U, p_rb = mannwhitneyu(g0, g1, alternative="two-sided")
                        rb = 1.0 - (2.0 * U) / (len(g0) * len(g1))
                        out.update({'RankBiserial': rb, 'RB_pval': p_rb})
                    else:
                        out.update({'RankBiserial': np.nan, 'RB_pval': np.nan})
                except Exception:
                    out.update({'RankBiserial': np.nan, 'RB_pval': np.nan})

                # Point-Biserial Correlation P-values per fold
                try:
                    r_pb, _ = pointbiserialr(y_val, X_t)
                    n = len(y_val)
                    if not np.isnan(r_pb) and abs(r_pb) < 1.0 and n > 2:
                        t_stat = r_pb * np.sqrt((n - 2) / (1 - r_pb ** 2))
                        p_pb = 2 * (1 - t.cdf(abs(t_stat), df=n - 2))
                        out.update({'PointBiserial_r': r_pb, 'RP_pval': p_pb})
                    else:
                        out.update({'PointBiserial_r': r_pb, 'RP_pval': np.nan})
                except Exception:
                    out.update({'PointBiserial_r': np.nan, 'RP_pval': np.nan})

                # Normality: Skew, Kurtosis, Anderson–Darling (Stat + ~ p-value)
                try:
                    skw = skew(X_t)
                    eks = kurtosis(X_t, fisher=True)
                    ad_res = anderson(X_t, dist='norm')
                    ad_stat = float(ad_res.statistic)
                    ad_p = _ad_normal_pvalue(ad_stat, len(X_t))
                    out.update({'Skewness': skw, 'Kurtosis': eks, 'AD_stat': ad_stat, 'AD_pval': ad_p})
                except Exception:
                    out.update({'Skewness': np.nan, 'Kurtosis': np.nan, 'AD_stat': np.nan, 'AD_pval': np.nan})

                results.append(out)

            except Exception as e:
                print(f"Transform '{name}' failed on fold {fold_num}: {e}")
                failed.add(name)

    if not results:
        raise ValueError("No transforms produced valid results")
    if failed:
        print(f"Failed transforms: {', '.join(sorted(failed))}")

    # Aggregate
    fold_df = pd.DataFrame(results)
    available = set(fold_df.columns)
    agg_cols = ['KendallTau', 'KT_pval', 'RankBiserial', 'RB_pval',
                'PointBiserial_r', 'RP_pval', 'Skewness', 'Kurtosis', 'AD_stat', 'AD_pval', 'n_fold']
    agg = {c: ('mean' if c == 'n_fold' else 'median') for c in agg_cols if c in available}
    summary = fold_df.groupby('Transform').agg(agg).reindex(selected)

    # Formatting
    for col in ['KendallTau', 'RankBiserial', 'PointBiserial_r', 'Skewness', 'Kurtosis', 'AD_stat', 'n_fold']:
        if col in summary.columns:
            summary[col] = summary[col].round(5)
    for col in ['KT_pval', 'RB_pval', 'RP_pval', 'AD_pval']:
        if col in summary.columns:
            summary[col] = summary[col].round(5)

    # Plot Distribution by Class Per Transform
    if plot and any(transform_data.values()):
        n_trans = len([n for n in selected if transform_data[n]])
        fig, ax = plt.subplots(figsize=(max(16, n_trans * 2), 10))
        fig.patch.set_facecolor('#FAFAFA'); ax.set_facecolor('#FFFFFF')

        plot_data, plot_labels, plot_positions, all_vals, pos = [], [], [], [], 0
        for name in selected:
            if not transform_data[name]:
                continue
            df_t = pd.DataFrame(transform_data[name])
            if df_t.empty or not df_t['value'].notna().any():
                continue
            for cls, suffix in [(0, '_0'), (1, '_1')]:
                vals = df_t[df_t['target'] == cls]['value'].values
                if len(vals) > 0:
                    plot_data.append(vals)
                    plot_labels.append(f"{name}{suffix}")
                    plot_positions.append(pos)
                    all_vals.extend(vals)
                    pos += 1
            pos += 1.0

        if plot_data:
            bp = ax.boxplot(plot_data, positions=plot_positions, labels=plot_labels,
                            patch_artist=True, widths=0.7, showfliers=True,
                            flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.6,
                                            markeredgecolor='black'))
            # style
            colors = {'0': {'face': '#87CEEB', 'edge': '#4682B4'},
                      '1': {'face': '#F08080', 'edge': '#CD5C5C'}}
            for patch, lab in zip(bp['boxes'], plot_labels):
                c = colors['0' if lab.endswith('_0') else '1']
                patch.set_facecolor(c['face']); patch.set_edgecolor(c['edge'])
                patch.set_alpha(0.8); patch.set_linewidth(1.5)
            for el in ['whiskers', 'caps']:
                for ln in bp[el]:
                    ln.set_color('black'); ln.set_linewidth(1.2)
            for med in bp['medians']:
                med.set_color('darkred'); med.set_linewidth(2.2)

            ax.set_title(f"Distribution of Transformed {feature_column} by Target Class",
                         fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
            ax.set_xlabel("Transform_TargetClass (0=Blue, 1=Red)", fontsize=12, color='#34495E')
            ax.set_ylabel("Transformed Values", fontsize=12, color='#34495E')
            ax.tick_params(axis='x', rotation=45, labelsize=10, colors='#2C3E50')
            ax.tick_params(axis='y', labelsize=10, colors='#2C3E50')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8); ax.set_axisbelow(True)
            if len(all_vals) > 0:
                y_min, y_max = np.min(all_vals), np.max(all_vals)
                pad = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
                ax.set_ylim(y_min - pad, y_max + pad)
            plt.suptitle(f"{feature_column} vs {target_column} — Cross-Validated Transform Analysis",
                         fontsize=18, fontweight='bold', color='#2C3E50', y=0.98)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    return (summary, fold_df) if return_fold_data else summary



# In[75]:


df.dtypes


# In[76]:


num_bin_trans(
    df,
    "num_strong_pos",
    "target", 
    no_transform=True,
    standardize=True,
    minmax=True,
    yeo_johnson=True,
    winsor_5pct=True,
    winsor_10pct=True,
    ln=True,
    log10=True, 
    root_2=True,
    root_3=True,
    n_splits=5,
    random_state=42,
    plot=True
)


# In[ ]:




