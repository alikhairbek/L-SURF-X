# ==================== L-SURF-X v10 | Q1-READY PIPELINE ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from mapie.regression import MapieRegressor
import shap

from phreeqpy.iphreeqc.phreeqc_dll import IPhreeqc

print("="*100)
print("L-SURF-X v10 | Q1 READY (NO LEAKAGE + CV + BASELINES)")
print("="*100)

# ====================== 1. LOAD DATA ======================
df = pd.read_csv("Data.csv")
df = df[df["Mineral"] == "quartz"].reset_index(drop=True)

# ====================== 2. BASIC FEATURES ======================
df["Log_I"] = np.log10((df["Electrolyte1_val"] + df.get("Electrolyte2_val", 0)).clip(1e-8))
df["Log_CO2"] = np.log10(df["Gas3_val"].clip(1e-12))
df["Total_U"] = df["Aq_val"] + df["Sorbed_val"]
df["Log_Total_U"] = np.log10(df["Total_U"].clip(1e-10))
df["Total_Sites"] = df["Mineral_val"] * df["MineralSA"] * df["Mineralsites"] * 1e18 / 6.022e23
df["Log_Total_Sites"] = np.log10(df["Total_Sites"].clip(1e-12))

# ====================== 3. TRAIN / TEST SPLIT (BEFORE PHREEQC) ======================
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ====================== 4. PHREEQC FUNCTION ======================
ph = IPhreeqc()
ph.load_database("/content/phreeqci-3.8.8/phreeqc3/database/llnl.dat")

def run_phreeqc(row):
    try:
        input_str = f"""
SOLUTION 1
    pH {row['pH']}
    units mol/kgw
    Na {row['Electrolyte1_val']}
    Cl {row['Electrolyte1_val']}
    U(6) {row['Total_U']}
    C(4) {10**row['Log_CO2']}
SELECTED_OUTPUT
    -reset true
    -molalities UO2+2 UO2CO3 UO2(CO3)3-4
    -ionic_strength true
END
"""
        ph.run_string(input_str)
        out = ph.get_selected_output_array()
        h, v = out[0], out[1]
        return [
            np.log10(float(v[h.index("m_UO2+2(mol/kgw)")])),
            np.log10(float(v[h.index("m_UO2CO3(mol/kgw)")])),
            np.log10(float(v[h.index("m_UO2(CO3)3-4(mol/kgw)")])),
            np.log10(float(v[h.index("ionic_strength")]))
        ]
    except:
        return [-30, -30, -30, -30]

# ====================== 5. APPLY PHREEQC SEPARATELY ======================
def add_speciation(df_part):
    spec = np.array([run_phreeqc(row) for _, row in df_part.iterrows()])
    df_part[["Log_UO2","Log_UO2CO3","Log_UO2CO33","Log_I_phreeqc"]] = spec
    return df_part

print("🔬 Running PHREEQC for TRAIN...")
train_df = add_speciation(train_df)

print("🔬 Running PHREEQC for TEST...")
test_df = add_speciation(test_df)

# ====================== 6. FEATURES ======================
features = ["pH","Log_I","Log_CO2","Log_Total_U","Log_Total_Sites",
            "Log_UO2","Log_UO2CO3","Log_UO2CO33","Log_I_phreeqc"]

X_train = train_df[features]
y_train = np.log10(train_df["Aq_val"].clip(1e-10))

X_test = test_df[features]
y_test = np.log10(test_df["Aq_val"].clip(1e-10))

# ====================== 7. MODELS ======================
xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
lgb = LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)
cat = CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05, verbose=0)

stack = StackingRegressor(
    estimators=[('xgb', xgb), ('lgb', lgb), ('cat', cat)],
    final_estimator=Ridge(alpha=1.0),
    cv=5
)

baseline_lr = LinearRegression()

models = {
    "Linear": baseline_lr,
    "XGBoost": xgb,
    "LightGBM": lgb,
    "CatBoost": cat,
    "Stacking": stack
}

# ====================== 8. CROSS-VALIDATION ======================
print("\n CROSS-VALIDATION RESULTS (Train only)")
cv = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
    results[name] = (scores.mean(), scores.std())
    print(f"{name:10s} → R² = {scores.mean():.4f} ± {scores.std():.4f}")

# ====================== 9. FINAL TRAIN + TEST ======================
print("\n FINAL TEST RESULTS")

final_results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    final_results[name] = (r2, rmse)
    print(f"{name:10s} → R² = {r2:.4f} | RMSE = {rmse:.4f}")

# ====================== 10. STACKING + UNCERTAINTY ======================
mapie = MapieRegressor(stack, method="plus", cv=10)
mapie.fit(X_train, y_train)

y_pred, y_pis = mapie.predict(X_test, alpha=0.05)

# ====================== 11. PLOTS ======================
plt.figure()
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Measured")
plt.ylabel("Predicted")
plt.title("Parity Plot")
plt.savefig("Parity.png", dpi=600)
plt.show()

# ====================== 12. SHAP ======================
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)

# ====================================================
# PROFESSIONAL FIGURES FOR L-SURF-X v10 (SEPARATE FILES)
# No modification to original code; uses existing variables
# ====================================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------
# 1.numpy arrays
# ----------------------------------------------------
y_test_np = np.array(y_test).ravel()
y_pred_np = np.array(y_pred).ravel()
residuals_np = y_test_np - y_pred_np

#  X_test DataFrame 
X_test_df = X_test.copy() if hasattr(X_test, 'columns') else pd.DataFrame(X_test)

# ----------------------------------------------------
# General sit
# ----------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 600
})

# Color
COLOR_MAIN = "#2c7bb6"   
COLOR_RES = "#d7191c"     
COLOR_IMP = "#2ca02c"    
COLOR_COMP = "#9467bd"    

# ====================================================
# FIGURE A: Parity Plot (scatter with density coloring)
# ====================================================
fig_a, ax_a = plt.subplots(figsize=(5, 5))

# cal dinsty
xy = np.vstack([y_test_np, y_pred_np])
z = gaussian_kde(xy)(xy)
# first
idx = np.argsort(z)

sc = ax_a.scatter(y_test_np[idx], y_pred_np[idx], c=z[idx], s=30, 
                  cmap='viridis', alpha=0.8, edgecolors='k', linewidth=0.3)

# orginal
lims = [min(y_test_np.min(), y_pred_np.min()), max(y_test_np.max(), y_pred_np.max())]
ax_a.plot(lims, lims, 'r--', linewidth=1.5, alpha=0.7, label='Ideal')

# stander
r2_val = r2_score(y_test_np, y_pred_np)
rmse_val = np.sqrt(mean_squared_error(y_test_np, y_pred_np))
ax_a.text(0.05, 0.92, f"R² = {r2_val:.3f}\nRMSE = {rmse_val:.3f}", 
          transform=ax_a.transAxes, fontsize=10,
          bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))

ax_a.set_xlabel("Experimental log₁₀(Aq_val)")
ax_a.set_ylabel("Predicted log₁₀(Aq_val)")
ax_a.set_title("(A) Parity Plot (Stacking Model)")
ax_a.grid(True, linestyle='--', alpha=0.4)
ax_a.legend(loc='lower right')
cbar = fig_a.colorbar(sc, ax=ax_a, shrink=0.7)
cbar.set_label('Point Density')

plt.tight_layout()
plt.savefig("Figure_A_Parity.png", dpi=600, bbox_inches='tight')
plt.show()
plt.close()

# ====================================================
# FIGURE B: Residuals Distribution with Normal Fit
# ====================================================
fig_b, ax_b = plt.subplots(figsize=(5, 4))

# H 
sns.histplot(residuals_np, bins=25, kde=True, stat='density', 
             color=COLOR_RES, alpha=0.6, edgecolor='black', ax=ax_b)

# b
mu, std = norm.fit(residuals_np)
x_norm = np.linspace(residuals_np.min(), residuals_np.max(), 200)
ax_b.plot(x_norm, norm.pdf(x_norm, mu, std), 'b--', linewidth=2, 
          label=f'Normal fit (μ={mu:.2f}, σ={std:.2f})')

ax_b.axvline(0, color='k', linestyle='--', linewidth=1.5, alpha=0.7)
ax_b.set_xlabel("Residuals (Predicted - Experimental)")
ax_b.set_ylabel("Density")
ax_b.set_title("(B) Residuals Distribution")
ax_b.legend(loc='upper right')
ax_b.text(0.05, 0.92, f"Mean = {mu:.3f}\nStd  = {std:.3f}", 
          transform=ax_b.transAxes, fontsize=9,
          bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))
ax_b.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig("Figure_B_Residuals.png", dpi=600, bbox_inches='tight')
plt.show()
plt.close()

# ====================================================
# FIGURE C: Feature Importance (XGBoost)
# ====================================================
#  feature_importances_
if hasattr(xgb, 'feature_importances_'):
    importances = xgb.feature_importances_
    feature_names = X_test_df.columns
    sorted_idx = np.argsort(importances)
    
    fig_c, ax_c = plt.subplots(figsize=(5, 4))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(importances)))
    ax_c.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx], 
              color=colors[sorted_idx], edgecolor='black')
    ax_c.set_xlabel("Importance")
    ax_c.set_title("(C) Feature Importance (XGBoost)")
    ax_c.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("Figure_C_FeatureImportance.png", dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()
else:
    print("Feature importance not available for XGBoost.")

# ====================================================
# FIGURE D: Model Comparison (Bar chart with CV error bars)
# ====================================================
model_names = list(final_results.keys())
r2_test = [final_results[m][0] for m in model_names]
#  CV results 
cv_means = [results[m][0] for m in model_names if m in results]
cv_stds = [results[m][1] for m in model_names if m in results]

fig_d, ax_d = plt.subplots(figsize=(6, 4))
x_pos = np.arange(len(model_names))
width = 0.6

#  R² with CV
bars = ax_d.bar(x_pos, r2_test, width, yerr=cv_stds if len(cv_stds)==len(model_names) else None,
                capsize=5, color=COLOR_COMP, alpha=0.8, edgecolor='black',
                label='Test R²')
ax_d.set_xticks(x_pos)
ax_d.set_xticklabels(model_names, rotation=45, ha='right')
ax_d.set_ylabel("R² Score")
ax_d.set_ylim(0, 1.05)
ax_d.set_title("(D) Model Performance Comparison")
ax_d.grid(axis='y', linestyle='--', alpha=0.3)

#  R² 
for bar, val in zip(bars, r2_test):
    ax_d.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', 
              ha='center', va='bottom', fontsize=9)

# c
ax_d.legend(loc='lower right')

plt.tight_layout()
plt.savefig("Figure_D_ModelComparison.png", dpi=600, bbox_inches='tight')
plt.show()
plt.close()

# ====================================================
# ADDITIONAL FIGURE 1: Prediction Intervals (Mapie)
# ====================================================
try:
    # y_pis  (y_pis shape: (n_samples, 2, 1))
    y_pred_mapie = np.array(y_pred).ravel()
    lower = y_pis[:, 0, 0]
    upper = y_pis[:, 1, 0]
    
    fig_e, ax_e = plt.subplots(figsize=(6, 5))
    # G
    order = np.argsort(y_pred_mapie)
    x_ord = np.arange(len(y_pred_mapie))
    ax_e.fill_between(x_ord, lower[order], upper[order], alpha=0.3, color='gray', label='95% PI')
    ax_e.scatter(x_ord, y_test_np[order], s=10, alpha=0.6, color=COLOR_MAIN, label='Experimental')
    ax_e.scatter(x_ord, y_pred_mapie[order], s=10, alpha=0.6, color='red', marker='x', label='Predicted')
    ax_e.set_xlabel("Sample Index (sorted by prediction)")
    ax_e.set_ylabel("log₁₀(Aq_val)")
    ax_e.set_title("Stacking with Prediction Intervals (α=0.05)")
    ax_e.legend(loc='best')
    ax_e.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("Figure_E_UncertaintyIntervals.png", dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()
except Exception as e:
    print(f"Could not create uncertainty plot: {e}")

# ====================================================
# ADDITIONAL FIGURE 2: Learning Curves (if possible)
# ====================================================
#  (Stacking)  validation curve
from sklearn.model_selection import learning_curve

def plot_learning_curve(model, X, y, title, figsize=(6,4)):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2', random_state=42, n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training R²')
    ax.plot(train_sizes, test_mean, 'o-', color='orange', label='Cross-validation R²')
    ax.set_xlabel('Training examples')
    ax.set_ylabel('R²')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig

#    (Stacking) 
try:
    fig_lc = plot_learning_curve(stack, X_train, y_train, "Learning Curve (Stacking)")
    fig_lc.savefig("Figure_F_LearningCurve.png", dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()
except Exception as e:
    print(f"Learning curve not generated: {e}")

# ====================================================
# ADDITIONAL FIGURE 3: Actual vs Predicted (sorted)
# ====================================================
fig_g, ax_g = plt.subplots(figsize=(6, 4))
order = np.argsort(y_test_np)
ax_g.plot(np.arange(len(y_test_np)), y_test_np[order], 'o-', label='Experimental', markersize=3, linewidth=0.5)
ax_g.plot(np.arange(len(y_test_np)), y_pred_np[order], 's-', label='Predicted', markersize=3, linewidth=0.5)
ax_g.set_xlabel("Sample Index (sorted by experimental value)")
ax_g.set_ylabel("log₁₀(Aq_val)")
ax_g.set_title("Actual vs Predicted (Sorted)")
ax_g.legend()
ax_g.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig("Figure_G_ActualVsPredicted.png", dpi=600, bbox_inches='tight')
plt.show()
plt.close()

print("\n All figures have been generated and saved as PNG files (600 dpi).")
print("Files: Figure_A_Parity.png, Figure_B_Residuals.png, Figure_C_FeatureImportance.png,\n"
      "       Figure_D_ModelComparison.png, Figure_E_UncertaintyIntervals.png,\n"
      "       Figure_F_LearningCurve.png, Figure_G_ActualVsPredicted.png")

# ====================== 13. SAVE TABLE ======================
table = pd.DataFrame({
    "Model": list(final_results.keys()),
    "R2": [v[0] for v in final_results.values()],
    "RMSE": [v[1] for v in final_results.values()]
})

table.to_excel("Results.xlsx", index=False)

print("\n EVERYTHING DONE - READY")