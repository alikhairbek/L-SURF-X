# L-SURF-X

# L-SURF-X: Ensemble Machine Learning Framework for U(VI) Adsorption Prediction

**L-SURF-X** is a chemistry-informed ensemble machine learning framework that integrates **PHREEQC aqueous speciation** with a **stacking regressor** (XGBoost + LightGBM + CatBoost + Ridge meta-learner) for high-accuracy prediction and uncertainty quantification of U(VI) adsorption on quartz.

This repository accompanies the manuscript:  
**"L-SURF-X: An Ensemble Machine Learning Framework Integrating PHREEQC Aqueous Speciation for Improved Prediction and Uncertainty Quantification of U(VI) Adsorption on Quartz"**

---

### ✨ Key Features

- Full integration with **PHREEQC** (llnl.dat) for aqueous speciation  
- Stacking ensemble model (XGBoost + LightGBM + CatBoost)  
- **Conformal Prediction** → calibrated 95% prediction intervals  
- **SHAP** interpretability analysis  
- Strict **no data leakage** (train/test split before speciation)  
- Publication-ready figures (4×3 inch, 800 dpi)  
- Automatic export of comparison tables to Excel & Word  

### 📊 Performance (Test Set)

- **Stacking Model**: **R² = 0.9632** | RMSE = 0.2041 (log₁₀[Aq])  
- Conformal Coverage (95% PI): **0.952**

---

### 📁 Data

**Data.csv** contains the complete high-quality U(VI)–quartz subset of the **L-SCIE** (LLNL Surface Complexation/Ion Exchange) database.

> This data upload includes a compilation of experiments for quantifying uranium adsorption onto quartz. The provided .csv file has been compiled from the literature in a findable, accessible, interoperable, reusable (**FAIR**) data format. This was accomplished using the Lawrence Livermore National Laboratory Surface Complexation Database Converter (SCDC) code written in the R programming language. This constitutes all current data compiled on uranium-quartz interactions in the L-SCIE database (as of 02/28/2022). This data was used to develop surface complexation models that fit the global community dataset. The FAIR-formatted dataset also enables the implementation of alternative machine-learning approaches.

**Permanent DOI**: [https://data.ess-dive.lbl.gov/view/doi:10.15485/1880687](https://data.ess-dive.lbl.gov/view/doi:10.15485/1880687)

**Original Publication**:  
Zavarin et al. (2022) – *Community Data Mining Approach for Surface Complexation Database Development*. Environ. Sci. Technol. https://doi.org/10.1021/acs.est.1c07109

---

### 🚀 Quick Start

```bash
git clone https://github.com/yourusername/L-SURF-X.git
cd L-SURF-X
pip install -r requirements.txt
python main.py
```

---

### 📁 Repository Structure

```
L-SURF-X/
├── Data.csv                          # U(VI)-quartz subset (L-SCIE, FAIR)
├── main.py                           # Full reproducible pipeline
├── lsurfx/                           # Core package (optional)
├── Fig1_Conformal.png
├── Fig2_Parity.png
├── Fig3_Residuals.png
├── Fig4_SHAP.png
├── Fig5_Model_Comparison.png
├── Model_Comparison_Table.xlsx
├── Model_Comparison_Table.docx
├── Results.xlsx
├── requirements.txt
├── README.md
└── LICENSE
```

---

### 📈 Publication-Ready Figures

All figures are generated at **800 dpi** and sized **4×3 inches** (or 4×4 for parity plots) with font size 12 — ready for direct insertion into Word or LaTeX.

---

### 📜 Citation

**Please cite both the paper and the data:**

```bibtex
@article{Khairbek2026,
  title   = {L-SURF-X: An Ensemble Machine Learning Framework Integrating PHREEQC Aqueous Speciation for Improved Prediction and Uncertainty Quantification of U(VI) Adsorption on Quartz},
  author  = {Khairbek, Ali A. and Thomas, Renjith },
  journal = {Environmental Science & Technology},
  year    = {2026},
  doi     = { }
}

@dataset{Zavarin2022,
  author    = {Zavarin, Mavrik and Chang, Eric and others},
  title     = {L-SCIE U(VI)-Quartz Adsorption Data},
  year      = {2022},
  doi       = {10.15485/1880687},
  url       = {https://data.ess-dive.lbl.gov/view/doi:10.15485/1880687}
}
```

---

### 📜 License

MIT License — see [LICENSE](LICENSE) file.

---

### 🧪 Reproducibility

- Fixed random seed (`random_state=42`)
- No data leakage (PHREEQC speciation applied **after** train/test split)
- Full environment setup script provided

---

### 📬 Contact & Contributions

- **Ali A. Khairbek** — `alikhairbek@gmail.com`
- **Prof. Renjith Thomas** — `renjith@sbcollege.ac.in`

Contributions and issues are welcome!

---


هذا الـ README الآن **مثالي** لمستودع يرافق ورقة Q1:
- يحتوي على رابط DOI الرسمي لـ Data.csv
- وصف FAIR كامل
- اقتباسات واضحة
- هيكل احترافي

هل تريد نسخة **بالعربية** أيضاً، أو تعديل أي جزء (مثل إضافة رابط GitHub الفعلي)؟  
اكتب "موافق" أو "عدّل ..." وسأعدّله فوراً.
