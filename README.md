# Transportation Infrastructure Crack Dataset

> A comprehensive, multi-scenario, and multi-modality dataset designed for robust infrastructure defect detection.

## 📖 Overview

This repository hosts the  **Transportation Infrastructure Crack Dataset** , the first of its kind to systematically cover multiple scenarios, granularities, and modalities.

After a rigorous cleaning process—removing duplicate, blurred, and invalid images—the final dataset comprises  **72,805 high-quality images** . To facilitate engineering-oriented applications, we propose a  **background-semantic-based stratified division method** , ensuring that models trained on this data are robust and generalizable across real-world environments.

## ⚙️ Data Processing & Standardization

To ensure the highest quality of annotations and data consistency, the following preprocessing steps were applied:

* **Resolution Standardization:** All images were resized to a uniform resolution of **640×640 pixels**^1^.
* **Data Balancing:** We addressed the "long-tail" problem to ensure balanced training. **Categories with annotation counts less than ****10%** of the most frequent category were excluded from the subsets^2^.
* **Cleaning:** Rigorous removal of invalid data (blurred or duplicate images)^3^.

## 📊 Dataset Statistics

### Subclass-Level Annotations

The table below details the distribution of crack categories, including box counts and small object ratios.

| **Primary Category** | **Subclass Name** | **Label Name** | **Box Count** | **Image Count** | **Small Object Ratio** | **Avg. Boxes/Image** |
| -------------------------- | ----------------------- | -------------------- | ------------------- | --------------------- | ---------------------------- | -------------------------- |
| **Crack**            | Network Crack           | `crack_web`        | 1,516               | 1,411                 | 0%                           | 1.07                       |
|                            | Block Crack             | `crack_block`      | 964                 | 961                   | 0%                           | 1.00                       |
|                            | Horizontal Crack        | `crack_horizontal` | 4,292               | 3,403                 | 24.4%                        | 1.26                       |
|                            | Vertical Crack          | `crack_vertical`   | 4,128               | 3,307                 | 24.9%                        | 1.25                       |
|                            | Edge Crack              | `crack_edge`       | 1,277               | 1,176                 | 7.8%                         | 1.09                       |
|                            | Linear/Irregular        | `crack_linear`     | 1,457               | 1,417                 | 0.4%                         | 1.03                       |
| **Conc. Wall Crack** | Concrete Crack          | `crack_concrete`   | 1,906               | 1,407                 | 13.6%                        | 1.35                       |
| **Patch**            | Patch                   | `patch`            | 1,975               | 1,679                 | 5.9%                         | 1.18                       |
| **Pothole**          | Dry Pothole             | `pothole_dry`      | 2,501               | 1,888                 | 35.2%                        | 1.33                       |
|                            | Water Pothole           | `pothole_watering` | 4,920               | 1,510                 | 41.3%                        | 3.26                       |
| **Spalling**         | Spalling                | `spalling`         | 2,814               | 1,781                 | 16.2%                        | 1.58                       |
| **Exposed Bars**     | Exposed Bars            | `exposed_bars`     | 5,033               | 1,548                 | 68.5%                        | 3.25                       |
| **Water Leakage**    | Water Leakage           | `water_leakage`    | 1,216               | 1,066                 | 3.9%                         | 1.14                       |
| **Efflorescence**    | Efflorescence           | `efflorescence`    | 1,196               | 525                   | 19.6%                        | 2.28                       |

^4^

---

## 🔀 Dataset Splitting Methodology

### The Philosophy: Why not Random Split?

Traditional methods typically employ a simple random split (e.g., 6:2:2 or 8:2). However, we argue that:

> Conventional approaches focus solely on data volume balance, neglecting the **semantic heterogeneity** inherent in infrastructure crack scenarios. **This leads to evaluation bias. **^5^

### Proposed Method: Stratified Scenario Division

To comprehensively assess model robustness, we implemented a  **background-semantic-based stratified dataset division method** :

1. **Scenario Isolation:** The dataset is first divided into four distinct scenario-based subsets.
2. **Balanced Allocation:** Within each subset, a stratified **8:2 split** is performed. **This balances training efficiency with evaluation reliability while preserving the independence of data distributions across scenarios**^6^^6^^6^^6^.

### Partition Overview

| **Subset Name**       | **Scenario Code** | **Annotation Granularity** | **Training Set** | **Validation Set** | **Total** |
| --------------------------- | ----------------------- | -------------------------------- | ---------------------- | ------------------------ | --------------- |
| **Close-up Static**   | `CS`                  | Primary / Subclass               | 17,045                 | 4,261                    | 21,306          |
| **Contextual Static** | `CTS`                 | Primary / Subclass               | 5,132                  | 1,282                    | 6,414           |
| **Dynamic Vehicle**   | `DV`                  | Primary / Subclass               | 15,783                 | 3,945                    | 22,580          |
| **Dynamic Aerial**    | `DA`                  | Primary / Subclass               | 1,573                  | 393                      | 1,966           |

^7^

### Category Distribution per Subset

The following charts illustrate the proportion of annotation categories within each specific sub-dataset:

**(Note: Figure 9 from the original document serves as the visual reference here. **^8^^8^^8^^8^)

---

## 📥 Download

The full dataset is available for download via Google Drive.

> [Download Dataset (Google Drive)](https://drive.google.com/file/d/15pi8nMlJp4P9VlVTRovpIKZLBa-pv9Bi/view?usp=drive_link)^9^

---

### 📝 Citation

If you use this dataset in your research, please cite the following paper:

**代码段**

```
@article{YourName2024,
  title={Transportation Infrastructure Crack Dataset Description},
  author={Author Names},
  journal={Journal Name},
  year={2024}
}
```

### 💡 Next Steps for You

**Would you like me to generate a Python script (`dataloader.py`) specifically tailored to this directory structure (CS, CTS, DV, DA) to help users quickly load this dataset?**
