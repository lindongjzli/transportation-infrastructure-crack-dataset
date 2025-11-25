# Transportation Infrastructure Crack Dataset

> A comprehensive and multi-scenario dataset designed for robust infrastructure defect detection.

## 📖 Overview

This repository hosts the Transportation Infrastructure Crack Dataset, a large-scale collection comprising 72,805 high-quality images across roads, bridges, and tunnels.

The dataset features 35,195 annotated instances, labeled based on a novel two-tier (coarse-to-fine) classification standard that is strictly aligned with engineering practice. Unlike traditional datasets, this collection captures infrastructure defects through diverse acquisition methods to ensure model robustness in real-world environments.

## ⚙️ Data Processing & Standardization

To ensure the highest quality of annotations and data consistency, the following preprocessing steps were applied:

* **Resolution Standardization:** All images were resized to a uniform resolution of **640×640 pixels**.
* **Data Balancing:** We addressed the "long-tail" problem to ensure balanced training. Categories with annotation counts less than **10%** of the most frequent category were excluded from the subsets.
* **Cleaning:** Rigorous removal of invalid data (blurred or duplicate images).

## 📊 Dataset Statistics

### Subclass-Level Annotations

The table below details the distribution of crack categories, including box counts and small object ratios.

| **Primary Category**     | **Subclass Name** | **Label Name** | **Box Count** | **Image Count** | **Small Object Count** | **Avg. Boxes/Image** | Small Object Ratio |
| ------------------------------ | ----------------------- | -------------------- | ------------------- | --------------------- | ---------------------------- | -------------------------- | ------------------ |
| **Crack**                | Network Crack           | `crack_web`        | 1,516               | 1,411                 | 0                            | 1.07                       | 0%                 |
|                                | Block Crack             | `crack_block`      | 964                 | 961                   | 0                            | 1.00                       | 0%                 |
|                                | Horizontal Crack        | `crack_horizontal` | 4,292               | 3,403                 | 1046                         | 1.26                       | 24.4%              |
|                                | Vertical Crack          | `crack_vertical`   | 4,128               | 3,307                 | 1027                         | 1.25                       | 24.9%              |
|                                | Edge Crack              | `crack_edge`       | 1,277               | 1,176                 | 99                           | 1.09                       | 7.8%               |
|                                | Linear/Irregular        | `crack_linear`     | 1,457               | 1,417                 | 6                            | 1.03                       | 0.4%               |
| **Concrete Wall Crack** | Concrete Crack          | `crack_concrete`   | 1,906               | 1,407                 | 259                          | 1.35                       | 13.6%              |
| **Patch**                | Patch                   | `patch`            | 1,975               | 1,679                 | 117                          | 1.18                       | 5.9%               |
| **Pothole**              | Dry Pothole             | `pothole_dry`      | 2,501               | 1,888                 | 880                          | 1.33                       | 35.2%              |
|                                | Water Pothole           | `pothole_watering` | 4,920               | 1,510                 | 2035                         | 3.26                       | 41.3%              |
| **Spalling**             | Spalling                | `spalling`         | 2,814               | 1,781                 | 455                          | 1.58                       | 16.2%              |
| **Exposed Bars**         | Exposed Bars            | `exposed_bars`     | 5,033               | 1,548                 | 3448                         | 3.25                       | 68.5%              |
| **Water Leakage**        | Water Leakage           | `water_leakage`    | 1,216               | 1,066                 | 48                           | 1.14                       | 3.9%               |
| **Efflorescence**        | Efflorescence           | `efflorescence`    | 1,196               | 525                   | 234                          | 2.28                       | 19.6%              |

---

## 🔀 Dataset Splitting Methodology

### The Philosophy: Why not Random Split?

Traditional methods typically employ a simple random split (e.g., 6:2:2 or 8:2). However, we argue that:

> Conventional approaches focus solely on data volume balance, neglecting the **semantic heterogeneity** inherent in infrastructure crack scenarios. **This leads to evaluation bias.

### Proposed Method: Stratified Scenario Division

To comprehensively assess model robustness, we implemented a  **background-semantic-based stratified dataset division method** :

1. **Scenario Isolation:** The dataset is first divided into four distinct scenario-based subsets.
2. **Balanced Allocation:** Within each subset, a stratified **8:2 split** is performed. **This balances training efficiency with evaluation reliability while preserving the independence of data distributions across scenarios**.

### Partition Overview

| **Subset Name**       | **Scenario Code** | **Annotation Granularity** | **Training Set** | **Validation Set** | **Total** |
| --------------------------- | ----------------------- | -------------------------------- | ---------------------- | ------------------------ | --------------- |
| **Close-up Static**   | `CS`                  | Primary / Subclass               | 17,045                 | 4,261                    | 21,306          |
| **Contextual Static** | `CTS`                 | Primary / Subclass               | 5,132                  | 1,282                    | 6,414           |
| **Dynamic Vehicle**   | `DV`                  | Primary / Subclass               | 15,783                 | 3,945                    | 22,580          |
| **Dynamic Aerial**    | `DA`                  | Primary / Subclass               | 1,573                  | 393                      | 1,966           |

### Category Distribution per Subset

The following charts illustrate the proportion of annotation categories within each specific sub-dataset:

![image](https://github.com/lindongjzli/transportation-infrastructure-crack-dataset/blob/master/image/Proportion%20of%20annotation%20categories%20in%20each%20sub-dataset.png)

---

## 📥 Download

The full dataset is available for download via Google Drive.

> [Download Dataset (Google Drive)](https://drive.google.com/file/d/15pi8nMlJp4P9VlVTRovpIKZLBa-pv9Bi/view?usp=drive_link)

---

### 📝 Citation

If you use this dataset in your research, please cite the following paper:

```
@article{YourName2025,
  title={A image dataset and benchmarks for transportation infrastructure crack detection},
  author={Linchao Li, Bangxing Li, Jiabao Xing and Bowen Du},
  journal={Scientific data},
  year={2025}
}
```
