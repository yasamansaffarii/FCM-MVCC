# MVCC-FCM: Multi-Layer Vulnerability and Cascading Collapse FCM Model for Educational Robotics Interventions


## Overview

This repository contains the complete implementation of the **Fuzzy Cognitive Map (FCM)** and scenario analysis framework described in the paper:

> **“A Multilayer Vulnerability and Cascading Collapse (MVCC) Model in Educational Robotics (STEM) Interventions”**  
> by Mohammad Pedrami, Yasaman Saffari, et al.

The model combines:

- A **meta‑synthesis** of 209 identified vulnerability factors (from a systematic literature review) organised into **five hierarchical layers** (from learner cognitive core to contextual/policy level).
- A **directed weighted graph** (FCM) where nodes represent layers, sub‑themes, and a final “breakdown point” (cascading collapse risk).
- Edge weights are derived from empirical frequencies, documented critical combinations, and semantic similarity (TF‑IDF) between sub‑themes.
- **Scenario analysis** to evaluate the steady‑state collapse risk under different intervention conditions (best‑case, worst‑case, and four specific critical‑combination scenarios).
- **Sensitivity analysis** of the FCM inference parameters (λ and γ).

All results—tables and publication‑ready figures (600 dpi PNG + vector PDF)—are automatically generated and saved.

---

## Authors

- **Mohammad Pedrami** — *First author* — Meta‑synthesis analysis & data preparation  
  [ORCID: 0009‑0002‑5611‑8238](https://orcid.org/0009-0002-5611-8238)

- **Dr. Yasaman Saffari** — *Corresponding author* — AI Researcher — FCM analysis & code implementation  
  [Personal website](https://yasamansaffarii.github.io)

---

## Requirements

- Python 3.9 or higher
- The following Python libraries (installed automatically if you run the setup cell):

```bash
pip install python-docx scikit-learn networkx scipy pandas matplotlib
```

All code is designed to run in **Google Colab** or a standard Jupyter notebook environment.

---

## Usage

### 1. Prepare the Input File

You need a Microsoft Word (`.docx`) file containing **two tables** in the exact order:

- **Table 0 (summary table)** : columns: `[main_theme_raw, sub_theme, total_frequency]`  
  (the official numeric source, grand total = 209).
- **Table 1 (full concepts table)** : columns: `[main_theme_raw, sub_theme, concepts_text, sources]`  
  where each sub‑theme entry lists individual concepts with their frequencies (e.g., “concept (frequency: X): description”).

The file name used in the code is `concepts.docx`.  
**Important:** The main theme names **must** exactly match the keywords used in the layer mapping (see code). The provided table structure must follow the reference version (both tables present, no extra header rows).

### 2. Run the Notebook

Execute the cells in order. In Colab, the first cell will install dependencies; then you will upload the Word document.

If you are running locally, place the file in the same directory and set:

```python
THEMES_DOCX = "concepts.docx"
```

Then run the entire script.

### 3. Output Files

All outputs are saved in the current working directory:

| File name                         | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| `table1_subtheme_stats.csv`       | Sub‑theme statistics: layer, total frequency, normalized vulnerability weight |
| `table2_weight_matrix.csv`        | Full FCM adjacency matrix (nodes × nodes) with edge weights                 |
| `table3_scenarios.csv`            | Scenario analysis: steady‑state collapse probability for each scenario      |
| `table4_sensitivity.csv`          | Sensitivity analysis over λ and γ parameters (baseline collapse risk)       |
| `table5_centrality.csv`           | Node centrality (in‑/out‑degree) and Kosko role (Transmitter/Receiver/Ordinary) |
| `table6_validation.txt`           | Pearson and Spearman correlations between centrality and raw frequency      |
| `fig1_fcm_network.png` / `.pdf`   | Multi‑layer concentric network layout (FCM graph)                          |
| `fig2_weight_heatmap.png` / `.pdf`| Heatmap of the weight matrix                                               |
| `fig3_scenario_comparison.png` / `.pdf`| Bar chart comparing collapse risk across scenarios                      |
| `fig4_convergence.png` / `.pdf`   | Convergence trajectories of the Tipping Point node                         |
| `fig5_centrality.png` / `.pdf`    | Horizontal bar chart of node centrality decomposition                      |
| `fig6_sensitivity.png` / `.pdf`   | Sensitivity plot for λ and γ parameters                                    |

All figures are rendered at **600 dpi** (print quality) and also as vector PDFs for publication.

---

## Code Structure

The implementation is organised in the following sequential blocks (as in the provided notebook):

1. **Install libraries & imports**  
2. **Load the Word document** – parse both tables and combine them.  
3. **Parse individual concepts** – extract name, frequency, and description.  
4. **Layer mapping** – assign each sub‑theme to one of the five theoretical layers (1 = core, 5 = outermost).  
5. **Sub‑theme statistics** – compute normalized vulnerability weights.  
6. **Define FCM nodes** – 5 layers + 12 sub‑themes + 1 output (breakdown point).  
7. **Build the weight matrix** – using:  
   - sub‑theme → layer edges (vulnerability weight)  
   - cascading layer edges (outer → inner)  
   - explicit feedback loop (L2 ↔ L3)  
   - core → output edge  
   - direct weak edges from each layer to output  
   - critical combination edges (documented synergies)  
   - TF‑IDF semantic similarity edges between cross‑layer sub‑themes.  
8. **FCM inference engine** – damped Kosko rule with sigmoid activation.  
9. **Run scenarios** – baseline, best‑case, worst‑case, and four critical‑combination scenarios.  
10. **Sensitivity analysis** – vary λ and γ.  
11. **Centrality metrics** – compute Kosko’s transmitter/receiver/ordinary roles.  
12. **Statistical validation** – correlate centrality with raw frequencies.  
13. **Generate all figures** with proper labels (English translations provided).

The code includes a commented‑out section for optional **LLM‑assisted edge refinement** using the Claude API (disabled by default).

---

## Customisation

- **Add or modify critical combinations** – edit the `CRITICAL_COMBOS` dictionary in the code.  
- **Adjust FCM parameters** – change `LAM` and `GAMMA` (default `0.7`, `0.1`).  
- **English labels** – the `SUB_LABELS_EN` dictionary maps Persian sub‑theme names to English translations; extend it if new sub‑themes appear.

---

## License

This project is made available under the **MIT License**. Feel free to use, modify, and distribute it, provided that you give appropriate credit to the authors.

---

## Citation

If you use this code in your research, please cite the original paper (once published) and/or this repository:

```bibtex
@article{PedramiSaffariMVCC,
  author = {Mohammad Pedrami and Yasaman Saffari},
  title  = {A Multilayer Vulnerability and Cascading Collapse (MVCC) Model in Educational Robotics (STEM) Interventions},
  journal= {},
  year   = {2026}
}
```

---

## Authors

Mohammad Pedrami — First-author — Meta-synthesis analysis & data preparing ORCID: https://orcid.org/0009-0002-5611-8238
Dr. Yasaman Saffari — Corresponding-author — Artificial Intelligence Researcher — FCM analysis & code implementation Personal website: https://yasamansaffarii.github.io
Iran
