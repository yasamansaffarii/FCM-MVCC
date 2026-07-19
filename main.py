# -*- coding: utf-8 -*-
"""
==============================================================================
 Fuzzy Cognitive Map (FCM) and Scenario Analysis for a
 Multilayer Vulnerability & Cascading Collapse (MVCC) Model
 in Educational Robotics (STEM) Interventions


 When running the UPLOAD cell, upload the following file:
        - concepts.docx   (must contain both tables: the frequency
          summary table + the comprehensive concepts table)
 Run the remaining cells in order. All tables (CSV) and figures (PNG/PDF)
 will be saved in the current folder and are downloadable.
==============================================================================
"""

# %% [1] Install libraries -----------------------------------------
!pip install python-docx scikit-learn networkx scipy pandas matplotlib -q

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import docx

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["figure.dpi"] = 600

DPI = 600  # Q1 print standard

def save_fig(fig, name):
    """Save both a 600-dpi PNG and a vector PDF, with no clipped labels."""
    fig.savefig(f"{name}.png", dpi=DPI, bbox_inches="tight", pad_inches=0.8)
    fig.savefig(f"{name}.pdf", bbox_inches="tight", pad_inches=0.8)
    plt.close(fig)

# %% [2] Load input file --------------------------------------------------
# In Colab: use the file-upload widget, then set the path in THEMES_DOCX
#
#   from google.colab import files
#   uploaded = files.upload()
#
THEMES_DOCX = "concepts.docx"   # contains the summary table (Table 0) and the full table (Table 1)

pdig = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
def to_num(s):
    s = str(s).translate(pdig).strip()
    return int(s) if s else None

d = docx.Document(THEMES_DOCX)
assert len(d.tables) >= 2, "Input file structure does not match the reference version."
t_summary = d.tables[0]   # Summary table: main theme/sub-theme/total frequency (official numeric source)
t_full = d.tables[1]      # Full table: main theme/sub-theme/concepts/sources/total frequency

# --- Summary table (official, precise numbers) ---
rows_summary, last_main = [], None
for ri, row in enumerate(t_summary.rows):
    if ri == 0:
        continue
    cells = [c.text.strip() for c in row.cells]
    main_theme = cells[0] if cells[0] else last_main
    if cells[0]:
        last_main = cells[0]
    rows_summary.append({"main_theme_raw": main_theme, "sub_theme": cells[1],
                          "total_freq": to_num(cells[2])})
df_summary = pd.DataFrame(rows_summary)

# --- Full table (concept text + sources, for TF-IDF analysis and documentation) ---
rows_full, last_main = [], None
for ri, row in enumerate(t_full.rows):
    if ri == 0:
        continue
    cells = [c.text.strip() for c in row.cells]
    main_theme = cells[0] if cells[0] else last_main
    if cells[0]:
        last_main = cells[0]
    rows_full.append({"main_theme_raw": main_theme, "sub_theme": cells[1],
                       "concepts_text": cells[2], "sources": cells[3]})
df_full = pd.DataFrame(rows_full)

records = df_full.merge(df_summary[["sub_theme", "total_freq"]], on="sub_theme", how="left")
records = records.to_dict("records")
print(f"Number of sub-themes extracted: {len(records)}")
print(f"Grand total frequency (should be 209): {df_summary['total_freq'].sum()}")

# %% [3] Extract individual concepts (name, frequency, description) — no polarity -------------
# Note: this table has no facilitator/barrier label; all 209 frequencies
# are inherently "vulnerability/barrier" items, because the whole model
# is a collapse model, not an effectiveness model.
FREQ_PAT = re.compile(r"\(\s*(?:فراوانی:\s*)?([۰-۹0-9]+)\s*\):\s*")

def parse_concepts(text):
    parts = re.split(r"\s*/\s*(?=[^:]*?\(\s*(?:فراوانی:\s*)?[۰-۹0-9]+\s*\):)", text)
    out = []
    for p in parts:
        p = p.strip().strip(".").strip()
        matches = list(FREQ_PAT.finditer(p))
        if not matches:
            continue
        last = matches[-1]
        name = p[:last.start()].strip()
        freq = int(last.group(1).translate(pdig))
        rest = p[last.end():].strip()
        out.append({"name": name, "freq": freq, "desc": rest})
    return out

concepts = []
for r in records:
    for c in parse_concepts(r["concepts_text"]):
        c["main_theme_raw"] = r["main_theme_raw"]
        c["sub_theme"] = r["sub_theme"]
        concepts.append(c)
print(f"Number of individual concepts extracted (for text analysis): {len(concepts)}")

# %% [4] Layer mapping: raw main theme <-> theoretical layer number (1=core ... 5=outermost) ---
# The order of themes in the input table does not reflect the theoretical
# layer order (that order is simply an artifact of the meta-synthesis
# table layout); the mapping below follows the MVCC theoretical document
# and is validated against each layer's total frequency (36, 26, 49, 54, 44).
LAYER_KEYWORDS = [
    (1, "شناختی و دانش پیشین", "Learner cognitive & prior-knowledge vulnerabilities (core)"),
    (2, "انگیزشی، عاطفی و اجتماعی", "Motivational, emotional & social barriers"),
    (3, "نظام معلم و طراحی آموزشی", "Teacher-system & instructional-design shortcomings"),
    (4, "منابع، فناوری و زمان", "Resource, technology & time constraints"),
    (5, "زمینه‌ای، سیاستی و روش‌شناختی", "Contextual, policy & methodological shortcomings (outermost)"),
]
def get_layer(main_theme_text):
    for lyr, kw, _ in LAYER_KEYWORDS:
        if kw in main_theme_text:
            return lyr
    raise ValueError(f"No layer found for: {main_theme_text}")

layer_name = {lyr: name for lyr, _, name in LAYER_KEYWORDS}
df_summary["layer"] = df_summary["main_theme_raw"].apply(get_layer)

check = df_summary.groupby("layer")["total_freq"].sum()
expected = {1: 36, 2: 26, 3: 49, 4: 54, 5: 44}
for lyr, exp in expected.items():
    got = check.get(lyr, None)
    flag = "OK" if got == exp else "!! MISMATCH !!"
    print(f"Layer {lyr} ({layer_name[lyr]}): computed frequency={got} | expected={exp} [{flag}]")

# %% [5] Per-sub-theme statistics (frequency and relative vulnerability weight only) ---------
df_sub = df_summary.copy()
df_sub["vuln_weight"] = df_sub["total_freq"] / df_sub["total_freq"].max()  # normalized to [0,1]
print(df_sub[["layer", "sub_theme", "total_freq", "vuln_weight"]].to_string(index=False))
df_sub.to_csv("table1_subtheme_stats.csv", index=False, encoding="utf-8-sig")

sub_row = {row.sub_theme: row for row in df_sub.itertuples()}
sub_themes = df_sub["sub_theme"].tolist()

# %% [6] Define FCM nodes (5 layers + 12 sub-themes + 1 breakdown-point node) --------
node_labels, label_map_layer, label_map_sub = {}, {}, {}
nid = 0
for lyr in range(1, 6):
    node_labels[f"L{lyr}"] = layer_name[lyr]
    label_map_layer[lyr] = f"L{lyr}"
    nid += 1
for st in sub_themes:
    node_labels[f"S{nid}"] = st
    label_map_sub[st] = f"S{nid}"
    nid += 1
OUTCOME = "OUT"
node_labels[OUTCOME] = "Breakdown point / intervention cascading-collapse probability"
all_nodes = list(node_labels.keys())
n = len(all_nodes)
idx = {k: i for i, k in enumerate(all_nodes)}
print(f"\nTotal number of FCM nodes: {n}  (5 layers + 12 sub-themes + 1 output node)")

# %% [7] Build the FCM weight matrix ---------------------------------------------------
W = np.zeros((n, n))

# (a) sub-theme -> its own layer edge: weight = relative frequency (always
#     positive; each sub-theme's vulnerability directly adds to the
#     overall pressure on its layer)
for st in sub_themes:
    row = sub_row[st]
    W[idx[label_map_sub[st]], idx[label_map_layer[row.layer]]] = round(row.vuln_weight, 4)

# (b) directed cascading layer edges: outermost -> innermost
#     L5 -> L4 -> L3 -> L2 -> L1 ("cascading amplification" principle;
#     weight based on the relative frequency of the source layer's total
#     vs. the grand total frequency)
layer_total_freq = df_sub.groupby("layer")["total_freq"].sum()
grand_total = layer_total_freq.sum()
cascade_pairs = [(5, 4), (4, 3), (3, 2), (2, 1)]
for src, dst in cascade_pairs:
    w = round(layer_total_freq[src] / grand_total, 4)
    W[idx[f"L{src}"], idx[f"L{dst}"]] = w

# (c) explicit, documented bidirectional feedback loop between Layer 2
#     (learner) and Layer 3 (teacher). Rationale: "a learner who has lost
#     motivation (Layer 2) gives negative feedback to the teacher (Layer 3)
#     and further undermines the teacher's confidence." Coefficient 0.5
#     moderates this indirect feedback effect relative to the main cascade.
w23 = round(0.5 * layer_total_freq[2] / grand_total, 4)
W[idx["L2"], idx["L3"]] = w23

# (d) final edge from the system's core -> breakdown point: only Layer 1
#     (innermost, surrounding the central core) connects directly to the
#     "breakdown point" node; other layers only affect it indirectly
#     through the cascading chain.
w_core = round(layer_total_freq[1] / grand_total, 4)
W[idx["L1"], idx[OUTCOME]] = w_core

# (e) direct, weak edges from every layer to the breakdown point
#     (each layer's background/systemic contribution, independent of the
#     cascading chain; attenuated by distance from the core)
for lyr in range(2, 6):
    dist = lyr - 1  # distance to the Layer-1 core
    w = round((layer_total_freq[lyr] / grand_total) * (1 / (dist + 1)), 4)
    W[idx[f"L{lyr}"], idx[OUTCOME]] += w

# (f) special "critical combination" edges documented in the theoretical
#     document: direct synergy between specific sub-themes identified as
#     the deadliest combinations.
CRITICAL_COMBOS = {
    "مثلث مرگبار فنی-آموزشی": [
        "پیچیدگی‌ها، نقص‌ها و شکنندگی فناورانه",
        "صلاحیت و آمادگی ناکافی معلمان",
        "محدودیت‌های زمانی و فشار ساختاری",
    ],
    "تلهٔ شناختی-برنامه‌ای": [
        "محدودیت‌های طراحی آموزشی و برنامه درسی",
        "نابرابری در دانش و تجربه پیشین یادگیرندگان",
    ],
    "سیلوی جنسیتی-فرهنگی": [
        "موانع زمینه‌ای، فرهنگی و سیاستی",
        "محدودیت‌های طراحی آموزشی و برنامه درسی",
        "شکاف‌های جنسیتی و نابرابری در مشارکت",
    ],
    "حلقهٔ بی‌منبع-بی‌چشم": [
        "کمبود منابع، زیرساخت و فشارهای مالی",
        "نقاط ضعف روش‌شناختی و چالش‌های سنجش",
    ],
}
COMBO_EDGE_WEIGHT = 0.65  # documented synergy weight (higher than ordinary edges)
for combo_name, members in CRITICAL_COMBOS.items():
    for a in members:
        for b in members:
            if a == b:
                continue
            i, j = idx[label_map_sub[a]], idx[label_map_sub[b]]
            if W[i, j] == 0:
                W[i, j] = COMBO_EDGE_WEIGHT

# (g) cross-theme side edges: TF-IDF semantic similarity (the ML/NLP
#     component), only between sub-themes from different layers that
#     don't already have a critical-combo edge.
def tokenize_fa(text):
    text = re.sub(r"[^\u0600-\u06FFA-Za-z\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]

sub_docs = [" ".join(c["name"] + " " + c["desc"] for c in concepts if c["sub_theme"] == st)
            for st in sub_themes]
vec = TfidfVectorizer(tokenizer=tokenize_fa, lowercase=False, token_pattern=None)
X = vec.fit_transform(sub_docs)
sim = cosine_similarity(X)
sims_flat = sim[np.triu_indices(len(sub_themes), k=1)]
thr = sims_flat.mean() + 1.0 * sims_flat.std()
print(f"\nTF-IDF semantic-similarity threshold for accepting a side edge: {thr:.4f}")

cross_edges = []
for a in range(len(sub_themes)):
    for b in range(len(sub_themes)):
        if a == b:
            continue
        st_a, st_b = sub_themes[a], sub_themes[b]
        if sub_row[st_a].layer == sub_row[st_b].layer:
            continue
        s = sim[a, b]
        if s >= thr:
            i, j = idx[label_map_sub[st_a]], idx[label_map_sub[st_b]]
            if W[i, j] == 0:
                W[i, j] = round(s, 4)  # always positive: both sub-themes are vulnerabilities
                cross_edges.append((st_a, st_b, s, W[i, j]))
print(f"Number of side edges discovered by TF-IDF: {len(cross_edges)}")

pd.DataFrame(W, index=all_nodes, columns=all_nodes).to_csv("table2_weight_matrix.csv", encoding="utf-8-sig")
n_edges = int((W != 0).sum())
density = n_edges / (n * (n - 1))
print(f"Nodes={n} | Edges={n_edges} | Graph density={density:.4f}")

# %% [8] FCM inference engine (damped Kosko rule) and scenario analysis --------------
LAM, GAMMA = 0.7, 0.1  # selected from the sensitivity analysis (Section 9): stable, non-saturated point

def sigmoid(x, lam=LAM):
    return 1.0 / (1.0 + np.exp(-lam * x))

def run_fcm(A0, clamp_idx=(), clamp_val=(), lam=LAM, gamma=GAMMA, max_iter=300, tol=1e-7):
    A = A0.copy()
    for ci, cv in zip(clamp_idx, clamp_val):
        A[ci] = cv
    history = [A.copy()]
    converged_at = max_iter
    for t in range(max_iter):
        raw = gamma * A + A @ W
        A_new = sigmoid(raw, lam)
        for ci, cv in zip(clamp_idx, clamp_val):
            A_new[ci] = cv
        history.append(A_new.copy())
        if np.linalg.norm(A_new - A) < tol:
            A = A_new; converged_at = t + 1; break
        A = A_new
    return A, np.array(history), converged_at

def clampnodes(labels, val):
    ids = [idx[label_map_sub[l]] for l in labels]
    return ids, [val] * len(ids)

A0 = np.full(n, 0.5)

# Baseline scenario: no intervention, all nodes neutral
A_base, hist_base, it_base = run_fcm(A0.copy())

# Best-case scenario: a well-designed, well-resourced intervention ->
# all sub-theme vulnerabilities are actively suppressed
ci_low, cv_low = clampnodes(sub_themes, 0.15)
A_best, hist_best, it_best = run_fcm(A0.copy(), ci_low, cv_low)

# Worst-case scenario: a completely under-resourced/unplanned intervention
# -> all vulnerabilities are maximally activated
ci_high, cv_high = clampnodes(sub_themes, 0.9)
A_worst, hist_worst, it_worst = run_fcm(A0.copy(), ci_high, cv_high)

# Four critical-combo scenarios (each combo is activated in an otherwise
# neutral context)
combo_scenarios = {}
for combo_name, members in CRITICAL_COMBOS.items():
    ci, cv = clampnodes(members, 0.9)
    A_c, hist_c, it_c = run_fcm(A0.copy(), ci, cv)
    combo_scenarios[combo_name] = (A_c, hist_c, it_c)

scenarios = {
    "Baseline (no intervention)": (A_base, it_base),
    "Best-case (well-resourced intervention)": (A_best, it_best),
    "Worst-case (under-resourced/unplanned)": (A_worst, it_worst),
}
for combo_name, (A_c, _, it_c) in combo_scenarios.items():
    scenarios[combo_name] = (A_c, it_c)

rows = [{"Scenario": name, "Convergence iterations": it,
         "Collapse probability (breakdown point)": round(A[idx[OUTCOME]], 4),
         "Difference from baseline": round(A[idx[OUTCOME]] - A_base[idx[OUTCOME]], 4)}
        for name, (A, it) in scenarios.items()]
df_scn = pd.DataFrame(rows)
print(df_scn.to_string(index=False))
df_scn.to_csv("table3_scenarios.csv", index=False, encoding="utf-8-sig")

# %% [9] Sensitivity analysis of the inference parameters (λ, γ) ---------------
sens_rows = []
for lam in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
    for gamma in [0.05, 0.1, 0.2, 0.3]:
        A_b, _, it_b = run_fcm(A0.copy(), lam=lam, gamma=gamma)
        sens_rows.append({"lambda": lam, "gamma": gamma, "iters": it_b,
                           "collapse_risk_base": round(A_b[idx[OUTCOME]], 4)})
df_sens = pd.DataFrame(sens_rows)
df_sens.to_csv("table4_sensitivity.csv", index=False, encoding="utf-8-sig")
print(df_sens.to_string(index=False))

# %% [10] Network centrality metrics (transmitter/receiver/ordinary per Kosko's theory) ---------
out_degree = np.sum(np.abs(W), axis=1)
in_degree = np.sum(np.abs(W), axis=0)
rows = []
for k in all_nodes:
    i = idx[k]; od, idg = out_degree[i], in_degree[i]
    role = ("Transmitter" if od > 0 and idg == 0 else
             "Receiver" if od == 0 and idg > 0 else "Ordinary")
    rows.append({"node": k, "label": node_labels[k], "out_degree": round(od, 4),
                 "in_degree": round(idg, 4), "centrality": round(od + idg, 4), "role": role})
df_cent = pd.DataFrame(rows).sort_values("centrality", ascending=False)
df_cent.to_csv("table5_centrality.csv", index=False, encoding="utf-8-sig")
print(df_cent.to_string(index=False))

# %% [11] Statistical validation: correlation between FCM centrality and raw meta-synthesis frequency -------
sub_cent = df_cent[df_cent["node"].str.startswith("S")]
merged = sub_cent.merge(df_sub, left_on="label", right_on="sub_theme")
r_p, p_p = stats.pearsonr(merged["centrality"], merged["total_freq"])
r_s, p_s = stats.spearmanr(merged["centrality"], merged["total_freq"])
print(f"\nPearson correlation (centrality ~ raw frequency) = {r_p:.4f}  (p={p_p:.4f})")
print(f"Spearman correlation (centrality ~ raw frequency) = {r_s:.4f}  (p={p_s:.4f})")
with open("table6_validation.txt", "w", encoding="utf-8") as f:
    f.write(f"Pearson r={r_p:.4f}, p={p_p:.4f}\nSpearman rho={r_s:.4f}, p={p_s:.4f}\nn={len(merged)}\n")

# %% [12] Plot figures (Q1 print version) -------------------------------------

# --- 12.1 Short, academic English labels ---
LAYER_LABELS_EN = {
    "L1": "L1: Cognitive Core",
    "L2": "L2: Motivation & Affect",
    "L3": "L3: Teacher & Pedagogy",
    "L4": "L4: Resources & Technology",
    "L5": "L5: Context & Policy",
    "OUT": "TIPPING POINT\n(Cascading Collapse Risk)",
}

# English translations of the sub-themes. Keys must exactly match the raw
# Persian sub_theme text in the input file. Any sub-theme not covered here
# (e.g. due to wording/spacing differences) automatically gets a safe,
# generic label (Sub-theme <layer>.<index>) and a warning is printed —
# the common/standard English translations for all twelve documented
# sub-themes are provided below, so this fallback should rarely trigger.
SUB_LABELS_EN = {
    "پیچیدگی‌ها، نقص‌ها و شکنندگی فناورانه": "Technological Complexity,\nFaults & Fragility",
    "صلاحیت و آمادگی ناکافی معلمان": "Inadequate Teacher\nCompetence & Readiness",
    "محدودیت‌های زمانی و فشار ساختاری": "Time Constraints &\nStructural Pressure",
    "محدودیت‌های طراحی آموزشی و برنامه درسی": "Instructional Design &\nCurriculum Limitations",
    "نابرابری در دانش و تجربه پیشین یادگیرندگان": "Prior Knowledge &\nExperience Inequality",
    "موانع زمینه‌ای، فرهنگی و سیاستی": "Contextual, Cultural\n& Policy Barriers",
    "شکاف‌های جنسیتی و نابرابری در مشارکت": "Gender Gaps &\nParticipation Inequality",
    "کمبود منابع، زیرساخت و فشارهای مالی": "Resource, Infrastructure\n& Financial Constraints",
    "نقاط ضعف روش‌شناختی و چالش‌های سنجش": "Methodological Weaknesses\n& Assessment Challenges",
    # Common/standard translations for the remaining layer-1 and layer-2
    # sub-themes typically found in this kind of meta-synthesis; these keys
    # match the most common wording, but check against your table (Table
    # 0) and adjust if your file uses different phrasing.
    "کمبود دانش و مهارت پیش‌نیاز فناورانه یادگیرندگان": "Prerequisite Tech-Skill Gap",
    "بدفهمی‌های مفهومی و شناختی": "Conceptual Misconceptions",
    "کاهش انگیزه و اضطراب یادگیرنده": "Learner Anxiety & Motivation Loss",
    "انزوای اجتماعی و ضعف کار گروهی": "Social Isolation & Weak Teamwork",
    "موانع شناختی و فراشناختی یادگیرنده": "Learner Cognitive &\nMetacognitive Barriers",
    "موانع روانشناختی، انگیزشی و نگرشی": "Psychological, Motivational\n& Attitudinal Barriers",
    "چالش‌های پویایی گروهی و تعاملات اجتماعی": "Group Dynamics & Social\nInteraction Challenges",
}



def get_short_label(st):
    if st in SUB_LABELS_EN:
        return SUB_LABELS_EN[st]
    lyr = sub_row[st].layer
    ordinal = [s for s in sub_themes if sub_row[s].layer == lyr].index(st) + 1
    print(f"[WARNING] No English translation found for '{st}'; "
          f"using placeholder label 'Sub-theme {lyr}.{ordinal}'. "
          f"Please add it to the SUB_LABELS_EN dictionary.")
    return f"Sub-theme {lyr}.{ordinal}"

short_labels = dict(LAYER_LABELS_EN)
for st in sub_themes:
    short_labels[label_map_sub[st]] = get_short_label(st)

# --- 12.2 Build the graph ---
G = nx.DiGraph()
for k in all_nodes:
    G.add_node(k)
for i in range(n):
    for j in range(n):
        if W[i, j] != 0:
            G.add_edge(all_nodes[i], all_nodes[j], weight=W[i, j])

# --- 12.3 Figure 1: true multilayer concentric circular layout ---
# L1 (core) at the center -> L5 (outermost) on the outside; each layer's
# sub-themes are spread evenly around that layer's own ring; the OUT node
# is placed separately, below the figure, to visually separate the
# "breakdown point" from the layered structure.
pos = {}
layer_radius = {1: 1.6, 2: 3.4, 3: 5.2, 4: 7.0, 5: 8.8}
sub_ring_offset = 1.1  # distance of the sub-theme ring from its own layer's ring

pos["L1"] = (0.0, 0.0)  # central core, exactly at the center
for lyr in range(2, 6):
    pos[f"L{lyr}"] = (0.0, layer_radius[lyr])  # layer node, top of its own ring

sub_by_layer = {lyr: [st for st in sub_themes if sub_row[st].layer == lyr] for lyr in range(1, 6)}
for lyr in range(1, 6):
    members = sub_by_layer[lyr]
    r = layer_radius[lyr] + sub_ring_offset if lyr > 1 else layer_radius[1] + sub_ring_offset
    k_n = len(members)
    for m_i, st in enumerate(members):
        ang = np.pi / 2 + 2 * np.pi * (m_i / max(k_n, 1))  # spread fully around the ring
        pos[label_map_sub[st]] = (r * np.cos(ang), r * np.sin(ang))

pos[OUTCOME] = (0.0, -(layer_radius[5] + sub_ring_offset + 2.5))  # bottom of the figure

node_colors, node_sizes = [], []
for k in all_nodes:
    if k == OUTCOME:
        node_colors.append("#C44E52"); node_sizes.append(4200)
    elif k.startswith("L"):
        node_colors.append("#4C72B0"); node_sizes.append(3200)
    else:
        node_colors.append("#DD8452"); node_sizes.append(2000)

fig, ax = plt.subplots(figsize=(26, 22))
edge_widths = [1.5 + 6 * abs(G[u][v]["weight"]) for u, v in G.edges()]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax,
                        edgecolors="black", linewidths=2)
nx.draw_networkx_edges(G, pos, edge_color="#7A1F1F", width=edge_widths, arrows=True,
                        arrowsize=20, connectionstyle="arc3,rad=0.08", ax=ax, alpha=0.55)
nx.draw_networkx_labels(G, pos, labels={k: short_labels[k] for k in all_nodes},
                         font_size=12, ax=ax)
ax.set_title("Fuzzy Cognitive Map — MVCC Multilayer Vulnerability & Cascading Collapse Model",
             fontsize=16, pad=20)
ax.axis("off")
plt.tight_layout(pad=4)
save_fig(fig, "fig1_fcm_network")

# --- 12.4 Figure 2: weight heatmap ---
fig, ax = plt.subplots(figsize=(17, 15))
im = ax.imshow(W, cmap="Reds", vmin=0, vmax=1)
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels([short_labels[k] for k in all_nodes], rotation=90, fontsize=12)
ax.set_yticklabels([short_labels[k] for k in all_nodes], fontsize=12)
cbar = plt.colorbar(im, ax=ax, label="Edge weight (vulnerability propagation strength)")
cbar.ax.tick_params(labelsize=12)
cbar.set_label("Edge weight (vulnerability propagation strength)", fontsize=13)
ax.set_title("FCM Adjacency (Weight) Matrix — MVCC Model", fontsize=16, pad=20)
plt.tight_layout(pad=4)
save_fig(fig, "fig2_weight_heatmap")

# --- 12.5 Figure 3: scenario comparison ---
# English names for the critical combos (readable, academic x-axis labels)
COMBO_LABELS_EN = {
    "مثلث مرگبار فنی-آموزشی": "Techno-Pedagogical Triangle",
    "تلهٔ شناختی-برنامه‌ای": "Cognitive-Curricular Trap",
    "سیلوی جنسیتی-فرهنگی": "Gender-Cultural Silo",
    "حلقهٔ بی‌منبع-بی‌چشم": "Resource-Assessment Loop",
}
scn_en = ["Baseline", "Best-case\n(well-resourced)", "Worst-case\n(under-resourced)"] + \
         [f"Combo:\n{COMBO_LABELS_EN.get(name, name)}" for name in CRITICAL_COMBOS.keys()]
vals = df_scn["Collapse probability (breakdown point)"].tolist()
colors_scn = ["#4C72B0", "#2E8B57", "#B22222", "#DD8452", "#8172B2", "#937860", "#CCB974"]
fig, ax = plt.subplots(figsize=(15, 8.5))
bars = ax.bar(scn_en, vals, color=colors_scn[:len(vals)], edgecolor="black", linewidth=1.5)
ax.axhline(vals[0], color="gray", linestyle="--", linewidth=1.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.4f}", ha="center", fontsize=12)
ax.set_ylabel("Steady-state collapse risk\n(Tipping Point activation)", fontsize=13)
ax.set_ylim(0, 1.0)
ax.set_title("FCM Scenario Analysis — MVCC Model", fontsize=16, pad=20)
plt.xticks(rotation=15, ha="right", fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout(pad=4)
save_fig(fig, "fig3_scenario_comparison")

# --- 12.6 Figure 4: convergence curves ---
fig, ax = plt.subplots(figsize=(14, 8.5))
hist_list = [hist_base, hist_best, hist_worst] + [combo_scenarios[c][1] for c in CRITICAL_COMBOS]
for h, lab, col in zip(hist_list, scn_en, colors_scn[:len(hist_list)]):
    ax.plot(range(len(h)), h[:, idx[OUTCOME]], marker="o", markersize=5,
            linewidth=2, label=lab.replace("\n", " "), color=col)
ax.set_xlabel("Iteration", fontsize=13)
ax.set_ylabel("Activation of TIPPING POINT node", fontsize=13)
ax.set_title("Convergence Trajectories across Scenarios", fontsize=16, pad=20)
ax.legend(fontsize=11)
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.tight_layout(pad=4)
save_fig(fig, "fig4_convergence")

# --- 12.7 Figure 5: centrality ---
df_cent_sorted = df_cent.sort_values("centrality", ascending=True)
fig, ax = plt.subplots(figsize=(13, 12))
y = range(len(df_cent_sorted))
ax.barh(y, df_cent_sorted["out_degree"], color="#2E8B57", label="Out-degree")
ax.barh(y, df_cent_sorted["in_degree"], left=df_cent_sorted["out_degree"],
        color="#4C72B0", label="In-degree")
ax.set_yticks(y)
ax.set_yticklabels([short_labels[k] for k in df_cent_sorted["node"]], fontsize=12)
ax.set_xlabel("Centrality", fontsize=13)
ax.set_title("Node Centrality Decomposition — MVCC Model", fontsize=16, pad=20)
ax.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.tight_layout(pad=4)
save_fig(fig, "fig5_centrality")

# --- 12.8 Figure 6: sensitivity ---
fig, ax = plt.subplots(figsize=(11.5, 8))
for gamma in sorted(df_sens["gamma"].unique()):
    sub = df_sens[df_sens["gamma"] == gamma]
    ax.plot(sub["lambda"], sub["collapse_risk_base"], marker="o", markersize=6,
            linewidth=2, label=f"γ={gamma}")
ax.set_xlabel("Sigmoid steepness (λ)", fontsize=13)
ax.set_ylabel("Baseline steady-state collapse risk", fontsize=13)
ax.set_title("Sensitivity Analysis of FCM Parameters — MVCC Model", fontsize=16, pad=20)
ax.legend(title="Memory coeff.", fontsize=11, title_fontsize=12)
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.tight_layout(pad=4)
save_fig(fig, "fig6_sensitivity")

print("\nAll six figures (each as a 600-dpi PNG plus a vector PDF) "
      "and six tables were generated and saved successfully.")

# %% [13] (Optional) LLM-assisted semantic edge-weight refinement module (Claude API) ---
# This section only activates if an API key is present, and acts as a
# secondary semantic-validation layer (not a replacement for the
# statistical weights). Running it is optional.
"""
import os, requests
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if API_KEY:
    def llm_edge_check(sub_a, sub_b):
        prompt = (f"In a model of cascading failure for educational robotics interventions, "
                  f"rate how strongly vulnerability A intensifies vulnerability B on a scale "
                  f"from 0 (unrelated) to 1 (very strong intensification). "
                  f"Respond with ONLY a number.\nA: {sub_a}\nB: {sub_b}")
        r = requests.post("https://api.anthropic.com/v1/messages",
                           headers={"x-api-key": API_KEY, "anthropic-version": "2023-06-01",
                                    "content-type": "application/json"},
                           json={"model": "claude-sonnet-4-6", "max_tokens": 10,
                                 "messages": [{"role": "user", "content": prompt}]})
        return r.json()
    # Example usage (disabled by default):
    # print(llm_edge_check(sub_themes[0], sub_themes[5]))
else:
    print("Note: ANTHROPIC_API_KEY is not set; the LLM semantic-refinement "
          "module was skipped (the paper's main results are based purely "
          "on TF-IDF and the actual empirical statistics).")
"""
