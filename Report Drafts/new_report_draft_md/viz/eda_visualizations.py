"""
EDA Visualizations — generates all figures for Chapter 3 §3.2.
Output: PNG files in the same directory.
"""
import json, math, re, sys, io
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Map backends — try contextily first, fall back to plain scatter
try:
    import contextily as ctx
    HAS_CTX = True
except ImportError:
    HAS_CTX = False
    print("[warn] contextily not available — geo maps will be plain scatter plots")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent.parent  # viz -> new_report_draft_md -> Report Drafts -> FYP Project

plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "axes.labelsize": 11})
# Increase default DPI for sharper output
plt.rcParams["figure.dpi"] = 150

# Arabic-to-English city name map for label rendering
CITY_EN = {
    "الكوفة": "Kufa", "المدينة": "Medina", "البصرة": "Basra",
    "مكة": "Mecca", "الشام": "al-Sham", "مصر": "Egypt",
    "دمشق": "Damascus", "بغداد": "Baghdad", "حمص": "Homs",
    "مرو": "Marw", "واسط": "Wasit", "الجزيرة": "al-Jazirah",
    "خراسان": "Khorasan", "الري": "Rayy", "حران": "Harran",
    "نيسابور": "Nishapur",
}

print("Generating EDA visualizations...\n")

# ── helpers ──
def save(fig, name):
    path = HERE / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  ✓ {name}")
    plt.close(fig)


# ========================================================================
# DATA LOADING
# ========================================================================
print("[load] Encyclopedia…")
enc = json.load(open(PROJECT / "output/shamela/narrator_encyclopedia.json", encoding="utf-8"))

print("[load] Alias index…")
alias = json.load(open(PROJECT / "output/shamela/alias_index.json", encoding="utf-8"))

print("[load] Graph nodes…")
nodes_df = pd.read_csv(PROJECT / "output/stage2/graph_nodes.csv")

print("[load] Geo data…")
geo = json.load(open(PROJECT / "output/geo/entry_locations.json", encoding="utf-8"))


# ========================================================================
# FIGURE 1 — Grade distribution (log-scale bar chart)
# ========================================================================
print("\n[fig1] Grade distribution…")
gc = Counter()
for e in enc:
    for g in e.get("grades", []):
        gc[g.get("grade", "") if isinstance(g, dict) else str(g)] += 1

grade_map = {"ثقة": "Thiqah", "صدوق": "Saduq", "مقبول": "Maqbul",
             "ضعيف": "Daif", "متروك": "Matruk", "مجهول": "Majhul"}
buckets = defaultdict(int)
for g, c in gc.items():
    m = False
    for ar, en in grade_map.items():
        if ar in g:
            buckets[en] += c; m = True; break
    if not m:
        buckets["Other"] += c

labels = ["Thiqah", "Saduq", "Maqbul", "Majhul", "Daif", "Matruk", "Other"]
values = [buckets[l] for l in labels]
colors = ["#2ecc71", "#27ae60", "#85c1e9", "#f39c12", "#e67e22", "#e74c3c", "#95a5a6"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.7)
ax.set_ylabel("Count (log scale)")
ax.set_title("Distribution of Jarḥ wa Taʿdīl Grade Annotations")
ax.set_yscale("log")
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width()/2, v*1.1, f"{v:,}", ha="center", va="bottom", fontsize=9)
ax.set_ylim(top=max(values) * 5)
save(fig, "fig1_grade_distribution.png")


# ========================================================================
# FIGURE 2 — Biography text length histogram
# ========================================================================
print("[fig2] Biography text lengths…")
text_lengths = np.array([len(e.get("full_text", "")) for e in enc])
text_lengths = text_lengths[text_lengths > 0]  # remove zeros

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(text_lengths, bins=80, color="#3498db", edgecolor="white", linewidth=0.3)
ax.set_xscale("log")
ax.set_xlabel("Character length (log scale)")
ax.set_ylabel("Number of biographies")
ax.set_title("Biography Text Length Distribution")
ax.axvline(200, color="red", linestyle="--", linewidth=1.5, label="Alignment challenge threshold (<200 chars)")
very_short = (text_lengths < 200).sum()
ax.text(150, ax.get_ylim()[1]*0.9, f"  {very_short:,} entries\n  ({very_short/len(text_lengths)*100:.1f}%)",
        color="red", fontsize=9, ha="right")
ax.legend(fontsize=9)
save(fig, "fig2_text_lengths.png")


# ========================================================================
# FIGURE 3 — Alias density histogram
# ========================================================================
print("[fig3] Alias density…")
if isinstance(next(iter(alias.values())), list):
    alias_counts = [len(v) for v in alias.values()]
else:
    alias_counts = [1] * len(alias)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(alias_counts, bins=50, color="#9b59b6", edgecolor="white", linewidth=0.3)
ax.set_xscale("log")
ax.set_xlabel("Number of alias forms per narrator (log scale)")
ax.set_ylabel("Number of narrators")
ax.set_title("Alias Form Density per Canonical Narrator")
ge5 = sum(1 for c in alias_counts if c >= 5)
ge20 = sum(1 for c in alias_counts if c >= 20)
ax.text(0.7, 0.95, f"≥5 forms: {ge5:,} ({ge5/len(alias_counts)*100:.1f}%)\n≥20 forms: {ge20:,} ({ge20/len(alias_counts)*100:.1f}%)",
        transform=ax.transAxes, fontsize=9, va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
save(fig, "fig3_alias_density.png")


# ========================================================================
# FIGURE 4 — Log-log degree distribution with power-law fit
# ========================================================================
print("[fig4] Degree distribution…")
degrees = nodes_df["in_degree"].fillna(0) + nodes_df["out_degree"].fillna(0)
deg_counter = Counter(degrees)
deg_vals = np.array(sorted(d for d in deg_counter.keys() if d > 0))
deg_freq = np.array([deg_counter[d] for d in deg_vals])

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(deg_vals, deg_freq, s=15, color="#e74c3c", alpha=0.7, label="Observed")

# Power-law fit
mask = (deg_vals >= 1) & (deg_freq >= 3)
if mask.sum() > 5:
    logk = np.log(deg_vals[mask])
    logp = np.log(deg_freq[mask])
    coeffs = np.polyfit(logk, logp, 1)
    gamma = -coeffs[0]
    fit_x = np.logspace(0, np.log10(deg_vals.max()), 100)
    fit_y = np.exp(np.polyval(coeffs, np.log(fit_x)))
    ax.plot(fit_x, fit_y, "--", color="black", linewidth=1.5,
            label=f"Power-law fit: γ ≈ {gamma:.3f}")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Degree (k, log scale)")
ax.set_ylabel("Frequency P(k), log scale")
ax.set_title("Degree Distribution of the Graded Transmission Subgraph")
ax.legend(fontsize=9)
save(fig, "fig4_degree_distribution.png")


# ========================================================================
# FIGURE 5 — Complementary CDF
# ========================================================================
print("[fig5] CCDF…")
sorted_degs = np.sort(degrees)
ccdf = 1.0 - np.arange(1, len(sorted_degs)+1) / len(sorted_degs)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sorted_degs, ccdf, linewidth=1.5, color="#2980b9")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Degree (k, log scale)")
ax.set_ylabel("P(X > k), log scale")
ax.set_title("Complementary CDF (CCDF) of Node Degree")
ax.grid(True, which="both", alpha=0.3)
save(fig, "fig5_ccdf.png")


# ========================================================================
# FIGURE 6 — Attribute coverage stacked bar
# ========================================================================
print("[fig6] Attribute coverage…")
graded = nodes_df["grade"].notna() & (nodes_df["grade"] != "")
tabaqa_n = nodes_df["tabaqa"].notna() & (nodes_df["tabaqa"] != "")
death_n = nodes_df["death_info"].notna() & (nodes_df["death_info"] != "")
N = len(nodes_df)

cats = ["Has grade", "Has tabaqa", "Has death info", "Missing all three"]
vals = [int(graded.sum()), int(tabaqa_n.sum()), int(death_n.sum()),
        int((~graded & ~tabaqa_n & ~death_n).sum())]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(cats, vals, color=["#2ecc71", "#3498db", "#f39c12", "#e74c3c"],
              edgecolor="white", linewidth=1.2)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + N*0.02, f"{v:,} ({v/N*100:.1f}%)",
            ha="center", fontsize=9)
ax.set_title("Attribute Coverage in the Graded Subgraph")
ax.set_ylabel("Number of nodes")
ax.set_ylim(0, N * 1.25)
save(fig, "fig6_attribute_coverage.png")


# ========================================================================
# FIGURE 7 — Tabaqa generation bar chart
# ========================================================================
print("[fig7] Tabaqa generations…")
tabaqa_gen_map = {
    "الثاني": 2, "الثالث": 3, "الرابع": 4, "الخامس": 5,
    "السادس": 6, "السابع": 7, "الثامن": 8, "التاسع": 9,
    "العا": 10, "الحادي": 11,
}
tabaqa_pops = Counter()
unparseable = 0
for _, row in nodes_df.iterrows():
    t = str(row.get("tabaqa", ""))
    if not t or t == "nan": continue
    matched = False
    for pat, gen in tabaqa_gen_map.items():
        if pat in t:
            tabaqa_pops[gen] += 1; matched = True; break
    if not matched:
        unparseable += 1

gens = list(range(2, 12))
gen_counts = [tabaqa_pops.get(g, 0) for g in gens]
gen_labels = [f"Gen {g}\n({622+(g-1)*21}–{622+g*21})" for g in gens]
ce_mid = [622 + (g-0.5)*21 for g in gens]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(gens, gen_counts, color="#1abc9c", edgecolor="white", width=0.7)
for b, c in zip(bars, gen_counts):
    ax.text(b.get_x() + b.get_width()/2, c + 5, f"{c}", ha="center", fontsize=8)
ax.set_xticks(gens)
ax.set_xticklabels(gen_labels, fontsize=8)
ax.set_xlabel("Tabaqa Generation (approximate CE period)")
ax.set_ylabel("Number of narrators")
ax.set_title("Narrator Distribution Across Tabaqa Generations")
if unparseable:
    ax.text(0.98, 0.95, f"Unparseable: {unparseable:,}", transform=ax.transAxes,
            fontsize=8, ha="right", va="top", style="italic")
save(fig, "fig7_tabaqa_distribution.png")


# ========================================================================
# FIGURE 8 — Timeline with compiler lifetimes (twin-axis layout)
# ========================================================================
print("[fig8] Timeline overlay…")
compilers = [
    ("Mālik b. Anas", 712, 795),
    ("Aḥmad b. Ḥanbal", 780, 855),
    ("al-Bukhārī", 810, 870),
    ("Muslim", 817, 875),
    ("Abū Dāwūd", 817, 889),
    ("al-Tirmidhī", 824, 892),
    ("Ibn Mājah", 824, 887),
    ("al-Nasāʾī", 829, 915),
]
compiler_colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71",
                   "#3498db", "#9b59b6", "#1abc9c", "#e84393"]

# Use a 2-panel layout: top=bar chart, bottom=timeline
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [1, 0.6]})

# ── Top panel: bar chart ──
ax1.bar(gens, gen_counts, color="#1abc9c", edgecolor="white", width=0.6)
for g, c in zip(gens, gen_counts):
    ax1.text(g, c + 8, f"{c}", ha="center", fontsize=8, fontweight="bold")
ax1.set_xticks(gens)
ax1.set_xticklabels([f"Gen {g}" for g in gens])
ax1.set_ylabel("Narrators")
ax1.set_title("Narrators per Generation with Major Compiler Lifespans")

# ── Bottom panel: compiler lifespans ──
for i, (name, birth, death) in enumerate(compilers):
    ax2.barh(i, death - birth, left=birth, height=0.6, color=compiler_colors[i],
             alpha=0.9, edgecolor="white", linewidth=0.5)
    mid = (birth + death) / 2
    ax2.text(mid, i, name, ha="center", va="center", fontsize=7.5,
             color="white", fontweight="bold")

ax2.set_yticks(range(len(compilers)))
ax2.set_yticklabels([""] * len(compilers))
ax2.set_xlabel("Year (CE)")
ax2.set_xlim(600, 950)
# Shade generation bands as vertical spans
gen_shades = ["#f0f0f0", "#ffffff"]
for gi, (g_start, g_next) in enumerate(zip(gens, gens[1:] + [12])):
    start_ce = 622 + (g_start - 1) * 21
    end_ce = 622 + (g_next - 1) * 21
    ax2.axvspan(start_ce, end_ce, color=gen_shades[gi % 2], alpha=0.3, zorder=-1)
    mid_ce = (start_ce + end_ce) / 2
    ax2.text(mid_ce, -0.6, f"Gen{g_start}", ha="center", fontsize=6.5, color="gray")

for _, _, death in compilers:
    ax2.axvline(death, color="gray", linewidth=0.3, linestyle="--", alpha=0.3)

fig.tight_layout()
save(fig, "fig8_compiler_timeline.png")


# ========================================================================
# FIGURE 9 — Geographic scatter plot on real basemap
# ========================================================================
print("[fig9] Geo scatter on basemap…")
city_coords = {}
for eid, loc_data in geo.items():
    if isinstance(loc_data, list) and len(loc_data) > 0 and isinstance(loc_data[0], list):
        city = loc_data[0][0]
        coords = tuple(loc_data[0][1]) if len(loc_data[0]) > 1 else None
        if coords:
            en_city = CITY_EN.get(city, city)
            if en_city not in city_coords:
                city_coords[en_city] = {"lat": coords[0], "lon": coords[1], "count": 0}
            city_coords[en_city]["count"] += 1

# Aggregate by unique coordinate
coord_groups = {}
for city, info in city_coords.items():
    key = (round(info["lat"], 2), round(info["lon"], 2))
    if key not in coord_groups:
        coord_groups[key] = {"lat": key[0], "lon": key[1], "count": 0, "cities": []}
    coord_groups[key]["count"] += info["count"]
    coord_groups[key]["cities"].append(city)

lats = [v["lat"] for v in coord_groups.values()]
lons = [v["lon"] for v in coord_groups.values()]
counts = [v["count"] for v in coord_groups.values()]
labels = [v["cities"][0] for v in coord_groups.values()]

fig, ax = plt.subplots(figsize=(11, 8))

scatter = ax.scatter(lons, lats, s=[c/6 for c in counts], c=counts,
                     cmap="YlOrRd", alpha=0.85, edgecolors="black", linewidth=0.6,
                     zorder=5)
for i, (lon, lat, label, cnt) in enumerate(zip(lons, lats, labels, counts)):
    if cnt > 50:
        dy = 0.5 if i % 2 == 0 else -0.8
        dx = 0.5 if i % 3 == 0 else -0.5
        ax.annotate(f"{label} ({cnt:,})", (lon, lat),
                    xytext=(dx, dy), textcoords="offset fontsize",
                    fontsize=8, ha="center", zorder=10,
                    arrowprops=dict(arrowstyle="->", color="gray", linewidth=0.6),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              alpha=0.85, edgecolor="gray"))

# Add basemap if available
if HAS_CTX:
    try:
        ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.CartoDB.Positron,
                        attribution=False)
        ax.set_xlim(min(lons) - 3, max(lons) + 3)
        ax.set_ylim(min(lats) - 2, max(lats) + 3)
    except Exception as e:
        print(f"  [warn] basemap failed: {e}")
        ax.grid(True, alpha=0.3)
else:
    ax.grid(True, alpha=0.3)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Geographic Distribution of Hadith Transmitters", pad=15)
cbar = plt.colorbar(scatter, ax=ax, label="Narrator count", shrink=0.7)
fig.tight_layout()
save(fig, "fig9_geo_map.png")

# Also generate an interactive HTML map via folium
try:
    import folium
    m = folium.Map(location=[30, 40], zoom_start=5,
                   tiles="CartoDB Positron", control_scale=True)
    for lon, lat, label, cnt in zip(lons, lats, labels, counts):
        radius = max(3, min(30, cnt / 30))
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color="#e74c3c",
            fill=True,
            fillOpacity=0.7,
            popup=f"<b>{label}</b><br>{cnt:,} entries",
            tooltip=label,
        ).add_to(m)
    m.save(str(HERE / "fig9_geo_interactive.html"))
    print(f"  ✓ fig9_geo_interactive.html")
except ImportError:
    print("  [skip] folium not available — no interactive map")


# ========================================================================
# FIGURE 10 — Regional concentration on basemap (bubble map)
# ========================================================================
print("[fig10] Regional concentration map…")
city_counter = Counter()
city_coords_map = {}
for eid, loc_data in geo.items():
    if isinstance(loc_data, list) and len(loc_data) > 0 and isinstance(loc_data[0], list):
        city_ar = loc_data[0][0]
        city_counter[city_ar] += 1
        coords = tuple(loc_data[0][1]) if len(loc_data[0]) > 1 else None
        if coords and city_ar not in city_coords_map:
            city_coords_map[city_ar] = coords

# Top cities with coordinates
top_ar = city_counter.most_common(15)
cities_plot = []
for city_ar, cnt in top_ar:
    en = CITY_EN.get(city_ar, city_ar)
    coords = city_coords_map.get(city_ar)
    if coords:
        cities_plot.append({"name": en, "lat": coords[0], "lon": coords[1], "count": cnt})

region_of_city = {
    "Kufa": "Iraq", "Basra": "Iraq", "Baghdad": "Iraq", "Wasit": "Iraq",
    "Medina": "Hejaz", "Mecca": "Hejaz",
    "Damascus": "Syria", "Homs": "Syria", "al-Sham": "Syria", "al-Jazirah": "Syria",
    "Egypt": "Egypt",
    "Khorasan": "Persia", "Marw": "Persia", "Rayy": "Persia", "Nishapur": "Persia",
    "Harran": "Anatolia",
}
region_colors = {"Iraq": "#e74c3c", "Hejaz": "#2ecc71", "Syria": "#3498db",
                 "Egypt": "#f39c12", "Persia": "#9b59b6", "Other": "#95a5a6"}

c_lats = [c["lat"] for c in cities_plot]
c_lons = [c["lon"] for c in cities_plot]
c_counts = [c["count"] for c in cities_plot]
c_names = [c["name"] for c in cities_plot]
c_regions = [region_of_city.get(c["name"], "Other") for c in cities_plot]
c_colors = [region_colors.get(r, "#95a5a6") for r in c_regions]

fig, ax = plt.subplots(figsize=(11, 8))
scatter = ax.scatter(c_lons, c_lats, s=[c/5 for c in c_counts],
                     c=c_colors, alpha=0.85, edgecolors="black", linewidth=0.6,
                     zorder=5)

# Label each city with name and count
for i, (lon, lat, name, cnt, reg) in enumerate(zip(c_lons, c_lats, c_names, c_counts, c_regions)):
    offset_yr = 0.5 if i % 2 == 0 else -0.7
    offset_xr = 0.4 if i % 3 == 1 else -0.4
    ax.annotate(f"{name}\n{cnt:,}", (lon, lat),
                xytext=(offset_xr, offset_yr), textcoords="offset fontsize",
                fontsize=8, ha="center", zorder=10,
                arrowprops=dict(arrowstyle="->", color="gray", linewidth=0.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor=c_colors[i],
                          alpha=0.7, edgecolor="gray"))

# Add basemap
if HAS_CTX:
    try:
        ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.CartoDB.Positron,
                        attribution=False)
        ax.set_xlim(min(c_lons) - 3, max(c_lons) + 3)
        ax.set_ylim(min(c_lats) - 2, max(c_lats) + 3)
    except Exception as e:
        print(f"  [warn] basemap failed: {e}")
        ax.grid(True, alpha=0.3)
else:
    ax.grid(True, alpha=0.3)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#e74c3c", label="Iraq"),
    Patch(facecolor="#2ecc71", label="Hejaz"),
    Patch(facecolor="#3498db", label="Syria/Levant"),
    Patch(facecolor="#f39c12", label="Egypt"),
    Patch(facecolor="#9b59b6", label="Persia/Khorasan"),
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=9, title="Region")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Major Hadith Transmission Centres (scaled by narrator count)", pad=15)
fig.tight_layout()
save(fig, "fig10_geo_bubble_map.png")

# Also generate an interactive HTML version
try:
    import folium
    m2 = folium.Map(location=[32, 42], zoom_start=5,
                    tiles="CartoDB Positron", control_scale=True)
    for c in cities_plot:
        popup_html = f"""
        <div style="font-family:sans-serif;min-width:120px">
            <b>{c['name']}</b><br>
            Narrator entries: <b>{c['count']:,}</b><br>
            Region: {region_of_city.get(c['name'], 'Other')}
        </div>
        """
        folium.CircleMarker(
            location=[c["lat"], c["lon"]],
            radius=max(5, min(40, c["count"] / 25)),
            color="#e74c3c",
            fill=True,
            fillOpacity=0.6,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=c["name"],
        ).add_to(m2)
    m2.save(str(HERE / "fig10_geo_interactive.html"))
    print(f"  ✓ fig10_geo_interactive.html")
except ImportError:
    print("  [skip] folium not available for interactive map")


print("\n✓ All 10 figures generated in:", HERE)
