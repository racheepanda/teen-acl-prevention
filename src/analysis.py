"""
Per-season ACL risk and warm-up benefit for US high-school athletes.

Second-order synthesis: sport/sex-specific baseline seasonal ACL risk
(Gornitzky et al. 2016) combined with published pooled effect estimates.
Uncertainty propagated by Monte Carlo; each published 95% CI is treated as a
log-normal (ratios, risks) or logit-normal (proportions) distribution.
Common random numbers: each parameter is drawn once and reused everywhere.

Run:  python src/analysis.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
N = 200_000
rng = np.random.default_rng(2026)

est = pd.read_csv(ROOT / "data" / "extracted_estimates.csv").set_index("param_id")
asm = pd.read_csv(ROOT / "data" / "assumptions.csv").set_index("assumption")["value"]
SEASONS = int(asm["seasons"])
ADHERENCE = [float(a) for a in asm["adherence_levels"].split(",")]
ROSTER = {s: int(asm[f"roster_{s}"]) for s in ["soccer", "basketball", "football", "lacrosse"]}


def lognormal(pid):
    r = est.loc[pid]
    s = (np.log(r.ci_high) - np.log(r.ci_low)) / 3.92
    return np.exp(rng.normal(np.log(r.point), s, N))


def logitnormal(pid):
    r = est.loc[pid]
    lg = lambda x: np.log(x / (1 - x))
    s = (lg(r.ci_high) - lg(r.ci_low)) / 3.92
    return 1 / (1 + np.exp(-rng.normal(lg(r.point), s, N)))


def q(x):
    return np.median(x), np.percentile(x, 2.5), np.percentile(x, 97.5)


def fmt(x, d=2):
    m, lo, hi = q(x)
    return f"{m:.{d}f} ({lo:.{d}f} to {hi:.{d}f})"


def nnt_fmt(arr):
    m, lo, hi = q(arr)
    if lo <= 0:
        return f"{1/m:.0f} (upper bound not estimable)"
    return f"{1/m:.0f} ({1/hi:.0f}-{1/lo:.0f})"


# ---- draw every parameter once (common random numbers) ----
groups = {
    "Girls soccer": ("base_soc_f", "F", "soccer"),
    "Girls basketball": ("base_bkb_f", "F", "basketball"),
    "Boys football": ("base_fb_m", "M", "football"),
    "Girls lacrosse": ("base_lax_f", "F", "lacrosse"),
    "Boys lacrosse": ("base_lax_m", "M", "lacrosse"),
    "Boys soccer": ("base_soc_m", "M", "soccer"),
}
base = {g: lognormal(pid) / 100 for g, (pid, _, _) in groups.items()}
RR = {k: lognormal(k) for k in ["nmt_female_team", "nmt_all_clar", "nmt_female_u18"]}
TURF = {"F": lognormal("turf_soc_f"), "M": lognormal("turf_soc_m")}
NONCONTACT = {"F": logitnormal("noncontact_f"), "M": logitnormal("noncontact_m")}
PRIMARY = "nmt_female_team"

# ---- Table 1: primary scenario ----
t1, sens, d_primary = [], [], {}
for g, (pid, sex, sport) in groups.items():
    p, rr = base[g], RR[PRIMARY]
    arr = p * (1 - rr)
    d_primary[g] = arr
    career = 1 - (1 - p) ** SEASONS
    career_arr = career - (1 - (1 - p * rr) ** SEASONS)
    t1.append({
        "Group": g,
        "Season risk (%)": fmt(p * 100),
        f"{SEASONS}-season risk (%)": fmt(career * 100, 1),
        "Warm-up risk cut (pts/season)": fmt(arr * 100),
        "NNT per season": nnt_fmt(arr),
        "Roster (assumed)": ROSTER[sport],
        "Team-seasons per ACL prevented": f"{(1/np.median(arr))/ROSTER[sport]:.1f}",
        f"Warm-up risk cut over {SEASONS} seasons (pts)": fmt(career_arr * 100),
    })
    # ---- sensitivity grid: effect source x mechanism x adherence ----
    for eid in RR:
        if eid == "nmt_female_u18" and sex == "M":
            continue
        for mech in ["all tears", "non-contact only"]:
            frac = 1.0 if mech == "all tears" else NONCONTACT[sex]
            for a in ADHERENCE:
                eff_rr = 1 - a * (1 - RR[eid])
                arr_s = p * frac * (1 - eff_rr)
                sens.append({"Group": g, "Effect source": eid, "Mechanism": mech,
                             "Adherence relative to trials": a, "ARR pts/season": fmt(arr_s * 100),
                             "NNT": nnt_fmt(arr_s), "_arr_med": np.median(arr_s) * 100})

t1 = pd.DataFrame(t1); sens = pd.DataFrame(sens)
t1.to_csv(OUT / "table1.csv", index=False)
sens.drop(columns="_arr_med").to_csv(OUT / "sensitivity_grid.csv", index=False)

# ---- turf (soccer only), upper bound ----
turf_rows = []
for g in ["Girls soccer", "Boys soccer"]:
    sex = groups[g][1]
    ex = base[g] * (TURF[sex] - 1)
    turf_rows.append({"Group": g, "Excess risk if all games on turf (pts/season, upper bound)": fmt(ex * 100)})
pd.DataFrame(turf_rows).to_csv(OUT / "turf_upper_bound.csv", index=False)

# ---- key numbers ----
gs = base["Girls soccer"]
warm_all = gs * (1 - RR[PRIMARY])
warm_nc = gs * NONCONTACT["F"] * (1 - RR[PRIMARY])
warm_nc_half = gs * NONCONTACT["F"] * 0.5 * (1 - RR[PRIMARY])
turf_gs = gs * (TURF["F"] - 1)

def rank(mech, a):
    s = sens[(sens["Effect source"] == PRIMARY) & (sens.Mechanism == mech) & (sens["Adherence relative to trials"] == a)]
    return list(s.sort_values("_arr_med", ascending=False).Group)

lines = [
    f"Girls soccer ARR, primary: {fmt(warm_all*100)} pts; NNT {nnt_fmt(warm_all)}",
    f"Girls soccer ARR, non-contact only: {fmt(warm_nc*100)} pts; NNT {nnt_fmt(warm_nc)}",
    f"Girls soccer ARR, non-contact only + half of trial-level adherence: {fmt(warm_nc_half*100)} pts; NNT {nnt_fmt(warm_nc_half)}",
    f"Girls soccer turf excess (upper bound): {fmt(turf_gs*100)} pts",
    f"Warm/turf ratio primary {np.median(warm_all)/np.median(turf_gs):.2f}x, P={np.mean(warm_all>turf_gs):.3f}",
    f"Warm/turf ratio non-contact {np.median(warm_nc)/np.median(turf_gs):.2f}x, P={np.mean(warm_nc>turf_gs):.3f}",
    f"Warm/turf ratio non-contact + half trial adherence {np.median(warm_nc_half)/np.median(turf_gs):.2f}x, P={np.mean(warm_nc_half>turf_gs):.3f}",
    f"Ranking (all tears, trial-level adherence): {rank('all tears', 1.0)}",
    f"Ranking (non-contact, half of trial-level adherence): {rank('non-contact only', 0.5)}",
    f"Range of girls-soccer ARR across full grid: {sens[sens.Group=='Girls soccer']._arr_med.min():.2f} to {sens[sens.Group=='Girls soccer']._arr_med.max():.2f}",
    f"Range of boys-soccer ARR across full grid: {sens[sens.Group=='Boys soccer']._arr_med.min():.2f} to {sens[sens.Group=='Boys soccer']._arr_med.max():.2f}",
]
# Ranking check if warm-up effect varied independently by sport
rr_indep = lognormal(PRIMARY)
lines.append(f"P(girls soccer ARR > girls basketball ARR): shared effect {np.mean(base['Girls soccer']*(1-RR[PRIMARY]) > base['Girls basketball']*(1-RR[PRIMARY])):.3f}; "
             f"independent per-sport effect {np.mean(base['Girls soccer']*(1-RR[PRIMARY]) > base['Girls basketball']*(1-rr_indep)):.3f}")
(OUT / "key_numbers.txt").write_text("\n".join(lines) + "\n")

# ---- Figure 1 ----
fig, ax = plt.subplots(figsize=(9, 6.2))
order = list(groups)
y = np.arange(len(order))[::-1]
styles = [("all tears", 1.0, "#1f6f8b", 0.18, "Warm-up, all tears, trial-level adherence"),
          ("non-contact only", 1.0, "#5fa8a0", 0.0, "Warm-up, non-contact tears only (alternative model)"),
          ("non-contact only", 0.5, "#b7d7cf", -0.18, "Non-contact only, half of trial-level adherence")]
for yi, g in zip(y, order):
    sex = groups[g][1]
    for mech, a, col, off, lab in styles:
        frac = 1.0 if mech == "all tears" else NONCONTACT[sex]
        arr = base[g] * frac * a * (1 - RR[PRIMARY]) * 100
        m, lo, hi = q(arr)
        ax.errorbar(m, yi + off, xerr=[[m - lo], [hi - m]], fmt="o", color=col, capsize=3,
                    label=lab if yi == y[0] else None)
    if "soccer" in g:
        m, lo, hi = q(-base[g] * (TURF[sex] - 1) * 100)
        ax.errorbar(m, yi, xerr=[[m - lo], [hi - m]], fmt="s", color="#c0392b", capsize=3,
                    label="Turf, games-only IRR applied to all injuries (contested)" if g == "Girls soccer" else None)
ax.axvline(0, color="grey", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels([g + ' (exploratory)' if g.startswith('Boys') else g for g in order])
ax.set_xlabel("Change in ACL risk per season (percentage points)\n← more risk     |     less risk →")
ax.set_title("Per-season ACL risk change for US high-school athletes\n(warm-up RR 0.50; medians and 95% uncertainty intervals)")
ax.text(0.99, 0.015, "Bracing & cleat type: no reliable ACL benefit shown",
        transform=ax.transAxes, ha="right", fontsize=8, style="italic", color="#555")
ax.legend(loc="upper center", fontsize=7.5, bbox_to_anchor=(0.45, -0.2), ncol=2, frameon=False)
plt.tight_layout(); plt.savefig(OUT / "figure1_decision_chart.png", dpi=200)

print(t1.to_string(index=False)); print(); print("\n".join(lines))
