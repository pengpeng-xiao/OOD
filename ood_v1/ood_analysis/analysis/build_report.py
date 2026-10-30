"""Build results/../report.html from the saved analysis outputs and the private split key."""
import base64, json, re, sys
from pathlib import Path
from string import Template
import numpy as np
from sklearn.metrics import roc_auc_score

A = Path(__file__).resolve().parent.parent
R = A / "results"
B = Path("/mnt/ssd/pengpengxiao/ood/anonymous_pde_benchmark/problems")
KEY = Path(sys.argv[1] if len(sys.argv) > 1 else
           "/mnt/ssd/pengpengxiao/deeponet-extrapolation/distribution/private/legacy_with_labels_20261013/anonymous_pde_benchmark_test_key.json")  # matches this bundle (manifest 0605dffb); the top-level key is for the re-shuffled release
key = json.load(open(KEY))["problems"]
SCALAR = ["diffusion_reaction", "burgers", "advection", "poisson"]
NICE = {"diffusion_reaction": "Diffusion–reaction", "burgers": "Burgers", "advection": "Advection",
        "poisson": "Poisson", "cavity_flow": "Cavity flow"}


def truth(prob):
    sid = np.load(B / prob / "test_data.npz")["sample_id"]
    m = {e["sample_id"]: e for e in key[prob]}
    y = np.array([m[s]["split"] == "ood" for s in sid])
    src = np.array([m[s]["source"].split("/")[-1] for s in sid])
    return sid, y, src


def f3(v):
    return f"{v:.3f}"


def fmt_lr(v):
    if v < 0.01:
        return "0"
    if v < 100:
        return f"{v:.1f}"
    return f"{v:,.0f}"


# ---------------------------------------------------------------- per-problem numbers
S = {}
for p in SCALAR:
    L = np.load(R / f"{p}_labels.npz")
    sid, y, src = truth(p)
    o = L["ood"]
    lr, lrt = 2 * L["lr_te"], 2 * L["lr_tr"]
    regimes = sorted({float(re.search(r"ls_([\d.]+)", s).group(1)) for s in src[y]})
    S[p] = dict(
        n=len(y), n_id=int((~o).sum()), n_ood=int(o.sum()), acc=float(np.mean(o == y)),
        auc=roc_auc_score(y, L["lr_te"]), auc_err=roc_auc_score(y, L["e_te"]),
        l_train=float(np.median(L["l_tr"])), regimes=regimes,
        lt=(L["l_tr"].min(), L["l_tr"].max()), lid=(L["l_hat"][~o].min(), L["l_hat"][~o].max()),
        lood=(L["l_hat"][o].min(), L["l_hat"][o].max()),
        lr_tr_max=lrt.max(), lr_id_max=lr[~o].max(), lr_ood=(lr[o].min(), lr[o].max()),
        e_tr=np.median(L["e_tr"]), e_id=np.median(L["e_te"][~o]), e_ood=np.median(L["e_te"][o]),
        sid=sid, lr=lr, lrt=lrt, lhat=L["l_hat"], ood=o, y=y,
    )
L = np.load(R / "cavity_flow_labels.npz")
sid, y, src = truth("cavity_flow")
o = L["ood"]
CAV = dict(n=len(y), n_id=int((~o).sum()), n_ood=int(o.sum()), acc=float(np.mean(o == y)),
           auc=roc_auc_score(y, L["U"]), auc_err=roc_auc_score(y, L["eu"]),
           U_id=(L["U"][~o].min(), L["U"][~o].max()), U_ood=(L["U"][o].min(), L["U"][o].max()),
           eu_tr_max=L["eu_tr"].max(), eu_id_max=L["eu"][~o].max(), eu_ood=(L["eu"][o].min(), L["eu"][o].max()))
TOTAL = sum(S[p]["n"] for p in SCALAR) + CAV["n"]
WRONG = sum(int(round((1 - S[p]["acc"]) * S[p]["n"])) for p in SCALAR) + int(round((1 - CAV["acc"]) * CAV["n"]))

# ---------------------------------------------------------------- results table rows
def regime_text(p):
    r = S[p]["regimes"]
    return ", ".join(f"{v:g}" for v in r)

# training-generator length scales from the pooled fit (gp_pooled.py, gp_poisson2.py)
L_TRAIN = {"diffusion_reaction": "l = 0.5", "burgers": "l = 1.0 (periodic)",
           "advection": "l = 0.5", "poisson": "l = 0.5 (11 modes)"}
rows = []
for p in SCALAR:
    s = S[p]
    rows.append(f"""<tr><th scope="row">{NICE[p]}</th><td class="num">{s['n_id']}</td><td class="num">{s['n_ood']}</td>
<td>{L_TRAIN[p]}</td><td>l ≈ {regime_text(p)}</td>
<td class="num strong">{f3(s['auc'])}</td><td class="num">{s['acc']*100:.0f}%</td><td class="num muted">{f3(s['auc_err'])}</td></tr>""")
rows.append(f"""<tr><th scope="row">{NICE['cavity_flow']}</th><td class="num">{CAV['n_id']}</td><td class="num">{CAV['n_ood']}</td>
<td>U ∈ [0, 0.40]</td><td>U = {CAV['U_ood'][0]:.2f}–{CAV['U_ood'][1]:.2f}</td>
<td class="num strong">{f3(CAV['auc'])}</td><td class="num">{CAV['acc']*100:.0f}%</td><td class="num muted">{f3(CAV['auc_err'])}</td></tr>""")
RESULTS_ROWS = "\n".join(rows)

# ---------------------------------------------------------------- evidence table rows
ev = []
for p in SCALAR:
    s = S[p]
    ev.append(f"""<tr><th scope="row">{NICE[p]}</th>
<td class="num">{s['lt'][0]:.2f}–{s['lt'][1]:.2f}</td><td class="num">{s['lid'][0]:.2f}–{s['lid'][1]:.2f}</td><td class="num">{s['lood'][0]:.2f}–{s['lood'][1]:.2f}</td>
<td class="num">{s['lr_tr_max']:.1f}</td><td class="num">{s['lr_id_max']:.1f}</td><td class="num">{fmt_lr(s['lr_ood'][0])}–{fmt_lr(s['lr_ood'][1])}</td>
<td class="num">{s['e_tr']*100:.2f}%</td><td class="num">{s['e_id']*100:.2f}%</td><td class="num">{s['e_ood']*100:.1f}%</td></tr>""")
EVIDENCE_ROWS = "\n".join(ev)

# ---------------------------------------------------------------- score-separation SVG
W, LEFT, RIGHT, TOP = 940, 150, 24, 40
ROW = 78
H = TOP + ROW * len(SCALAR) + 44
XMIN, XMAX = -2.0, 4.0                       # log10(2 logLR)
def X(v):
    v = np.clip(np.log10(max(v, 10 ** XMIN)), XMIN, XMAX)
    return LEFT + (v - XMIN) / (XMAX - XMIN) * (W - LEFT - RIGHT)
parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="sep-title sep-desc" class="sep">',
         '<title id="sep-title">Detection score for every test sample</title>',
         '<desc id="sep-desc">For each scalar PDE, the grey band spans the training samples\' scores and each dot is a test sample, '
         'blue if labelled in-distribution and orange if labelled out-of-distribution. A dashed line marks the threshold 2·logLR = 15.1.</desc>']
for e in range(int(XMIN), int(XMAX) + 1):
    x = X(10.0 ** e)
    lab = {-2: "≤0.01", -1: "0.1", 0: "1", 1: "10", 2: "100", 3: "1,000", 4: "10,000"}[e]
    parts.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{TOP-8}" y2="{H-38}"/>')
    parts.append(f'<text class="tick" x="{x:.1f}" y="{H-20}" text-anchor="middle">{lab}</text>')
parts.append(f'<text class="axis" x="{(LEFT+W-RIGHT)/2:.0f}" y="{H-2}" text-anchor="middle">2·log likelihood ratio against the training kernel (log scale)</text>')
rng = np.random.default_rng(7)
for i, p in enumerate(SCALAR):
    s = S[p]
    y0 = TOP + i * ROW
    parts.append(f'<text class="rowlab" x="{LEFT-14}" y="{y0+16}" text-anchor="end">{NICE[p]}</text>')
    parts.append(f'<text class="rowsub" x="{LEFT-14}" y="{y0+32}" text-anchor="end">{s["n_id"]} ID · {s["n_ood"]} OOD</text>')
    # training band
    x0, x1 = X(s["lrt"].min()), X(s["lrt"].max())
    xm = X(float(np.median(s["lrt"])))
    parts.append(f'<g><title>Training samples (n=1000): median {np.median(s["lrt"]):.2f}, max {s["lrt"].max():.1f}</title>'
                 f'<rect class="band" x="{x0:.1f}" y="{y0+2}" width="{x1-x0:.1f}" height="10" rx="3"/>'
                 f'<line class="bandmed" x1="{xm:.1f}" x2="{xm:.1f}" y1="{y0}" y2="{y0+14}"/></g>')
    # test dots
    order = np.argsort(s["ood"])                  # draw ID first, OOD on top
    for j in order:
        v = s["lr"][j]
        cls = "ood" if s["ood"][j] else "id"
        cy = y0 + 22 + rng.uniform(0, 30)
        tip = (f'{s["sid"][j]} · 2·logLR {fmt_lr(v)} · fitted l {s["lhat"][j]:.3f} · '
               f'{"OOD" if s["ood"][j] else "ID"}')
        parts.append(f'<circle class="{cls}" cx="{X(v):.1f}" cy="{cy:.1f}" r="3.6"><title>{tip}</title></circle>')
xt = X(15.1)
parts.append(f'<line class="thr" x1="{xt:.1f}" x2="{xt:.1f}" y1="{TOP-14}" y2="{H-38}"/>')
parts.append(f'<text class="thrlab" x="{xt+6:.1f}" y="{TOP-18}">threshold 15.1 (p = 10⁻⁴)</text>')
parts.append("</svg>")
SEP_SVG = "\n".join(parts)

FIG_B64 = base64.b64encode((R / "ood_summary.png").read_bytes()).decode()

html = Template(open(A / "analysis" / "report_template.html").read()).substitute(
    TOTAL=f"{TOTAL:,}", WRONG=WRONG, CORRECT=f"{TOTAL-WRONG:,}",
    RESULTS_ROWS=RESULTS_ROWS, EVIDENCE_ROWS=EVIDENCE_ROWS, SEP_SVG=SEP_SVG, FIG_B64=FIG_B64,
    CAV_U_ID=f"{CAV['U_id'][0]:.2f}–{CAV['U_id'][1]:.2f}", CAV_U_OOD=f"{CAV['U_ood'][0]:.2f}–{CAV['U_ood'][1]:.2f}",
    CAV_EU_TR=f"{CAV['eu_tr_max']*100:.2f}%", CAV_EU_ID=f"{CAV['eu_id_max']*100:.2f}%",
    CAV_EU_OOD=f"{CAV['eu_ood'][0]*100:.1f}–{CAV['eu_ood'][1]*100:.1f}%",
    PO_OOD_MIN=fmt_lr(S["poisson"]["lr_ood"][0]),
    BU_AUC_ERR=f3(S["burgers"]["auc_err"]),
)
(A / "report.html").write_text(html)
print("wrote", A / "report.html", f"({len(html)/1024:.0f} KB)")
