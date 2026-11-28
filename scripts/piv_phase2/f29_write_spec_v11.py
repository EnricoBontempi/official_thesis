"""F2.8 - write the ADDITIVE spec v1.1 (decisione_R2 sec. 7). v1 is read, never modified or overwritten."""
import copy, hashlib, subprocess, sys
from pathlib import Path
import yaml
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "piv_phase1"))
import common as C
sys.path.insert(0, str(C.REPO / "src"))
from combustion_recon.data import piv_observations as P

v1p = C.REPO / "configs/observations/piv_observation_spec_v1.yaml"
v1sha = (v1p.with_suffix(".sha256")).read_text().split()[0]
assert hashlib.sha256(v1p.read_bytes()).hexdigest() == v1sha, "v1 spec hash mismatch"
s = copy.deepcopy(yaml.safe_load(v1p.read_bytes()))
s["spec_version"] = 1.1
s["date"] = "2026-11-28"
s["status"] = "v1.1: additive correction after R2 (decisione_R2 sec. 7); every v1 value and configuration is unchanged; v1 stays in the repository"
s["parent"] = {"spec": "configs/observations/piv_observation_spec_v1.yaml", "file_sha256": v1sha, "content_sha256": s["frozen"]["content_sha256"]}
s["roadmap"]["decision_R2"] = "mese3/02_virtual_piv/decisione_R2.md"; s["roadmap"]["commit_decision_R2"] = "82e37e0"
base = {k: s["configs"][k] for k in ("PIV-A", "PIV-B-in")}
new = {}
for f in ("A", "B-in"):
    c = copy.deepcopy(base[f"PIV-{f}"]); c.update({"noise": "corr_t", "t_nu": 7, "role": "sensitivity",
        "note": "heavy tails: marginal map to Student t (nu=7, excess kurtosis 2), same standardised correlated realisation, same nodes (v1.1, R2 sec. 4); not calibrated"})
    new[f"SENS-TAIL-{f}"] = c
for rej in ("iso", "clu"):
    c = copy.deepcopy(s["configs"][f"SENS-REJ-{rej}-sub-A"]); c["substitution"] = "mean5x5_iterative"
    c["note"] = "same rejected positions as SENS-REJ-%s-sub, substituted with the iterative mean of the known 5x5 neighbours (area declared in the Cardiff XML, not validated; v1.1, R2 sec. 5)" % rej
    new[f"SENS-REJ-{rej}-sub5-A"] = c
assert not set(new) & set(s["configs"])
s["configs"].update(new)
s["error"]["tail_sensitivity"] = {"construction": "standardised correlated Gaussian field of the matching PIV config -> z_t = F_t^-1(Phi(z); nu) / sqrt(nu/(nu-2)) node by node; times sigma_u,c",
    "nu": 7, "excess_kurtosis_marginal": "6/(nu-4) = 2", "motivation": "Neal residuals after the 5-MAD mask: excess kurtosis 1.6-2.8 (different processing and reference: not transferred as a PIV-GTMC property)",
    "keeps": "sigma_u, rank order (rank correlation identical to the PIV config), spatial structure; Pearson correlation changes slightly and is reported", "class": "sensibilita (not calibrated for flames)"}
s["rejection_sensitivity"]["substitution_rules"] = {"mean3x3_iterative": "principal rule of the `sub` level (TUM micro-PIV, GTMC practice)",
    "mean5x5_iterative": "variant: iterative mean of known neighbours in a 5x5 area; area declared in the Cardiff processing XML; the comparison with real substituted values depends on an unofficial decoder and is NOT a validation",
    "passes": "the passes are repeated until every rejected vector is filled; meta.sub_passes and meta.sub_passes_gt3 are recorded"}
s["rejection_sensitivity"]["applied_to"] = "PIV-A only"
cl = s["parameter_classification"]
for e in cl:
    if e["id"] == "error_correlation":
        e["source"] += ". D-rho closed at R2: Neal (different processing, residual includes the reference) does not confirm the model at the first lag; principal unchanged; SENS-RHO0 mandatory next to the principal on A and B-in"
cl += [
 {"id": "tail_sensitivity", "value": "SENS-TAIL: Student t, nu = 7 (excess kurtosis 2), same realisation", "class": "sensibilita", "source": "Neal residual kurtosis 1.6-2.8; not calibrated"},
 {"id": "substitution_5x5", "value": "mean5x5_iterative", "class": "sensibilita", "source": "area declared in the Cardiff XML (PIVD:S2D-01); comparison with real values is not a validation (unofficial decoder)"},
 {"id": "rho0_mandatory", "value": "SENS-RHO0 reported next to the principal in every phase-3 comparison on A and B-in", "class": "decisione R2", "source": "decisione_R2 sec. 3"},
]
s["provenance"]["v1_1_written_by_code_sha"] = subprocess.check_output(["git", "-C", str(C.REPO), "rev-parse", "HEAD"], text=True).strip()
s["provenance"]["parent_file_sha256"] = v1sha
s["frozen"] = {"content_sha256": P.spec_content_hash(s), "note": "sha256 of the canonical JSON of every key except 'frozen'; file sha256 in piv_observation_spec_v1_1.sha256"}
out = C.REPO / "configs/observations/piv_observation_spec_v1_1.yaml"
out.write_text(yaml.safe_dump(s, sort_keys=False, allow_unicode=True, width=140))
sha = hashlib.sha256(out.read_bytes()).hexdigest()
out.with_suffix(".sha256").write_text(f"{sha}  {out.name}\n")
spec = P.load_piv_spec(out, expected_sha256=sha)
print("v1.1 written", sha, "configs", len(spec.raw["configs"]), "(v1: 31)")
assert hashlib.sha256(v1p.read_bytes()).hexdigest() == v1sha
