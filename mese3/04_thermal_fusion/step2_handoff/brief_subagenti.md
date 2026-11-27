# Common brief for literature subagents (mission 04, step 2, thermal measurement models)

Context: MSc thesis (MIT Deng Lab / PoliMi). We reconstruct instantaneous temperature fields from velocity observations on LES data
(RealPDEBench: NH3/CH4/air swirl flame, 1 bar, LES 2D slice 128x128, x in [-37,37] mm, y in [63,159] mm, dt = 0.25 ms, 2001 frames = 0.5 s,
channels: T, 3 velocity comps, pressure, HRR, mole fractions CH4 CO CO2 H2O NH2 NH3 OH; mean T ~1860-2190 K; |u| O(1-40) m/s; wall
thermal BCs and geometry undeclared). We want to build realistic THERMAL OBSERVATIONS from LES fields, grounded in documented experiments.
Priority: THERMOCOUPLES IN THE GAS (also hot gas) in combustion rigs comparable with PIV rigs.
Step 1 found: hot-gas thermocouples documented only in campaigns separate from PIV (Virginia Tech DOE report OSTI 1463254; TU Delft jet
combustor Sampat Fuel 2025; TU Delft closed JHC van den Berg 2024; EM2C Mercier 2016 cold ORZ), PoliMi (Olivani 2007) and UMD (Archer 2005)
only at abstract level; simultaneous compensated thermocouple + LDV in premixed flames (Heitor, Taylor, Whitelaw 1985; Ferrao & Heitor 1998;
Yanagi & Mimura 1981) only at abstract level.

## Rules (mandatory)
- Ground decisive claims in PRIMARY, verifiable sources (full text whenever possible). Web search/secondary only to find leads.
- For every claim give: full reference, DOI/URL actually used, section/page/figure/equation/table, short verbatim quote for decisive points,
  and READ LEVEL: TI (whole doc), TS (full text available; relevant sections read fully), PZ (partial/keyword), AB (abstract only), SEC (secondary/snippet).
- Label: [D] documented in the source; [I] your inference; [M] missing/not reported/not accessible.
- "not reported" != "not measured"; "same paper" != "simultaneous"; a traverse over many points is NOT a simultaneous sensor network.
- Distinguish: documented mounting vs reconstructed; measured parameter vs estimated by validated correlation vs transferred from another
  source vs unknown. A declared uncertainty is NOT automatically independent Gaussian noise. Physical turbulent fluctuation is NOT measurement noise.
- Do NOT invent "typical" values. If a number is not in the source, say [M].
- Record every query, every access attempt that FAILED (host + error), and alternative legitimate copies tried
  (OSTI, arXiv, HAL, institutional repositories, author pages, PMC, Semantic Scholar/OpenAlex open-access links, theses repositories).
- Networking: use curl via Bash (proxy preconfigured; CA bundle /root/.ccr/ca-bundle.crt). Known: sciencedirect 403, crossref 429 often,
  re.public.polimi.it / web.archive.org / europepmc unreachable; osti.gov, elib.dlr.de, hal.science, repository.tudelft.nl, arxiv,
  api.openalex.org, link.springer.com landing pages, iopscience, pmc.ncbi.nlm.nih.gov, semanticscholar reachable. Use WebSearch (load via
  ToolSearch "select:WebSearch") for discovery. Do not bypass permission denials. Do not contact authors.
- Save downloaded files ONLY under your own scratch dir given below, each download in its own new empty subdirectory; extract PDF text with
  `pdftotext` (or python -I with pypdf if needed). Never write into the git repository /home/user/official_thesis.
- Final answer: a structured report in markdown (it is what I will read; be complete and precise, no padding), with sections:
  1. Findings per source/configuration (with all the fields requested in your specific task); 2. Models/equations exactly as the authors
  write them (variables, units, assumptions, validity); 3. Parameter table with provenance class (measured / validated correlation /
  transferred / unknown); 4. Bibliography with DOI/URL + read level + sections/pages/figures used; 5. Queries run; 6. Failed accesses;
  7. What you could not verify. Also save the same report to <your scratch dir>/REPORT.md.
