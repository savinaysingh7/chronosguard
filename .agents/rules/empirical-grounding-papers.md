---
trigger: model_decision
description: Apply when writing or updating research papers, technical reports, or benchmark tables in this workspace.
---

# Empirical & Bibliographic Grounding for Research Papers

1. **Source of Truth Is Live Code, Not Legacy Docs:** Never copy benchmark numbers from older Markdown/PDF reports without re-verifying against `src/` and freshly executed multi-seed scripts (minimum 5 seeds for stochastic splits/retraining ablations).
2. **Zero Interpolated Table Cells:** Every cell in every empirical table must trace directly to a verified script run or test log. If a condition was not measured, write `--` or run the benchmark — never extrapolate or pad rows.
3. **Verify Every Citation Externally:** Before finalizing `.bib` entries, verify exact title, full author list, venue name, volume/issue/pages, and DOI/URL against authoritative metadata APIs (Crossref, DBLP, Semantic Scholar, arXiv, or publisher proceedings).
