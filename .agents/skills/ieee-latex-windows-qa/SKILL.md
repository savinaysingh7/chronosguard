---
name: ieee-latex-windows-qa
description: >-
  Compile IEEEtran LaTeX + BibTeX papers on Windows without admin/UAC using standalone
  Tectonic, and run visual double-column QA to catch equation gutter collisions and
  table overflows before delivery.
---

# IEEE LaTeX Compilation & Visual QA (Windows, No Admin)

## 1. Install Standalone Tectonic (When pdflatex/MiKTeX Is Missing)
If `pdflatex`, `xelatex`, and `tectonic` are not on `PATH` (and `winget` fails on UAC):
1. Query the latest `x86_64-pc-windows-msvc.zip` asset from `tectonic-typesetting/tectonic` releases.
2. Extract `tectonic.exe` to `%USERPROFILE%\.local\bin\tectonic.exe`.
3. Compile `main.tex` + `references.bib` directly:
   ```powershell
   & "$env:USERPROFILE\.local\bin\tectonic.exe" main.tex
   ```

## 2. Mandatory Post-Compile Quality Checks
1. **Zero Overfull Hboxes:** Verify Tectonic output reports zero `Overfull \hbox` warnings on equations or tables.
2. **BibTeX Casing & Resolution:** Protect acronyms in `references.bib` titles with braces (e.g., `{AI}`, `{GEO}`, `{SHAP}`, `{MMD}`) and scan extracted PDF text for `[?]`, `??`, `undefined`, or `NaN`.
3. **Visual Double-Column Inspection:** Do NOT rely solely on page-width bounding boxes. Render key pages (Page 1, every page containing display equations or tables, and References) to PNGs via PyMuPDF (`fitz`) and visually inspect them to verify zero column-gutter overlaps, clean `aligned`/`multline` equation wrapping, and crisp table alignment.
