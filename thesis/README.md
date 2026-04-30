# Thesis

LaTeX manuscript and defence materials.

## Layout
- `manuscript/`  HCMIU Overleaf class file + `main.tex` + chapters/
- `defense/`     Slide deck for June 8-26 defence
- `proposal/`    Earlier IU drafts (frozen)

## Building
```bash
cd manuscript
latexmk -pdf -output-directory=build main.tex
```

Figures and tables are pulled from `../../reports/`. Run
`make sync-figures` (from project root) before final compile to ensure
local copies under `manuscript/figures/` and `manuscript/tables/` are
fresh.
