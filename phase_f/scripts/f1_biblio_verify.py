"""
f1_biblio_verify.py  —  D11 Task A
Prints verified ground truth for 5 memory-flagged .bib entries and dumps
matching entries from your actual .bib file for direct comparison.

Place in:  phase_f/
Run:       python3 f1_biblio_verify.py

Output:
    phase_f/data/biblio_anchor_verification.md
"""

import os
import re

PHASE_F_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_ROOT = os.path.dirname(PHASE_F_DIR)
DATA_DIR    = os.path.join(PHASE_F_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Ground truth — web-verified 2026-06-02
# ------------------------------------------------------------------
GROUND_TRUTH = {
    "fremer-pvldb": {
        "lead_author":  "Hengyu Ye",
        "all_authors":  "Hengyu Ye, Jiadong Chen, Fuxin Jiang, Xiao He, "
                        "Tieying Zhang, Jianjun Chen, Xiaofeng Gao",
        "venue":        "Proceedings of the VLDB Endowment (PVLDB)",
        "volume":       "18", "number": "11", "pages": "3812--3825", "year": "2025",
        "doi":          "10.14778/3749646.3749656",
        "code":         "PUBLIC — github.com/YHYHYHYHYHY/Fremer",
        "WATCH":        "If .bib has Jiadong Chen first -> WRONG for PVLDB entry. "
                        "Jiadong Chen leads arXiv version only.",
    },
    "fremer-arxiv": {
        "lead_author":  "Jiadong Chen",
        "all_authors":  "Jiadong Chen, Hengyu Ye, Fuxin Jiang, Xiao He, "
                        "Tieying Zhang, Jianjun Chen, Xiaofeng Gao",
        "venue":        "arXiv preprint", "eprint": "2507.12908", "year": "2025",
        "WATCH":        "Same paper as PVLDB but arXiv lists Jiadong Chen first. "
                        "Confirm eprint=2507.12908, NOT the PVLDB DOI.",
    },
    "aapa": {
        "lead_author":  "Guilin Zhang",
        "all_authors":  "Guilin Zhang, Srinivas Vippagunta, Raghavendra Nandagopal, "
                        "Suchitra Raman, Jeff Xu, Marcus Pfeiffer, "
                        "Shreeshankar Chatterjee, Ziqi Tan, Wulan Guo, Hailong Jiang",
        "affiliations": "George Washington University, Workday Inc., Youngstown State University",
        "venue":        "arXiv preprint", "eprint": "2507.05653", "year": "2025",
        "code":         "PUBLIC — github.com/GuilinDev/aapa-simulator",
        "WATCH":        "Confirm lead=Guilin Zhang and affiliations GWU+Workday+YSU.",
    },
    "optscaler": {
        "lead_author":  "Ding Zou",
        "all_authors":  "Ding Zou, Wei Lu, Zhibo Zhu, Xingyu Lu, Jun Zhou, "
                        "Xiaojin Wang, KangYu Liu, Haiqing Wang, Kefan Wang, Renen Sun",
        "venue":        "Proceedings of the VLDB Endowment (PVLDB)",
        "volume":       "17", "number": "12", "year": "2024",
        "doi":          "10.14778/3685800.3685829",
        "code":         "NO PUBLIC CODE — Papers With Code confirms 'No code available'",
        "WATCH":        "DELETE any url= or note= field pointing to GitHub — no code exists.",
    },
    "hcmiu-qd719": {
        "finding":      "Publications = OPTIONAL Rector-discretion bonus, NOT graduation gate.",
        "source":       "Memory anchor + Dr. Ho Long Van acceptance 2026-05-22.",
        "WATCH":        "No .bib entry usually needed. If one exists, confirm publications optional.",
    },
}

BIB_KEYS = list(GROUND_TRUTH.keys())

# ------------------------------------------------------------------
# Try to find .bib file at thesis root
# ------------------------------------------------------------------
bib_content    = {}
bib_file_found = None

for name in ["thesis.bib", "references.bib", "main.bib",
             "submission/thesis.bib", "submission/references.bib"]:
    path = os.path.join(THESIS_ROOT, name)
    if os.path.isfile(path):
        bib_file_found = path
        break

if bib_file_found:
    print(f"Found .bib file: {bib_file_found}")
    raw = open(bib_file_found, encoding="utf-8", errors="replace").read()
    for key in BIB_KEYS:
        m = re.search(rf"(@\w+\{{{re.escape(key)},[^@]*)", raw, re.DOTALL | re.IGNORECASE)
        bib_content[key] = m.group(1).strip() if m else f"[KEY '{key}' NOT FOUND in bib file]"
else:
    print("WARNING: .bib file not found at thesis root. Searched:")
    print(f"  {THESIS_ROOT}/thesis.bib  references.bib  main.bib  submission/...")
    print("  Paste your .bib entries manually into the output file.\n")
    for key in BIB_KEYS:
        bib_content[key] = "[BIB FILE NOT FOUND — paste entry manually]"

# ------------------------------------------------------------------
# Print comparison + save
# ------------------------------------------------------------------
lines = [
    "# Biblio Anchor Verification — D11 (2026-06-02)",
    f"Bib file: {bib_file_found or 'NOT FOUND'}",
    "",
]

for key, truth in GROUND_TRUTH.items():
    lines += [f"{'='*60}", f"KEY: {key}", ""]
    lines.append("VERIFIED GROUND TRUTH:")
    for k, v in truth.items():
        lines.append(f"  {k:<14}: {v}")
    lines += ["", "YOUR .BIB ENTRY:"]
    lines.append(bib_content.get(key, "[not extracted]"))
    lines.append("")

lines += [
    "="*60,
    "ACTIONS:",
    "  Match -> mark OK",
    "  Mismatch author/venue/year/doi -> open ERRATA-013+ in ERRATA.md",
    "  fremer-pvldb : first author must be Hengyu Ye",
    "  fremer-arxiv : must use eprint 2507.12908 not PVLDB DOI",
    "  aapa         : lead Guilin Zhang, affiliations GWU+Workday+YSU",
    "  optscaler    : NO url/note field pointing to code",
    "  hcmiu-qd719  : publications optional not mandatory",
]

out_text = "\n".join(lines)
print(out_text)

out_path = os.path.join(DATA_DIR, "biblio_anchor_verification.md")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(out_text)
print(f"\nSaved: {out_path}")
print("\nf1_biblio_verify.py DONE")
