# Biblio Anchor Verification — D11 (2026-06-02)
Bib file: NOT FOUND

============================================================
KEY: fremer-pvldb

VERIFIED GROUND TRUTH:
  lead_author   : Hengyu Ye
  all_authors   : Hengyu Ye, Jiadong Chen, Fuxin Jiang, Xiao He, Tieying Zhang, Jianjun Chen, Xiaofeng Gao
  venue         : Proceedings of the VLDB Endowment (PVLDB)
  volume        : 18
  number        : 11
  pages         : 3812--3825
  year          : 2025
  doi           : 10.14778/3749646.3749656
  code          : PUBLIC — https://github.com/YHYHYHYHYHY/Fremer
  watch_for     : If your .bib has Jiadong Chen as first author for the PVLDB entry, that is wrong. Jiadong Chen leads the arXiv version only.

YOUR .BIB ENTRY:
[BIB FILE NOT FOUND — paste entry manually]

============================================================
KEY: fremer-arxiv

VERIFIED GROUND TRUTH:
  lead_author   : Jiadong Chen
  all_authors   : Jiadong Chen, Hengyu Ye, Fuxin Jiang, Xiao He, Tieying Zhang, Jianjun Chen, Xiaofeng Gao
  venue         : arXiv preprint
  eprint        : 2507.12908
  year          : 2025
  doi           : arXiv:2507.12908
  watch_for     : Same paper as fremer-pvldb but arXiv listing puts Jiadong Chen first (equal contribution). Confirm your .bib key for the arXiv entry uses eprint=2507.12908, NOT the PVLDB DOI.

YOUR .BIB ENTRY:
[BIB FILE NOT FOUND — paste entry manually]

============================================================
KEY: aapa

VERIFIED GROUND TRUTH:
  lead_author   : Guilin Zhang
  all_authors   : Guilin Zhang, Srinivas Vippagunta, Raghavendra Nandagopal, Suchitra Raman, Jeff Xu, Marcus Pfeiffer, Shreeshankar Chatterjee, Ziqi Tan, Wulan Guo, Hailong Jiang
  affiliations  : George Washington University (GWU), Workday Inc., Youngstown State University (YSU)
  venue         : arXiv preprint
  eprint        : 2507.05653
  year          : 2025
  code          : PUBLIC — https://github.com/GuilinDev/aapa-simulator
  watch_for     : Confirm lead author is Guilin Zhang and affiliations include GWU, Workday, Youngstown State.

YOUR .BIB ENTRY:
[BIB FILE NOT FOUND — paste entry manually]

============================================================
KEY: optscaler

VERIFIED GROUND TRUTH:
  lead_author   : Ding Zou
  all_authors   : Ding Zou, Wei Lu, Zhibo Zhu, Xingyu Lu, Jun Zhou, Xiaojin Wang, KangYu Liu, Haiqing Wang, Kefan Wang, Renen Sun
  venue         : Proceedings of the VLDB Endowment (PVLDB)
  volume        : 17
  number        : 12
  year          : 2024
  doi           : 10.14778/3685800.3685829
  code          : NO PUBLIC CODE — Papers With Code confirms 'No code available'
  watch_for     : If your .bib entry has a url= or note= field pointing to a GitHub repo, DELETE that field. The paper is real; the code claim is not.

YOUR .BIB ENTRY:
[BIB FILE NOT FOUND — paste entry manually]

============================================================
KEY: hcmiu-qd719

VERIFIED GROUND TRUTH:
  finding       : Publications are an OPTIONAL Rector-discretion bonus, NOT a graduation gate.
  source        : Memory anchor confirmed by Dr. Ho Long Van acceptance 2026-05-22.
  watch_for     : No .bib entry typically needed for a university regulation. If one exists, confirm it does not claim publications are mandatory.

YOUR .BIB ENTRY:
[BIB FILE NOT FOUND — paste entry manually]

============================================================
ACTION REQUIRED:
  For each key, compare 'YOUR .BIB ENTRY' against 'VERIFIED GROUND TRUTH'.
  If mismatch on author/venue/year/doi: mark as ERRATA-013+ candidate.
  Special checks:
  - fremer-pvldb: first author must be Hengyu Ye
  - fremer-arxiv: must cite eprint 2507.12908, NOT PVLDB DOI
  - aapa: lead Guilin Zhang, affiliations GWU+Workday+YSU
  - optscaler: NO url/note field pointing to code (no code exists)
  - hcmiu-qd719: publications optional, not mandatory