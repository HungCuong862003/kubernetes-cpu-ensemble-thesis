# Data

Three external datasets are used in this thesis. Each lives under its own
folder in `raw/` with a `PROVENANCE.md` containing source URL, download
date, SHA-256 checksum, and license.

## Dataset summary
| Dataset    | Granularity | Purpose                       |
|------------|-------------|-------------------------------|
| Alibaba    | 5-min       | Primary (NNLS ensemble train) |
| Bitbrains  | 5-min       | Cross-dataset validation      |
| ByteDance  | 10-min      | Third validation              |

## Subdirectory roles
- `raw/`       Immutable inputs. Never edit.
- `interim/`   Reversible transforms (cleaned, joined). Regenerable.
- `processed/` Model-ready feature tables (per dataset x horizon).
- `external/`  Reference series, calendars, third-party tables.
