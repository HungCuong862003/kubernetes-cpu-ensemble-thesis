"""
_paths.py  —  shared path resolver for ALL phase_f/scripts/
Import this at the top of every F-task script.

Usage:
    from _paths import P
    omega = pd.read_csv(os.path.join(P['results'], 'omega_summary.csv'))
    out   = os.path.join(P['data'], 'myfile.csv')
"""

import os

def _build():
    # This file lives in phase_f/scripts/
    # So: scripts/ -> phase_f/ -> thesis_root/
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    phase_f_dir = os.path.dirname(scripts_dir)
    thesis_root = os.path.dirname(phase_f_dir)

    paths = {
        'scripts':  scripts_dir,
        'phase_f':  phase_f_dir,
        'thesis':   thesis_root,
        'data':     os.path.join(phase_f_dir, 'data'),
        'results':  os.path.join(thesis_root, 'results'),
        'src':      os.path.join(thesis_root, 'src'),
        'reports':  os.path.join(thesis_root, 'reports'),
        'bcf':      os.path.join(thesis_root, 'results', 'bcf'),
        'bcf_v2':   os.path.join(thesis_root, 'results', 'bcf_v2'),
        'foundation': os.path.join(thesis_root, 'results', 'foundation_comparison'),
    }

    # create data dir if missing
    os.makedirs(paths['data'], exist_ok=True)

    return paths

P = _build()

if __name__ == "__main__":
    print("Path resolver check:")
    for k, v in P.items():
        exists = "OK" if os.path.exists(v) else "MISSING"
        print(f"  {k:<12}: {v}  [{exists}]")
