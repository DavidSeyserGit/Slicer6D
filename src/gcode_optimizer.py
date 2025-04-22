#!/usr/bin/env python3
"""
gcode_optimizer.py

Reads a G‑code file, removes redundant or zero‑length moves,
collapses repeated feed rates and E‑values, and rounds coordinates.

Usage:
    python gcode_optimizer.py input.gcode output.gcode
"""

import re
import sys

# how many decimals to keep on X/Y/Z/E
PREC = 3
# minimal extrusion to bother
MIN_E_DELTA = 1e-5

# regex to pick out G1 parameters
RE_PARAM = re.compile(r'([XYZEFS])([-+]?\d*\.?\d+)')

def parse_params(line):
    """Return dict of params in a G1 line."""
    return {m.group(1): float(m.group(2)) for m in RE_PARAM.finditer(line)}

def fmt_param(letter, value):
    """Format a parameter with fixed precision, strip trailing zeros."""
    s = f"{value:.{PREC}f}".rstrip('0').rstrip('.')
    return f"{letter}{s}"

def optimize_gcode(lines):
    out = []
    last = {'X':None, 'Y':None, 'Z':None, 'E':None, 'F':None}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith(';'):
            # preserve empty/comments
            out.append(line)
            continue

        parts = line.split(None, 1)
        cmd = parts[0]
        rest = parts[1] if len(parts)>1 else ''

        if cmd in ('G0','G1'):
            p = parse_params(line)
            # compute deltas
            dx = ('X' in p and last['X'] is not None and abs(p['X']-last['X'])<1e-6)
            dy = ('Y' in p and last['Y'] is not None and abs(p['Y']-last['Y'])<1e-6)
            dz = ('Z' in p and last['Z'] is not None and abs(p['Z']-last['Z'])<1e-6)
            de = ('E' in p and last['E'] is not None and abs(p['E']-last['E'])<MIN_E_DELTA)
            # if it does nothing: skip
            if cmd=='G1' and all([
                dx or 'X' not in p,
                dy or 'Y' not in p,
                dz or 'Z' not in p,
                de or 'E' not in p
            ]):
                continue

            pieces = [cmd]
            # include only changed X/Y/Z/E
            for L in ('X','Y','Z','E'):
                if L in p:
                    if last[L] is None or abs(p[L]-last[L])> (MIN_E_DELTA if L=='E' else 1e-6):
                        pieces.append(fmt_param(L, p[L]))
                        last[L] = p[L]
            # include feedrate only if it changed
            if 'F' in p:
                if last['F'] is None or abs(p['F']-last['F'])>1e-3:
                    pieces.append(fmt_param('F', p['F']))
                    last['F'] = p['F']
            out.append(' '.join(pieces))
        else:
            # other commands: passthrough
            out.append(line)
    return out

def main():
    if len(sys.argv) != 3:
        print("Usage: python gcode_optimizer.py in.gcode out.gcode")
        sys.exit(1)

    inp, outp = sys.argv[1], sys.argv[2]
    with open(inp) as f:
        src = f.readlines()
    optimized = optimize_gcode(src)
    with open(outp, 'w') as f:
        f.write('\n'.join(optimized))
    print(f"Optimized {len(src)} → {len(optimized)} lines.")

if __name__ == '__main__':
    main()
