#!/usr/bin/env python3
"""
Date: 11 March 2026 16:36 GMT

Neutron / neutral anchor search

Code: "model_g_particle_1d_neutron_batch_search_1a.py"

What it does:

- searches for the most neutral-looking branch it can find
- scores candidates by neutrality, outer-bias suppression, shell regularity, and stability
- writes best_candidate_neutron_anchor.json
- that JSON contains neutron_length_scale_m_per_unit, which is the anchor for the proton run

............
- Written by Brendan Darrer aided by ChatGPT 5.4 (extensive thinking)
- adapted from: @ https://github.com/blue-science/subquantum_kinetics/blob/master/particle.nb and
https://github.com/frostburn/tf2-model-g with https://github.com/frostburn/tf2-model-g/blob/master/docs/overview.pdf
- with ChatGPT 5.4 writing it and Brendan guiding it to produce a clean code.

Install:
    pip3 install numpy scipy matplotlib imageio imageio[ffmpeg]

Tested for: Ubuntu 24.04.3 LTS on i7-4790 (Optiplex 7020/9020), Python 3.10+
............

"The best next step is to make a tighter local scan around this winner and add the neutron-anchor workflow, so the length calibration stops being tautological." Yes, do this please!

STEP-BY-STEP COMMENTARY
=======================

What this script is for
-----------------------
This script searches for a neutral or neutron-like 1D Model G solution.
Its main job is not to find a proton-like charged particle, but to find a
candidate that is as electrically neutral as possible while still showing a
stable particle-like core-shell structure.

Why this matters
----------------
In the SQK calibration workflow, the neutron-like branch is useful because it
can provide a length-scale anchor without needing a net charge. The script
measures a shell/Turing wavelength in simulation units and then maps that to
the known neutron Compton wavelength in metres. That gives an estimated
conversion factor from simulation length units to SI metres.

High-level flow of the script
-----------------------------
1. Scan over a grid of trial parameters such as dy, b, g, amp, sx, st, Tseed.
2. For each trial, run the existing 1D shifted Model G solver from
   model_g_particle_1d_proton_batch_search_1c.py.
3. Measure neutral-branch diagnostics from the final pG, pX, pY profiles.
4. Build a neutron-anchor score that rewards:
   - low net charge proxy,
   - low outer electric bias,
   - balanced pX/pY core behaviour,
   - regular shell spacing,
   - late-time stability.
5. Keep the best candidate and write out summary files, CSV tables, plots,
   and a neutron-anchor JSON file for later proton refinement runs.

What the most important outputs mean
------------------------------------
- score_neutron_anchor:
    Main ranking score for the neutral search. Lower is better.
- neutron_length_scale_m_per_unit:
    Estimated metres-per-simulation-unit conversion obtained by forcing the
    measured simulation wavelength to match the neutron Compton wavelength.
- best_candidate_neutron_anchor.json:
    Small JSON file containing the neutron anchor information that can be fed
    into the proton refinement script.
- branch_label:
    Text label saying whether the best result looks genuinely neutral-like or
    still has unwanted positive/negative charge bias.

Important caution
-----------------
This script uses the shifted perturbation variables pG, pX, pY from the
working 1D solver as practical SQK proxies. So this is best read as a search
and calibration tool, not as a final proof that the full neutron branch has
been physically derived.

""""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.constants import h, m_n, c

_THIS_DIR = Path(__file__).resolve().parent
import sys
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from model_g_particle_1d_proton_batch_search_1c import (
    GridParams,
    ModelG1D,
    ModelParams,
    SeedParams,
    HAVE_SQK_MODULE,
    parse_float_list,
    compute_sqk_proxy_metrics,
    symmetrize_profile_to_radial,
    write_csv,
)

NEUTRON_COMPTON_WAVELENGTH_M = h / (m_n * c)


def score_neutron_candidate(model: ModelG1D, sol, pG: np.ndarray, pX: np.ndarray, pY: np.ndarray) -> dict:
    diag = model.diagnostics(sol.y[:, -1])
    sqk = compute_sqk_proxy_metrics(model, sol, pG, pX, pY, neutron_length_scale_m_per_unit=None)

    r, pX_r = symmetrize_profile_to_radial(model.x, pX)
    _, pY_r = symmetrize_profile_to_radial(model.x, pY)
    _, pG_r = symmetrize_profile_to_radial(model.x, pG)

    amp = max(float(np.max(np.abs(pX_r))), float(np.max(np.abs(pY_r))), 1.0e-30)
    neutrality_ratio = abs(float(sqk['sqk_q_core_proxy'])) / (abs(float(sqk['sqk_q_abs_core_proxy'])) + 1.0e-30)
    outer_bias_error = (abs(float(sqk['sqk_outer_bias_x'])) + abs(float(sqk['sqk_outer_bias_y']))) / amp
    core_pair_balance_error = abs(float(sqk['sqk_core_bias_x']) + float(sqk['sqk_core_bias_y'])) / amp
    center_bias_error = abs(float(diag['pY_core'])) / amp
    qproxy_balance_error = abs(float(diag['Qproxy_int_pYdx'])) / (np.trapezoid(np.abs(pY_r), r) + 1.0e-30)
    shell_err = float(sqk['sqk_shell_spacing_error'])
    stability_error = float(sqk['sqk_stability_error'])
    lambda_sim = float(sqk['sqk_lambda_sim']) if np.isfinite(sqk['sqk_lambda_sim']) else float('nan')
    lambda_missing_penalty = 0.0 if np.isfinite(lambda_sim) and lambda_sim > 0.0 else 1.0

    score_neutron = (
        0.30 * neutrality_ratio
        + 0.20 * outer_bias_error
        + 0.15 * core_pair_balance_error
        + 0.10 * center_bias_error
        + 0.10 * qproxy_balance_error
        + 0.10 * shell_err
        + 0.03 * stability_error
        + 0.02 * lambda_missing_penalty
    )

    if np.isfinite(lambda_sim) and lambda_sim > 0.0:
        length_scale_m_per_unit = NEUTRON_COMPTON_WAVELENGTH_M / lambda_sim
        neutron_lambda_m = lambda_sim * length_scale_m_per_unit
        neutron_lambda_error_frac = abs(neutron_lambda_m - NEUTRON_COMPTON_WAVELENGTH_M) / NEUTRON_COMPTON_WAVELENGTH_M
    else:
        length_scale_m_per_unit = float('nan')
        neutron_lambda_m = float('nan')
        neutron_lambda_error_frac = float('nan')

    if neutrality_ratio < 0.20 and outer_bias_error < 0.25:
        branch_label = 'neutral-like (neutron-anchor candidate)'
    elif float(sqk['sqk_q_core_proxy']) > 0.0:
        branch_label = 'positive-bias (not very neutral)'
    elif float(sqk['sqk_q_core_proxy']) < 0.0:
        branch_label = 'negative-bias (not very neutral)'
    else:
        branch_label = 'indeterminate'

    return {
        **diag,
        **sqk,
        'neutron_length_scale_m_per_unit': float(length_scale_m_per_unit) if np.isfinite(length_scale_m_per_unit) else float('nan'),
        'neutron_length_scale_source': 'neutron-self-anchor' if np.isfinite(length_scale_m_per_unit) else 'unavailable',
        'neutron_lambda_m': float(neutron_lambda_m) if np.isfinite(neutron_lambda_m) else float('nan'),
        'neutron_lambda_error_frac': float(neutron_lambda_error_frac) if np.isfinite(neutron_lambda_error_frac) else float('nan'),
        'neutron_neutrality_ratio': float(neutrality_ratio),
        'neutron_outer_bias_error': float(outer_bias_error),
        'neutron_core_pair_balance_error': float(core_pair_balance_error),
        'neutron_center_bias_error': float(center_bias_error),
        'neutron_qproxy_balance_error': float(qproxy_balance_error),
        'score_neutron_anchor': float(score_neutron),
        'branch_label': branch_label,
    }


def make_neutron_summary_plot(path: str, x: np.ndarray, pG: np.ndarray, pX: np.ndarray, pY: np.ndarray, row: dict) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(x, pY, label='pY final')
    axes[0].plot(x, pX, label='pX final')
    axes[0].axhline(0.0, linewidth=0.8)
    axes[0].axvline(0.0, linewidth=0.8)
    axes[0].set_ylabel('shifted electric sector')
    axes[0].set_title('Neutral / neutron-anchor candidate')
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(x, pG, label='pG final')
    axes[1].axhline(0.0, linewidth=0.8)
    axes[1].axvline(0.0, linewidth=0.8)
    axes[1].set_xlabel('x [simulation units]')
    axes[1].set_ylabel('shifted gravity sector')
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    text = (
        f"anchor score = {row['score_neutron_anchor']:.4f}\n"
        f"neutrality={row['neutron_neutrality_ratio']:.4f}, outer_bias={row['neutron_outer_bias_error']:.4f}\n"
        f"shell_err={row['sqk_shell_spacing_error']:.4f}, stability={row['sqk_stability_error']:.4f}\n"
        f"dy={row['dy']}, b={row['b']}, g={row['g']}\n"
        f"amp={row['amp']}, sx={row['sx']}, st={row['st']}, Tseed={row['Tseed']}\n"
        f"lambda_sim={row['sqk_lambda_sim']:.6g}, anchor={row['neutron_length_scale_m_per_unit']:.6g} m/unit"
    )
    axes[0].text(0.02, 0.98, text, transform=axes[0].transAxes, va='top',
                 bbox=dict(boxstyle='round,pad=0.35', alpha=0.12))
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Batch-search for a neutral / neutron-anchor 1D Model G profile')
    parser.add_argument('--outdir', default='./model_g_neutron_batch_search_out_1a', help='Output folder')
    parser.add_argument('--L', type=float, default=20.0)
    parser.add_argument('--nx', type=int, default=51)
    parser.add_argument('--tfinal', type=float, default=8.0)
    parser.add_argument('--max-step', type=float, default=0.1)
    parser.add_argument('--rtol', type=float, default=1e-4)
    parser.add_argument('--atol', type=float, default=1e-6)
    parser.add_argument('--nframes', type=int, default=30)
    parser.add_argument('--sign', type=int, default=-1)
    parser.add_argument('--a', type=float, default=14.0)
    parser.add_argument('--dx', type=float, default=1.0)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--q', type=float, default=1.0)
    parser.add_argument('--s', type=float, default=0.0)
    parser.add_argument('--u', type=float, default=0.0)
    parser.add_argument('--v', type=float, default=0.0)
    parser.add_argument('--w', type=float, default=0.0)
    parser.add_argument('--dy', default='9.5,10.5,12.0', help='Comma list')
    parser.add_argument('--b', default='27,28,29', help='Comma list')
    parser.add_argument('--g', default='0.09,0.10,0.11', help='Comma list')
    parser.add_argument('--amp', default='0.9,1.0,1.1', help='Comma list')
    parser.add_argument('--sx', default='0.9,1.0,1.1', help='Comma list')
    parser.add_argument('--st', default='1.5', help='Comma list')
    parser.add_argument('--Tseed', default='3.0', help='Comma list')
    parser.add_argument('--topk', type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    gp = GridParams(L=args.L, nx=args.nx, Tfinal=args.tfinal, max_step=args.max_step, rtol=args.rtol, atol=args.atol, dense=False)

    dy_vals = parse_float_list(args.dy)
    b_vals = parse_float_list(args.b)
    g_vals = parse_float_list(args.g)
    amp_vals = parse_float_list(args.amp)
    sx_vals = parse_float_list(args.sx)
    st_vals = parse_float_list(args.st)
    tseed_vals = parse_float_list(args.Tseed)

    rows = []
    best = None
    best_profiles = None
    combos = [(dy, b, g, amp, sx, st, Tseed) for dy in dy_vals for b in b_vals for g in g_vals for amp in amp_vals for sx in sx_vals for st in st_vals for Tseed in tseed_vals]
    t0 = time.time()

    for dy, b, g, amp, sx, st, Tseed in combos:
        mp = ModelParams(a=args.a, b=b, dx=args.dx, dy=dy, p=args.p, q=args.q, g=g, s=args.s, u=args.u, v=args.v, w=args.w)
        sp = SeedParams(sign=args.sign, amp=amp, sx=sx, st=st, Tseed=Tseed, nseeds=1)
        row = {
            'dy': dy, 'b': b, 'g': g, 'amp': amp, 'sx': sx, 'st': st, 'Tseed': Tseed,
            'solver_success': False, 'solver_message': ''
        }
        try:
            model = ModelG1D(mp, gp, sp)
            sol = model.run(nframes=args.nframes)
            pG, pX, pY = model.unpack(sol.y[:, -1])
            scored = score_neutron_candidate(model, sol, pG, pX, pY)
            row.update(scored)
            row.update({
                'solver_success': bool(sol.success),
                'solver_message': str(sol.message),
                'nfev': int(getattr(sol, 'nfev', -1)),
                'njev': int(getattr(sol, 'njev', -1)),
                'nlu': int(getattr(sol, 'nlu', -1)),
                'rank_score': float(scored['score_neutron_anchor']),
            })
            if best is None or row['rank_score'] < best['rank_score']:
                best = row.copy()
                best_profiles = (model.x.copy(), pG.copy(), pX.copy(), pY.copy())
        except Exception as exc:
            row.update({
                'solver_success': False,
                'solver_message': repr(exc),
                'rank_score': 1e9,
                'score_neutron_anchor': 1e9,
            })
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda rr: float(rr.get('rank_score', 1e9)))
    top_rows = rows_sorted[: max(1, min(args.topk, len(rows_sorted)))]

    csv_all = os.path.join(args.outdir, 'all_candidates.csv')
    csv_top = os.path.join(args.outdir, 'top_candidates.csv')
    anchor_json = os.path.join(args.outdir, 'best_candidate_neutron_anchor.json')
    best_json = os.path.join(args.outdir, 'best_candidate.json')
    plot_png = os.path.join(args.outdir, 'best_candidate_plot.png')
    best_npz = os.path.join(args.outdir, 'best_candidate_profiles.npz')
    summary_txt = os.path.join(args.outdir, 'summary.txt')

    preferred = [
        'rank_score', 'score_neutron_anchor', 'dy', 'b', 'g', 'amp', 'sx', 'st', 'Tseed',
        'pG_core', 'pX_core', 'pY_core', 'Qproxy_int_pYdx', 'pY_peak_abs', 'pY_fwhm_abs', 'polarity_label',
        'sqk_lambda_sim', 'sqk_r_core_sim', 'sqk_shell_spacing_error',
        'sqk_core_bias_x', 'sqk_core_bias_y', 'sqk_outer_bias_x', 'sqk_outer_bias_y',
        'sqk_q_core_proxy', 'sqk_q_abs_core_proxy', 'sqk_sg_core_proxy', 'sqk_q_far_proxy',
        'sqk_charge_consistency', 'sqk_tail_rel_error', 'sqk_proton_charge_bias_error', 'sqk_stability_error',
        'sqk_length_scale_m_per_unit', 'sqk_length_scale_source', 'sqk_proton_lambda_m', 'sqk_proton_lambda_error_frac',
        'sqk_charge_scale_c_per_proxy', 'sqk_active_mass_scale_kg_per_proxy', 'score_sqk_proxy',
        'neutron_neutrality_ratio', 'neutron_outer_bias_error',
        'neutron_core_pair_balance_error', 'neutron_center_bias_error', 'neutron_qproxy_balance_error',
        'neutron_length_scale_m_per_unit', 'neutron_length_scale_source', 'neutron_lambda_m', 'neutron_lambda_error_frac',
        'branch_label', 'solver_success', 'solver_message', 'nfev', 'njev', 'nlu'
    ]
    extra = sorted({k for row in rows_sorted for k in row.keys()} - set(preferred))
    fieldnames = preferred + extra
    write_csv(csv_all, [{k: row.get(k, '') for k in fieldnames} for row in rows_sorted], fieldnames)
    write_csv(csv_top, [{k: row.get(k, '') for k in fieldnames} for row in top_rows], fieldnames)

    if best is not None:
        with open(best_json, 'w', encoding='utf-8') as f:
            json.dump(best, f, indent=2)
        anchor_payload = {
            'neutron_length_scale_m_per_unit': best.get('neutron_length_scale_m_per_unit'),
            'neutron_lambda_sim': best.get('sqk_lambda_sim'),
            'neutron_lambda_m': best.get('neutron_lambda_m'),
            'dy': best.get('dy'), 'b': best.get('b'), 'g': best.get('g'),
            'amp': best.get('amp'), 'sx': best.get('sx'), 'st': best.get('st'), 'Tseed': best.get('Tseed'),
            'score_neutron_anchor': best.get('score_neutron_anchor'),
            'branch_label': best.get('branch_label'),
        }
        with open(anchor_json, 'w', encoding='utf-8') as f:
            json.dump(anchor_payload, f, indent=2)
        if best_profiles is not None:
            x, pG, pX, pY = best_profiles
            np.savez(best_npz, x=x, pG=pG, pX=pX, pY=pY)
            make_neutron_summary_plot(plot_png, x, pG, pX, pY, best)

    elapsed = time.time() - t0
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write('Model G 1D neutral / neutron-anchor batch search summary\n')
        f.write(f'trials = {len(rows)}\n')
        f.write(f'elapsed_sec = {elapsed:.3f}\n')
        f.write(f'sqk_module_available = {HAVE_SQK_MODULE}\n')
        f.write(f'neutron_compton_wavelength_m = {NEUTRON_COMPTON_WAVELENGTH_M:.15e}\n')
        if best is not None:
            f.write('Best candidate\n')
            for key in fieldnames:
                if key in best:
                    f.write(f'  {key} = {best[key]}\n')
            f.write('\nInterpretation\n')
            f.write('  Lower score_neutron_anchor is better.\n')
            f.write('  neutron_length_scale_m_per_unit is the anchor to pass into the proton refinement script.\n')
            f.write('  branch_label should ideally read neutral-like (neutron-anchor candidate).\n')

    print(f'Wrote: {csv_all}')
    print(f'Wrote: {csv_top}')
    print(f'Wrote: {anchor_json}')
    print(f'Wrote: {best_json}')
    print(f'Wrote: {best_npz}')
    print(f'Wrote: {plot_png}')
    print(f'Wrote: {summary_txt}')


if __name__ == '__main__':
    main()
