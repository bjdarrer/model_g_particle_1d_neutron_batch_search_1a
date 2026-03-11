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

"The best next step is to make a tighter local scan around this winner and add the neutron-anchor 
workflow, so the length calibration stops being tautological." Yes, do this please!

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
