#!/usr/bin/env python3
"""
Gurobi-Native BARON-Inspired SCDCOPF Solver (Innovations A, B, C)
+ Dr. Mujahid's Normalized Weighted Slack Sweep:
    min  w * sum((s^+ + s^-) / g_max) + sum(eta_pu)

Imports data loading and independent physical verification from check_apr_eta_limits.py.
Uses a persistent gurobipy model for simplex warm-starting and reduced-cost probing.
"""
import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import gurobipy as gp
import numpy as np
import pandas as pd

from check_apr_eta_limits import (
    resolve_case, load_case, select_outages, read_fixed_dispatch,
    build_model, add_apr_slacks, apply_eta_limits,
    solve_model, analyze, recover, eta_limit_diagnostics
)


class PersistentGurobiRunner:
    """Keeps a single gurobipy Model in memory for fast basis warm-starts and RC probing."""
    def __init__(self, model, args):
        self.base_model = model
        self.args = args
        self.env = gp.Env(empty=True)
        self.env.setParam('OutputFlag', int(args.tee))
        self.env.start()

        self.gm = gp.Model(env=self.env)
        self.gm.Params.Threads = args.threads
        self.gm.Params.FeasibilityTol = 1e-9
        self.gm.Params.OptimalityTol = 1e-9
        self.gm.Params.MIPGap = args.mip_gap
        self.gm.Params.TimeLimit = args.time_limit

        n = len(model.c0)
        lb = np.where(np.isfinite(model.lb), model.lb, -gp.GRB.INFINITY)
        ub = np.where(np.isfinite(model.ub), model.ub, gp.GRB.INFINITY)
        vtype = np.where(model.integer > 0, gp.GRB.BINARY, gp.GRB.CONTINUOUS)
        self.v = self.gm.addMVar(n, lb=lb, ub=ub, vtype=vtype)

        eq = np.isfinite(model.lower) & (model.lower == model.upper)
        up = np.isfinite(model.upper) & ~eq
        low = np.isfinite(model.lower) & ~eq
        if eq.any():
            self.gm.addMConstr(model.A[eq], self.v, '=', model.upper[eq])
        if up.any():
            self.gm.addMConstr(model.A[up], self.v, '<', model.upper[up])
        if low.any():
            self.gm.addMConstr(model.A[low], self.v, '>', model.lower[low])
        self.gm.update()

    def solve(self, q_obj, ub_override=None, constant=0.0):
        t0 = perf_counter()
        scale = float(max(1.0, np.max(np.abs(q_obj), initial=0.0)))
        qscaled = q_obj / scale

        # In-place upper bound and objective updates (preserves simplex basis!)
        target_ub = self.base_model.ub if ub_override is None else ub_override
        self.v.UB = np.where(np.isfinite(target_ub), target_ub, gp.GRB.INFINITY)
        self.gm.setObjective(qscaled @ self.v, gp.GRB.MINIMIZE)
        self.gm.optimize()

        if self.gm.Status == gp.GRB.INF_OR_UNBD:
            self.gm.Params.DualReductions = 0
            self.gm.reset()
            self.gm.optimize()

        status = {
            gp.GRB.OPTIMAL: 'optimal',
            gp.GRB.INFEASIBLE: 'infeasible',
            gp.GRB.UNBOUNDED: 'unbounded',
            gp.GRB.TIME_LIMIT: 'limit'
        }.get(self.gm.Status, f'gurobi_status_{self.gm.Status}')

        vec, rc, bound = None, None, None
        if self.gm.SolCount > 0:
            vec = np.asarray(self.v.X, dtype=float)
        if status == 'optimal':
            bound = float(self.gm.ObjVal) * scale + constant
            if not self.base_model.integer.any():
                rc = np.asarray(self.v.RC, dtype=float) * scale

        res = dict(
            status=status, solver='gurobi', total_seconds=perf_counter() - t0,
            solver_best_bound_penalized=bound,
            solver_penalized_objective=( None if vec is None else float(q_obj @ vec + constant) ),
            solver_point_verified=(vec is not None and status == 'optimal')
        )
        return res, vec, rc

    def close(self):
        self.gm.dispose()
        self.env.dispose()


# ==============================================================================
# INNOVATION A: Analytical FBBT & Monotonicity Bounds on s+, s- (BARON Presolve)
# ==============================================================================
def innovation_a_s_bounds(model, case, outages, fix_regimes=True):
    """
    Propagates analytical bounds [z_k^L, z_k^U] -> [U_plus, U_minus] on (s^+, s^-)
    and fixes guaranteed proportional (s^- = 0) or saturated (s^+ = 0) pairs.
    """
    ng, nk = len(case['lo']), len(outages)
    g_L, g_U, a = case['lo'], case['hi'], case['a']

    z_L, z_U = np.zeros(nk), np.ones(nk)
    for s, k in enumerate(outages):
        surv = [i for i in range(ng) if i != k]
        # Lower bound z_L: even with max headroom (g_L), response must cover g_L[k]
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if np.sum(np.minimum(a[surv] * mid, case['hi'][surv] - g_L[surv])) >= g_L[k]:
                hi = mid
            else:
                lo = mid
        z_L[s] = lo

        # Upper bound z_U: with min headroom (g_U), pickup needed is at most g_U[k]
        if np.sum(case['hi'][surv] - g_U[surv]) > g_U[k] + 1e-7:
            lo, hi = z_L[s], 1.0
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if np.sum(np.minimum(a[surv] * mid, case['hi'][surv] - g_U[surv])) >= g_U[k]:
                    hi = mid
                else:
                    lo = mid
            z_U[s] = hi

    ub = model.ub.copy()
    n_pairs = len(model.slack_pairs)
    U_plus, U_minus = np.zeros(n_pairs), np.zeros(n_pairs)
    n_fixed_prop, n_fixed_sat = 0, 0

    for t, (s, i) in enumerate(model.slack_pairs):
        U_plus[t] = max(1e-4, float(case['hi'][i] - g_L[i]))
        U_minus[t] = max(1e-4, float(g_U[i] + a[i] * z_U[s] - g_L[i]))
        ub[model.splus[t]] = min(ub[model.splus[t]], U_plus[t])
        ub[model.sminus[t]] = min(ub[model.sminus[t]], U_minus[t])

        if fix_regimes:
            delta_L = case['hi'][i] - g_U[i] - a[i] * z_U[s]
            delta_U = case['hi'][i] - g_L[i] - a[i] * z_L[s]
            if delta_L >= 1e-6:
                ub[model.sminus[t]] = 0.0
                n_fixed_prop += 1
            elif delta_U <= -1e-6:
                ub[model.splus[t]] = 0.0
                n_fixed_sat += 1

    stats = dict(fixed_proportional=n_fixed_prop, fixed_saturated=n_fixed_sat, total_pairs=n_pairs)
    return replace(model, ub=ub), U_plus, U_minus, stats


# ==============================================================================
# INNOVATION B: Difference-of-Convex (DC) Selective Slack Reweighting
# ==============================================================================
def build_weighted_objective(model, weight, theta_plus, theta_minus, include_gen_cost=False, penalty=1500.0):
    """
    Iter 0: theta_plus = theta_minus = 1 / g_max (Dr. Mujahid's normalized slack objective).
    Iter >= 1: DC subgradient weights penalize ONLY the smaller slack in each (s^+, s^-) pair.
    """
    q = np.zeros_like(model.c0)
    if include_gen_cost:
        q[model.g] = model.original_cost[model.g] / penalty
    q[model.splus] = weight * theta_plus
    q[model.sminus] = weight * theta_minus
    q[model.eta_line] = 1.0
    q[model.eta_unit] = 1.0
    return q


def dc_update_weights(vec, model, base_scale_p, base_scale_m, tau=1e-5):
    """Subgradient linearization of min(s^+, s^-) = (s^+ + s^-) - max(s^+, s^-)."""
    sp = np.maximum(0.0, vec[model.splus])
    sm = np.maximum(0.0, vec[model.sminus])
    denom = sp + sm + 2.0 * tau
    return base_scale_p * (sm + tau) / denom, base_scale_m * (sp + tau) / denom


# ==============================================================================
# INNOVATION C: Fixed-Regime Polishing + Gurobi Reduced-Cost (RC) Probing
# ==============================================================================
def innovation_c_gurobi_polish(runner, model, case, outages, cand_vec, max_probe_rounds=3):
    """
    1. Locks the active complementarity pattern (s^- = 0 or s^+ = 0) from cand_vec.
    2. Minimizes the TRUE operating cost F = C_g + C_l + C_u in Gurobi.
    3. Probes Gurobi's Reduced Costs (RC) at degenerate breakpoints (s^+ == 0 and s^- == 0)
       to flip any binding regime that further lowers operating cost F.
    """
    ub = model.ub.copy()
    sp, sm = cand_vec[model.splus], cand_vec[model.sminus]
    g_cand = cand_vec[model.g]

    # Identify physical regime from candidate dispatch (using scalar APR recovery if needed)
    regime = np.ones(len(model.slack_pairs), dtype=int)  # +1: prop (s^-=0), -1: sat (s^+=0)
    for s, k in enumerate(outages):
        _, zr = recover(case, g_cand, k, 1e-10)
        zr = zr if zr is not None else cand_vec[model.z[s]]
        for t, (s_idx, i) in enumerate(model.slack_pairs):
            if s_idx != s:
                continue
            diff = (case['hi'][i] - (g_cand[i] + case['a'][i] * zr)) * case['base']
            if abs(diff) <= 1e-2:
                regime[t] = 1 if sm[t] <= sp[t] else -1
            else:
                regime[t] = 1 if diff > 0 else -1

    total_time = 0.0
    best_res, best_vec = None, None

    for _ in range(max_probe_rounds):
        ub_iter = ub.copy()
        for t in range(len(model.slack_pairs)):
            if regime[t] == 1:
                ub_iter[model.sminus[t]] = 0.0
            else:
                ub_iter[model.splus[t]] = 0.0

        res, vec, rc = runner.solve(
            model.original_cost, ub_override=ub_iter, constant=model.original_constant
        )
        total_time += res['total_seconds']
        if vec is None or res['status'] != 'optimal':
            break
        best_res, best_vec = res, vec

        if rc is None:
            break

        # Gurobi Reduced-Cost Breakpoint Probing:
        # Find pairs at the saturation boundary (both s^+ ~ 0 and s^- ~ 0) where
        # releasing the fixed slack has a strictly negative reduced cost (improves F).
        sp_cur, sm_cur = vec[model.splus], vec[model.sminus]
        flip_candidates = []
        for t in range(len(model.slack_pairs)):
            if sp_cur[t] <= 1e-6 and sm_cur[t] <= 1e-6:
                fixed_col = model.sminus[t] if regime[t] == 1 else model.splus[t]
                if rc[fixed_col] < -1e-3:
                    flip_candidates.append((rc[fixed_col], t))

        if not flip_candidates:
            break
        # Flip up to 5 most negative reduced-cost breakpoint pairs and warm-start Gurobi
        flip_candidates.sort()
        for _, t_flip in flip_candidates[:5]:
            regime[t_flip] *= -1

    if best_res is not None:
        best_res['total_seconds'] = total_time
    return best_res, best_vec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True)
    p.add_argument('--case-dir', default='../excel_outputs')
    p.add_argument('--output-dir', default='output_baron_gurobi')
    p.add_argument('--solver', choices=['gurobi', 'highs'], default='gurobi')
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--demand-scale', type=float, default=1.0)
    p.add_argument('--penalty', type=float, default=1500.0)
    p.add_argument('--weights', nargs='+', type=float, default=[10000, 1000, 100, 10, 1])
    p.add_argument('--dc-iters', type=int, default=3, help='Innovation B DC reweighting iterations per w')
    p.add_argument('--bound-norm', choices=['gmax', 'tight'], default='gmax',
                   help='gmax: Advisor 1/g_max; tight: Innovation A 1/U_bound')
    p.add_argument('--include-gen-cost', action='store_true',
                   help='Also include c^T g / M_eta in the weighted LP objective')
    p.add_argument('--stop-on-apr-loss', action='store_true',
                   help='Stop decreasing w once direct APR complementarity is lost')
    p.add_argument('--compare-exact', action='store_true', help='Solve exact MILP reference in Gurobi')
    p.add_argument('--generator-contingencies', default='all')
    p.add_argument('--no-line-contingencies', action='store_true')
    p.add_argument('--islanding', choices=['error', 'skip'], default='skip')
    p.add_argument('--thermal-mode', choices=['soft', 'hard'], default='soft')
    p.add_argument('--eta-line-fraction', type=float, default=None)
    p.add_argument('--eta-unit-fraction', type=float, default=None)
    p.add_argument('--eta-line-mw', type=float, default=None)
    p.add_argument('--eta-unit-mw', type=float, default=None)
    p.add_argument('--eta-line-total-mw', type=float, default=None)
    p.add_argument('--eta-unit-total-mw', type=float, default=None)
    p.add_argument('--time-limit', type=float, default=300.0)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--mip-gap', type=float, default=1e-7)
    p.add_argument('--apr-tol-mw', type=float, default=1e-4)
    p.add_argument('--feas-tol-mw', type=float, default=1e-4)
    p.add_argument('--fixed-dispatch', default=None)
    p.add_argument('--fixed-method', default='central')
    p.add_argument('--fixed-run', type=int, default=1)
    p.add_argument('--tee', action='store_true')
    args = p.parse_args()
    args.backend = args.solver

    t_start = perf_counter()
    path = resolve_case(args.case, args.case_dir)
    case = load_case(path, args.demand_scale)
    case['a'] = args.gamma * (case['hi'] - case['lo'])
    args.base = case['base']
    outages = select_outages(case, args.generator_contingencies)
    fixed = read_fixed_dispatch(case, args)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # 1. Build Base Continuous Model + s^+, s^- + Optional Direct Eta Limits
    raw_lp = build_model(case, outages, args, exact=False, fixed=fixed)
    slack_lp = add_apr_slacks(raw_lp, case, outages)
    n_pad = len(slack_lp.c0) - len(raw_lp.c0)
    slack_lp = replace(
        slack_lp,
        original_cost=np.r_[raw_lp.c0, np.zeros(n_pad)],
        original_constant=raw_lp.constant
    )
    base_lp = apply_eta_limits(slack_lp, case, args)

    # 2. Innovation A: Analytical s-Bounds & Monotonicity Presolve
    tight_lp, U_plus, U_minus, bound_stats = innovation_a_s_bounds(base_lp, case, outages, fix_regimes=True)
    print(f"=== Innovation A (s-Bounds Presolve) ===")
    print(f"Total (k,i) pairs: {bound_stats['total_pairs']} | "
          f"Fixed Proportional (s^-=0): {bound_stats['fixed_proportional']} | "
          f"Fixed Saturated (s^+=0): {bound_stats['fixed_saturated']}")

    gmax_denoms = np.array([case['hi'][i] for _, i in tight_lp.slack_pairs])
    base_scale_p = (1.0 / gmax_denoms) if args.bound_norm == 'gmax' else (1.0 / U_plus)
    base_scale_m = (1.0 / gmax_denoms) if args.bound_norm == 'gmax' else (1.0 / U_minus)

    rows = []

    # 3. Optional Exact MILP Benchmark in Gurobi (under identical eta limits)
    if args.compare_exact:
        print("\nSolving exact_APR MILP benchmark in Gurobi...")
        exact_mdl = apply_eta_limits(
            add_apr_slacks(build_model(case, outages, args, exact=True, fixed=fixed), case, outages),
            case, args
        )
        r_ex, v_ex = solve_model(exact_mdl, 0.0, args)
        if v_ex is not None:
            m_ex, _, _, _ = analyze(case, outages, exact_mdl, v_ex, args, 'exact_APR', 0.0)
            eta_ex, _ = eta_limit_diagnostics(case, outages, v_ex[exact_mdl.g], v_ex[exact_mdl.y], args)
            r_ex.update(m_ex, **eta_ex, model='exact_APR', weight=None, dc_iter=0, both_pos_pairs=0)
            rows.append(r_ex)
            print(f"  [exact_APR MILP] Operating Cost = {r_ex['operating_objective']:.2f} | "
                  f"APR Error = {r_ex['max_APR_L1_error_MW']:.4e} MW | Time = {r_ex['total_seconds']:.3f}s")

    # 4. Persistent Gurobi Sweep over w (High-to-Low) with Innovations B & C
    weights = sorted(set(args.weights), reverse=True)
    print(f"\n=== Running Gurobi High-to-Low w Sweep {weights} + Innovations B & C ===")
    runner = PersistentGurobiRunner(tight_lp, args)

    try:
        for w in weights:
            theta_p, theta_m = base_scale_p.copy(), base_scale_m.copy()
            last_valid_vec = None

            for dc_it in range(args.dc_iters):
                tag = f"LP_w{w:g}_DC{dc_it}"
                q_w = build_weighted_objective(
                    tight_lp, w, theta_p, theta_m,
                    include_gen_cost=args.include_gen_cost, penalty=args.penalty
                )
                res, vec, _ = runner.solve(q_w)
                if vec is None:
                    print(f"  [{tag}] Gurobi status: {res['status']}")
                    break

                metrics, _, _, _ = analyze(case, outages, tight_lp, vec, args, tag, w)
                eta_diag, _ = eta_limit_diagnostics(case, outages, vec[tight_lp.g], vec[tight_lp.y], args)
                res.update(metrics, **eta_diag, model=tag, weight=w, dc_iter=dc_it)
                res['both_pos_pairs'] = int(np.count_nonzero(
                    np.minimum(vec[tight_lp.splus], vec[tight_lp.sminus]) * case['base'] > args.apr_tol_mw
                ))
                rows.append(res)
                last_valid_vec = vec

                print(f"  [{tag}] Cost = {metrics['operating_objective']:.2f} | "
                      f"APR Err = {metrics['max_APR_L1_error_MW']:.4e} MW | "
                      f"Violating Pairs = {res['both_pos_pairs']} | Time = {res['total_seconds']:.3f}s")

                # Innovation B: DC subgradient weight update
                theta_p, theta_m = dc_update_weights(vec, tight_lp, base_scale_p, base_scale_m)

            # Innovation C: Fixed-Regime Polishing + Gurobi Reduced-Cost Probing
            if last_valid_vec is not None:
                p_tag = f"Polished_LP_w{w:g}"
                r_pol, v_pol = innovation_c_gurobi_polish(runner, tight_lp, case, outages, last_valid_vec)
                if v_pol is not None:
                    m_p, _, _, _ = analyze(case, outages, tight_lp, v_pol, args, p_tag, w)
                    eta_p, _ = eta_limit_diagnostics(case, outages, v_pol[tight_lp.g], v_pol[tight_lp.y], args)
                    r_pol.update(m_p, **eta_p, model=p_tag, weight=w, dc_iter=args.dc_iters)
                    r_pol['both_pos_pairs'] = int(np.count_nonzero(
                        np.minimum(v_pol[tight_lp.splus], v_pol[tight_lp.sminus]) * case['base'] > args.apr_tol_mw
                    ))
                    rows.append(r_pol)
                    print(f"  --> [{p_tag} (Innovation C)] Operating Cost = {m_p['operating_objective']:.2f} | "
                          f"APR Err = {m_p['max_APR_L1_error_MW']:.4e} MW | "
                          f"Violating Pairs = {r_pol['both_pos_pairs']} | Time = {r_pol['total_seconds']:.3f}s")

                if args.stop_on_apr_loss and rows[-2]['both_pos_pairs'] > 0:
                    print(f"  [Advisor Stopping Rule] Complementarity lost at w={w:g}; stopping sweep.")
                    break
    finally:
        runner.close()

    summary = pd.DataFrame(rows)
    summary.to_csv(output / 'summary_baron_gurobi.csv', index=False)
    cols = [c for c in ['model', 'weight', 'dc_iter', 'status', 'operating_objective',
                        'generation_cost', 'line_thermal_penalty', 'unit_thermal_penalty',
                        'max_APR_L1_error_MW', 'both_pos_pairs', 'eta_limits_satisfied', 'total_seconds']
            if c in summary.columns]
    print("\n=== Final Gurobi Summary ===")
    print(summary[cols].to_string(index=False))
    print(f"\nTotal wall time: {perf_counter() - t_start:.3f}s | Saved to: {output.resolve()}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())