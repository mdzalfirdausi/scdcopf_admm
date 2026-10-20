#!/usr/bin/env python3
"""APR experiments with individual and/or aggregate thermal-slack limits.

New options in this standalone version:
  --no-generation-budget: diagnostic with NO C_g cap. Requires direct eta
    limits and --thermal-cost-budgets off; generation output bounds remain.
  --thermal-cost-budgets off: replace C_l/C_u caps with direct eta limits;
    keep the generation-cost cap. Requires --budget-mode split.
  --eta-line-fraction DELTA: eta <= DELTA*rating on EACH monitored line
    in nominal and line-outage states. 0.05 means 5% extra line flow.
  --eta-unit-fraction DELTA: same, for lines after generator outages.
  --eta-line-mw / --eta-unit-mw: alternatively, individual absolute MW caps.
  --eta-line-total-mw / --eta-unit-total-mw: optional aggregate overload
    sums across all modeled states of that family (not simultaneous MW).

Individual and aggregate limits can be combined. These new limits apply to
budget candidates, including budget_exact_APR, NOT reference or unrestricted
envelope_LP/exact_APR controls. Every candidate is checked against actual line
flows before and after APR recovery; see eta_limit_checks.csv. Original cost
is still evaluated with the specified thermal penalty for comparison.

The following describes the inherited cost-budget experiments:

This is a NEW centralized experiment, not ADMM and not the line-only benchmark.
LP: min c'g + M_eta sum(eta_pu) + W sum(z_k), with one common z_k per
    generator outage, y_kk=0, sum(y_k)=demand, g_i<=y_ki<=Pmax_i,
    y_ki<=g_i+a_i*z_k, a_i=gamma*(Pmax_i-Pmin_i), 0<=z_k<=1.
Includes nominal network, generator outages, and optional single-line outages.
Slacks default to SOFT, nonnegative, with no upper bounds.
--compare-exact adds the original APR MILP (W=0).
--compare-weighted-exact also solves exact-APR MILPs with each requested W.

NEW: --budget-increase 0.10 enables the professor's cost-budget experiment.
First solve a reference OPF without generator outages (--reference-opf line,
the default, or nominal). Then enforce ORIGINAL generation cost + ALL thermal
penalties <= 1.10 * reference cost, and minimize sum(z) by default.
This is the professor's revised objective: thermal penalties remain ONLY in
the cost budget. --budget-objective thermal-signal restores the earlier
minimization of thermal penalties + W*sum(z). In signal-only mode W is fixed
to 1; multiplying the sole objective by any positive W cannot select a
different optimal set, so no W sweep or W=0 budget candidate is performed.

NEW: --budget-mode split separates the cost budget into C_g, C_l, and C_u.
C_l = nominal + line-outage thermal penalties; C_u = line thermal penalties
under GENERATOR outages. Generator output bounds remain hard. Use
--reference-opf envelope: one full unpenalized continuous APR relaxation
defines all three components at the SAME reference optimum without binaries.
This differs from the earlier line-only OPF and is labeled in the metadata.
--epsilon-g, --epsilon-line, --epsilon-unit override --budget-increase for each
component. A zero reference penalty cannot be relaxed by a percentage.
Optional --line-allowance/--unit-allowance add explicit absolute allowances in
OBJECTIVE-COST units (not MW); both default to zero.
--epsilon-g-grid, --epsilon-line-grid, --epsilon-unit-grid accept lists and
test their Cartesian product while reusing input matrices and one reference.
Do not supply both a scalar epsilon and its grid. Equivalent effective caps
(for example with a zero reference penalty) share one solve, with aliases in
metadata.json. epsilon_results.csv lists all distinct LP experiments and
identifies candidates verified against APR AND every separate budget.

NEW: --budget-objective apr-slack implements the handwritten proposal:
 splus-sminus = Pmax-(g+a*z), y+splus=Pmax, y+sminus=g+a*z,
 splus,sminus>=0, minimize sum(splus+sminus), with NO direct signal penalty.
All powers/slacks and the minimized slack sum are per unit; an MW-converted
slack sum is also reported. All three equalities are included, although the
first is implied by the other two. Cost budgets and APR checks are retained.
This slack objective is not the original APR mismatch. In fact,
 min(splus,sminus)=min(Pmax,g+a*z)-y for feasible envelope outputs,
 splus+sminus=abs(Pmax-g-a*z)+2*min(splus,sminus).
Thus even a physical APR solution can have a positive slack objective.

TWO-STAGE: --budget-objective apr-slack --lexicographic-cost first minimizes
APR slacks, then minimizes original operating cost subject to the optimal
slack sum plus --slack-preserve-tol (default 1e-7 per unit). Individual slacks
and dispatches remain free. Stage two can lose APR consistency; both stages
are independently validated. Results: budget_LP and budget_LP_cost;
lexicographic_comparison.csv compares cost, slack sum, APR error and timing.

DIAGNOSTIC: --budget-objective operating minimizes the ORIGINAL generation
plus thermal cost under the selected budgets. Add --compare-weighted-exact
to solve an exact-APR MILP with the same budgets and operating objective.
Compare exact_APR (unrestricted original optimum) with budget_exact_APR
(best original cost under the budgets). This quantifies the cost restriction
from the budgets, rather than confusing it with an APR-slack objective value.
The preventive dispatch is optimized again, not fixed to the reference OPF.
--reference-cost VALUE reuses a previously computed cost and skips that solve;
its external solve time is then excluded and must be reported separately.
Recovery is checked for APR feasibility AND cost-budget feasibility separately.
The previous sufficient-W argument does not apply with a cost-budget constraint.
pipeline_seconds charges input preprocessing, reference stage, model building,
solve, and validation/recovery to each candidate; it excludes other sweep points
and optional MILP benchmarks. This is not a guaranteed exact MILP replacement.

Linear c1 costs, c0 constant; c2 is deliberately ignored, as in original P1.
Power variables are per unit. Penalty is per unit slack, NOT per MW.
W is in original objective units per normalized signal; it is NOT per MW.
The constraint y_ki>=g_i strengthens the envelope to upward-only response.
A shared signal does not by itself prove APR consistency. Check apr_checks.csv.
The W>max_k M_eta*sum_i(a_i)*sum_j(spread(H_j,survivors)) condition is
reported as the candidate sufficient threshold for SOFT, unbounded slacks.
Even if APR is tight, the signal penalty may change the original optimum.

Install: python -m pip install numpy pandas scipy openpyxl
Optional commercial backend: gurobipy, with a usable license.

Small modified test (original case5 demand 1000 MW is generator-N-1 infeasible):
 python check_apr_signal_penalty.py --case pglib_opf_case5_pjm.xlsx \
   --gamma 1 --demand-scale 0.9 --no-line-contingencies --solver highs \
   --weights 0 1 10 100 1000 --include-threshold --compare-exact

Cost-budget example, including non-islanding line contingencies:
 python check_apr_signal_penalty.py --case pglib_opf_case5_pjm.xlsx \
   --gamma 1 --demand-scale 0.9 --solver highs --thermal-mode soft \
   --budget-increase 0.10 --reference-opf line --budget-objective signal \
   --compare-exact --output-dir output_apr_budget_case5

Input Excel schema: baseMVA, bus, gen, gencost, branch, matching the P1 script.
A case stem without .xlsx is accepted. Default folder: ../excel_outputs.
--fixed-dispatch CSV accepts gen_ID,generation_MW and, if present, --fixed-method
and --fixed-run filters matching the previous benchmark dispatch.csv schema.

Outputs: summary.csv, apr_checks.csv, dispatch.csv, apr_dispatch.csv,
capacity_screen.csv, excluded_outages.csv, and metadata.json.
Candidate outputs are labeled separately from recovered original-APR outputs.
A numerical sweep illustrates a result; it does not prove a theorem or novelty.
"""
import argparse
from dataclasses import dataclass
from dataclasses import replace
import copy
from itertools import product
import importlib.util
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

def max0(a):
    return float(np.max(a, initial=0.0))


def resolve_case(name, case_dir):
    if not Path(name).suffix:
        name += ".xlsx"
    for p in (Path(name), Path(case_dir) / name, Path(__file__).parent / name):
        if p.is_file():
            return p.resolve()
    raise ValueError(f"Case not found: {name}. Use --case with the full Excel path.")


def load_case(path, demand_scale):
    sheets = pd.read_excel(path, sheet_name=None)
    for name in ('baseMVA', 'bus', 'gen', 'gencost', 'branch'):
        if name not in sheets or sheets[name].empty:
            raise ValueError(f"Missing/empty Excel sheet: {name}")
    base = float(sheets['baseMVA'].iloc[0, 0])
    bus, gen, cost, branch = [sheets[k].copy() for k in ('bus', 'gen', 'gencost', 'branch')]
    if not np.isfinite(base) or base <= 0:
        raise ValueError('baseMVA must be positive and finite')
    if bus.bus_i.duplicated().any() or gen.gen_ID.duplicated().any() or cost.gen_ID.duplicated().any():
        raise ValueError('Bus IDs and generator IDs must be unique')
    if 'c2' not in cost:
        cost['c2'] = 0.0
    gen = gen.merge(cost[['gen_ID', 'c2', 'c1', 'c0']], on='gen_ID', how='left', validate='one_to_one')
    gen = gen[(gen.status > 0) & (gen.Pmax > 0) & (gen.Pmin >= 0)].reset_index(drop=True)
    branch = branch[branch.status > 0].reset_index(drop=True)
    if gen.empty or branch.empty:
        raise ValueError('No active generators or lines')
    if gen[['c2', 'c1', 'c0', 'Pmin', 'Pmax']].isna().any().any():
        raise ValueError('Missing generator costs/bounds')
    bus_index = {v: i for i, v in enumerate(bus.bus_i)}
    refs = np.flatnonzero(bus.type.to_numpy() == 3)
    if len(refs) != 1:
        raise ValueError('Exactly one reference bus is required')
    n, m, ng = len(bus), len(branch), len(gen)
    incidence = np.zeros((m, n))
    susceptance = np.empty(m)
    for j, r in branch.iterrows():
        if r.bus_i not in bus_index or r.bus_j not in bus_index or r.bus_i == r.bus_j:
            raise ValueError('Invalid line endpoints')
        if abs(float(r.angle)) > 1e-10:
            raise ValueError('Nonzero transformer phase shifts are unsupported (affine flow offsets needed)')
        tap = 1.0 if float(r.ratio) == 0 else float(r.ratio)
        x = float(r.x)
        if not np.isfinite(x) or x == 0 or not np.isfinite(tap) or tap <= 0:
            raise ValueError('Invalid branch reactance/tap')
        incidence[j, bus_index[r.bus_i]] = 1.0
        incidence[j, bus_index[r.bus_j]] = -1.0
        susceptance[j] = 1.0 / (x * tap)
    # Explicit connectivity, not just a numerical inverse test.
    from scipy.sparse.csgraph import connected_components
    adjacency = sparse.csr_matrix(np.abs(incidence).T @ np.abs(incidence))
    if connected_components(adjacency, directed=False, return_labels=False) != 1:
        raise ValueError('Base network is disconnected')
    ref = int(refs[0])
    keep = np.delete(np.arange(n), ref)
    bbus = incidence.T @ (susceptance[:, None] * incidence)
    inverse = np.zeros((n, n))
    inverse[np.ix_(keep, keep)] = np.linalg.solve(bbus[np.ix_(keep, keep)], np.eye(n - 1))
    ptdf = susceptance[:, None] * (incidence @ inverse)
    transfer = ptdf @ incidence.T
    den = 1.0 - np.diag(transfer)
    eligible = np.flatnonzero(np.abs(den) >= 1e-8)
    excluded = np.flatnonzero(np.abs(den) < 1e-8)
    lodf = np.zeros((m, m))
    lodf[:, eligible] = transfer[:, eligible] / den[eligible]
    lodf[np.arange(m), np.arange(m)] = -1.0
    B = np.zeros((n, ng))
    for j, r in gen.iterrows():
        if r.bus_i not in bus_index:
            raise ValueError('Generator references unknown bus')
        B[bus_index[r.bus_i], j] = 1.0
    pd_pu = bus.Pd.to_numpy(float) * demand_scale / base
    lo, hi = gen.Pmin.to_numpy(float) / base, gen.Pmax.to_numpy(float) / base
    if np.any(lo > hi):
        raise ValueError('Pmin > Pmax')
    for values in (pd_pu, lo, hi, gen[['c2', 'c1', 'c0']].to_numpy(float), branch.rateA.to_numpy(float)):
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite input data')
    if sum(lo) > sum(pd_pu) + 1e-9 or sum(hi) < sum(pd_pu) - 1e-9:
        raise ValueError('Demand cannot be balanced within generator bounds even without line limits')
    return dict(base=base, bus=bus, gen=gen, branch=branch, A=incidence, b=susceptance,
                B=B, K=ptdf, L=lodf, eligible=eligible, excluded=excluded, ref=ref,
                keep=keep, d=pd_pu, D=float(sum(pd_pu)), lo=lo, hi=hi,
                rates=branch.rateA.to_numpy(float) / base)


def id_key(value):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    try:
        number = float(text)
        return str(int(number)) if number.is_integer() else text
    except ValueError:
        return text


def select_outages(case, selection):
    if selection.lower() == 'all':
        return list(range(len(case['gen'])))
    lookup = {id_key(x): i for i, x in enumerate(case['gen'].gen_ID)}
    names = [id_key(x) for x in selection.split(',') if x.strip()]
    if not names or any(x not in lookup for x in names):
        raise ValueError('Unknown generator-contingency IDs; use active gen_ID values separated by commas, or all')
    return list(dict.fromkeys(lookup[x] for x in names))


def read_fixed_dispatch(case, args):
    if not args.fixed_dispatch:
        return None
    table = pd.read_csv(args.fixed_dispatch)
    if 'method' in table:
        table = table[table['method'] == args.fixed_method]
    if 'run' in table:
        table = table[table['run'] == args.fixed_run]
    if not {'gen_ID', 'generation_MW'}.issubset(table.columns):
        raise ValueError('Fixed-dispatch CSV requires gen_ID and generation_MW columns')
    table = table.assign(key=table.gen_ID.map(id_key))
    if table.key.duplicated().any():
        raise ValueError('Fixed-dispatch CSV has multiple rows per generator after method/run filtering')
    lookup = dict(zip(table.key, table.generation_MW))
    keys = [id_key(x) for x in case['gen'].gen_ID]
    if set(lookup) != set(keys):
        raise ValueError('Fixed-dispatch CSV must contain exactly the active generator IDs in this case')
    g = np.array([lookup[x] for x in keys], dtype=float) / case['base']
    err = max(max0(case['lo'] - g), max0(g - case['hi']), abs(g.sum() - case['D'])) * case['base']
    if not np.isfinite(g).all() or err > args.feas_tol_mw:
        raise ValueError('Fixed dispatch violates generation bounds or balance for this demand instance')
    return g


@dataclass
class LinearModel:
    A: object
    lower: np.ndarray
    upper: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    integer: np.ndarray
    c0: np.ndarray
    g: np.ndarray
    y: np.ndarray
    z: np.ndarray
    x: dict
    constant: float
    row_is_power: object = None
    eta_line: object = None
    eta_unit: object = None
    splus: object = None
    sminus: object = None
    slack_pairs: object = None
    original_cost: object = None
    original_constant: object = None
    eta_records: object = None


def network_blocks(case, include_lines):
    """Positive-injection PTDF orientation; symmetric rateA constraints."""
    ng = len(case['lo'])
    mon = np.flatnonzero(case['rates'] > 0)
    H = case['K'] @ case['B']
    offset = -case['K'] @ case['d']
    blocks = [(-1, mon, H[mon], offset[mon])]
    if include_lines:
        for ell in case['eligible']:
            keep = mon[mon != ell]
            T = case['K'][keep] + np.outer(case['L'][keep, ell], case['K'][ell])
            blocks.append((int(ell), keep, T @ case['B'], -(T @ case['d'])))
    assert all(F.shape[1] == ng for _, _, F, _ in blocks)
    return H, offset, mon, blocks


def build_model(case, outages, args, exact=False, fixed=None):
    ng, nk = len(case['lo']), len(outages)
    soft = args.thermal_mode == 'soft'
    H, offset, mon, blocks = network_blocks(case, not args.no_line_contingencies)
    lb, ub, integrality, cost = [], [], [], []

    def variables(low, high, obj=0.0, integer=0):
        low, high = np.atleast_1d(low).astype(float), np.atleast_1d(high).astype(float)
        low, high = np.broadcast_arrays(low, high)
        index = np.arange(len(lb), len(lb) + low.size)
        lb.extend(low.ravel()); ub.extend(high.ravel())
        cost.extend(np.broadcast_to(obj, low.shape).ravel())
        integrality.extend([integer] * low.size)
        return index

    # LaTeX (bounds): \underline{g}_i \le g_i \le \overline{g}_i.
    # If --fixed-dispatch is supplied: g_i = g_i^{fixed} instead.
    # LaTeX (original generation cost): C_g = c^T g + C_0.
    # g is per unit, so c_i = c_{1,i} * baseMVA; C_0 is added separately.
    g = variables(case['lo'] if fixed is None else fixed, case['hi'] if fixed is None else fixed,
                  case['gen'].c1.to_numpy(float) * case['base'])
    y = np.empty((nk, ng), dtype=int)
    for s, k in enumerate(outages):
        # LaTeX: \underline{g}_i \le g_{k,i} \le \overline{g}_i, i\ne k.
        low, high = case['lo'].copy(), case['hi'].copy()
        # LaTeX: g_{k,k}=0 (the outaged generator).
        low[k] = high[k] = 0.0
        y[s] = variables(low, high)
    # LaTeX: 0 \le z_k \le 1, k\in\mathcal{K}_g.
    z = variables(np.zeros(nk), np.ones(nk))
    x = {}
    if exact:
        for s, k in enumerate(outages):
            for i in range(ng):
                if i != k:
                    # LaTeX: x_{k,i}\in\{0,1\}; exact MILP benchmark ONLY.
                    x[s, i] = int(variables(0.0, 1.0, integer=1)[0])

    row, col, data, lower, upper = [], [], [], [], []
    eta_line, eta_unit, eta_records = [], [], []

    def constraint(indices, values, low=-np.inf, high=np.inf):
        indices, values = np.asarray(indices, dtype=int), np.asarray(values, dtype=float)
        nz = values != 0
        row.extend([len(lower)] * int(nz.sum()))
        col.extend(indices[nz]); data.extend(values[nz])
        lower.append(low); upper.append(high)

    def flow_constraints(indices, monitored, F, b, family='line', state=-1):
        # LaTeX: \eta_{s,j}\ge0, with no finite upper bound (soft mode).
        # LaTeX (thermal objective term): M_\eta\sum_{s,j}\eta_{s,j}.
        # Hard mode omits eta, equivalently setting all thermal slacks to zero.
        eta = variables(np.zeros(len(monitored)), np.full(len(monitored), np.inf), args.penalty) if soft else None
        if eta is not None:
            (eta_unit if family == 'unit' else eta_line).extend(eta.tolist())
            eta_records.extend((int(index), family, int(state), int(j))
                               for index, j in zip(eta, monitored))
        for t, j in enumerate(monitored):
            idx = list(indices) + ([int(eta[t])] if soft else [])
            # LaTeX: f_{s,j}=F_{s,j}u_s+b_{s,j}\le\bar f_j+\eta_{s,j}.
            constraint(idx, list(F[t]) + ([-1.0] if soft else []), high=case['rates'][j] - b[t])
            # LaTeX: -f_{s,j}\le\bar f_j+\eta_{s,j}.
            # Together: -\bar f_j-\eta_{s,j}\le f_{s,j}\le\bar f_j+\eta_{s,j}.
            constraint(idx, list(-F[t]) + ([-1.0] if soft else []), high=case['rates'][j] + b[t])

    # LaTeX: \mathbf{1}^T g = D (nominal power balance).
    constraint(g, np.ones(ng), case['D'], case['D'])
    for ell, monitored, F, b in blocks:
        # LaTeX (base): f^0=K(Bg-d).
        # LaTeX (line outage ell): f_j^{L,ell}=f_j^0+L_{j,ell}f_ell^0,
        # j\ne ell; only non-islanding outages and rateA>0 lines are monitored.
        flow_constraints(g, monitored, F, b, state=ell)
    for s, k in enumerate(outages):
        # LaTeX: \mathbf{1}^T g_k = D (generator-outage power balance).
        constraint(y[s], np.ones(ng), case['D'], case['D'])
        # LaTeX: f^{G,k}=K(Bg_k-d), with soft/hard thermal bounds above.
        flow_constraints(y[s], mon, H[mon], offset[mon], family='unit', state=int(k))
        for i in range(ng):
            if i == k:
                continue
            # Upward-only envelope; also valid for the exact APR model.
            # LaTeX: g_{k,i}-g_i\ge0, i\ne k.
            constraint([y[s, i], g[i]], [1.0, -1.0], low=0.0)
            if not exact:
                # LaTeX: g_{k,i}-g_i-a_i z_k\le0,
                # a_i=\gamma_i(\bar g_i-\underline g_i).
                constraint([y[s, i], g[i], z[s]], [1.0, -1.0, -case['a'][i]], high=0.0)
            else:
                # x=1: proportional; x=0: saturation. Valid for gamma in [0,1].
                M = case['hi'][i]
                # LaTeX: g_{k,i}-g_i-a_i z_k\le\bar g_i(1-x_{k,i}).
                constraint([y[s, i], g[i], z[s], x[s, i]], [1.0, -1.0, -case['a'][i], M], high=M)
                # LaTeX: g_{k,i}-g_i-a_i z_k\ge-\bar g_i(1-x_{k,i}).
                constraint([y[s, i], g[i], z[s], x[s, i]], [1.0, -1.0, -case['a'][i], -M], low=-M)
                # LaTeX: g_i+a_i z_k\ge\bar g_i(1-x_{k,i}).
                constraint([g[i], z[s], x[s, i]], [1.0, case['a'][i], M], low=M)
                # LaTeX: g_{k,i}\ge\bar g_i(1-x_{k,i}).
                constraint([y[s, i], x[s, i]], [1.0, M], low=M)
    A = sparse.coo_matrix((data, (row, col)), shape=(len(lower), len(lb))).tocsc()
    return LinearModel(A, np.array(lower), np.array(upper), np.array(lb), np.array(ub),
                       np.array(integrality), np.array(cost), g, y, z, x, float(case['gen'].c0.sum()),
                       eta_line=np.array(eta_line, dtype=int), eta_unit=np.array(eta_unit, dtype=int),
                       eta_records=eta_records)


def solve_model(model, weight, args):
    t = perf_counter()
    # LaTeX: min q^T v + W\sum_k z_k, where q is the incoming model objective.
    # Signal-only budget model: q^T v=0, W=1, so min \sum_k z_k.
    # Earlier thermal-signal budget model: min C_eta + W\sum_k z_k.
    # Unbudgeted model: q^T v + C_0 = C_g + C_eta.
    q = model.c0.copy(); q[model.z] += weight
    scale = float(max(1.0, max0(np.abs(q))))
    # LaTeX: argmin_v q^T v = argmin_v (q/scale)^T v for scale>0.
    qscaled = q / scale
    result = dict(status='solver_failed', solver=args.backend, weight=weight, objective_scale=scale)
    vector, bound = None, None
    if args.backend == 'highs':
        if model.integer.any():
            r = milp(qscaled, integrality=model.integer, bounds=Bounds(model.lb, model.ub),
                     constraints=LinearConstraint(model.A, model.lower, model.upper),
                     options={'time_limit': args.time_limit, 'mip_rel_gap': args.mip_gap, 'disp': args.tee})
            status = {0: 'optimal', 1: 'limit', 2: 'infeasible', 3: 'unbounded'}.get(r.status, 'solver_failed')
            b = getattr(r, 'mip_dual_bound', None)
            if b is not None and np.isfinite(b):
                bound = float(b) * scale + model.constant
        else:
            eq = np.isfinite(model.lower) & (model.lower == model.upper)
            up = np.isfinite(model.upper) & ~eq
            low = np.isfinite(model.lower) & ~eq
            r = linprog(qscaled, A_ub=sparse.vstack([model.A[up], -model.A[low]], format='csc'),
                        b_ub=np.r_[model.upper[up], -model.lower[low]], A_eq=model.A[eq], b_eq=model.upper[eq],
                        bounds=list(zip(model.lb, model.ub)), method='highs',
                        options={'time_limit': args.time_limit, 'primal_feasibility_tolerance': 1e-9,
                                 'dual_feasibility_tolerance': 1e-9, 'disp': args.tee})
            status = {0: 'optimal', 1: 'limit', 2: 'infeasible', 3: 'unbounded'}.get(r.status, 'solver_failed')
            if r.status == 0:
                bound = float(r.fun) * scale + model.constant
        result.update(status=status, message=r.message)
        if getattr(r, 'x', None) is not None and np.isfinite(r.x).all():
            vector = np.asarray(r.x)
    else:
        import gurobipy as gp
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', int(args.tee)); env.start()
            with gp.Model(env=env) as gm:
                gm.Params.Threads = args.threads
                gm.Params.FeasibilityTol = 1e-9
                gm.Params.OptimalityTol = 1e-9
                gm.Params.MIPGap = args.mip_gap
                gm.Params.TimeLimit = args.time_limit
                v = gm.addMVar(len(q), lb=np.where(np.isfinite(model.lb), model.lb, -gp.GRB.INFINITY),
                              ub=np.where(np.isfinite(model.ub), model.ub, gp.GRB.INFINITY),
                              vtype=np.where(model.integer > 0, gp.GRB.BINARY, gp.GRB.CONTINUOUS))
                eq = np.isfinite(model.lower) & (model.lower == model.upper)
                up = np.isfinite(model.upper) & ~eq; low = np.isfinite(model.lower) & ~eq
                if eq.any(): gm.addMConstr(model.A[eq], v, '=', model.upper[eq])
                if up.any(): gm.addMConstr(model.A[up], v, '<', model.upper[up])
                if low.any(): gm.addMConstr(model.A[low], v, '>', model.lower[low])
                # LaTeX: \min_v (q/scale)^T v (Gurobi defaults to minimization).
                gm.setObjective(qscaled @ v); gm.optimize()
                if gm.Status == gp.GRB.INF_OR_UNBD:
                    gm.Params.DualReductions = 0; gm.reset(); gm.optimize()
                status = {gp.GRB.OPTIMAL: 'optimal', gp.GRB.INFEASIBLE: 'infeasible',
                          gp.GRB.UNBOUNDED: 'unbounded', gp.GRB.INF_OR_UNBD: 'infeasible_or_unbounded',
                          gp.GRB.TIME_LIMIT: 'limit'}.get(gm.Status, f'gurobi_status_{gm.Status}')
                result.update(status=status, message=f'Gurobi status {gm.Status}')
                if gm.SolCount:
                    vector = np.asarray(v.X)
                if status == 'optimal':
                    bound = float(gm.ObjVal) * scale + model.constant
                elif model.integer.any():
                    try:
                        if np.isfinite(gm.ObjBound): bound = float(gm.ObjBound) * scale + model.constant
                    except gp.GurobiError:
                        pass
    result['total_seconds'] = perf_counter() - t
    result['solver_best_bound_penalized'] = bound
    if vector is not None:
        result['solver_penalized_objective'] = float(q @ vector + model.constant)
        dimensionless = np.zeros(len(vector), dtype=bool)
        dimensionless[model.z] = True
        if model.x: dimensionless[list(model.x.values())] = True
        power = ~dimensionless
        row_power = np.ones(model.A.shape[0], dtype=bool) if model.row_is_power is None else model.row_is_power
        row_error = np.maximum(model.lower - model.A @ vector, model.A @ vector - model.upper)
        violation = max(max0(row_error[row_power]),
                        max0(model.lb[power] - vector[power]), max0(vector[power] - model.ub[power]))
        other_error = max0(row_error[~row_power])
        domain_error = max(max0(model.lb[dimensionless] - vector[dimensionless]),
                           max0(vector[dimensionless] - model.ub[dimensionless]))
        int_error = max0(np.abs(vector[model.integer > 0] - np.rint(vector[model.integer > 0])))
        result['power_constraint_violation_MW'] = violation * args.base
        result['signal_and_binary_domain_error'] = domain_error
        result['integrality_error'] = int_error
        result['normalized_nonpower_constraint_error'] = other_error
        result['solver_point_verified'] = bool(violation * args.base <= args.feas_tol_mw and domain_error <= 1e-7 and int_error <= 1e-6 and other_error <= 1e-8)
    return result, vector


def physical(case, g, z, k):
    p = np.minimum(g + case['a'] * z, case['hi'])
    p[k] = 0.0
    return p


def recover(case, g, k, tol_pu):
    p0, p1 = physical(case, g, 0, k), physical(case, g, 1, k)
    if p0.sum() > case['D'] + tol_pu or p1.sum() < case['D'] - tol_pu:
        return None, None
    if abs(p0.sum() - case['D']) <= tol_pu: return p0, 0.0
    if abs(p1.sum() - case['D']) <= tol_pu: return p1, 1.0
    low, high = 0.0, 1.0
    for _ in range(120):
        z = (low + high) / 2; p = physical(case, g, z, k)
        if abs(p.sum() - case['D']) <= tol_pu: return p, z
        if p.sum() > case['D']: high = z
        else: low = z
    return None, None


def operating_metrics(case, outages, g, outputs, include_lines, penalty):
    H, offset, mon, blocks = network_blocks(case, include_lines)
    overloads = [np.maximum(np.abs(F @ g + b) - case['rates'][monitored], 0.0)
                 for _, monitored, F, b in blocks]
    line_eta = sum(float(e.sum()) for e in overloads)
    unit_overloads = [np.maximum(np.abs(H[mon] @ y + offset[mon]) - case['rates'][mon], 0.0) for y in outputs]
    unit_eta = sum(float(e.sum()) for e in unit_overloads)
    overloads += unit_overloads
    eta = sum(float(e.sum()) for e in overloads)
    generation = float(case['gen'].c1.to_numpy(float) @ (g * case['base']) + case['gen'].c0.sum())
    bound_error = max(max0(case['lo'] - g), max0(g - case['hi']))
    for k, y in zip(outages, outputs):
        survivors = np.arange(len(g)) != k
        bound_error = max(bound_error, max0(case['lo'][survivors] - y[survivors]),
                          max0(y[survivors] - case['hi'][survivors]), abs(y[k]))
    return dict(operating_objective=generation + penalty * eta, generation_cost=generation,
                line_thermal_penalty=penalty*line_eta, unit_thermal_penalty=penalty*unit_eta,
                line_sum_overload_MW=line_eta*case['base'], unit_sum_overload_MW=unit_eta*case['base'],
                sum_overload_MW=eta * case['base'], max_overload_MW=max(max0(e) for e in overloads) * case['base'],
                balance_error_MW=max([abs(g.sum() - case['D'])] + [abs(y.sum() - case['D']) for y in outputs]) * case['base'],
                generator_bound_error_MW=bound_error * case['base'])


def analyze(case, outages, model, vector, args, tag, weight):
    g, y, z = vector[model.g], vector[model.y], vector[model.z]
    checks, details, recovered = [], [], []
    for s, k in enumerate(outages):
        q = physical(case, g, z[s], k)
        p, zr = recover(case, g, k, min(args.feas_tol_mw / case['base'] / 100, 1e-10))
        delta = float(np.abs(y[s] - q).sum()) * case['base']
        checks.append(dict(model=tag, weight=weight, gen_ID=case['gen'].iloc[k].gen_ID,
                           signal=z[s], apr_L1_error_MW=delta, apr_max_error_MW=max0(np.abs(y[s] - q)) * case['base'],
                           physical_imbalance_at_reported_signal_MW=(q.sum() - case['D']) * case['base'],
                           apr_consistent=bool(delta <= args.apr_tol_mw), recovery_exists=p is not None,
                           recovered_signal=zr, recovery_change_MW=None if p is None else float(np.abs(p-y[s]).sum())*case['base']))
        if p is not None: recovered.append(p)
        for i in range(len(g)):
            details.append(dict(model=tag, weight=weight, outage_gen_ID=case['gen'].iloc[k].gen_ID,
                                gen_ID=case['gen'].iloc[i].gen_ID, provisional_MW=y[s,i]*case['base'],
                                physical_at_signal_MW=q[i]*case['base'], recovered_MW=None if p is None else p[i]*case['base']))
    metrics = operating_metrics(case, outages, g, y, not args.no_line_contingencies, args.penalty)
    metrics['signal_sum'] = float(z.sum()); metrics['signal_penalty'] = weight * float(z.sum())
    metrics['penalized_objective'] = metrics['operating_objective'] + metrics['signal_penalty']
    metrics['max_APR_L1_error_MW'] = max(c['apr_L1_error_MW'] for c in checks)
    metrics['all_APR_consistent'] = all(c['apr_consistent'] for c in checks)
    metrics['provisional_feasible_for_mode'] = bool(metrics['balance_error_MW'] <= args.feas_tol_mw
        and metrics['generator_bound_error_MW'] <= args.feas_tol_mw
        and (args.thermal_mode == 'soft' or metrics['max_overload_MW'] <= args.feas_tol_mw))
    metrics['provisional_original_APR_feasible'] = bool(metrics['provisional_feasible_for_mode'] and metrics['all_APR_consistent'])
    metrics['recovery_success'] = len(recovered) == len(outages)
    if metrics['recovery_success']:
        rec = operating_metrics(case, outages, g, recovered, not args.no_line_contingencies, args.penalty)
        metrics.update({f'recovered_{k}': v for k,v in rec.items()})
        metrics['recovered_original_APR_feasible'] = bool(rec['balance_error_MW'] <= args.feas_tol_mw
            and rec['generator_bound_error_MW'] <= args.feas_tol_mw
            and (args.thermal_mode == 'soft' or rec['max_overload_MW'] <= args.feas_tol_mw))
    dispatch = [dict(model=tag, weight=weight, gen_ID=case['gen'].iloc[i].gen_ID,
                     bus_i=case['gen'].iloc[i].bus_i, generation_MW=g[i]*case['base']) for i in range(len(g))]
    if model.splus is not None:
        sp, sm = vector[model.splus], vector[model.sminus]
        metrics['apr_slack_sum_pu'] = float(sp.sum()+sm.sum())
        metrics['apr_slack_sum_MW'] = metrics['apr_slack_sum_pu']*case['base']
        lookup = {pair:t for t,pair in enumerate(model.slack_pairs)}
        by_contingency = [[] for _ in outages]
        per_contingency_gap = np.zeros(len(outages))
        errors, equivalent = [], 0.0
        for t,(s,i) in enumerate(model.slack_pairs):
            by_contingency[s].append(t)
            target = g[i]+case['a'][i]*z[s]
            errors.extend([abs(sp[t]-sm[t]-(case['hi'][i]-target)),
                           abs(y[s,i]+sp[t]-case['hi'][i]), abs(y[s,i]+sm[t]-target)])
            per_contingency_gap[s] += min(sp[t],sm[t])
        for s,k in enumerate(outages):
            survivors = np.arange(len(g)) != k
            equivalent += float(case['hi'][survivors].sum()+g[survivors].sum()+case['a'][survivors].sum()*z[s]-2*case['D'])
            indices = by_contingency[s]
            checks[s]['splus_sum_MW'] = float(sp[indices].sum())*case['base']
            checks[s]['sminus_sum_MW'] = float(sm[indices].sum())*case['base']
            checks[s]['apr_gap_from_slacks_MW'] = float(per_contingency_gap[s])*case['base']
        metrics['apr_slack_equivalent_objective_pu'] = equivalent
        metrics['apr_slack_equation_error_MW'] = max(errors,default=0.0)*case['base']
        metrics['APR_gap_from_slacks_MW'] = max0(per_contingency_gap)*case['base']
        s_by_id = {case['gen'].iloc[k].gen_ID:s for s,k in enumerate(outages)}
        i_by_id = {case['gen'].iloc[i].gen_ID:i for i in range(len(g))}
        for record in details:
            s = s_by_id[record['outage_gen_ID']]
            i = i_by_id[record['gen_ID']]
            t = lookup.get((s,i))
            record['splus_MW'] = None if t is None else float(sp[t])*case['base']
            record['sminus_MW'] = None if t is None else float(sm[t])*case['base']
    return metrics, checks, details, dispatch


def penalty_threshold(case, outages):
    H = case['K'] @ case['B']; mon = case['rates'] > 0
    records = []
    for k in outages:
        survivors = np.arange(len(case['lo'])) != k
        spread = np.ptp(H[mon][:, survivors], axis=1)
        S = float(case['a'][survivors].sum())
        records.append(dict(outage_gen_ID=case['gen'].iloc[k].gen_ID,
                            response_slope_sum_pu=S, transfer_spread_sum=float(spread.sum()),
                            threshold_per_penalty_unit=S*float(spread.sum())))
    return records


def add_apr_slacks(model, case, outages):
    """Professor's handwritten LP extension, for every survivor i != k.

    s+ - s- = Pmax - (g+a*z), y+s+ = Pmax, y+s- = g+a*z.
    All s variables are nonnegative. The first equality is redundant but is
    retained to match the proposal. Projection onto (g,y,z) is unchanged.
    Original generation/thermal coefficients on new s variables are zero.
    """
    if model.original_cost is not None or model.splus is not None:
        raise ValueError('APR slacks must be added once, before cost budgets')
    pairs = [(s,i) for s,k in enumerate(outages) for i in range(len(case['lo'])) if i != k]
    n, count = len(model.lb), len(pairs)
    plus = np.arange(n, n+count)
    minus = np.arange(n+count, n+2*count)
    rr, cc, vv, rhs = [], [], [], []
    def equality(indices, values, value):
        row = len(rhs)
        for index, coef in zip(indices, values):
            if coef != 0: rr.append(row); cc.append(index); vv.append(coef)
        rhs.append(value)
    for t,(s,i) in enumerate(pairs):
        # LaTeX: s^+_{k,i}-s^-_{k,i}=\bar g_i-(g_i+a_i z_k).
        equality([plus[t],minus[t],model.g[i],model.z[s]], [1.,-1.,1.,case['a'][i]], case['hi'][i])
        # LaTeX: g_{k,i}+s^+_{k,i}=\bar g_i.
        equality([model.y[s,i],plus[t]], [1.,1.], case['hi'][i])
        # LaTeX: g_{k,i}+s^-_{k,i}=g_i+a_i z_k.
        equality([model.y[s,i],minus[t],model.g[i],model.z[s]], [1.,1.,-1.,-case['a'][i]], 0.)
    definitions = sparse.coo_matrix((vv,(rr,cc)), shape=(len(rhs),n+2*count)).tocsc()
    A = sparse.vstack([sparse.hstack([model.A,sparse.csc_matrix((model.A.shape[0],2*count))],format='csc'),definitions],format='csc')
    return replace(model,A=A,lower=np.r_[model.lower,rhs],upper=np.r_[model.upper,rhs],
                   lb=np.r_[model.lb,np.zeros(2*count)],ub=np.r_[model.ub,np.full(2*count,np.inf)],
                   integer=np.r_[model.integer,np.zeros(2*count,dtype=int)],
                   c0=np.r_[model.c0,np.zeros(2*count)],splus=plus,sminus=minus,slack_pairs=pairs,
                   row_is_power=np.ones(A.shape[0],dtype=bool))


def cost_component_rows(model, original=False):
    """C_g (with C0), C_l (base+line outages), C_u (generator outages).

    C_u prices LINE thermal slack under generator outages. It does not soften
    generator output bounds, which remain hard in every formulation.
    """
    coefficients = model.original_cost if original else model.c0
    constant = model.original_constant if original else model.constant
    if coefficients is None: raise ValueError('Original cost coefficients are unavailable')
    cg, cl, cu = [np.zeros_like(coefficients) for _ in range(3)]
    cg[model.g] = coefficients[model.g]
    cl[model.eta_line] = coefficients[model.eta_line]
    cu[model.eta_unit] = coefficients[model.eta_unit]
    if not np.allclose(cg+cl+cu, coefficients, atol=0, rtol=0):
        raise ValueError('Cost components do not reproduce the original objective')
    return {'generation': (cg, constant), 'line': (cl, 0.0), 'unit': (cu, 0.0)}


def budget_diagnostics(metrics, caps, args, prefix=''):
    """Independently recompute minimal physical slack costs before/after recovery."""
    if not isinstance(caps, dict):
        caps = {'total': caps}
    names = {'total': 'operating_objective', 'generation': 'generation_cost',
             'line': 'line_thermal_penalty', 'unit': 'unit_thermal_penalty'}
    result = {}
    for name, cap in caps.items():
        value = metrics[prefix+names[name]]
        error = max(0.0, value-cap)
        tolerance = max(args.budget_tol, 1e-8*max(1.0, abs(cap)))
        result[prefix+name+'_budget_cap'] = cap
        result[prefix+name+'_budget_violation'] = error
        result[prefix+name+'_budget_satisfied'] = bool(error <= tolerance)
    result[prefix+'budget_violation'] = max((result[prefix+n+'_budget_violation'] for n in caps), default=0.0)
    result[prefix+'budget_satisfied'] = all(result[prefix+n+'_budget_satisfied'] for n in caps)
    return result


def budget_model(model, cap, objective='signal'):
    """Keep ORIGINAL cost in the budget; select the new objective coefficients.

    signal: zero ALL coefficients, then solve_model adds sum(z), using W=1.
    thermal-signal: retain thermal coefficients, then solve_model adds W*sum(z).
    This does not fix or warm-start g.
    The normalized budget row is dimensionless, not a power residual in MW.
    """
    if isinstance(cap, dict):
        parts = cost_component_rows(model)
        specifications = [(parts[name][0], parts[name][1], value) for name, value in cap.items()]
        # LaTeX: C_g <= cap_g; C_l <= cap_l; C_u <= cap_u.
    else:
        specifications = [(model.c0, model.constant, cap)]
        # LaTeX (joint mode): C_g+C_l+C_u <= cap.
    budget_rows, budget_upper = [], []
    for coefficients, constant, limit in specifications:
        scale = max(1.0, abs(limit), abs(constant))
        budget_rows.append(sparse.csr_matrix(coefficients.reshape(1, -1)/scale))
        budget_upper.append((limit-constant)/scale)
    A = sparse.vstack([model.A]+budget_rows, format='csc')
    if objective == 'signal':
        # LaTeX (revised objective): \min \sum_k z_k.
        # Erase BOTH generation and thermal coefficients; keep them in A above.
        q = np.zeros_like(model.c0)
    elif objective == 'thermal-signal':
        # LaTeX (earlier objective before signal term): \min C_eta.
        q = model.c0.copy()
        q[model.g] = 0.0
    elif objective == 'apr-slack':
        if model.splus is None or model.sminus is None:
            raise ValueError('apr-slack objective requires add_apr_slacks() before budget_model()')
        # LaTeX: min sum_{k,i!=k}(s^+_{k,i}+s^-_{k,i}); no direct signal term.
        q = np.zeros_like(model.c0)
        q[model.splus] = 1.0
        q[model.sminus] = 1.0
    elif objective == 'operating':
        # LaTeX (diagnostic): min C_g+C_l+C_u under the selected cost caps.
        q = model.c0.copy()
    else:
        raise ValueError('Unknown budget objective')
    return replace(model, A=A, lower=np.r_[model.lower, np.full(len(specifications), -np.inf)],
                   upper=np.r_[model.upper, budget_upper], c0=q, constant=model.constant if objective=='operating' else 0.0,
                   original_cost=model.c0.copy(),original_constant=model.constant,
                   row_is_power=np.r_[np.ones(model.A.shape[0], dtype=bool), np.zeros(len(specifications), dtype=bool)])


def preserve_slack_optimum(model, optimum, tolerance):
    """Stage two: minimize original cost with sum(splus+sminus)<=Sstar+tau.

    Keep every budget, bound, and physical row unchanged. The new row is
    normalized and independently checked as a nonpower constraint.
    """
    if model.splus is None or model.original_cost is None:
        raise ValueError('Stage two requires an APR-slack budget model')
    coefficients = np.zeros_like(model.c0)
    coefficients[model.splus] = 1.0
    coefficients[model.sminus] = 1.0
    limit = optimum+tolerance
    scale = max(1.0, abs(limit))
    row = sparse.csr_matrix(coefficients.reshape(1, -1)/scale)
    return replace(model, A=sparse.vstack([model.A, row], format='csc'),
                   lower=np.r_[model.lower, -np.inf], upper=np.r_[model.upper, limit/scale],
                   row_is_power=np.r_[model.row_is_power, False],
                   c0=model.original_cost.copy(), constant=model.original_constant)


def individual_eta_cap_pu(case, family, monitored, args):
    """Individual overload cap: absolute MW OR fraction of each line rating.

    LaTeX: 0 <= eta_{k,j} <= h_{k,j}, with h = H_MW/baseMVA
    or h = delta_family * rateA_j/baseMVA. A fraction of 0.05 means 5%.
    The 'unit' family still constrains LINE overloads after generator outages.
    """
    absolute = getattr(args, 'eta_'+family+'_mw')
    fraction = getattr(args, 'eta_'+family+'_fraction')
    if absolute is not None:
        return np.full(len(monitored), absolute/case['base'])
    if fraction is not None:
        return fraction*case['rates'][monitored]
    return np.full(len(monitored), np.inf)


def apply_eta_limits(model, case, args):
    """Apply new limits ONLY to budget candidates; baseline controls stay unchanged."""
    ub = model.ub.copy()
    for index, family, state, j in model.eta_records:
        ub[index] = min(ub[index], individual_eta_cap_pu(case, family, np.array([j]), args)[0])
    extra, upper = [], []
    for family, indices in [('line', model.eta_line), ('unit', model.eta_unit)]:
        total = getattr(args, 'eta_'+family+'_total_mw')
        if total is not None:
            # LaTeX: sum_{states,j in family} eta_{k,j} <= H_family_MW/baseMVA.
            row = sparse.csr_matrix((np.ones(len(indices)),
                                     (np.zeros(len(indices), dtype=int), indices)),
                                    shape=(1, len(model.c0)))
            extra.append(row); upper.append(total/case['base'])
    if not extra:
        return replace(model, ub=ub)
    return replace(model, ub=ub, A=sparse.vstack([model.A]+extra, format='csc'),
                   lower=np.r_[model.lower, np.full(len(extra), -np.inf)],
                   upper=np.r_[model.upper, upper],
                   row_is_power=np.r_[model.row_is_power, np.ones(len(extra), dtype=bool)])


def eta_limit_diagnostics(case, outages, g, outputs, args, prefix=''):
    """Check limits against actual flows, independently of solver slack values."""
    H, offset, mon, blocks = network_blocks(case, not args.no_line_contingencies)
    states = [('line', ell, monitored, F@g+b)
              for ell, monitored, F, b in blocks]
    states.extend(('unit', int(k), mon, H[mon]@y+offset[mon])
                  for k, y in zip(outages, outputs))
    records, sums, violation = [], {'line': 0.0, 'unit': 0.0}, 0.0
    for family, state, monitored, flows in states:
        eta_mw = np.maximum(np.abs(flows)-case['rates'][monitored], 0.0)*case['base']
        limits = individual_eta_cap_pu(case, family, monitored, args)*case['base']
        sums[family] += float(eta_mw.sum())
        violation = max(violation, max0(eta_mw-limits))
        kind = 'generator_outage' if family=='unit' else 'nominal' if state==-1 else 'line_outage'
        outage_id = (id_key(case['gen'].iloc[state].gen_ID) if family=='unit' else
                     None if state==-1 else id_key(case['branch'].iloc[state].line_ID))
        for j, flow, eta, limit in zip(monitored, flows, eta_mw, limits):
            records.append(dict(phase='recovered' if prefix else 'provisional', family=family,
                                state_kind=kind, outage_ID=outage_id,
                                line_ID=id_key(case['branch'].iloc[j].line_ID),
                                flow_MW=float(flow)*case['base'],
                                rating_MW=float(case['rates'][j])*case['base'],
                                minimum_eta_MW=float(eta),
                                individual_eta_cap_MW=float(limit) if np.isfinite(limit) else None,
                                individual_violation_MW=max(0.0, float(eta-limit))))
    result = {}
    for family in ('line', 'unit'):
        total = getattr(args, 'eta_'+family+'_total_mw')
        extra = max(0.0, sums[family]-total) if total is not None else 0.0
        violation = max(violation, extra)
        result[prefix+family+'_aggregate_eta_MW'] = sums[family]
        result[prefix+family+'_aggregate_eta_cap_MW'] = total
        result[prefix+family+'_aggregate_eta_violation_MW'] = extra
    result[prefix+'eta_limit_violation_MW'] = violation
    result[prefix+'eta_limits_satisfied'] = bool(violation <= args.feas_tol_mw)
    return result, records


def total_cost_cap(cap):
    # A generation-only cap is not a cap on total operating cost.
    if isinstance(cap, dict):
        return sum(cap.values()) if set(cap)=={'generation','line','unit'} else None
    return cap


def run_budget_experiment(case, outages, args, fixed, output, metadata, preprocess, weights):
    """Reference OPF -> cost-budget LP -> physical recovery; optional MILP controls."""
    wall_start = perf_counter()
    save_meta = lambda: (output/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    reference_start = perf_counter()
    reference_bound = None
    if args.reference_cost is not None:
        reference = dict(model='supplied_reference', status='supplied_not_independently_verified',
                         operating_objective=args.reference_cost, total_seconds=0.0,
                         note='External reference solve time is not included; report it separately.')
        refcost = args.reference_cost
        print(f'Using supplied reference cost {refcost:.8f}; external reference runtime is excluded.')
    else:
        refargs = copy.copy(args)
        refargs.no_line_contingencies = args.no_line_contingencies if args.reference_opf == 'envelope' else args.reference_opf == 'nominal'
        refoutages = outages if args.reference_opf == 'envelope' else []
        print(f'Solving reference {args.reference_opf} OPF ({len(refoutages)} generator-outage blocks).', flush=True)
        # LaTeX: v_ref in argmin_{v in X_env} C_g+C_l+C_u for envelope reference.
        # Other reference choices omit generator-outage/APR blocks.
        t = perf_counter(); refmodel = build_model(case, refoutages, refargs); build = perf_counter()-t
        reference, v = solve_model(refmodel, 0.0, refargs)
        reference.update(model=f'reference_{args.reference_opf}_OPF', model_build_seconds=build)
        if v is not None:
            tv = perf_counter()
            metrics = operating_metrics(case, refoutages, v[refmodel.g], v[refmodel.y], not refargs.no_line_contingencies, args.penalty)
            reference.update(metrics); reference['validation_seconds'] = perf_counter()-tv
            pd.DataFrame(dict(gen_ID=case['gen'].gen_ID, bus_i=case['gen'].bus_i,
                              generation_MW=v[refmodel.g]*case['base'])).to_csv(output/'reference_dispatch.csv', index=False)
        pd.DataFrame([reference]).to_csv(output/'reference_summary.csv', index=False)
        if reference['status'] != 'optimal' or not reference.get('solver_point_verified'):
            print('Reference OPF did not produce a verified optimum; no automatic cost budget was formed.')
            pd.DataFrame([reference]).to_csv(output/'summary.csv', index=False)
            metadata['reference_result'] = reference; save_meta()
            return 2
        refcost = reference['operating_objective']
        # A reference with additional line constraints is not a lower bound for
        # a target that explicitly omitted those constraints.
        if refargs.no_line_contingencies or not args.no_line_contingencies:
            reference_bound = reference['solver_best_bound_penalized']
        print(f'  Reference cost={refcost:.8f}; solve={reference["total_seconds"]:.4f}s')
    pd.DataFrame([reference]).to_csv(output/'reference_summary.csv', index=False)
    if not np.isfinite(refcost) or refcost <= 0:
        raise ValueError('Percentage budgets require a positive finite reference cost; inspect cost conventions.')
    reference_components = None
    if args.budget_mode == 'split':
        reference_components = {'generation': reference['generation_cost'],
                                'line': reference['line_thermal_penalty'],
                                'unit': reference['unit_thermal_penalty']}
        if any(not np.isfinite(value) or value < 0 for value in reference_components.values()):
            raise ValueError('Split percentage budgets require finite nonnegative reference components')
        print('Reference components: ' + ', '.join(f'{name}={value:.8f}' for name,value in reference_components.items()))
        if args.thermal_cost_budgets == 'on' and (reference_components['line'] <= args.budget_tol or reference_components['unit'] <= args.budget_tol):
            print('NOTE: A zero reference penalty stays zero for ANY relative epsilon. Optional absolute allowances are in objective-cost units.')
    reference_seconds = perf_counter()-reference_start
    metadata.update(lexicographic_cost=args.lexicographic_cost, slack_preserve_tolerance_pu=args.slack_preserve_tol, reference_result=reference, reference_cost=refcost, reference_stage_seconds=reference_seconds,
                    reference_runtime_scope='External reference solve excluded' if args.reference_cost is not None else 'Reference construction, solve, validation, and reference output',
                    budget_mode=args.budget_mode, reference_components=reference_components,
                    budget_definition='Separate C_g, C_l (base+line outages), C_u (generator-outage line overloads) caps' if args.budget_mode=='split' else 'Original generation cost plus all modeled thermal penalties <= (1+epsilon)*reference_cost',
                    budget_objective={'signal':'sum(z); generation and thermal costs only appear in budget',
                                      'thermal-signal':'Thermal penalties + W*sum(z)',
                                      'apr-slack':'sum(splus+sminus) in per unit; no direct signal or thermal objective',
                                      'operating':'Original C_g+C_l+C_u under selected budgets (diagnostic)'}[args.budget_objective],
                    threshold_applies=False,
                    threshold_note='The earlier sufficient-weight argument does not cover the added cost constraint.',
                    pipeline_timing_definition='Budget candidates: input preprocessing + reference stage + own model build + own solve + own validation/recovery. Unbudgeted controls exclude the reference stage. Other sweep points and comparison solves excluded; complete experiment wall time separately recorded.')
    save_meta()
    print('Budget objective: ' + {'signal':'sum(z) ONLY','thermal-signal':'thermal penalties + W*sum(z)',
                                 'apr-slack':'sum(splus+sminus) ONLY (per unit)',
                                 'operating':'ORIGINAL operating cost C_g+C_l+C_u (diagnostic)'}[args.budget_objective] + f'; budget mode={args.budget_mode}.')
    if args.budget_objective == 'signal':
        print('Signal-only objective: W=1; no penalty-weight sweep is needed. Optimal dispatch/slack values can be nonunique.')
    elif args.budget_objective == 'apr-slack':
        print('APR-slack objective: W=0, no direct signal penalty. All three handwritten equalities and nonnegative APR slacks are included.')
        print('NOTE: sum(splus+sminus) is NOT APR mismatch and need not be zero for an APR-consistent solution.')
    elif args.budget_objective == 'operating':
        print('Operating-cost diagnostic: W=0. Only budget_exact_APR certifies the best original SCOPF cost under these budgets; budget_LP is a relaxation.')
        if not args.compare_weighted_exact: print('Use --compare-weighted-exact to quantify the restriction from the budgets with an exact-APR benchmark.')
    print('NOTE: Recovery can violate the cost budget; original APR feasibility and budget feasibility are checked separately.')
    print('NOTE: The earlier sufficient-weight argument does NOT establish APR tightness with the added budget constraint.')
    rows, all_checks, all_details, all_dispatch = [], [], [], []
    all_eta_checks = []
    metadata['thermal_cost_budgets'] = args.thermal_cost_budgets
    metadata['eta_limit_scope'] = 'Budget candidates only; reference and unrestricted controls retain their original constraints'
    metadata['eta_limit_settings'] = {key:value for key,value in vars(args).items() if key.startswith('eta_')}
    if args.thermal_cost_budgets == 'off':
        metadata['budget_definition'] = 'Generation-cost cap plus explicit individual and/or aggregate thermal-slack limits; no C_l or C_u cost caps'
        print('Thermal COST budgets OFF: using direct eta limits instead.' +
              (' C_g cap retained.' if not args.no_generation_budget else ''), flush=True)
    metadata['generation_budget_enabled'] = not args.no_generation_budget
    if args.no_generation_budget:
        metadata['budget_definition'] = 'Direct eta limits ONLY; no generation or thermal cost caps. Generator output bounds, balance and APR envelope remain.'
        print('DIAGNOSTIC: generation COST cap also OFF; only direct eta limits restrict overloads. Physical generation bounds remain.', flush=True)
    print('Direct eta settings: ' + json.dumps(metadata['eta_limit_settings']), flush=True)
    save_meta()

    def save_progress():
        pd.DataFrame(rows).to_csv(output/'summary.csv', index=False)
        pd.DataFrame(all_checks).to_csv(output/'apr_checks.csv', index=False)
        pd.DataFrame(all_details).to_csv(output/'apr_dispatch.csv', index=False)
        pd.DataFrame(all_dispatch).to_csv(output/'dispatch.csv', index=False)
        if all_eta_checks:
            pd.DataFrame(all_eta_checks).to_csv(output/'eta_limit_checks.csv', index=False)

    def run(tag, model, weight, build_seconds, epsilon=None, cap=None, eps_setting=None, point_id=None, first_stage=None):
        budgeted = cap is not None
        cap_text = json.dumps(cap) if isinstance(cap, dict) else (f'{cap:.8f}' if budgeted else '')
        print(f'Solving {tag}, W={weight:g}' + (f', point={point_id}, base increase={100*epsilon:g}%, cap={cap_text}' if budgeted else ''), flush=True)
        r, vector = solve_model(model, weight, args)
        objective = 'operating' if first_stage is not None else args.budget_objective
        r.update(model=tag, objective_kind={'signal':'signal_only','thermal-signal':'thermal_plus_signal','apr-slack':'apr_slack_sum','operating':'original_cost_under_budgets'}[objective] if budgeted else 'original_plus_signal',
                 shared_matrix_build_seconds=build_seconds, thermal_mode=args.thermal_mode,
                 gamma=args.gamma, demand_scale=args.demand_scale, reference_cost=refcost,
                 budget_increase=epsilon, budget_mode=args.budget_mode if budgeted else None,
                 budget_point=point_id,
                 cost_budget=total_cost_cap(cap), candidate_guarantee_applicable=False,
                 external_reference_time_excluded=args.reference_cost is not None)
        if budgeted and isinstance(cap, dict):
            r.update(epsilon_generation=eps_setting['generation'] if not args.no_generation_budget else None,
                     epsilon_line=eps_setting['line'] if args.thermal_cost_budgets=='on' else None,
                     epsilon_unit=eps_setting['unit'] if args.thermal_cost_budgets=='on' else None,
                     absolute_line_allowance=args.line_allowance, absolute_unit_allowance=args.unit_allowance)
        if vector is not None:
            tv = perf_counter()
            metrics, checks, details, dispatch = analyze(case, outages, model, vector, args, tag, weight)
            r.update(metrics)
            if budgeted:
                eta_diagnostics, eta_checks = eta_limit_diagnostics(case, outages, vector[model.g], vector[model.y], args)
                r.update(eta_diagnostics)
                if metrics.get('recovery_success'):
                    recovered_outputs = [recover(case, vector[model.g], k,
                                         min(args.feas_tol_mw/case['base']/100, 1e-10))[0] for k in outages]
                    recovered_diagnostics, recovered_checks = eta_limit_diagnostics(
                        case, outages, vector[model.g], recovered_outputs, args, prefix='recovered_')
                    r.update(recovered_diagnostics); eta_checks.extend(recovered_checks)
                for record in eta_checks:
                    record.update(model=tag, budget_point=point_id, weight=weight)
                all_eta_checks.extend(eta_checks)
            if model.splus is not None:
                r['both_APR_slacks_positive_pairs'] = int(np.count_nonzero(np.minimum(vector[model.splus], vector[model.sminus])*case['base'] > args.apr_tol_mw))
            r['original_plus_signal_objective'] = metrics['penalized_objective']
            if budgeted:
                r['penalized_objective'] = metrics['apr_slack_sum_pu'] if objective=='apr-slack' else metrics['signal_penalty']
                if objective == 'thermal-signal':
                    r['penalized_objective'] += metrics['operating_objective']-metrics['generation_cost']
                elif objective == 'operating':
                    r['penalized_objective'] = metrics['operating_objective']
                r['optimized_budget_objective'] = r['penalized_objective']
                r['optimized_budget_objective_units'] = {'signal':'dimensionless','thermal-signal':'original_objective_cost_units','apr-slack':'per_unit_power_sum','operating':'original_objective_cost_units'}[objective]
                r['budget_constraint_cost_with_solver_slacks'] = float(model.original_cost@vector+model.original_constant)
                diagnostics = budget_diagnostics(metrics, cap, args)
                r.update({'provisional_'+key: value for key,value in diagnostics.items()})
                if isinstance(cap, dict):
                    for name, (coef, constant) in cost_component_rows(model, original=True).items():
                        r[name+'_cost_with_solver_slacks'] = float(coef@vector+constant)
                r['provisional_APR_and_budget_feasible'] = bool(metrics['provisional_original_APR_feasible'] and r['provisional_budget_satisfied'] and r['solver_point_verified'] and r['eta_limits_satisfied'])
                if metrics.get('recovery_success'):
                    r.update(budget_diagnostics(metrics, cap, args, prefix='recovered_'))
                    r['recovered_APR_and_budget_feasible'] = bool(metrics.get('recovered_original_APR_feasible') and r['recovered_budget_satisfied'] and r['recovered_eta_limits_satisfied'])
            if first_stage is not None:
                limit = first_stage['apr_slack_sum_pu']+args.slack_preserve_tol
                actual = metrics['apr_slack_sum_pu']
                allowance = 1e-8*max(1.0, abs(limit))
                preserved = actual <= limit+allowance
                r.update(stage1_slack_optimum_pu=first_stage['apr_slack_sum_pu'],
                         slack_preserve_tolerance_pu=args.slack_preserve_tol,
                         slack_preserve_limit_pu=limit, slack_preserved=preserved,
                         slack_preserve_validation_allowance_pu=allowance,
                         stage1_operating_objective=first_stage['operating_objective'],
                         operating_cost_reduction=first_stage['operating_objective']-metrics['operating_objective'],
                         stage1_solve_seconds=first_stage['total_seconds'])
                r['solver_point_verified'] = bool(r['solver_point_verified'] and preserved)
                r['provisional_APR_and_budget_feasible'] &= r['solver_point_verified']
            r['provisional_original_APR_feasible'] &= r['solver_point_verified']
            if not r['solver_point_verified']: r['status'] = 'validation_failed'
            r['validation_seconds'] = perf_counter()-tv
            for table in (checks, details, dispatch):
                for record in table:
                    record.update(budget_increase=epsilon, budget_mode=args.budget_mode if budgeted else None,
                                  budget_point=point_id, cost_budget=total_cost_cap(cap))
                    if isinstance(cap, dict):
                        record.update({name+'_budget_cap': value for name,value in cap.items()})
                        record.update({'epsilon_'+name: value if (name=='generation' and not args.no_generation_budget) or (name!='generation' and args.thermal_cost_budgets=='on') else None for name,value in eps_setting.items()})
            all_checks.extend(checks); all_details.extend(details); all_dispatch.extend(dispatch)
        r['pipeline_seconds'] = preprocess+(reference_seconds if budgeted else 0.0)+build_seconds+r['total_seconds']+r.get('validation_seconds', 0.0)
        if first_stage is not None:
            r['pipeline_seconds'] = first_stage['pipeline_seconds']+build_seconds+r['total_seconds']+r.get('validation_seconds', 0.0)
        r['pipeline_is_candidate_method'] = tag in ('budget_LP','budget_LP_cost')
        rows.append(r); save_progress()
        print(f'  status={r["status"]}; solve={r["total_seconds"]:.4f}s; pipeline={r["pipeline_seconds"]:.4f}s; original cost={r.get("operating_objective", float("nan")):.8f}; APR error={r.get("max_APR_L1_error_MW", float("nan")):.6g} MW; optimized objective={r.get("optimized_budget_objective", r.get("penalized_objective", float("nan"))):.10g}', flush=True)
        if budgeted and isinstance(cap, dict) and vector is not None:
            print('  Components: ' + ', '.join(f'{name}={r[key]:.8f} / cap=' + (f'{cap[name]:.8f}' if name in cap else 'none') for name,key in [('generation','generation_cost'),('line','line_thermal_penalty'),('unit','unit_thermal_penalty')]), flush=True)
            print(f'  Direct eta limits: provisional passed={r["eta_limits_satisfied"]}; max violation={r["eta_limit_violation_MW"]:.6g} MW', flush=True)
        if budgeted and r.get('recovery_success'):
            print(f'  Recovered cost={r["recovered_operating_objective"]:.8f}; recovered APR+budget feasible={r["recovered_APR_and_budget_feasible"]}', flush=True)
        return r

    # Unbudgeted W=0 LP gives a valid lower bound for the original operating objective.
    t = perf_counter(); original_lp = build_model(case, outages, args, False, fixed); lp_build = perf_counter()-t
    run('envelope_LP', original_lp, 0.0, lp_build)
    original_exact = None
    if args.compare_exact or args.compare_weighted_exact:
        t = perf_counter(); original_exact = build_model(case, outages, args, True, fixed); exact_build = perf_counter()-t
        run('exact_APR', original_exact, 0.0, exact_build)
    candidate_lp, candidate_lp_build = original_lp, lp_build
    candidate_exact, candidate_exact_build = original_exact, exact_build if original_exact is not None else None
    if args.budget_objective == 'apr-slack':
        t=perf_counter(); candidate_lp=add_apr_slacks(original_lp,case,outages); candidate_lp_build=lp_build+perf_counter()-t
        if args.compare_weighted_exact:
            t=perf_counter(); candidate_exact=add_apr_slacks(original_exact,case,outages); candidate_exact_build=exact_build+perf_counter()-t
    settings, settings_by_caps, requested_points = [], {}, 0
    for epsilon in dict.fromkeys(args.budget_increase):
        if args.budget_mode == 'split':
            grids = [getattr(args, grid) or [getattr(args, scalar) if getattr(args, scalar) is not None else epsilon]
                     for grid,scalar in [('epsilon_g_grid','epsilon_g'),('epsilon_line_grid','epsilon_line'),('epsilon_unit_grid','epsilon_unit')]]
            epsilons = [dict(zip(['generation','line','unit'], values)) for values in product(*grids)]
        else:
            epsilons = [None]
        for eps in epsilons:
            requested_points += 1
            if eps is not None:
                # LaTeX: C_g<=(1+eps_g)C_g_ref;
                # C_l<=(1+eps_l)C_l_ref+Delta_l; C_u<=(1+eps_u)C_u_ref+Delta_u.
                cap = {name: (1+eps[name])*value for name,value in reference_components.items()}
                cap['line'] += args.line_allowance; cap['unit'] += args.unit_allowance
                if args.thermal_cost_budgets == 'off':
                    cap = {'generation': cap['generation']}
                if args.no_generation_budget:
                    cap.pop('generation', None)
                key = tuple(cap.values())
            else:
                cap = (1.0+epsilon)*refcost
                key = (cap,)
            if key in settings_by_caps:
                settings_by_caps[key]['equivalent_epsilon_settings'].append(dict(base_epsilon=epsilon, epsilons=eps))
                continue
            point = dict(budget_point=f'P{len(settings)+1:04d}', base_epsilon=epsilon, epsilons=eps,
                         absolute_line_allowance=args.line_allowance, absolute_unit_allowance=args.unit_allowance,
                         caps=cap, equivalent_epsilon_settings=[])
            settings.append(point); settings_by_caps[key]=point
    metadata['budget_settings'] = settings
    metadata['requested_epsilon_combinations'] = requested_points
    metadata['distinct_budget_combinations'] = len(settings)
    save_meta()
    print(f'Epsilon sweep: {requested_points} requested combinations, {len(settings)} distinct budget combinations; reference and inputs reused.')
    if requested_points > len(settings):
        print('Identical budgets share one solve; equivalent epsilon settings are recorded in metadata.json (e.g. zero reference penalties).')
    for point in settings:
        epsilon, eps, cap, point_id = point['base_epsilon'], point['epsilons'], point['caps'], point['budget_point']
        t = perf_counter(); model = apply_eta_limits(budget_model(candidate_lp, cap, args.budget_objective), case, args); build = candidate_lp_build+perf_counter()-t
        for weight in weights:
            first = run('budget_LP', model, weight, build, epsilon, cap, eps, point_id)
            if args.lexicographic_cost:
                if first['status']=='optimal' and first.get('solver_point_verified'):
                    t=perf_counter()
                    second_model=preserve_slack_optimum(model, first['apr_slack_sum_pu'], args.slack_preserve_tol)
                    second_build=perf_counter()-t
                    run('budget_LP_cost', second_model, 0.0, second_build, epsilon, cap, eps, point_id, first_stage=first)
                else:
                    print('Stage two skipped: stage one did not produce a verified slack optimum.', flush=True)
        if args.compare_weighted_exact:
            t = perf_counter(); emodel = apply_eta_limits(budget_model(candidate_exact, cap, args.budget_objective), case, args); ebuild = candidate_exact_build+perf_counter()-t
            for weight in weights:
                run('budget_exact_APR', emodel, weight, ebuild, epsilon, cap, eps, point_id)
    bounds = [r['solver_best_bound_penalized'] for r in rows
              if r['objective_kind']=='original_plus_signal' and r['weight']==0 and r['solver_best_bound_penalized'] is not None]
    if reference_bound is not None: bounds.append(reference_bound)
    lower = max(bounds) if bounds else None
    benchmark = next((r for r in rows if r['model']=='exact_APR' and r['status']=='optimal' and r.get('provisional_original_APR_feasible')), None)
    for r in rows:
        r['bound_scope'] = 'fixed_dispatch_problem' if fixed is not None else 'full_preventive_problem'
        r['valid_lower_bound_original' if fixed is None else 'valid_lower_bound_fixed_dispatch'] = lower
        incumbent = r.get('recovered_operating_objective') if r.get('recovered_original_APR_feasible') else None
        if r.get('provisional_original_APR_feasible'):
            incumbent = min(incumbent, r['operating_objective']) if incumbent is not None else r['operating_objective']
        r['verified_original_upper_bound'] = incumbent
        if incumbent is not None and lower is not None:
            r['certified_original_gap' if fixed is None else 'certified_fixed_dispatch_gap'] = incumbent-lower
        if benchmark and r.get('recovery_success'):
            r['recovered_difference_vs_original_exact'] = r['recovered_operating_objective']-benchmark['operating_objective']
            r['recovered_relative_difference_vs_original_exact'] = r['recovered_difference_vs_original_exact']/max(1.0, abs(benchmark['operating_objective']))
    save_progress()
    if args.budget_objective == 'operating' and args.compare_weighted_exact:
        diagnostic = []
        for r in rows:
            if r['model'] != 'budget_exact_APR': continue
            d = {key:r.get(key) for key in ['budget_point','epsilon_generation','epsilon_line','epsilon_unit','status','operating_objective','solver_best_bound_penalized']}
            if benchmark:
                d['unrestricted_MILP_optimum'] = benchmark['operating_objective']
                if r['status']=='optimal' and r.get('provisional_APR_and_budget_feasible'):
                    gap = r['operating_objective']-benchmark['operating_objective']
                    tolerance = max(1e-4,1e-7*abs(benchmark['operating_objective']))
                    d['cost_increase_due_to_budgets'] = gap
                    d['relative_cost_increase_due_to_budgets'] = gap/max(1.,abs(benchmark['operating_objective']))
                    d['conclusion'] = 'Original optimum attainable within solver tolerance' if abs(gap)<=tolerance else 'Budgets raise the minimum original operating cost' if gap>0 else 'Inspect solver tolerances and validation'
                elif r['status']=='infeasible':
                    d['conclusion'] = 'Budgets exclude every exact-APR feasible solution'
                else: d['conclusion'] = 'No certified budget-restricted optimum; inspect status and bounds'
            else: d['conclusion'] = 'Unrestricted MILP optimum unavailable; no certified cost-gap comparison'
            diagnostic.append(d)
        pd.DataFrame(diagnostic).to_csv(output/'budget_cost_diagnostics.csv',index=False)
        print('Budget restriction diagnostic (ORIGINAL operating objective):')
        print(pd.DataFrame(diagnostic).to_string(index=False))
    metadata['experiment_seconds_including_reference_and_all_sweep_and_benchmark_jobs'] = preprocess+perf_counter()-wall_start
    save_meta()
    summary = pd.DataFrame(rows)
    sweep_rows = []
    for r in rows:
        if r['model'] not in ('budget_LP','budget_LP_cost'): continue
        point = r.copy()
        candidates = []
        if r.get('provisional_APR_and_budget_feasible'): candidates.append(r['operating_objective'])
        if r.get('recovered_APR_and_budget_feasible'): candidates.append(r['recovered_operating_objective'])
        point['has_APR_and_budget_feasible_candidate'] = bool(candidates)
        point['budget_feasible_candidate_cost'] = min(candidates) if candidates else None
        sweep_rows.append(point)
    if args.lexicographic_cost:
        comparison = []
        for second in rows:
            if second['model'] != 'budget_LP_cost': continue
            first = next(r for r in rows if r['model']=='budget_LP' and r['budget_point']==second['budget_point'])
            comparison.append(dict(budget_point=second['budget_point'],
                stage1_slack_objective_pu=first.get('apr_slack_sum_pu'),
                stage2_slack_objective_pu=second.get('apr_slack_sum_pu'),
                slack_preserved=second.get('slack_preserved'),
                stage1_operating_cost=first.get('operating_objective'),
                stage2_operating_cost=second.get('operating_objective'),
                operating_cost_reduction=second.get('operating_cost_reduction'),
                stage1_both_slacks_positive_pairs=first.get('both_APR_slacks_positive_pairs'),
                stage2_both_slacks_positive_pairs=second.get('both_APR_slacks_positive_pairs'),
                stage1_APR_mismatch_MW=first.get('max_APR_L1_error_MW'),
                stage2_APR_mismatch_MW=second.get('max_APR_L1_error_MW'),
                stage2_APR_and_budget_feasible=second.get('provisional_APR_and_budget_feasible'),
                stage1_solve_seconds=first['total_seconds'], stage2_solve_seconds=second['total_seconds'],
                both_stage_solve_seconds=first['total_seconds']+second['total_seconds'],
                complete_candidate_pipeline_seconds=second['pipeline_seconds']))
        pd.DataFrame(comparison).to_csv(output/'lexicographic_comparison.csv', index=False)
        print('Two-stage comparison (APR consistency must be checked again):')
        print(pd.DataFrame(comparison).to_string(index=False))
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(output/'epsilon_results.csv', index=False)
    feasible = sweep[sweep.has_APR_and_budget_feasible_candidate].sort_values('budget_feasible_candidate_cost')
    print(f'APR-consistent or recovered candidates satisfying ALL budgets: {len(feasible)} of {len(sweep)} LP solves.')
    if len(feasible):
        print('Lowest operating costs among verified budget-feasible candidates:')
        print(feasible[[c for c in ['budget_point','epsilon_generation','epsilon_line','epsilon_unit','optimized_budget_objective','budget_feasible_candidate_cost','max_APR_L1_error_MW'] if c in feasible]].head(5).to_string(index=False))
    columns = [c for c in ['model','budget_point','epsilon_generation','epsilon_line','epsilon_unit','weight','status','optimized_budget_objective','operating_objective','max_APR_L1_error_MW','recovered_operating_objective','recovered_APR_and_budget_feasible','total_seconds','pipeline_seconds'] if c in summary]
    print(summary[columns].to_string(index=False)); print(f'Saved: {output.resolve()}')
    return 0 if all(r['status']=='optimal' for r in rows) else 2


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--case', required=True)
    p.add_argument('--case-dir', default='../excel_outputs')
    p.add_argument('--output-dir', default='output_apr_signal')
    p.add_argument('--solver', choices=['auto','highs','gurobi'], default='auto')
    p.add_argument('--gamma', type=float, default=0.05)
    p.add_argument('--demand-scale', type=float, default=1.0)
    p.add_argument('--penalty', type=float, default=1500.0)
    p.add_argument('--weights', nargs='+', type=float, default=[0,1,10,100,1000])
    p.add_argument('--include-threshold', action='store_true', help='Also test W=1.01*B (or W=1 if B=0); B is the candidate soft-limit threshold')
    p.add_argument('--compare-exact', action='store_true', help='Solve original exact-APR MILP with W=0')
    p.add_argument('--compare-weighted-exact', action='store_true', help='Also solve exact-APR MILP for each requested weight')
    p.add_argument('--generator-contingencies', default='all', help='Active gen_ID values separated by commas, or all')
    p.add_argument('--no-line-contingencies', action='store_true')
    p.add_argument('--islanding', choices=['error','skip'], default='error')
    p.add_argument('--thermal-mode', choices=['soft','hard'], default='soft')
    p.add_argument('--time-limit', type=float, default=300.0, help='Seconds per optimization solve')
    p.add_argument('--threads', type=int, default=1, help='Gurobi threads; HiGHS via SciPy uses its default threading')
    p.add_argument('--mip-gap', type=float, default=1e-7)
    p.add_argument('--apr-tol-mw', type=float, default=1e-4)
    p.add_argument('--feas-tol-mw', type=float, default=1e-4)
    p.add_argument('--fixed-dispatch')
    p.add_argument('--fixed-method', default='central')
    p.add_argument('--fixed-run', type=int, default=1)
    p.add_argument('--tee', action='store_true')
    p.add_argument('--budget-increase', nargs='+', type=float, help='Enable budget experiments; 0.10 means at most 10 percent above the reference cost. Accepts several fractions.')
    p.add_argument('--budget-objective', choices=['signal','thermal-signal','apr-slack','operating'], default='signal', help='signal: min sum(z). apr-slack: min sum(splus+sminus). operating: diagnostic min original cost under budgets. thermal-signal: earlier weighted objective.')
    p.add_argument('--lexicographic-cost', action='store_true', help='After a verified apr-slack LP optimum, minimize original operating cost preserving its slack sum. Both stages validated independently.')
    p.add_argument('--slack-preserve-tol', type=float, default=1e-7, help='Allowed absolute increase in stage-one slack sum, in per-unit aggregate power (default 1e-7).')
    p.add_argument('--reference-opf', choices=['nominal','line','envelope'], default='line', help='nominal/line omit generator outages; envelope includes full continuous generator-APR relaxation and is required for split budgets.')
    p.add_argument('--budget-mode', choices=['joint','split'], default='joint', help='joint: combined cost cap; split: separate generation, base+line thermal, and generator-outage thermal caps.')
    p.add_argument('--thermal-cost-budgets', choices=['on','off'], default='on', help='off: remove C_l and C_u cost caps in split mode; retain C_g cap and require explicit direct eta limits. on preserves existing cost budgets.')
    p.add_argument('--no-generation-budget', action='store_true', help='Diagnostic: remove C_g cost cap too. Requires --thermal-cost-budgets off and direct eta limits. Physical generation bounds stay active.')
    for family, description in [('line','nominal and line-outage states'), ('unit','generator-outage states')]:
        individual = p.add_mutually_exclusive_group()
        individual.add_argument('--eta-'+family+'-mw', type=float, help='Maximum overload in MW on EACH monitored line in '+description+'.')
        individual.add_argument('--eta-'+family+'-fraction', type=float, help='Maximum overload as fraction of EACH monitored line rating in '+description+'; 0.05 means 5 percent, NOT a percentage of reference slack.')
        p.add_argument('--eta-'+family+'-total-mw', type=float, help='Optional SUM of overloads in MW over all lines and all '+description+'; can combine with individual caps.')
    p.add_argument('--epsilon-g', type=float, help='Generation-cost relative increase in split mode; otherwise use --budget-increase.')
    p.add_argument('--epsilon-line', type=float, help='Base+line-outage thermal-cost relative increase in split mode.')
    p.add_argument('--epsilon-unit', type=float, help='Generator-outage thermal-cost relative increase in split mode.')
    p.add_argument('--epsilon-g-grid', nargs='+', type=float, help='Generation-cost epsilon values; Cartesian product with the other epsilon grids.')
    p.add_argument('--epsilon-line-grid', nargs='+', type=float, help='Base+line thermal-cost epsilon values.')
    p.add_argument('--epsilon-unit-grid', nargs='+', type=float, help='Generator-outage thermal-cost epsilon values; 1 means a 100 percent increase.')
    p.add_argument('--line-allowance', type=float, default=0.0, help='ADDITIVE base+line thermal-cost allowance; objective-cost units, NOT MW.')
    p.add_argument('--unit-allowance', type=float, default=0.0, help='ADDITIVE generator-outage thermal-cost allowance; objective-cost units, NOT MW.')
    p.add_argument('--reference-cost', type=float, help='Use an already computed positive reference cost; its external solve time and cost convention must be reported separately.')
    p.add_argument('--budget-tol', type=float, default=1e-4, help='Absolute original-cost budget validation tolerance (also allows 1e-8 relative).')
    return p


def main():
    args = parser().parse_args()
    eta_values = [value for key,value in vars(args).items() if key.startswith('eta_')]
    eta_requested = any(value is not None for value in eta_values)
    if any(value is not None and (not np.isfinite(value) or value < 0) for value in eta_values):
        raise ValueError('Direct eta limits must be finite and nonnegative')
    if eta_requested and (args.budget_increase is None or args.thermal_mode != 'soft'):
        raise ValueError('Direct eta limits require --thermal-mode soft and a budget experiment (--budget-increase)')
    if args.thermal_cost_budgets == 'off':
        if args.budget_mode != 'split' or not eta_requested:
            raise ValueError('--thermal-cost-budgets off requires --budget-mode split and at least one direct eta limit')
        if any(getattr(args,key) is not None for key in ('epsilon_line','epsilon_unit','epsilon_line_grid','epsilon_unit_grid')) or args.line_allowance or args.unit_allowance:
            raise ValueError('Thermal cost-budget epsilons/allowances do not apply when --thermal-cost-budgets off. Use --eta-... options instead.')
    if args.no_generation_budget:
        if args.thermal_cost_budgets != 'off':
            raise ValueError('--no-generation-budget requires --thermal-cost-budgets off and direct eta limits')
        if args.epsilon_g is not None or args.epsilon_g_grid is not None:
            raise ValueError('Generation epsilon does not apply with --no-generation-budget; remove --epsilon-g / --epsilon-g-grid')
    if args.lexicographic_cost and (args.budget_objective != 'apr-slack' or args.budget_increase is None):
        raise ValueError('--lexicographic-cost requires --budget-objective apr-slack and --budget-increase')
    if not np.isfinite(args.slack_preserve_tol) or args.slack_preserve_tol < 0:
        raise ValueError('--slack-preserve-tol must be finite and nonnegative')
    if not np.isfinite(args.gamma) or not 0 <= args.gamma <= 1:
        raise ValueError('gamma must be in [0,1] for this APR model and its exact MILP linearization')
    for key in ('demand_scale','penalty','time_limit','apr_tol_mw','feas_tol_mw'):
        if not np.isfinite(getattr(args,key)) or getattr(args,key) <= 0: raise ValueError(f'{key} must be positive and finite')
    if args.threads < 1 or args.fixed_run < 1 or not np.isfinite(args.mip_gap) or args.mip_gap < 0:
        raise ValueError('Invalid threads, fixed-run, or MIP gap')
    if any(not np.isfinite(w) or w < 0 for w in args.weights): raise ValueError('Weights must be finite and nonnegative')
    if args.budget_increase is not None and any(not np.isfinite(e) or e < 0 for e in args.budget_increase):
        raise ValueError('Budget increases must be finite nonnegative fractions; 0.10 means 10 percent.')
    if args.reference_cost is not None and (not np.isfinite(args.reference_cost) or args.reference_cost <= 0):
        raise ValueError('reference-cost must be positive and finite')
    if args.reference_cost is not None and args.budget_increase is None:
        raise ValueError('reference-cost requires --budget-increase')
    if args.budget_objective in ('apr-slack','operating') and args.budget_increase is None:
        raise ValueError('Selected objective requires a budget experiment, e.g. --budget-increase 0.10')
    for key in ('epsilon_g','epsilon_line','epsilon_unit','line_allowance','unit_allowance'):
        value=getattr(args,key)
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError(f'{key} must be a finite nonnegative number')
    for grid,scalar in [('epsilon_g_grid','epsilon_g'),('epsilon_line_grid','epsilon_line'),('epsilon_unit_grid','epsilon_unit')]:
        values = getattr(args,grid)
        if values is not None:
            if args.budget_mode != 'split': raise ValueError('Epsilon grids require --budget-mode split')
            if getattr(args,scalar) is not None: raise ValueError(f'Use either {grid} or {scalar}, not both')
            if any(not np.isfinite(v) or v < 0 for v in values):
                raise ValueError(f'{grid} must contain finite nonnegative values')
    if args.budget_mode == 'split':
        if args.budget_increase is None: raise ValueError('split mode requires --budget-increase, e.g. 0.10')
        if args.reference_opf != 'envelope' or args.reference_cost is not None:
            raise ValueError('Split budgets require --reference-opf envelope and an automatically solved reference so all three reference components are defined.')
    elif any(getattr(args,key) is not None for key in ('epsilon_g','epsilon_line','epsilon_unit')) or args.line_allowance or args.unit_allowance:
        raise ValueError('Component epsilons and absolute allowances require --budget-mode split')
    if not np.isfinite(args.budget_tol) or args.budget_tol <= 0: raise ValueError('budget-tol must be positive and finite')
    args.backend = args.solver if args.solver != 'auto' else ('gurobi' if importlib.util.find_spec('gurobipy') else 'highs')
    started = perf_counter(); path = resolve_case(args.case, args.case_dir)
    case = load_case(path, args.demand_scale); case['a'] = args.gamma * (case['hi'] - case['lo'])
    args.base = case['base']; outages = select_outages(case, args.generator_contingencies)
    if len(case['lo']) < 2: raise ValueError('Generator-contingency APR experiments require a surviving generator')
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    excluded = case['branch'].iloc[case['excluded']].copy()
    excluded['reason'] = 'islanding outage: excluded from LODF formulation'
    excluded.to_csv(output/'excluded_outages.csv', index=False)
    reference_uses_lines = args.budget_increase is not None and args.reference_cost is None and (args.reference_opf == 'line' or (args.reference_opf == 'envelope' and not args.no_line_contingencies))
    if len(excluded) and (not args.no_line_contingencies or reference_uses_lines) and args.islanding == 'error':
        raise ValueError('Islanding outages found. Use --islanding skip explicitly to study only the non-islanding subset.')
    fixed = read_fixed_dispatch(case,args)
    screen = [dict(gen_ID=case['gen'].iloc[k].gen_ID, surviving_capacity_MW=float(case['hi'].sum()-case['hi'][k])*case['base'],
                   demand_MW=case['D']*case['base']) for k in outages]
    for r in screen: r['capacity_shortage_MW']=max(0.0,r['demand_MW']-r['surviving_capacity_MW'])
    pd.DataFrame(screen).to_csv(output/'capacity_screen.csv',index=False)
    thresholds = penalty_threshold(case,outages)
    for r in thresholds: r['B_k']=args.penalty*r['threshold_per_penalty_unit']
    pd.DataFrame(thresholds).to_csv(output/'penalty_thresholds.csv',index=False)
    B = max(r['B_k'] for r in thresholds)
    weights = list(dict.fromkeys([0.0]+args.weights+([1.01*B if B>0 else 1.0] if args.include_threshold else [])))
    if args.budget_increase is not None and args.budget_objective == 'signal':
        weights = [1.0]
    elif args.budget_increase is not None and args.budget_objective in ('apr-slack','operating'):
        weights = [0.0]
    preprocess = perf_counter()-started
    metadata = dict(arguments=vars(args), case_path=str(path), python=sys.version, platform=platform.platform(),
                    numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,
                    input_preprocessing_seconds=preprocess, demand_MW=case['D']*case['base'],
                    generator_contingencies=[id_key(case['gen'].iloc[k].gen_ID) for k in outages],
                    line_contingencies=0 if args.no_line_contingencies else len(case['eligible']),
                    excluded_islanding=len(excluded) if not args.no_line_contingencies else 0,
                    candidate_penalty_threshold=B, threshold_applies=args.thermal_mode=='soft' and args.budget_increase is None,
                    fixed_preventive_dispatch=fixed is not None, weights=weights,
                    cost_convention='c1 in money/MW, powers per unit, penalty per unit thermal slack; c2 ignored',
                    interpretation='Centralized model test, not ADMM; original cost excludes W sum(z); numerical tightness is not a proof')
    if args.backend == 'gurobi':
        import gurobipy as gp
        metadata['gurobi_version'] = list(gp.gurobi.version())
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(f'Case={path.name}; demand={case["D"]*case["base"]:g} MW; gamma={args.gamma:g}; solver={args.backend}')
    print(f'Generator outages={len(outages)}; line outages={metadata["line_contingencies"]}; excluded islanding={metadata["excluded_islanding"]}; mode={args.thermal_mode}')
    print(f'Candidate soft-limit threshold B={B:.8g}; tested W={weights}')
    print('NOTE: The proposed sufficient-weight argument concerns only SOFT limits with unbounded slacks and NO cost budget.')
    if np.any(case['gen'].c2.to_numpy(float)!=0): print('NOTE: c2 ignored; using linear c1 generation costs.')
    if any(r['capacity_shortage_MW']>args.feas_tol_mw for r in screen):
        print('INFEASIBLE: at least one outage leaves insufficient total surviving capacity. See capacity_screen.csv; thermal slacks cannot replace generation.')
        pd.DataFrame([dict(model='capacity_screen',status='infeasible_capacity')]).to_csv(output/'summary.csv',index=False)
        return 2
    if args.budget_increase is not None:
        return run_budget_experiment(case,outages,args,fixed,output,metadata,preprocess,weights)
    t=perf_counter(); lp=build_model(case,outages,args,False,fixed); lp_build=perf_counter()-t
    exact=None; exact_build=None
    if args.compare_exact or args.compare_weighted_exact:
        t=perf_counter(); exact=build_model(case,outages,args,True,fixed); exact_build=perf_counter()-t
    rows=[]; all_checks=[]; all_details=[]; all_dispatch=[]
    jobs=[]
    if exact is not None: jobs.append(('exact_APR',exact,0.0,exact_build))
    jobs.extend(('envelope_LP',lp,w,lp_build) for w in weights)
    if args.compare_weighted_exact: jobs.extend(('exact_APR',exact,w,exact_build) for w in weights if w!=0)
    for tag,model,weight,build_seconds in jobs:
        print(f'Solving {tag}, W={weight:.8g}',flush=True)
        r,vector=solve_model(model,weight,args)
        r.update(model=tag,shared_matrix_build_seconds=build_seconds,thermal_mode=args.thermal_mode,
                 gamma=args.gamma,demand_scale=args.demand_scale,above_candidate_threshold=weight>B,
                 candidate_guarantee_applicable=args.thermal_mode=='soft' and weight>B)
        if vector is not None:
            tv=perf_counter(); metrics,checks,details,dispatch=analyze(case,outages,model,vector,args,tag,weight)
            r.update(metrics);r['validation_seconds']=perf_counter()-tv
            # An invalid numerical solver point is never promoted to an incumbent.
            r['provisional_original_APR_feasible'] &= r['solver_point_verified']
            if not r['solver_point_verified']: r['status']='validation_failed'
            all_checks.extend(checks);all_details.extend(details);all_dispatch.extend(dispatch)
        rows.append(r)
        # Save completed solves incrementally, preserving progress on time limits.
        pd.DataFrame(rows).to_csv(output/'summary.csv',index=False)
        pd.DataFrame(all_checks).to_csv(output/'apr_checks.csv',index=False)
        pd.DataFrame(all_details).to_csv(output/'apr_dispatch.csv',index=False)
        pd.DataFrame(all_dispatch).to_csv(output/'dispatch.csv',index=False)
        print(f'  status={r["status"]}; time={r["total_seconds"]:.4f}s; original cost={r.get("operating_objective",float("nan")):.8f}; max APR L1 error={r.get("max_APR_L1_error_MW",float("nan")):.6g} MW',flush=True)
    bounds=[r['solver_best_bound_penalized'] for r in rows if r['weight']==0 and r['solver_best_bound_penalized'] is not None]
    lower=max(bounds) if bounds else None
    original=next((r for r in rows if r['model']=='exact_APR' and r['weight']==0 and r['status']=='optimal'
                   and r.get('provisional_original_APR_feasible')),None)
    for r in rows:
        r['bound_scope']='fixed_dispatch_problem' if fixed is not None else 'full_preventive_problem'
        r['valid_lower_bound_original']=lower if fixed is None else None
        r['valid_lower_bound_fixed_dispatch']=lower if fixed is not None else None
        if original and 'operating_objective' in r:
            # A signed difference, not an optimality gap for APR-inconsistent points.
            r['provisional_cost_difference_vs_original_exact']=r['operating_objective']-original['operating_objective']
        incumbent=r.get('recovered_operating_objective') if r.get('recovered_original_APR_feasible') else None
        if r.get('provisional_original_APR_feasible'):
            incumbent=min(incumbent,r['operating_objective']) if incumbent is not None else r['operating_objective']
        r['verified_original_upper_bound']=incumbent
        if incumbent is not None and lower is not None:
            r['certified_original_gap' if fixed is None else 'certified_fixed_dispatch_gap']=incumbent-lower
    summary=pd.DataFrame(rows);summary.to_csv(output/'summary.csv',index=False)
    columns=[c for c in ['model','weight','status','operating_objective','max_APR_L1_error_MW','all_APR_consistent','recovered_operating_objective','total_seconds'] if c in summary]
    print(summary[columns].to_string(index=False));print(f'Saved: {output.resolve()}')
    return 0 if all(r['status']=='optimal' for r in rows) else 2


if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        sys.exit(1)


