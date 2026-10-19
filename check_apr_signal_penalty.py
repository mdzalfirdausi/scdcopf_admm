#!/usr/bin/env python3
"""Test the APR inequalities with an explicit common-signal penalty.

This is a NEW centralized experiment, not ADMM and not the line-only benchmark.
LP: min c'g + M_eta sum(eta_pu) + W sum(z_k), with one common z_k per
    generator outage, y_kk=0, sum(y_k)=demand, g_i<=y_ki<=Pmax_i,
    y_ki<=g_i+a_i*z_k, a_i=gamma*(Pmax_i-Pmin_i), 0<=z_k<=1.
Includes nominal network, generator outages, and optional single-line outages.
Slacks default to SOFT, nonnegative, with no upper bounds.
--compare-exact adds the original APR MILP (W=0).
--compare-weighted-exact also solves exact-APR MILPs with each requested W.

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

    g = variables(case['lo'] if fixed is None else fixed, case['hi'] if fixed is None else fixed,
                  case['gen'].c1.to_numpy(float) * case['base'])
    y = np.empty((nk, ng), dtype=int)
    for s, k in enumerate(outages):
        low, high = case['lo'].copy(), case['hi'].copy()
        low[k] = high[k] = 0.0
        y[s] = variables(low, high)
    z = variables(np.zeros(nk), np.ones(nk))
    x = {}
    if exact:
        for s, k in enumerate(outages):
            for i in range(ng):
                if i != k:
                    x[s, i] = int(variables(0.0, 1.0, integer=1)[0])

    row, col, data, lower, upper = [], [], [], [], []

    def constraint(indices, values, low=-np.inf, high=np.inf):
        indices, values = np.asarray(indices, dtype=int), np.asarray(values, dtype=float)
        nz = values != 0
        row.extend([len(lower)] * int(nz.sum()))
        col.extend(indices[nz]); data.extend(values[nz])
        lower.append(low); upper.append(high)

    def flow_constraints(indices, monitored, F, b):
        eta = variables(np.zeros(len(monitored)), np.full(len(monitored), np.inf), args.penalty) if soft else None
        for t, j in enumerate(monitored):
            idx = list(indices) + ([int(eta[t])] if soft else [])
            constraint(idx, list(F[t]) + ([-1.0] if soft else []), high=case['rates'][j] - b[t])
            constraint(idx, list(-F[t]) + ([-1.0] if soft else []), high=case['rates'][j] + b[t])

    constraint(g, np.ones(ng), case['D'], case['D'])
    for _, monitored, F, b in blocks:
        flow_constraints(g, monitored, F, b)
    for s, k in enumerate(outages):
        constraint(y[s], np.ones(ng), case['D'], case['D'])
        flow_constraints(y[s], mon, H[mon], offset[mon])
        for i in range(ng):
            if i == k:
                continue
            # Upward-only envelope; also valid for the exact APR model.
            constraint([y[s, i], g[i]], [1.0, -1.0], low=0.0)
            if not exact:
                constraint([y[s, i], g[i], z[s]], [1.0, -1.0, -case['a'][i]], high=0.0)
            else:
                # x=1: proportional; x=0: saturation. Valid for gamma in [0,1].
                M = case['hi'][i]
                constraint([y[s, i], g[i], z[s], x[s, i]], [1.0, -1.0, -case['a'][i], M], high=M)
                constraint([y[s, i], g[i], z[s], x[s, i]], [1.0, -1.0, -case['a'][i], -M], low=-M)
                constraint([g[i], z[s], x[s, i]], [1.0, case['a'][i], M], low=M)
                constraint([y[s, i], x[s, i]], [1.0, M], low=M)
    A = sparse.coo_matrix((data, (row, col)), shape=(len(lower), len(lb))).tocsc()
    return LinearModel(A, np.array(lower), np.array(upper), np.array(lb), np.array(ub),
                       np.array(integrality), np.array(cost), g, y, z, x, float(case['gen'].c0.sum()))


def solve_model(model, weight, args):
    t = perf_counter()
    q = model.c0.copy(); q[model.z] += weight
    scale = float(max(1.0, max0(np.abs(q))))
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
        violation = max(max0(model.lower - model.A @ vector), max0(model.A @ vector - model.upper),
                        max0(model.lb[power] - vector[power]), max0(vector[power] - model.ub[power]))
        domain_error = max(max0(model.lb[dimensionless] - vector[dimensionless]),
                           max0(vector[dimensionless] - model.ub[dimensionless]))
        int_error = max0(np.abs(vector[model.integer > 0] - np.rint(vector[model.integer > 0])))
        result['power_constraint_violation_MW'] = violation * args.base
        result['signal_and_binary_domain_error'] = domain_error
        result['integrality_error'] = int_error
        result['solver_point_verified'] = bool(violation * args.base <= args.feas_tol_mw and domain_error <= 1e-7 and int_error <= 1e-6)
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
    overloads += [np.maximum(np.abs(H[mon] @ y + offset[mon]) - case['rates'][mon], 0.0) for y in outputs]
    eta = sum(float(e.sum()) for e in overloads)
    generation = float(case['gen'].c1.to_numpy(float) @ (g * case['base']) + case['gen'].c0.sum())
    bound_error = max(max0(case['lo'] - g), max0(g - case['hi']))
    for k, y in zip(outages, outputs):
        survivors = np.arange(len(g)) != k
        bound_error = max(bound_error, max0(case['lo'][survivors] - y[survivors]),
                          max0(y[survivors] - case['hi'][survivors]), abs(y[k]))
    return dict(operating_objective=generation + penalty * eta, generation_cost=generation,
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
    return p


def main():
    args = parser().parse_args()
    if not np.isfinite(args.gamma) or not 0 <= args.gamma <= 1:
        raise ValueError('gamma must be in [0,1] for this APR model and its exact MILP linearization')
    for key in ('demand_scale','penalty','time_limit','apr_tol_mw','feas_tol_mw'):
        if not np.isfinite(getattr(args,key)) or getattr(args,key) <= 0: raise ValueError(f'{key} must be positive and finite')
    if args.threads < 1 or args.fixed_run < 1 or not np.isfinite(args.mip_gap) or args.mip_gap < 0:
        raise ValueError('Invalid threads, fixed-run, or MIP gap')
    if any(not np.isfinite(w) or w < 0 for w in args.weights): raise ValueError('Weights must be finite and nonnegative')
    args.backend = args.solver if args.solver != 'auto' else ('gurobi' if importlib.util.find_spec('gurobipy') else 'highs')
    started = perf_counter(); path = resolve_case(args.case, args.case_dir)
    case = load_case(path, args.demand_scale); case['a'] = args.gamma * (case['hi'] - case['lo'])
    args.base = case['base']; outages = select_outages(case, args.generator_contingencies)
    if len(case['lo']) < 2: raise ValueError('Generator-contingency APR experiments require a surviving generator')
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    excluded = case['branch'].iloc[case['excluded']].copy()
    excluded['reason'] = 'islanding outage: excluded from LODF formulation'
    excluded.to_csv(output/'excluded_outages.csv', index=False)
    if len(excluded) and not args.no_line_contingencies and args.islanding == 'error':
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
    preprocess = perf_counter()-started
    metadata = dict(arguments=vars(args), case_path=str(path), python=sys.version, platform=platform.platform(),
                    numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,
                    input_preprocessing_seconds=preprocess, demand_MW=case['D']*case['base'],
                    generator_contingencies=[id_key(case['gen'].iloc[k].gen_ID) for k in outages],
                    line_contingencies=0 if args.no_line_contingencies else len(case['eligible']),
                    excluded_islanding=len(excluded) if not args.no_line_contingencies else 0,
                    candidate_penalty_threshold=B, threshold_applies=args.thermal_mode=='soft',
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
    print('NOTE: Only the SOFT formulation with unbounded slacks is covered by the proposed sufficient-weight argument.')
    if np.any(case['gen'].c2.to_numpy(float)!=0): print('NOTE: c2 ignored; using linear c1 generation costs.')
    if any(r['capacity_shortage_MW']>args.feas_tol_mw for r in screen):
        print('INFEASIBLE: at least one outage leaves insufficient total surviving capacity. See capacity_screen.csv; thermal slacks cannot replace generation.')
        pd.DataFrame([dict(model='capacity_screen',status='infeasible_capacity')]).to_csv(output/'summary.csv',index=False)
        return 2
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


