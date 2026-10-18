#!/usr/bin/env python3
"""Compare one extensive preventive line-only DC SCOPF with consensus ADMM.

Standalone: does not import or modify solve_scdcopf_p1.py. No generator outages,
APR, gamma, or binary variables. Each scenario is ONE line outage; one common
preventive dispatch must work in the base case and every included scenario.

Install: python -m pip install numpy pandas scipy openpyxl osqp
Run: python benchmark_line_scopf.py --case pglib_opf_case5_pjm.xlsx --repeat 3
Parallel: add --workers auto --qp-solver gurobi (or osqp).

Default: hard thermal limits, linear c1 generation costs (c2 explicitly ignored,
as in the original P1 script). --cost quadratic retains nonnegative c2 values.
--thermal-mode soft permits eta and uses penalty * SUM(eta in PER UNIT), matching
the original script's convention, NOT penalty per MW. Soft results can overload.
--islanding skip explicitly excludes bridge outages; default error prevents
silently claiming security against all lines when bridges are present.

Central: HiGHS LP via SciPy, or optional Gurobi; convex QP uses --qp-solver.
ADMM: parallelizable scenario QPs + one analytic generation-cost coordination
QP. Local QP backends: OSQP, Gurobi, or SciPy SLSQP (small-case fallback only).
All local QPs together form ONE ADMM block. Fixed rho; no central warm start.

Timing: reports model setup + iterations/solve, independent validation separately,
and end-to-end method time. Input/PTDF preprocessing is shared and reported
separately. Cold starts per repeat; subproblem warm starts only WITHIN a run.
Thread workers are actual execution, not an estimated parallel time. Dense PTDF
construction/contingency blocks are intended for first experiments, not proof of
large-grid scalability. No speedup or convergence is assumed in the report.

Outputs: summary.csv, dispatch.csv, flows.csv, excluded_outages.csv,
admm_history_runN.csv, metadata.json. Every final dispatch is checked by fresh
DC power-flow solves on each post-outage topology, independent of the LODF model.
If ADMM fails or stops early, its iterate is labeled accordingly, never optimal.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter, process_time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize


def max0(a):
    return float(np.max(a, initial=0.0))


def resolve_case(name, case_dir):
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


@dataclass
class Scenario:
    outage: int
    label: str
    monitored: np.ndarray
    F: np.ndarray
    offset: np.ndarray
    rates: np.ndarray


def scenarios(case):
    result = []
    for outage in [-1] + list(case['eligible']):
        T = case['K'] if outage == -1 else case['K'] + np.outer(case['L'][:, outage], case['K'][outage])
        mon = np.flatnonzero((case['rates'] > 0) & (np.arange(len(case['rates'])) != outage))
        label = 'base' if outage == -1 else f"line_outage_{case['branch'].iloc[outage]['line_ID']}"
        result.append(Scenario(int(outage), label, mon, T[mon] @ case['B'],
                               -(T[mon] @ case['d']), case['rates'][mon]))
    return result


def scenario_constraints(case, scene, soft):
    """QP variables [local dispatch, local eta]; l <= A v <= u."""
    ng, ne = len(case['lo']), len(scene.monitored) if soft else 0
    nv = ng + ne
    identity = sparse.eye(nv, format='csc')
    balance = sparse.csc_matrix(np.r_[np.ones(ng), np.zeros(ne)][None, :])
    F = sparse.csc_matrix(scene.F)
    if soft:
        e = sparse.eye(ne, format='csc')
        line_matrix = sparse.vstack([sparse.hstack([F, -e]), sparse.hstack([-F, -e])], format='csc')
    else:
        line_matrix = sparse.vstack([F, -F], format='csc')
    A = sparse.vstack([identity, balance, line_matrix], format='csc')
    lower = np.r_[case['lo'], np.zeros(ne), case['D'], np.full(2 * len(scene.rates), -np.inf)]
    upper = np.r_[case['hi'], np.full(ne, np.inf), case['D'],
                  scene.rates - scene.offset, scene.rates + scene.offset]
    return A, lower, upper


class QP:
    """Reusable convex QP with diagonal Hessian and changing linear objective."""
    def __init__(self, diagonal, q, A, lower, upper, backend, tol, threads=1, env=None):
        self.diagonal, self.q = diagonal, q.copy()
        self.A, self.lower, self.upper, self.backend = A, lower, upper, backend
        self.last = None
        self.tol = tol
        if backend == 'osqp':
            import osqp
            self.model = osqp.OSQP()
            self.model.setup(P=sparse.diags(diagonal, format='csc'), q=q, A=A, l=lower, u=upper,
                             verbose=False, eps_abs=tol, eps_rel=tol, max_iter=100000,
                             polishing=True, warm_starting=True)
        elif backend == 'gurobi':
            import gurobipy as gp
            self.gp = gp
            self.owns_env = env is None
            self.env = env if env is not None else create_gurobi_env()
            self.model = gp.Model(env=self.env)
            self.model.Params.Threads = threads
            self.model.Params.FeasibilityTol = max(tol, 1e-9)
            self.model.Params.OptimalityTol = max(tol, 1e-9)
            self.v = self.model.addMVar(len(q), lb=-gp.GRB.INFINITY)
            eq = np.isfinite(lower) & (lower == upper)
            upper_rows = np.isfinite(upper) & ~eq
            lower_rows = np.isfinite(lower) & ~eq
            if eq.any():
                self.model.addMConstr(A[eq], self.v, '=', upper[eq])
            if upper_rows.any():
                self.model.addMConstr(A[upper_rows], self.v, '<', upper[upper_rows])
            if lower_rows.any():
                self.model.addMConstr(A[lower_rows], self.v, '>', lower[lower_rows])
            self.model.setObjective(0.5 * self.v @ sparse.diags(diagonal) @ self.v + q @ self.v)
        else:
            # Deliberately a documented fallback, not a scalable QP implementation.
            dense = A.toarray()
            self.constraints = []
            eq = np.isfinite(lower) & (lower == upper)
            if eq.any():
                self.constraints.append(LinearConstraint(dense[eq], lower[eq], upper[eq]))
            if (~eq).any():
                self.constraints.append(LinearConstraint(dense[~eq], lower[~eq], upper[~eq]))
            # A feasible initial point avoids SLSQP line-search failures from
            # heavily overloaded initial dispatches. This extra LP is charged
            # to setup time and is NOT a warm start from the joint solution.
            up = np.isfinite(upper) & ~eq
            low = np.isfinite(lower) & ~eq
            feasible = linprog(np.zeros(len(q)),
                               A_ub=sparse.vstack([A[up], -A[low]], format='csc'),
                               b_ub=np.r_[upper[up], -lower[low]],
                               A_eq=A[eq], b_eq=upper[eq],
                               bounds=[(None, None)] * len(q), method='highs')
            if not feasible.success:
                raise RuntimeError(f'SLSQP initialization LP failed: {feasible.message}')
            self.last = feasible.x

    def solve(self, q, initial=None):
        if self.backend == 'osqp':
            self.model.update(q=q)
            result = self.model.solve()
            if result.info.status_val != 1:
                raise RuntimeError(f'OSQP subproblem not solved accurately: {result.info.status}')
            answer = result.x
        elif self.backend == 'gurobi':
            self.v.Obj = q
            self.model.optimize()
            if self.model.Status != self.gp.GRB.OPTIMAL:
                raise RuntimeError(f'Gurobi subproblem status: {self.model.Status}')
            answer = self.v.X
        else:
            start = self.last if self.last is not None else initial
            if start is None:
                start = np.zeros(len(q))
            result = minimize(lambda v: 0.5 * np.dot(self.diagonal * v, v) + q @ v,
                              start, jac=lambda v: self.diagonal * v + q,
                              method='SLSQP', constraints=self.constraints,
                              options={'ftol': 1e-11, 'maxiter': 2000})
            if not result.success:
                raise RuntimeError(f'SLSQP subproblem failed: {result.message}')
            answer = result.x
        violation = max(max0(self.lower - self.A @ answer), max0(self.A @ answer - self.upper))
        if not np.isfinite(answer).all() or violation > max(20 * self.tol, 1e-7):
            raise RuntimeError(f'QP constraint violation {violation:.3g} pu exceeds inner tolerance')
        self.last = answer.copy()
        return answer

    def close(self):
        if self.backend == 'gurobi':
            self.model.dispose()
            if self.owns_env:
                self.env.dispose()


def create_gurobi_env():
    """Called only on the main thread, including during parallel ADMM setup."""
    import gurobipy as gp
    env = gp.Env(empty=True)
    try:
        env.setParam('OutputFlag', 0)
        env.start()
    except Exception:
        env.dispose()
        raise
    return env


def available_cpus():
    """Respect the current process CPU affinity where the OS exposes it."""
    if hasattr(os, 'sched_getaffinity'):
        return max(1, len(os.sched_getaffinity(0)))
    process_count = getattr(os, 'process_cpu_count', os.cpu_count)
    return max(1, process_count() or 1)


def positive_or_auto(value):
    if str(value).lower() == 'auto':
        return 'auto'
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError('Use a positive integer or auto') from exc
    if number < 1:
        raise argparse.ArgumentTypeError('Use a positive integer or auto')
    return number


def choose_qp_backend(requested):
    if requested != 'auto':
        return requested
    for name in ('osqp', 'gurobipy'):
        if importlib.util.find_spec(name):
            return 'gurobi' if name == 'gurobipy' else name
    return 'scipy'


def economic_projection(center, diagonal, linear, lo, hi, demand):
    """Solve separable strictly convex QP with box and sum constraints."""
    left = float(np.min(diagonal * (center - hi) - linear) - 1.0)
    right = float(np.max(diagonal * (center - lo) - linear) + 1.0)
    for _ in range(100):
        multiplier = (left + right) / 2
        value = np.clip(center - (linear + multiplier) / diagonal, lo, hi)
        if value.sum() > demand:
            left = multiplier
        else:
            right = multiplier
    return np.clip(center - (linear + (left + right) / 2) / diagonal, lo, hi)


def generation_cost(case, g, quadratic):
    mw = g * case['base']
    gen = case['gen']
    return float(gen.c1.to_numpy() @ mw + gen.c0.sum() +
                 (gen.c2.to_numpy() @ (mw * mw) if quadratic else 0.0))


def assess(case, scenes, g, soft, penalty, quadratic, tol_mw, independent=True):
    """Recompute minimal eta for THIS common dispatch, never local ADMM copies."""
    records, eta_sum, maximum, outage_error = [], 0.0, 0.0, 0.0
    injection = case['B'] @ g - case['d']
    for scene in scenes:
        flow = scene.F @ g + scene.offset
        if independent:
            active = np.arange(len(case['b'])) != scene.outage
            A, b = case['A'][active], case['b'][active]
            bbus = A.T @ (b[:, None] * A)
            theta = np.zeros(len(case['d']))
            keep = case['keep']
            theta[keep] = np.linalg.solve(bbus[np.ix_(keep, keep)], injection[keep])
            physical = np.zeros(len(case['b']))
            physical[active] = b * (A @ theta)
            outage_error = max(outage_error, max0(np.abs(flow - physical[scene.monitored])) * case['base'])
            flow = physical[scene.monitored]
        overload = np.maximum(np.abs(flow) - scene.rates, 0.0)
        eta_sum += float(sum(overload))
        maximum = max(maximum, max0(overload) * case['base'])
        if independent:
            for j, f, e, rate in zip(scene.monitored, flow, overload, scene.rates):
                records.append(dict(scenario=scene.label, monitored_line_ID=case['branch'].iloc[j]['line_ID'],
                                    flow_MW=f * case['base'], limit_MW=rate * case['base'],
                                    overload_MW=e * case['base']))
    balance = abs(float(sum(g)) - case['D']) * case['base']
    bound_error = max(max0(case['lo'] - g), max0(g - case['hi'])) * case['base']
    secure = balance <= tol_mw and bound_error <= tol_mw and maximum <= tol_mw
    valid = balance <= tol_mw and bound_error <= tol_mw and (soft or maximum <= tol_mw)
    return dict(objective=generation_cost(case, g, quadratic) + penalty * eta_sum,
                generation_cost=generation_cost(case, g, quadratic), eta_sum_MW=eta_sum * case['base'],
                max_overload_MW=maximum, balance_error_MW=balance, gen_bound_error_MW=bound_error,
                lodf_vs_topology_error_MW=outage_error,
                feasible_for_mode=bool(valid), zero_overload_secure=bool(secure)), records


def solve_central(case, scenes, args, scale):
    started, cpu = perf_counter(), process_time()
    ng, soft = len(case['lo']), args.thermal_mode == 'soft'
    counts = [len(s.monitored) for s in scenes]
    ne = sum(counts) if soft else 0
    F = sparse.vstack([sparse.csc_matrix(s.F) for s in scenes], format='csc')
    offset, rates = np.concatenate([s.offset for s in scenes]), np.concatenate([s.rates for s in scenes])
    if soft:
        I = sparse.eye(ne, format='csc')
        inequalities = sparse.vstack([sparse.hstack([F, -I]), sparse.hstack([-F, -I])], format='csc')
    else:
        inequalities = sparse.vstack([F, -F], format='csc')
    rhs = np.r_[rates - offset, rates + offset]
    q = np.r_[case['gen'].c1.to_numpy(float) * case['base'], np.full(ne, args.penalty)] / scale
    balance = sparse.csc_matrix(np.r_[np.ones(ng), np.zeros(ne)][None, :])
    bounds = list(zip(case['lo'], case['hi'])) + [(0.0, None)] * ne
    qp = None
    if args.cost == 'linear' and args.lp_solver == 'highs':
        setup_seconds = perf_counter() - started
        t = perf_counter()
        result = linprog(q, A_ub=inequalities, b_ub=rhs, A_eq=balance, b_eq=[case['D']],
                         bounds=bounds, method='highs', options={'dual_feasibility_tolerance': 1e-8,
                                                               'primal_feasibility_tolerance': 1e-8})
        solve_seconds = perf_counter() - t
        vector = result.x if result.success else None
        status = 'optimal' if result.success else ('infeasible' if result.status == 2 else 'solver_failed')
        message = result.message
        backend = 'highs'
    else:
        backend = 'gurobi' if args.cost == 'linear' else args.qp_backend
        A = sparse.vstack([sparse.eye(ng + ne), balance, inequalities], format='csc')
        lower = np.r_[case['lo'], np.zeros(ne), case['D'], np.full(len(rhs), -np.inf)]
        upper = np.r_[case['hi'], np.full(ne, np.inf), case['D'], rhs]
        diagonal = np.r_[2 * case['gen'].c2.to_numpy(float) * case['base'] ** 2
                         if args.cost == 'quadratic' else np.zeros(ng), np.zeros(ne)] / scale
        qp = QP(diagonal, q, A, lower, upper, backend, args.inner_tol, threads=args.central_threads)
        setup_seconds = perf_counter() - started
        t = perf_counter()
        try:
            vector = qp.solve(q)
            status, message = 'optimal', 'Convex centralized model solved'
        except RuntimeError as exc:
            vector, status, message = None, 'solver_failed', str(exc)
        solve_seconds = perf_counter() - t
        qp.close()
    return dict(method='central', status=status, message=message, backend=backend, iterations=0,
                setup_seconds=setup_seconds, solve_seconds=solve_seconds,
                total_seconds=perf_counter() - started, cpu_seconds=process_time() - cpu,
                variables=ng + ne, thermal_inequalities=2 * len(rates), workers=1,
                solver_threads=args.central_threads if backend == 'gurobi' else None), \
        (None if vector is None else vector[:ng]), []


def solve_admm(case, scenes, args, scale):
    started, cpu = perf_counter(), process_time()
    ng, ns, soft = len(case['lo']), len(scenes), args.thermal_mode == 'soft'
    z = economic_projection((case['lo'] + case['hi']) / 2, np.ones(ng), np.zeros(ng),
                            case['lo'], case['hi'], case['D'])
    x = np.tile(z, (ns, 1))
    u = np.zeros_like(x)
    models, linear_costs, starts, environments, executors = [], [], [], [], []
    # A worker always handles the SAME group of scenarios. Models in that group
    # share one environment, exclusively used by that worker. Create/dispose all
    # Gurobi environments on the main thread, per Gurobi's threading guidance.
    groups = [list(range(w, ns, args.workers)) for w in range(args.workers)]
    try:
        if args.qp_backend == 'gurobi':
            for _ in groups:
                environments.append(create_gurobi_env())
        for j, scene in enumerate(scenes):
            A, lower, upper = scenario_constraints(case, scene, soft)
            ne = len(scene.monitored) if soft else 0
            q = np.r_[np.zeros(ng), np.full(ne, args.penalty / scale)]
            diagonal = np.r_[np.full(ng, args.rho), np.zeros(ne)]
            env = environments[j % args.workers] if environments else None
            models.append(QP(diagonal, q, A, lower, upper, args.qp_backend, args.inner_tol,
                             threads=args.qp_threads, env=env))
            linear_costs.append(q)
            starts.append(np.r_[z, np.maximum(np.abs(scene.F @ z + scene.offset) - scene.rates, 0)] if soft else z)
        if args.workers > 1:
            for _ in groups:
                pool = ThreadPoolExecutor(max_workers=1)
                executors.append(pool)
                pool.submit(lambda: None).result()  # Charge thread startup to setup.
    except Exception:
        for pool in executors:
            pool.shutdown(wait=True)
        for model in models:
            model.close()
        for env in environments:
            env.dispose()
        raise
    setup_seconds = perf_counter() - started
    history, status, message = [], 'iteration_limit', 'ADMM iteration limit reached'
    t = perf_counter()
    iteration = 0
    try:
        for iteration in range(1, args.max_iter + 1):
            def update_group(indices):
                values = []
                for j in indices:
                    q = linear_costs[j].copy()
                    q[:ng] = -args.rho * (z - u[j])
                    values.append((j, models[j].solve(q, starts[j])[:ng]))
                return values
            if executors:
                futures = [pool.submit(update_group, group) for pool, group in zip(executors, groups)]
                # Wait for ALL workers even if one fails before cleaning up.
                results, failure = [], None
                for future in futures:
                    try:
                        results.extend(future.result())
                    except Exception as exc:
                        failure = failure or exc
                if failure is not None:
                    raise RuntimeError(f'Parallel QP update failed: {failure}') from failure
            else:
                results = update_group(groups[0])
            for j, value in results:
                x[j] = value
            previous = z.copy()
            curvature = (2 * case['gen'].c2.to_numpy(float) * case['base'] ** 2 / scale
                         if args.cost == 'quadratic' else np.zeros(ng))
            hessian = args.rho * ns + curvature
            mean = (x + u).mean(axis=0)
            center = args.rho * ns * mean / hessian
            z = economic_projection(center, hessian, case['gen'].c1.to_numpy(float) * case['base'] / scale,
                                    case['lo'], case['hi'], case['D'])
            u += x - z
            primal = float(np.linalg.norm(x - z))
            dual = float(args.rho * np.sqrt(ns) * np.linalg.norm(z - previous))
            eps_p = np.sqrt(ns * ng) * args.abs_tol + args.rel_tol * max(np.linalg.norm(x), np.sqrt(ns) * np.linalg.norm(z))
            eps_d = np.sqrt(ns * ng) * args.abs_tol + args.rel_tol * args.rho * np.linalg.norm(u)
            check, _ = assess(case, scenes, z, soft, args.penalty, args.cost == 'quadratic',
                              args.feas_tol_mw, independent=False)
            history.append(dict(iteration=iteration, primal_residual_pu=primal, dual_residual_scaled=dual,
                                primal_threshold=eps_p, dual_threshold=eps_d,
                                objective_at_common_dispatch=check['objective'],
                                max_overload_MW=check['max_overload_MW'], elapsed_seconds=perf_counter() - t))
            if args.log_every and (iteration == 1 or iteration % args.log_every == 0):
                print(f"  ADMM {iteration}: r={primal:.3g}, s={dual:.3g}, common-dispatch overload="
                      f"{check['max_overload_MW']:.6g} MW", flush=True)
            if primal <= eps_p and dual <= eps_d and check['feasible_for_mode']:
                status, message = 'converged', 'Residual tests and common-dispatch feasibility passed'
                break
    except RuntimeError as exc:
        status, message = 'subproblem_failed', str(exc)
    finally:
        for pool in executors:
            pool.shutdown(wait=True)
        for model in models:
            model.close()
        for env in environments:
            env.dispose()
    solve_seconds = perf_counter() - t
    final = history[-1] if history else {}
    return dict(method='admm', status=status, message=message, backend=args.qp_backend, iterations=iteration,
                setup_seconds=setup_seconds, solve_seconds=solve_seconds,
                total_seconds=perf_counter() - started, cpu_seconds=process_time() - cpu,
                variables=sum(len(q) for q in linear_costs) + ng,
                thermal_inequalities=2 * sum(len(s.monitored) for s in scenes), workers=args.workers,
                solver_threads=args.qp_threads if args.qp_backend == 'gurobi' else 1,
                primal_residual_pu=final.get('primal_residual_pu'),
                dual_residual_scaled=final.get('dual_residual_scaled')), z, history


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--case', required=True)
    p.add_argument('--case-dir', default='../excel_outputs')
    p.add_argument('--output-dir', default='output_line_scopf')
    p.add_argument('--method', choices=['central', 'admm', 'both'], default='both')
    p.add_argument('--cost', choices=['linear', 'quadratic'], default='linear')
    p.add_argument('--thermal-mode', choices=['hard', 'soft'], default='hard')
    p.add_argument('--penalty', type=float, default=1500.0, help='Cost per unit eta (NOT per MW) in soft mode')
    p.add_argument('--islanding', choices=['error', 'skip'], default='skip')
    p.add_argument('--lp-solver', choices=['highs', 'gurobi'], default='highs')
    p.add_argument('--qp-solver', choices=['auto', 'osqp', 'gurobi', 'scipy'], default='auto')
    p.add_argument('--rho', type=float, default=1.0, help='Fixed ADMM rho for scaled objective, per-unit dispatch')
    p.add_argument('--abs-tol', type=float, default=1e-7)
    p.add_argument('--rel-tol', type=float, default=1e-7)
    p.add_argument('--inner-tol', type=float, default=1e-9)
    p.add_argument('--feas-tol-mw', type=float, default=1e-4)
    p.add_argument('--max-iter', type=int, default=5000)
    p.add_argument('--workers', type=positive_or_auto, default=1,
                   help='Parallel scenario workers; auto = min(scenarios, CPU budget / QP threads)')
    p.add_argument('--cpu-budget', type=positive_or_auto, default='auto',
                   help='Logical CPU budget for automatic worker allocation; default uses process affinity')
    p.add_argument('--qp-threads', type=int, default=1,
                   help='Gurobi threads per scenario QP; 1 recommended for small QPs')
    p.add_argument('--central-threads', type=positive_or_auto, default=1,
                   help='Gurobi threads for the central LP/QP; auto uses the same CPU budget')
    p.add_argument('--repeat', type=int, default=1)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--demand-scale', type=float, default=1.0)
    return p


def main():
    args = parser().parse_args()
    for name in ('rho', 'abs_tol', 'rel_tol', 'inner_tol', 'feas_tol_mw', 'penalty', 'demand_scale'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f'{name} must be positive and finite')
    if min(args.repeat, args.qp_threads, args.max_iter) < 1 or args.log_every < 0:
        raise ValueError('repeat, qp-threads, max-iter must be positive; log-every nonnegative')
    path = resolve_case(args.case, args.case_dir)
    t = perf_counter()
    case = load_case(path, args.demand_scale)
    if args.cost == 'quadratic' and np.any(case['gen'].c2.to_numpy(float) < 0):
        raise ValueError('Quadratic generation costs must be convex: c2 >= 0')
    scenes = scenarios(case)
    preprocessing = perf_counter() - t
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    excluded = case['branch'].iloc[case['excluded']].copy()
    excluded['reason'] = 'islanding outage: LODF not applicable'
    excluded.to_csv(output / 'excluded_outages.csv', index=False)
    if len(excluded) and args.islanding == 'error':
        raise ValueError(f'{len(excluded)} islanding line outages found. See excluded_outages.csv. '
                         'Use --islanding skip only to study the non-islanding subset explicitly.')
    args.qp_backend = choose_qp_backend(args.qp_solver)
    cpu_count = available_cpus()
    args.cpu_budget = min(cpu_count, cpu_count if args.cpu_budget == 'auto' else args.cpu_budget)
    if args.qp_backend != 'gurobi' and args.qp_threads != 1:
        raise ValueError('--qp-threads applies only to Gurobi; use 1 for other backends')
    if args.qp_threads > args.cpu_budget:
        raise ValueError('--qp-threads exceeds the CPU budget')
    args.workers_requested = args.workers
    maximum_workers = min(len(scenes), args.cpu_budget // args.qp_threads)
    args.workers = maximum_workers if args.workers == 'auto' else min(args.workers, maximum_workers)
    args.central_threads = args.cpu_budget if args.central_threads == 'auto' else min(args.central_threads, args.cpu_budget)
    if args.qp_backend == 'scipy' and (args.method != 'central' or args.cost == 'quadratic'):
        print('NOTE: SciPy SLSQP fallback is for small-case verification. Install osqp or use --qp-solver gurobi for timing studies.')
    c1 = case['gen'].c1.to_numpy(float) * case['base']
    curvature = case['gen'].c2.to_numpy(float) * case['base'] ** 2 if args.cost == 'quadratic' else np.zeros(len(c1))
    scale = float(max(1.0, max0(np.abs(c1)), max0(np.abs(curvature)), args.penalty if args.thermal_mode == 'soft' else 0.0))
    print(f"Case: {path.name}; demand={case['D'] * case['base']:.6g} MW; generators={len(c1)}")
    print(f"Line contingencies={len(scenes)-1}; excluded islanding={len(excluded)}; thermal mode={args.thermal_mode}")
    print(f"Generation cost={args.cost}; shared preprocessing={preprocessing:.6f} s; objective scale={scale:g}")
    print(f"CPU budget={args.cpu_budget}; ADMM workers={args.workers}; QP threads={args.qp_threads}; "
          f"central Gurobi threads={args.central_threads}")
    if args.cost == 'linear' and np.any(case['gen'].c2.to_numpy(float) != 0):
        print('NOTE: nonzero c2 is ignored in linear mode, matching your existing linear P1 objective.')
    rows, dispatch_rows, flow_rows = [], [], []
    methods = ['central', 'admm'] if args.method == 'both' else [args.method]
    for repeat in range(1, args.repeat + 1):
        # Alternate order to reduce systematic first-method timing bias.
        order = methods if repeat % 2 else methods[::-1]
        for method in order:
            print(f'Run {repeat}: {method}', flush=True)
            fn = solve_central if method == 'central' else solve_admm
            method_start = perf_counter()
            try:
                result, g, history = fn(case, scenes, args, scale)
            except (RuntimeError, ImportError) as exc:
                result = dict(method=method, status='solver_failed', message=str(exc),
                              backend=args.lp_solver if method == 'central' and args.cost == 'linear' else args.qp_backend,
                              total_seconds=perf_counter() - method_start, workers=args.workers if method == 'admm' else 1)
                g, history = None, []
            result.update(run=repeat, case=path.name, cost=args.cost, thermal_mode=args.thermal_mode,
                          demand_scale=args.demand_scale, line_contingencies=len(scenes) - 1,
                          excluded_islanding=len(excluded), shared_preprocessing_seconds=preprocessing)
            if g is not None:
                tv = perf_counter()
                check, flows = assess(case, scenes, g, args.thermal_mode == 'soft', args.penalty,
                                      args.cost == 'quadratic', args.feas_tol_mw)
                result.update(check)
                result['validation_seconds'] = perf_counter() - tv
                result['end_to_end_seconds'] = result['total_seconds'] + result['validation_seconds']
                if check['lodf_vs_topology_error_MW'] > max(args.feas_tol_mw, 1e-6):
                    result['status'] = 'validation_failed'
                    result['message'] += '; LODF and independent topology flows disagree'
                elif not check['feasible_for_mode'] and result['status'] in ('optimal', 'converged'):
                    result['status'] = 'validation_failed'
                for j, value in enumerate(g):
                    dispatch_rows.append(dict(run=repeat, method=method, gen_ID=case['gen'].iloc[j]['gen_ID'],
                                              bus_i=case['gen'].iloc[j]['bus_i'], generation_MW=value * case['base']))
                flow_rows.extend(dict(run=repeat, method=method, **row) for row in flows)
            else:
                result['end_to_end_seconds'] = result['total_seconds']
            rows.append(result)
            if history:
                pd.DataFrame(history).to_csv(output / f'admm_history_run{repeat}.csv', index=False)
            print(f"  status={result['status']}; total={result['total_seconds']:.6f} s; "
                  f"objective={result.get('objective', float('nan')):.8f}; "
                  f"max overload={result.get('max_overload_MW', float('nan')):.6g} MW", flush=True)
            print(f"  {result['message']}")
    for row in rows:
        reference = next((r for r in rows if r['run'] == row['run'] and r['method'] == 'central'
                          and r['status'] == 'optimal' and r.get('feasible_for_mode')), None)
        if reference and row.get('objective') is not None:
            row['objective_difference_vs_central'] = row['objective'] - reference['objective']
            row['relative_objective_difference'] = row['objective_difference_vs_central'] / max(1.0, abs(reference['objective']))
    summary = pd.DataFrame(rows)
    summary.to_csv(output / 'summary.csv', index=False)
    pd.DataFrame(dispatch_rows).to_csv(output / 'dispatch.csv', index=False)
    pd.DataFrame(flow_rows).to_csv(output / 'flows.csv', index=False)
    import scipy
    backend_versions = {}
    for module_name in ('osqp', 'gurobipy'):
        if module_name in sys.modules:
            module = sys.modules[module_name]
            backend_versions[module_name] = (str(module.gurobi.version()) if module_name == 'gurobipy'
                                             else getattr(module, '__version__', 'unknown'))
    metadata = dict(arguments=vars(args), python=sys.version, platform=platform.platform(),
                    numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                    backend_versions=backend_versions,
                    objective_scale=scale, case_path=str(path), generation_contingencies=0,
                    timing='Cold runs; shared input/PTDF preprocessing excluded from per-method total; validation separate',
                    parallelism='Fixed scenario groups per worker; Gurobi environment per worker; central HiGHS uses its own default threading',
                    security_scope='Base case plus all NON-ISLANDING single-line outages; rateA <= 0 means unmonitored')
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    cols = [c for c in ['run', 'method', 'status', 'workers', 'solver_threads', 'total_seconds', 'iterations', 'objective',
                        'max_overload_MW', 'zero_overload_secure', 'relative_objective_difference'] if c in summary]
    print(summary[cols].to_string(index=False))
    print(f'Saved results: {output.resolve()}')
    if any(r['status'] not in ('optimal', 'converged') for r in rows):
        return 2
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, ImportError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
