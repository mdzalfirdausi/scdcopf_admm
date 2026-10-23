"""Compare APR-weight LP bisection with the original cost-minimizing SC-DC-OPF MIP.

Self-contained; no check_apr_eta_limits.py import is needed.
All methods use the same zero-phase-shift, soft-limit network and linear costs.
Optional --with-lp-start also solves a fresh MIP using the verified LP candidate.
"""
import argparse
from array import array
import shutil
from dataclasses import dataclass
from dataclasses import replace
import copy
import csv
from itertools import product
import importlib.util
import json
from pathlib import Path
import platform
import sys
import faulthandler
import traceback

if __name__ == "__main__":
    faulthandler.enable(all_threads=True)
from contextlib import contextmanager, ExitStack
from tempfile import NamedTemporaryFile
from time import perf_counter
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

IMPLEMENTATION_VERSION = '2026-10-22-apr-comparison-checkpoint-diagnostic-v2'

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
    eta_delta: object = None
    nominal_flow: object = None


class LineFlowBlocks:
    """Re-iterable contingency blocks without retaining dense outage matrices."""
    def __init__(self, case, H, offset, monitored, include_lines):
        self.case, self.H, self.offset = case, H, offset
        self.monitored, self.include_lines = monitored, include_lines

    def __iter__(self):
        H, offset, mon, case = self.H, self.offset, self.monitored, self.case
        yield -1, mon, H[mon], offset[mon]
        if self.include_lines:
            for ell in case['eligible']:
                keep = mon[mon != ell]
                lodf = case['L'][keep, ell]
                yield int(ell), keep, H[keep]+np.outer(lodf,H[ell]), offset[keep]+lodf*offset[ell]


def network_blocks(case, include_lines):
    """Positive-injection PTDF flows; zero phase shifts; lazy outage blocks."""
    H = case['K'] @ case['B']
    offset = -case['K'] @ case['d']
    mon = np.flatnonzero(case['rates'] > 0)
    return H, offset, mon, LineFlowBlocks(case,H,offset,mon,include_lines)


def line_flow_states(case, g, include_lines):
    """Evaluate line outages from nominal flows without dense outage matrices."""
    mon = np.flatnonzero(case['rates'] > 0)
    f = case['K'] @ (case['B'] @ g-case['d'])
    yield -1, mon, f[mon]
    if include_lines:
        for ell in case['eligible']:
            keep = mon[mon != ell]
            yield int(ell), keep, f[keep]+case['L'][keep,ell]*f[ell]


class SparseRows:
    """Build CSR directly from typed buffers, avoiding Python COO triplet lists."""
    def __init__(self):
        self.values = array('d')
        self.columns = array('q')
        self.starts = array('q', [0])

    def append(self, indices, values):
        indices, values = np.asarray(indices,dtype=np.int64), np.asarray(values,dtype=np.float64)
        nonzero = values != 0
        self.columns.frombytes(np.ascontiguousarray(indices[nonzero]).tobytes())
        self.values.frombytes(np.ascontiguousarray(values[nonzero]).tobytes())
        self.starts.append(len(self.values))

    def matrix(self, nvars):
        return sparse.csr_matrix((np.frombuffer(self.values,dtype=np.float64),
                                  np.frombuffer(self.columns,dtype=np.int64),
                                  np.frombuffer(self.starts,dtype=np.int64)),
                                 shape=(len(self.starts)-1,nvars),copy=False)


def build_model(case, outages, args, exact=False, fixed=None):
    ng, nk = len(case['lo']), len(outages)
    soft = args.thermal_mode == 'soft'
    H, offset, mon, blocks = network_blocks(case, not args.no_line_contingencies)
    lb, ub, cost = array('d'), array('d'), array('d')
    integrality = array('b')

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

    coefficients = SparseRows()
    lower, upper = array('d'), array('d')
    eta_line, eta_unit, eta_records = [], [], []

    def constraint(indices, values, low=-np.inf, high=np.inf):
        coefficients.append(indices, values)
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
            if F is None:
                # Factor line-outage flows through the shared nominal f variables.
                idx, values = [int(nominal_flow[j])], [1.0]
                if state != -1:
                    idx.append(int(nominal_flow[state]))
                    values.append(float(case['L'][j,state]))
            else:
                idx, values = list(indices), list(F[t])
            if soft:
                idx.append(int(eta[t]))
            # LaTeX: f_{s,j}=F_{s,j}u_s+b_{s,j}\le\bar f_j+\eta_{s,j}.
            constraint(idx, values + ([-1.0] if soft else []), high=case['rates'][j] - b[t])
            # LaTeX: -f_{s,j}\le\bar f_j+\eta_{s,j}.
            # Together: -\bar f_j-\eta_{s,j}\le f_{s,j}\le\bar f_j+\eta_{s,j}.
            constraint(idx, [-v for v in values] + ([-1.0] if soft else []), high=case['rates'][j] + b[t])

    # LaTeX: \mathbf{1}^T g = D (nominal power balance).
    constraint(g, np.ones(ng), case['D'], case['D'])
    # f0 = H*g + offset. Eliminating these auxiliary variables recovers the
    # original PTDF/LODF inequalities exactly; all contingencies are retained.
    nominal_flow = variables(np.full(len(case['rates']),-np.inf),
                             np.full(len(case['rates']),np.inf))
    for j in range(len(case['rates'])):
        constraint(np.r_[g,nominal_flow[j]],np.r_[-H[j],1.0],offset[j],offset[j])
    flow_constraints(None,mon,None,np.zeros(len(mon)),state=-1)
    if not args.no_line_contingencies:
        for ell in case['eligible']:
            monitored = mon[mon != ell]
            flow_constraints(None,monitored,None,np.zeros(len(monitored)),state=int(ell))
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
    print(f'Assembling sparse model: rows={len(lower):,}; variables={len(lb):,}; '
          f'nonzeros={len(coefficients.values):,} (typed CSR buffers)',flush=True)
    A = coefficients.matrix(len(lb))
    return LinearModel(A, np.array(lower), np.array(upper), np.array(lb), np.array(ub),
                       np.array(integrality), np.array(cost), g, y, z, x, float(case['gen'].c0.sum()),
                       eta_line=np.array(eta_line, dtype=int), eta_unit=np.array(eta_unit, dtype=int),
                       eta_records=eta_records, nominal_flow=nominal_flow)


class GurobiSession:
    """Keep one native Gurobi model; subsequent solves update only its objective.

    Constraint arrays must be shared by identity with the original LinearModel.
    Use as a context manager so the model and environment close on every exit.
    """

    def __init__(self, model, args):
        import gurobipy as gp
        started = perf_counter()
        self.gp = gp
        self.structure = model
        self.solve_count = 0
        self.closed = False
        self.resources = ExitStack()
        try:
            env = self.resources.enter_context(gp.Env(empty=True))
            env.setParam('OutputFlag', int(args.tee))
            env.start()
            gm = self.resources.enter_context(gp.Model(env=env))
            self.gm = gm
            gm.Params.Threads = args.threads
            gm.Params.FeasibilityTol = 1e-9
            gm.Params.OptimalityTol = 1e-9
            gm.Params.MIPGap = args.mip_gap
            gm.Params.TimeLimit = args.time_limit
            gm.ModelSense = gp.GRB.MINIMIZE
            self.v = gm.addMVar(len(model.c0),
                lb=np.where(np.isfinite(model.lb), model.lb, -gp.GRB.INFINITY),
                ub=np.where(np.isfinite(model.ub), model.ub, gp.GRB.INFINITY),
                vtype=np.where(model.integer > 0, gp.GRB.BINARY, gp.GRB.CONTINUOUS))
            eq = np.isfinite(model.lower) & (model.lower == model.upper)
            up = np.isfinite(model.upper) & ~eq
            low = np.isfinite(model.lower) & ~eq
            if eq.any(): gm.addMConstr(model.A[eq], self.v, '=', model.upper[eq])
            if up.any(): gm.addMConstr(model.A[up], self.v, '<', model.upper[up])
            if low.any(): gm.addMConstr(model.A[low], self.v, '>', model.lower[low])
            gm.update()
            self.build_seconds = perf_counter()-started
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if not self.closed:
            self.closed = True
            self.resources.close()

    def solve(self, model, qscaled, scale):
        if self.closed:
            raise RuntimeError('Gurobi session has already been closed')
        if any(getattr(model, name) is not getattr(self.structure, name)
               for name in ('A', 'lower', 'upper', 'lb', 'ub', 'integer')):
            raise ValueError('Persistent Gurobi session allows objective changes only')
        gp, gm = self.gp, self.gm
        # No reset, variable creation, or constraint upload between weight tests.
        # optimize() applies the pending Obj attribute changes automatically.
        self.v.setAttr(gp.GRB.Attr.Obj, qscaled)
        self.solve_count += 1
        gm.optimize()
        if gm.Status == gp.GRB.INF_OR_UNBD:
            # Exceptional diagnostic retry, matching the original behavior.
            gm.Params.DualReductions = 0
            gm.reset()
            gm.optimize()
        status = {gp.GRB.OPTIMAL: 'optimal', gp.GRB.INFEASIBLE: 'infeasible',
                  gp.GRB.UNBOUNDED: 'unbounded', gp.GRB.INF_OR_UNBD: 'infeasible_or_unbounded',
                  gp.GRB.TIME_LIMIT: 'limit'}.get(gm.Status, f'gurobi_status_{gm.Status}')
        info = dict(status=status, message=f'Gurobi status {gm.Status}',
                    gurobi_session_solve_index=self.solve_count,
                    gurobi_model_reused=self.solve_count > 1,
                    solver_runtime_seconds=float(gm.Runtime),
                    simplex_iterations=float(gm.IterCount))
        vector = np.array(self.v.X, copy=True) if gm.SolCount else None
        bound = None
        if model.integer.any():
            try:
                if np.isfinite(gm.ObjBound):
                    bound = float(gm.ObjBound)*scale + model.constant
            except gp.GurobiError:
                pass
            info['mip_relative_gap'] = float(gm.MIPGap) if gm.SolCount else None
            info['mip_node_count'] = float(gm.NodeCount)
        elif status == 'optimal':
            bound = float(gm.ObjVal)*scale + model.constant
        return info, vector, bound


def solve_model(model, weight, args, gurobi_session=None):
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
            result['mip_relative_gap'] = getattr(r,'mip_gap',None)
            result['mip_node_count'] = getattr(r,'mip_node_count',None)
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
        if gurobi_session is None:
            with GurobiSession(model, args) as temporary_session:
                info, vector, bound = temporary_session.solve(model, qscaled, scale)
        else:
            info, vector, bound = gurobi_session.solve(model, qscaled, scale)
        result.update(info)
    result['total_seconds'] = perf_counter() - t
    result['solver_best_bound_penalized'] = bound
    if vector is not None:
        result['solver_penalized_objective'] = float(q @ vector + model.constant)
        dimensionless = np.zeros(len(vector), dtype=bool)
        dimensionless[model.z] = True
        if model.eta_delta is not None:
            dimensionless[model.eta_delta] = True
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
    overloads = [np.maximum(np.abs(flows)-case['rates'][monitored],0.0)
                 for _,monitored,flows in line_flow_states(case,g,include_lines)]
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


def eta_limit_diagnostics(case, outages, g, outputs, args, prefix='', record_sink=None):
    """Check limits against actual flows, independently of solver slack values."""
    line_ids = tuple(
        id_key(value)
        for value in case['branch']['line_ID'].to_numpy(copy=True)
    )
    gen_ids = tuple(
        id_key(value)
        for value in case['gen']['gen_ID'].to_numpy(copy=True)
    )
    H, offset, mon, blocks = network_blocks(case, not args.no_line_contingencies)
    states = [('line',ell,monitored,flows)
              for ell,monitored,flows in line_flow_states(case,g,not args.no_line_contingencies)]
    states.extend(('unit', int(k), mon, H[mon]@y+offset[mon])
                  for k, y in zip(outages, outputs))
    records, sums, violation = [], {'line': 0.0, 'unit': 0.0}, 0.0
    for family, state, monitored, flows in states:
        eta_mw = np.maximum(np.abs(flows)-case['rates'][monitored], 0.0)*case['base']
        limits = individual_eta_cap_pu(case, family, monitored, args)*case['base']
        sums[family] += float(eta_mw.sum())
        violation = max(violation, max0(eta_mw-limits))
        kind = 'generator_outage' if family=='unit' else 'nominal' if state==-1 else 'line_outage'
        outage_id = (
            gen_ids[int(state)] if family == 'unit'
            else None if state == -1
            else line_ids[int(state)]
        )
        for j, flow, eta, limit in zip(monitored, flows, eta_mw, limits):
            record = dict(phase='recovered' if prefix else 'provisional', family=family,
                                state_kind=kind, outage_ID=outage_id,
                                line_ID=line_ids[int(j)],
                                flow_MW=float(flow)*case['base'],
                                rating_MW=float(case['rates'][j])*case['base'],
                                minimum_eta_MW=float(eta),
                                individual_eta_cap_MW=float(limit) if np.isfinite(limit) else None,
                                individual_violation_MW=max(0.0, float(eta-limit)))
            if record_sink is None:
                records.append(record)
            else:
                record_sink(record)
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


def minimum_eta_fraction_model(model, case):
    """min delta >= 0, eta_{k,j} - rating_j*delta <= 0. No cost caps.

    delta is dimensionless. The LP version retains the APR envelope; the
    optional exact version retains all original APR binaries/constraints.
    """
    if not model.eta_records:
        raise ValueError('Minimum-fraction diagnostic requires monitored lines and soft thermal slacks')
    n = len(model.c0)
    count = len(model.eta_records)
    rr = np.repeat(np.arange(count), 2)
    cc = np.array([[index, n] for index, family, state, j in model.eta_records]).ravel()
    vv = np.array([[1.0, -case['rates'][j]] for index, family, state, j in model.eta_records]).ravel()
    extra = sparse.coo_matrix((vv, (rr, cc)), shape=(count, n+1)).tocsc()
    old = sparse.hstack([model.A, sparse.csc_matrix((model.A.shape[0], 1))], format='csc')
    row_power = np.ones(model.A.shape[0], dtype=bool) if model.row_is_power is None else model.row_is_power
    return replace(model, A=sparse.vstack([old, extra], format='csc'),
                   lower=np.r_[model.lower, np.full(count, -np.inf)],
                   upper=np.r_[model.upper, np.zeros(count)],
                   lb=np.r_[model.lb, 0.0], ub=np.r_[model.ub, np.inf],
                   integer=np.r_[model.integer, 0], c0=np.r_[np.zeros(n), 1.0], constant=0.0,
                   original_cost=np.r_[model.c0, 0.0], original_constant=model.constant,
                   row_is_power=np.r_[row_power, np.ones(count, dtype=bool)], eta_delta=n)


def run_minimum_eta_fraction(case, outages, args, fixed, output, metadata, preprocess):
    started = perf_counter()
    rows, all_checks, all_details, all_dispatch, all_eta = [], [], [], [], []
    metadata.update(diagnostic='minimum_uniform_eta_fraction',
                    optimized_objective='delta, with eta_{state,line} <= delta*rating_line',
                    objective_units='dimensionless_fraction_of_each_line_rating',
                    cost_caps_enabled=False, threshold_applies=False,
                    bound_scope='fixed_dispatch' if fixed is not None else 'full_preventive_dispatch',
                    warning='LP optimum is a lower bound on the allowance required by exact APR; it is not an operating-cost optimum.')
    print('DIAGNOSTIC: minimize uniform overload fraction delta. NO cost caps; physical generation bounds remain.')
    print('LP objective is delta, not sum(splus+sminus) and not operating cost.')
    jobs = [('eta_fraction_LP', False)]
    if args.compare_exact:
        jobs.append(('eta_fraction_exact_APR', True))
    for tag, exact in jobs:
        t = perf_counter()
        model = minimum_eta_fraction_model(build_model(case, outages, args, exact, fixed), case)
        build_seconds = perf_counter()-t
        print('Solving '+tag, flush=True)
        result, vector = solve_model(model, 0.0, args)
        result.update(model=tag, objective_kind='minimum_uniform_eta_fraction',
                      optimized_objective_units='dimensionless', model_build_seconds=build_seconds,
                      gamma=args.gamma, demand_scale=args.demand_scale,
                      bound_scope=metadata['bound_scope'],
                      uniform_fraction_solver_lower_bound=result['solver_best_bound_penalized'])
        if vector is not None:
            t = perf_counter()
            fraction = float(vector[model.eta_delta])
            metrics, checks, details, dispatch = analyze(case, outages, model, vector, args, tag, 0.0)
            result.update(metrics)
            result.update(optimized_uniform_fraction=fraction, optimized_uniform_percent=100*fraction,
                          penalized_objective=fraction, optimized_objective=fraction)
            check_args = copy.copy(args)
            check_args.eta_line_fraction = fraction
            check_args.eta_unit_fraction = fraction
            diagnostics, eta_records = eta_limit_diagnostics(
                case, outages, vector[model.g], vector[model.y], check_args)
            result.update(diagnostics)
            result['provisional_APR_and_eta_feasible'] = bool(result['solver_point_verified'] and
                result['provisional_original_APR_feasible'] and result['eta_limits_satisfied'])
            if metrics['recovery_success']:
                recovered = [recover(case, vector[model.g], k,
                    min(args.feas_tol_mw/case['base']/100, 1e-10))[0] for k in outages]
                rec_diagnostics, rec_records = eta_limit_diagnostics(
                    case, outages, vector[model.g], recovered, check_args, prefix='recovered_')
                result.update(rec_diagnostics); eta_records.extend(rec_records)
                result['recovered_APR_and_eta_feasible'] = bool(result['recovered_original_APR_feasible'] and
                    result['recovered_eta_limits_satisfied'])
            for record in eta_records:
                record.update(model=tag, optimized_uniform_fraction=fraction)
                record['relative_overload_fraction'] = record['minimum_eta_MW']/record['rating_MW']
                record['at_fraction_limit'] = bool(abs(record['minimum_eta_MW']-fraction*record['rating_MW']) <= args.feas_tol_mw)
            recovered_ratios = [record['relative_overload_fraction'] for record in eta_records if record['phase']=='recovered']
            if recovered_ratios:
                result['recovered_required_uniform_fraction'] = max(recovered_ratios)
            all_checks.extend(checks); all_details.extend(details); all_dispatch.extend(dispatch); all_eta.extend(eta_records)
            result['validation_seconds'] = perf_counter()-t
            if not result['solver_point_verified']:
                result['status'] = 'validation_failed'
            print(f'  status={result["status"]}; minimum fraction={fraction:.10g} ({100*fraction:.6f}%); '
                  f'APR mismatch={result["max_APR_L1_error_MW"]:.6g} MW; '
                  f'APR+eta verified={result["provisional_APR_and_eta_feasible"]}', flush=True)
        else:
            print(f'  status={result["status"]}; {result.get("message", "")}', flush=True)
        result['pipeline_seconds'] = preprocess+build_seconds+result['total_seconds']+result.get('validation_seconds', 0.0)
        rows.append(result)
        pd.DataFrame(rows).to_csv(output/'summary.csv', index=False)
        pd.DataFrame(all_checks).to_csv(output/'apr_checks.csv', index=False)
        pd.DataFrame(all_details).to_csv(output/'apr_dispatch.csv', index=False)
        pd.DataFrame(all_dispatch).to_csv(output/'dispatch.csv', index=False)
        if all_eta:
            table = pd.DataFrame(all_eta)
            table.to_csv(output/'eta_limit_checks.csv', index=False)
            table[table.phase=='provisional'].sort_values('relative_overload_fraction', ascending=False).to_csv(output/'critical_lines.csv', index=False)
    metadata['diagnostic_seconds_including_all_requested_solves'] = preprocess+perf_counter()-started
    metadata['results'] = rows
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    columns = ['model','status','optimized_uniform_fraction','optimized_uniform_percent','operating_objective',
               'max_APR_L1_error_MW','provisional_APR_and_eta_feasible','recovered_APR_and_eta_feasible','total_seconds']
    summary = pd.DataFrame(rows)
    print(summary[[column for column in columns if column in summary]].to_string(index=False))
    print('Saved: '+str(output.resolve()))
    print('The LP allowance is a lower bound for exact APR. Check APR consistency and recovery limits separately.')
    return 0 if all(row['status']=='optimal' for row in rows) else 2


def normalized_slack_objective(
    model, case, weight, thermal_weight=1.0, normalization='none'
):
    """Minimize original MIP objective + weight * sum(splus+sminus).

    Retains generation costs, M_eta thermal penalties, and the constant.
    Response slacks are unnormalized.
    """
    if normalization != 'none':
        raise ValueError("Use normalization='none' for this formulation.")

    if thermal_weight != 1.0:
        raise ValueError(
            "This formulation retains the original thermal penalties. "
            "Use thermal_weight=1.0."
        )

    if not np.isfinite(weight) or weight < 0:
        raise ValueError("weight must be finite and nonnegative.")

    # Always recover the original objective, avoiding accumulated penalties.
    if model.original_cost is None:
        original = model.c0.copy()
        original_constant = model.constant
    else:
        original = model.original_cost.copy()
        original_constant = model.original_constant

    q = original.copy()
    q[model.splus] += weight
    q[model.sminus] += weight

    return replace(
        model,
        c0=q,
        constant=original_constant,
        original_cost=original,
        original_constant=original_constant,
    )

def run_normalized_slack_sweep(case, outages, args, fixed, output, metadata, preprocess):
    started = perf_counter()
    pure = args.normalized_slack_only
    normalization = args.slack_normalization
    stem = 'normalized' if normalization == 'capacity' else 'unnormalized'
    response_expression = 'sum((splus+sminus)/gmax)' if normalization == 'capacity' else 'sum(splus+sminus)'
    lp_tag = stem + ('_slack_only_LP' if pure else '_slack_LP')
    exact_tag = stem + '_slack_exact_APR' if pure else 'same_limits_exact_APR'
    weights = [1.0] if pure else sorted(set(args.weights), reverse=True)
    t = perf_counter()
    base_lp = apply_eta_limits(add_apr_slacks(build_model(case,outages,args,False,fixed),case,outages),case,args)
    shared_build = perf_counter()-t
    jobs = []
    if args.compare_exact or pure:
        t = perf_counter()
        control = apply_eta_limits(add_apr_slacks(build_model(case,outages,args,True,fixed),case,outages),case,args)
        if pure:
            control = normalized_slack_objective(control,case,1.0,thermal_weight=0.0,normalization=normalization)
        jobs.append((exact_tag,
                     1.0 if pure else None,control,perf_counter()-t))
    for weight in weights:
        t = perf_counter()
        jobs.append((lp_tag,weight,
                     normalized_slack_objective(base_lp,case,weight,thermal_weight=0.0 if pure else 1.0,normalization=normalization),
                     shared_build+perf_counter()-t))
    if pure:
        jobs.sort(key=lambda job:job[0]==exact_tag)
    metadata.update(experiment=stem+'_response_slack_weight_sweep', weights=weights,
                    slack_normalization=normalization,
                    optimized_objective='w*'+response_expression+' + sum(eta_pu)',
                    thermal_objective_units='per-unit overload summed over modeled lines and states',
                    direct_eta_limits={key:value for key,value in vars(args).items() if key.startswith('eta_')},
                    generation_cost_in_objective=False, cost_caps_enabled=False, threshold_applies=False,
                    comparator='Exact APR minimum ORIGINAL operating cost under the same direct eta limits, if requested',
                    warning='Large w does not guarantee APR consistency. All requested weights are tested; no monotone acceptance claim.')
    if pure:
        metadata.update(experiment='same_objective_'+stem+'_slack_LP_MILP_diagnostic',
                        optimized_objective=response_expression+' ONLY; eta and generation cost have zero coefficients',
                        comparator='Exact APR MILP minimizes the SAME response-slack objective as the LP',
                        weights=[], warning='This compares response-slack optima, NOT original operating-cost optima.')
        print('DIAGNOSTIC: BOTH models minimize '+response_expression+' ONLY.')
        print('No thermal-slack objective, generation-cost objective, cost budgets or signal penalty.')
    else:
        print('OBJECTIVE: w*'+response_expression+' + sum(eta_pu). No generation-cost term or cost caps.')
        print('Weight order (high to low): '+str(weights))
        print('Every weight is checked; a failed APR check does not stop the sweep.')
    rows, all_checks, all_details, all_dispatch, all_eta = [], [], [], [], []
    denominators = np.array([case['hi'][i] for s,i in base_lp.slack_pairs])
    for tag, weight, model, build_seconds in jobs:
        print(f'Solving {tag}' + ('' if pure else f', w={weight:g}' if weight is not None else ', original operating objective'), flush=True)
        # w is already in q on s+/s-. Passing w here would wrongly add w*sum(z).
        result, vector = solve_model(model, 0.0, args)
        result.update(model=tag, weight=None if pure else weight, slack_weight=None if pure else weight,
                      objective_kind=stem+'_response_slacks_ONLY' if pure else stem+'_response_slacks_plus_eta' if weight is not None else 'original_cost_same_eta_limits',
                      slack_normalization=normalization,
                      model_build_seconds=build_seconds, gamma=args.gamma, demand_scale=args.demand_scale,
                      cost_caps_enabled=False, exact_APR_enforced=tag==exact_tag)
        if vector is not None:
            t = perf_counter()
            metrics, checks, details, dispatch = analyze(case,outages,model,vector,args,tag,0.0)
            result.update(metrics)
            normalized = float(((vector[model.splus]+vector[model.sminus])/denominators).sum())
            unnormalized = float((vector[model.splus]+vector[model.sminus]).sum())
            selected_slacks = normalized if normalization == 'capacity' else unnormalized
            solver_eta = float(vector[model.eta_line].sum()+vector[model.eta_unit].sum())
            physical_eta = metrics['sum_overload_MW']/case['base']
            result.update(normalized_response_slack_sum=normalized,
                          unnormalized_response_slack_sum_pu=unnormalized,
                          objective_response_slack_sum=selected_slacks,
                          thermal_slack_sum_pu=physical_eta, solver_thermal_slack_sum_pu=solver_eta,
                          both_APR_slacks_positive_pairs=int(np.count_nonzero(
                              np.minimum(vector[model.splus],vector[model.sminus])*case['base']>args.apr_tol_mw)))
            value = selected_slacks if pure else weight*selected_slacks+solver_eta if weight is not None else metrics['operating_objective']
            result.update(optimized_objective=value, penalized_objective=value,
                          optimized_objective_units=('dimensionless_normalized_slack_sum' if normalization=='capacity' else 'per_unit_aggregate_response_slack') if pure else 'weighted_response_slacks_plus_pu_overload' if weight is not None else 'operating_cost',
                          solver_objective_reconstruction_error=abs(value-result['solver_penalized_objective']))
            diagnostics, eta_records = eta_limit_diagnostics(case,outages,vector[model.g],vector[model.y],args)
            result.update(diagnostics)
            result['provisional_APR_and_eta_feasible'] = bool(result['solver_point_verified'] and
                metrics['provisional_original_APR_feasible'] and result['eta_limits_satisfied'])
            if metrics['recovery_success']:
                recovered = [recover(case,vector[model.g],k,min(args.feas_tol_mw/case['base']/100,1e-10))[0] for k in outages]
                rec_diagnostics, rec_records = eta_limit_diagnostics(case,outages,vector[model.g],recovered,args,prefix='recovered_')
                result.update(rec_diagnostics);eta_records.extend(rec_records)
                result['recovered_APR_and_eta_feasible'] = bool(metrics['recovered_original_APR_feasible'] and result['recovered_eta_limits_satisfied'])
            for table in (checks,details,dispatch,eta_records):
                for record in table:record.update(model=tag,weight=None if pure else weight)
            all_checks.extend(checks);all_details.extend(details);all_dispatch.extend(dispatch);all_eta.extend(eta_records)
            candidates = []
            if result['provisional_APR_and_eta_feasible']:candidates.append(metrics['operating_objective'])
            if result.get('recovered_APR_and_eta_feasible'):candidates.append(metrics['recovered_operating_objective'])
            result['verified_candidate_operating_cost'] = min(candidates) if candidates else None
            result['has_verified_candidate'] = bool(candidates)
            result['validation_seconds'] = perf_counter()-t
            if not result['solver_point_verified']:result['status']='validation_failed'
            print(f'  status={result["status"]}; {stem} slacks={selected_slacks:.8g}; eta={physical_eta:.8g} pu; '
                  f'APR mismatch={metrics["max_APR_L1_error_MW"]:.6g} MW; both-positive pairs={result["both_APR_slacks_positive_pairs"]}; '
                  f'direct accepted={result["provisional_APR_and_eta_feasible"]}; cost={metrics["operating_objective"]:.8f}',flush=True)
        else:print(f'  status={result["status"]}; {result.get("message", "")}',flush=True)
        result['pipeline_seconds'] = preprocess+build_seconds+result['total_seconds']+result.get('validation_seconds',0.0)
        rows.append(result)
        pd.DataFrame(rows).to_csv(output/'summary.csv',index=False)
        for name,data in [('apr_checks',all_checks),('apr_dispatch',all_details),('dispatch',all_dispatch),('eta_limit_checks',all_eta)]:
            pd.DataFrame(data).to_csv(output/(name+'.csv'),index=False)
    benchmark = next((row for row in rows if row['model']=='same_limits_exact_APR' and row['status']=='optimal'
                      and row.get('provisional_APR_and_eta_feasible')),None)
    if benchmark:
        for row in rows:
            if row['model']==lp_tag and row.get('has_verified_candidate'):
                row['verified_cost_difference_vs_same_limits_MILP'] = row['verified_candidate_operating_cost']-benchmark['operating_objective']
                row['verified_relative_cost_difference_vs_same_limits_MILP'] = row['verified_cost_difference_vs_same_limits_MILP']/max(1.0,abs(benchmark['operating_objective']))
    table = pd.DataFrame(rows)
    table.to_csv(output/'summary.csv',index=False)
    if not pure:
        table[table.model==lp_tag].to_csv(output/'weight_results.csv',index=False)
    else:
        lp = next(row for row in rows if row['model']==lp_tag)
        mip = next(row for row in rows if row['model']==exact_tag)
        diagnostic = dict(LP_status=lp['status'], exact_APR_status=mip['status'],
                          slack_normalization=normalization,
                          LP_response_slack_objective=lp.get('objective_response_slack_sum'),
                          exact_APR_response_slack_objective=mip.get('objective_response_slack_sum'),
                          LP_normalized_slack=lp.get('normalized_response_slack_sum'),
                          exact_APR_normalized_slack=mip.get('normalized_response_slack_sum'),
                          exact_APR_solver_lower_bound=mip.get('solver_best_bound_penalized'),
                          conclusion='Inconclusive: inspect solver statuses, bounds and feasibility checks')
        if lp['status']=='optimal' and lp.get('solver_point_verified'):
            lower = lp['objective_response_slack_sum']
            candidate = mip.get('objective_response_slack_sum')
            mip_bound = mip.get('solver_best_bound_penalized')
            tolerance = 1e-7*max(1.0,abs(lower),abs(candidate) if candidate is not None else 0.0)
            diagnostic['comparison_tolerance'] = tolerance
            if candidate is not None:
                diagnostic['incumbent_objective_difference'] = candidate-lower
            if mip_bound is not None:
                diagnostic['certified_separation_using_MILP_bound'] = mip_bound-lower
            if mip.get('provisional_APR_and_eta_feasible') and candidate<=lower+tolerance:
                diagnostic['conclusion'] = 'An APR-consistent response-slack optimum exists within numerical tolerance; inspect direct LP APR diagnostics too'
            elif mip_bound is not None and mip_bound>lower+tolerance:
                diagnostic['conclusion'] = 'Strict response-slack relaxation gap: no APR-consistent point attains the LP minimum within numerical tolerance'
        metadata['same_objective_diagnostic'] = diagnostic
        pd.DataFrame([diagnostic]).to_csv(output/(stem+'_slack_diagnostic.csv'),index=False)
        print('SAME-OBJECTIVE DIAGNOSTIC: '+json.dumps(diagnostic),flush=True)
    metadata.update(results=rows,
                    sweep_solve_seconds=sum(row['total_seconds'] for row in rows if row['model']==lp_tag),
                    complete_experiment_seconds=preprocess+perf_counter()-started)
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    columns=['model','weight','status','objective_response_slack_sum','normalized_response_slack_sum','unnormalized_response_slack_sum_pu','thermal_slack_sum_pu','max_APR_L1_error_MW',
             'both_APR_slacks_positive_pairs','provisional_APR_and_eta_feasible','recovered_APR_and_eta_feasible',
             'operating_objective','verified_candidate_operating_cost','total_seconds']
    print(table[[column for column in columns if column in table]].to_string(index=False))
    print('Saved: '+str(output.resolve()))
    return 0 if all(row['status']=='optimal' for row in rows) else 2


@contextmanager
def atomic_output(path):
    """Replace a checkpoint only after its new contents were written in full."""
    path = Path(path)
    temporary = None
    try:
        with NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                dir=path.parent, prefix='.'+path.name+'.',
                                suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            yield stream
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def snapshot_data(value):
    """Detach nested records before any output code iterates their keys."""
    if isinstance(value, dict):
        return {key: snapshot_data(item) for key, item in value.copy().items()}
    if isinstance(value, (list, tuple)):
        return [snapshot_data(item) for item in tuple(value)]
    if isinstance(value, np.ndarray):
        return snapshot_data(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_search_json(path, value):
    payload = snapshot_data(value)
    with atomic_output(path) as stream:
        json.dump(payload, stream, indent=2)
        stream.write('\n')


def write_search_csv(path, records):
    """Export private records and identify the export active during a crash.

    Explicit schema loops avoid the generator expression reported in the
    native-crash traceback. This is a diagnostic workaround, not evidence
    that CSV field discovery caused the underlying native memory fault.
    """
    path = Path(path)
    print(f'Checkpoint CSV: {path.name}; stage=snapshot', flush=True)
    records = snapshot_data(records)
    fields = []
    seen = set()
    print(f'Checkpoint CSV: {path.name}; rows={len(records)}; stage=columns', flush=True)
    for index, record in enumerate(records):
        if type(record) is not dict:
            raise TypeError(f'{path.name}: row {index} is not a plain dictionary')
        for key in tuple(record.keys()):
            if not isinstance(key, str):
                raise TypeError(f'{path.name}: row {index} has a non-string column name')
            if key not in seen:
                seen.add(key)
                fields.append(key)
    print(f'Checkpoint CSV: {path.name}; stage=write', flush=True)
    with atomic_output(path) as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for record in records:
                writer.writerow(record)
    print(f'Checkpoint CSV: {path.name}; stage=complete', flush=True)


def bisect_apr_weight(evaluate, start=1.0, absolute_tol=1e-3,
                      relative_tol=0.0, max_weight=1e6, max_solves=40,
                      checkpoint=None):
    """Bracket and bisect DIRECT optimal-LP APR acceptance.

    evaluate(weight, phase) returns valid_probe and accepted_for_search.
    A nonoptimal/unverified solve is inconclusive, not a failed APR test.
    A fail/pass interval describes observations; it does not certify a
    globally minimal APR-consistent weight without a monotonicity proof.
    """
    if (not np.isfinite(start) or start <= 0 or
            not np.isfinite(max_weight) or max_weight < start or
            not np.isfinite(absolute_tol) or absolute_tol <= 0 or
            not np.isfinite(relative_tol) or relative_tol < 0 or max_solves < 1):
        raise ValueError('Invalid weight search limits or tolerances')
    observations = []
    lower, upper = None, None

    def probe(weight, phase):
        if len(observations) >= max_solves:
            return None
        finish('running', current_weight=float(weight), current_phase=phase)
        try:
            row = dict(evaluate(float(weight), phase))
        except BaseException as exc:
            # Keep the previous completed probes, and do not turn an exception
            # into an APR failure or silently move a bracket endpoint.
            try:
                finish('interrupted' if isinstance(exc, KeyboardInterrupt) else 'error',
                       failed_weight=float(weight), failed_phase=phase,
                       error_type=type(exc).__name__, error_message=str(exc),
                       error_traceback=traceback.format_exc())
            except Exception as save_error:
                print(f'ERROR saving failure checkpoint: {save_error}', file=sys.stderr)
            raise
        row.update(weight=float(weight), search_phase=phase)
        observations.append(row)
        return row

    def finish(reason, **extra):
        passing = [row['weight'] for row in observations
                   if row.get('valid_probe') and row.get('accepted_for_search')]
        width = None if lower is None or upper is None else upper-lower
        summary = dict(termination=reason, solve_count=len(observations),
                    lower_failed_weight=lower, upper_accepted_weight=upper,
                    interval_width=width,
                    best_passing_weight=min(passing) if passing else None,
                    absolute_weight_tolerance=absolute_tol,
                    relative_weight_tolerance=relative_tol,
                    search_completed=reason in ('zero_weight_accepted', 'interval_tolerance'),
                    global_minimum_certified=bool(passing and min(passing)==0.0),
                    search_in_progress=reason=='running',
                    interpretation='Smallest verified tested weight; positive-weight bracket is a numerical bisection result, not a proof of a global threshold')
        summary.update(extra)
        if checkpoint is not None:
            checkpoint(snapshot_data(summary))
        return summary

    first = probe(start, 'start')
    if not first.get('valid_probe'):
        return finish('solver_inconclusive')
    if first.get('accepted_for_search'):
        upper = float(start)
    else:
        lower = float(start)
    # Search only strictly positive weights.
    while lower is None:
        weight = upper / 2.0
        if weight <= 0.0 or weight == upper:
            return finish('floating_point_limit')

        row = probe(weight, 'decrease_positive')
        if row is None:
            return finish('solve_limit')
        if not row.get('valid_probe'):
            return finish('solver_inconclusive')

        if row.get('accepted_for_search'):
            upper = weight
        else:
            lower = weight
    while upper is None:
        if lower >= max_weight:
            return finish('maximum_weight_reached')
        weight = min(max_weight, 2.0*lower)
        row = probe(weight, 'expand_upper')
        if row is None:
            return finish('solve_limit')
        if not row.get('valid_probe'):
            return finish('solver_inconclusive')
        if row.get('accepted_for_search'):
            upper = weight
        else:
            lower = weight
    while upper-lower > max(absolute_tol, relative_tol*abs(upper)):
        midpoint = lower + (upper-lower)/2.0
        if midpoint == lower or midpoint == upper:
            return finish('floating_point_limit')
        row = probe(midpoint, 'bisect')
        if row is None:
            return finish('solve_limit')
        if not row.get('valid_probe'):
            return finish('solver_inconclusive')
        if row.get('accepted_for_search'):
            upper = midpoint
        else:
            lower = midpoint
    return finish('interval_tolerance')


def run_weight_bisection(case, outages, args, fixed, output, metadata, preprocess):
    """Retain a readable summary and traceback even when setup or saving fails."""
    write_search_json(output/'weight_search_summary.json',
                      dict(termination='initializing', search_completed=False,
                           search_in_progress=True, solve_count=0,
                           best_passing_weight=None, global_minimum_certified=False))
    try:
        with ExitStack() as solver_resources:
            return _run_weight_bisection(case, outages, args, fixed, output, metadata,
                                         preprocess, solver_resources)
    except BaseException as exc:
        error_trace = traceback.format_exc()
        try:
            summary = json.loads((output/'weight_search_summary.json').read_text(encoding='utf-8'))
            summary.update(termination='interrupted' if isinstance(exc, KeyboardInterrupt) else 'error',
                           search_completed=False, search_in_progress=False,
                           error_type=type(exc).__name__, error_message=str(exc),
                           error_traceback=error_trace)
            write_search_json(output/'weight_search_summary.json', summary)
            with atomic_output(output/'weight_search_error.txt') as stream:
                stream.write(error_trace)
        except Exception as save_error:
            print(f'ERROR saving failure details: {save_error}', file=sys.stderr)
        raise


def _run_weight_bisection(case, outages, args, fixed, output, metadata, preprocess,
                           solver_resources):
    """Solve one shared continuous model at adaptively chosen weights."""
    started = perf_counter()
    # Reuse contingency coefficient matrices in model construction, validation
    # and recovery. The cache is scoped to this case and search.
    case.pop('_network_blocks_cache', None)
    t = perf_counter()
    base_lp = apply_eta_limits(add_apr_slacks(
        build_model(case,outages,args,False,fixed),case,outages),case,args)
    shared_build = perf_counter()-t
    if np.any(base_lp.integer):
        raise ValueError('Weight bisection requires a continuous LP')
    gurobi_session = None
    gurobi_build = 0.0
    if args.backend == 'gurobi':
        print('Building persistent Gurobi model once for the weight search...', flush=True)
        gurobi_session = solver_resources.enter_context(GurobiSession(base_lp, args))
        gurobi_build = gurobi_session.build_seconds
        print(f'Persistent Gurobi model ready; native build={gurobi_build:.4f}s', flush=True)
    normalized = args.slack_normalization == 'capacity'
    expression = 'sum((splus+sminus)/gmax)' if normalized else 'sum(splus+sminus)'
    metadata.update(experiment='adaptive_APR_weight_bisection',
                    optimized_objective='original MIP objective + w*sum(splus+sminus)',
                    slack_normalization=args.slack_normalization,
                    weights=[], generation_cost_in_objective=True,
                    cost_caps_enabled=False, threshold_applies=False,
                    shared_model_build_seconds=shared_build,
                    memory_efficient_construction=True,
                    persistent_gurobi_model=gurobi_session is not None,
                    gurobi_model_build_seconds=gurobi_build,
                    solve_timing_definition='total_solve_seconds sums objective-update, optimize and result-extraction time; one-time Python and native Gurobi builds reported separately; complete experiment includes all stages',
                    direct_eta_limits={key:value for key,value in vars(args).items() if key.startswith('eta_')},
                    acceptance='Optimal verified DIRECT LP solution, APR error <= apr_tol_mw, no both-positive slack pairs, and all requested eta limits satisfied',
                    search_interpretation='Numerical bisection on returned-solution APR acceptance; a global monotone threshold has not been established')
    print(
        'OBJECTIVE: original generation cost + M_eta*sum(eta_pu) '
        '+ w*sum(splus+sminus).'
    )
    print(f'Adaptive weight search: start={args.weight_start:g}; absolute tolerance={args.weight_tol:g}; '
          f'maximum weight={args.weight_max:g}; solve limit={args.weight_max_solves}')
    rows, checks_all, details_all, dispatch_all, eta_all = [], [], [], [], []
    best, best_tables = None, None
    saved_count = -1
    saved_eta_count = 0
    # Large per-line/state diagnostics are streamed, never accumulated in RAM.
    with atomic_output(output/'eta_limit_checks.csv') as stream:
        pass

    def checkpoint_search(search):
        nonlocal saved_count, saved_eta_count
        search.update(total_solve_seconds=sum(row['total_seconds'] for row in rows),
                      completed_result_count=len(rows),
                      shared_model_build_seconds=shared_build,
                      persistent_gurobi_model=gurobi_session is not None,
                      gurobi_model_build_seconds=gurobi_build,
                      gurobi_model_build_count=0 if gurobi_session is None else 1,
                      gurobi_weight_solve_count=0 if gurobi_session is None else gurobi_session.solve_count,
                      complete_experiment_seconds=preprocess+perf_counter()-started,
                      apr_tolerance_MW=args.apr_tol_mw,
                      best_candidate_operating_cost=None if best is None else best['operating_objective'])
        # Write the small summary first, including on an exception. A failure
        # while exporting a large table cannot leave this file absent.
        write_search_json(output/'weight_search_summary.json', search)
        if saved_count != len(rows):
            if rows:
                print(f'Checkpoint: saving completed test {len(rows)}...',flush=True)
            for name in ('summary', 'weight_results', 'weight_bisection'):
                write_search_csv(output/(name+'.csv'), rows)
            for name, table in [('apr_checks', checks_all), ('apr_dispatch', details_all),
                                ('dispatch', dispatch_all)]:
                write_search_csv(output/(name+'.csv'), table)
            if best is not None:
                write_search_csv(output/'best_weight.csv', [best])
                for name, table in best_tables:
                    write_search_csv(output/(name+'.csv'), table)
                source = output/best['eta_limit_checks_file']
                with source.open(encoding='utf-8',newline='') as incoming, atomic_output(output/'best_eta_limit_checks.csv') as stream:
                    shutil.copyfileobj(incoming,stream)
            for record in rows[saved_eta_count:]:
                filename = record.get('eta_limit_checks_file')
                if filename:
                    target = output/'eta_limit_checks.csv'
                    with (output/filename).open(encoding='utf-8',newline='') as incoming:
                        header = incoming.readline()
                        with target.open('a',encoding='utf-8',newline='') as stream:
                            if stream.tell()==0:
                                stream.write(header)
                            shutil.copyfileobj(incoming,stream)
                saved_eta_count += 1
            saved_count = len(rows)
        payload = dict(metadata)
        payload.update(search=search, best_result=best, results=rows,
                       weights=[row['weight'] for row in rows])
        write_search_json(output/'metadata.json', payload)
        if rows:
            print(f'Checkpoint: completed test {len(rows)} saved.',flush=True)

    def evaluate(weight, phase):
        nonlocal best, best_tables
        iteration = len(rows)+1
        print(f'Weight test {iteration}: phase={phase}; w={weight:.12g}',flush=True)
        t = perf_counter()
        model = normalized_slack_objective(base_lp,case,weight,
                                           thermal_weight=1.0,normalization=args.slack_normalization)
        objective_build = perf_counter()-t
        # The search weight already multiplies s+/s-. The solver's separate
        # signal-weight argument must remain zero.
        if gurobi_session is None:
            result, vector = solve_model(model,0.0,args)
        else:
            result, vector = solve_model(model,0.0,args,gurobi_session=gurobi_session)
        result.update(model='weight_bisection_LP',weight=weight,slack_weight=weight,
                      search_iteration=iteration,search_phase=phase,
                      slack_normalization=args.slack_normalization,
                      cost_caps_enabled=False,exact_APR_enforced=False,
                      objective_kind='original_objective_plus_weighted_response_slacks',
                      objective_build_seconds=objective_build,
                      model_build_seconds=shared_build if iteration==1 else 0.0,
                      gamma=args.gamma,demand_scale=args.demand_scale,
                      valid_probe=False,accepted_for_search=False)
        local_checks, local_details, local_dispatch, local_eta = [], [], [], []
        if vector is not None:
            t = perf_counter()
            metrics, local_checks, local_details, local_dispatch = analyze(
                case,outages,model,vector,args,'weight_bisection_LP',0.0)
            result.update(metrics)
            slack_values=vector[model.splus]+vector[model.sminus]
            capacities=np.array([case['hi'][i] for s,i in model.slack_pairs])
            unnormalized_sum=float(slack_values.sum())
            normalized_sum=float((slack_values/capacities).sum())
            selected_sum=normalized_sum if normalized else unnormalized_sum
            solver_eta=float(vector[model.eta_line].sum()+vector[model.eta_unit].sum())
            original_value = float(
                model.original_cost @ vector + model.original_constant
            )
            objective = weight * selected_sum + original_value
            both=int(np.count_nonzero(np.minimum(vector[model.splus],vector[model.sminus])
                                     *case['base']>args.apr_tol_mw))
            reconstruction_error=abs(objective-result['solver_penalized_objective'])
            result.update(objective_response_slack_sum=selected_sum,
                          normalized_response_slack_sum=normalized_sum,
                          unnormalized_response_slack_sum_pu=unnormalized_sum,
                          thermal_slack_sum_pu=metrics['sum_overload_MW']/case['base'],
                          solver_thermal_slack_sum_pu=solver_eta,
                          optimized_objective=objective,penalized_objective=objective,
                          solver_objective_reconstruction_error=reconstruction_error,
                          both_APR_slacks_positive_pairs=both)
            eta_path = output/f'eta_limit_checks_weight_{iteration:04d}.csv'
            with atomic_output(eta_path) as stream:
                writer = None
                def save_eta(record):
                    nonlocal writer
                    record.update(model='weight_bisection_LP',weight=weight,
                                  search_iteration=iteration,search_phase=phase)
                    if writer is None:
                        writer = csv.DictWriter(stream,fieldnames=list(record))
                        writer.writeheader()
                    writer.writerow(record)
                diagnostics, local_eta = eta_limit_diagnostics(
                    case,outages,vector[model.g],vector[model.y],args,record_sink=save_eta)
            result['eta_limit_checks_file'] = eta_path.name
            result.update(diagnostics)
            objective_verified=bool(reconstruction_error<=1e-7*max(1.0,abs(objective)))
            result['valid_probe']=bool(result['status']=='optimal' and
                                       result['solver_point_verified'] and objective_verified)
            accepted=bool(result['valid_probe'] and
                          metrics['provisional_original_APR_feasible'] and
                          metrics['max_APR_L1_error_MW']<=args.apr_tol_mw and
                          both==0 and diagnostics['eta_limits_satisfied'])
            result.update(provisional_APR_and_eta_feasible=accepted,
                          accepted_for_search=accepted,
                          has_verified_candidate=accepted,
                          verified_candidate_operating_cost=metrics['operating_objective'] if accepted else None)
            for table in (local_checks,local_details,local_dispatch,local_eta):
                for record in table:
                    record.update(model='weight_bisection_LP',weight=weight,
                                  search_iteration=iteration,search_phase=phase)
            result['validation_seconds']=perf_counter()-t
            if not result['solver_point_verified'] or not objective_verified:
                result['status']='validation_failed'
            if accepted and (best is None or weight<best['weight']):
                best=dict(result)
                best_tables = [('best_dispatch',local_dispatch),('best_apr_dispatch',local_details),
                               ('best_apr_checks',local_checks)]
            label='PASS' if accepted else 'APR_FAIL' if result['valid_probe'] else 'INCONCLUSIVE'
            print(f'  {label}; status={result["status"]}; APR mismatch={metrics["max_APR_L1_error_MW"]:.6g} MW; '
                  f'both-positive pairs={both}; eta={result["thermal_slack_sum_pu"]:.8g} pu; '
                  f'cost={metrics["operating_objective"]:.8f}; solve={result["total_seconds"]:.4f}s',flush=True)
        else:
            print(f'  INCONCLUSIVE; status={result["status"]}; {result.get("message", "")}',flush=True)
        result['evaluation_seconds']=objective_build+result['total_seconds']+result.get('validation_seconds',0.0)
        # Persist the small verified result before bulk diagnostic/checkpoint work.
        write_search_json(output/'last_completed_weight.json',result)
        rows.append(dict(result))
        checks_all.extend(local_checks);details_all.extend(local_details)
        dispatch_all.extend(local_dispatch);eta_all.extend(local_eta)
        return result

    search=bisect_apr_weight(evaluate,start=args.weight_start,
                            absolute_tol=args.weight_tol,relative_tol=args.weight_rtol,
                            max_weight=args.weight_max,max_solves=args.weight_max_solves,
                            checkpoint=checkpoint_search)
    search.update(total_solve_seconds=sum(row['total_seconds'] for row in rows),
                  shared_model_build_seconds=shared_build,
                  persistent_gurobi_model=gurobi_session is not None,
                  gurobi_model_build_seconds=gurobi_build,
                  gurobi_model_build_count=0 if gurobi_session is None else 1,
                  gurobi_weight_solve_count=0 if gurobi_session is None else gurobi_session.solve_count,
                  complete_experiment_seconds=preprocess+perf_counter()-started,
                  apr_tolerance_MW=args.apr_tol_mw,
                  best_candidate_operating_cost=None if best is None else best['operating_objective'])
    metadata.update(search=search,best_result=best,results=rows)
    write_search_json(output/'weight_search_summary.json', search)
    write_search_json(output/'metadata.json', metadata)
    print('Search termination: '+search['termination'])
    if search['best_passing_weight'] is None:
        print('No directly APR-consistent optimal LP candidate found within the search limits.')
    else:
        print(f'Smallest verified tested weight: {search["best_passing_weight"]:.12g}')
        if search['lower_failed_weight'] is not None:
            print(f'Observed fail/pass interval: [{search["lower_failed_weight"]:.12g}, '
                  f'{search["upper_accepted_weight"]:.12g}]; width={search["interval_width"]:.6g}')
        print(f'Selected operating cost: {best["operating_objective"]:.8f}; '
              f'APR mismatch={best["max_APR_L1_error_MW"]:.6g} MW')
    print(f'Total LP solve time: {search["total_solve_seconds"]:.6f} s; '
          f'complete experiment: {search["complete_experiment_seconds"]:.6f} s')
    print('Saved: '+str(output.resolve()))
    return 0 if search['search_completed'] and best is not None else 2


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
    p.add_argument('--version', action='version', version='%(prog)s '+IMPLEMENTATION_VERSION)
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
    p.add_argument('--min-eta-fraction', action='store_true', help='Standalone diagnostic: minimize delta with eta <= delta*rating on every monitored line/state, no cost caps. Optional --compare-exact solves the exact APR minimum-delta MILP too.')
    p.add_argument('--normalized-slack-sweep', action='store_true', help='Standalone high-to-low sweep: min w*sum((splus+sminus)/gmax)+sum(eta_pu). Uses --weights, optional direct eta caps and --compare-exact (original cost, same caps). No cost budgets.')
    p.add_argument('--normalized-slack-only', action='store_true', help='Targeted diagnostic: solve LP AND exact-APR MILP minimizing ONLY sum((splus+sminus)/gmax). No thermal or generation cost objective. Optional direct eta caps apply equally to both.')
    p.add_argument('--weight-bisection', action='store_true', help='Adaptive LP search: start at --weight-start, bracket a passing weight, and bisect the APR-fail/APR-pass interval. No weight grid or cost budgets. Use --slack-normalization none for the current method.')
    p.add_argument('--weight-start', type=float, default=1.0, help='Positive initial weight for adaptive search (default 1).')
    p.add_argument('--weight-tol', type=float, default=1e-3, help='Absolute tolerance on final fail/pass WEIGHT interval width (default 0.001). Distinct from --apr-tol-mw.')
    p.add_argument('--weight-rtol', type=float, default=0.0, help='Optional relative weight interval tolerance; stop at max(weight-tol,weight-rtol*upper). Default 0.')
    p.add_argument('--weight-max', type=float, default=1e6, help='Largest weight allowed during upper-bracket expansion (default 1e6).')
    p.add_argument('--weight-max-solves', type=int, default=40, help='Maximum LP solves including start, zero endpoint and expansion (default 40).')
    p.add_argument('--slack-normalization', choices=['capacity','none'], default='capacity', help='Standalone slack modes: capacity divides each slack pair by its generator Pmax (default); none minimizes the unnormalized sum instead. Applies equally to LP and exact-APR comparator.')
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
    print('Implementation: '+IMPLEMENTATION_VERSION, flush=True)
    standalone_slack = args.normalized_slack_sweep or args.normalized_slack_only or args.weight_bisection
    if args.slack_normalization!='capacity' and not standalone_slack:
        raise ValueError('--slack-normalization none requires --normalized-slack-only, --normalized-slack-sweep, or --weight-bisection')
    if args.weight_bisection:
        for key in ('weight_start','weight_tol','weight_max'):
            if not np.isfinite(getattr(args,key)) or getattr(args,key)<=0:
                raise ValueError(f'{key} must be positive and finite')
        if not np.isfinite(args.weight_rtol) or args.weight_rtol<0:
            raise ValueError('weight-rtol must be finite and nonnegative')
        if args.weight_max<args.weight_start or args.weight_max_solves<2:
            raise ValueError('weight-max must be at least weight-start, and weight-max-solves must be at least 2')
        if args.compare_exact:
            raise ValueError('--weight-bisection uses LP solves only; run an exact-APR comparison separately')
    eta_values = [value for key,value in vars(args).items() if key.startswith('eta_')]
    eta_requested = any(value is not None for value in eta_values)
    if standalone_slack:
        if sum((args.normalized_slack_sweep,args.normalized_slack_only,args.weight_bisection))>1:
            raise ValueError('Choose one of --normalized-slack-sweep, --normalized-slack-only, or --weight-bisection')
        if args.thermal_mode!='soft' or args.min_eta_fraction or args.budget_increase is not None or args.no_generation_budget or args.thermal_cost_budgets!='on':
            raise ValueError('--normalized-slack-sweep requires soft mode and replaces budget modes. Remove --budget-increase, --no-generation-budget and --thermal-cost-budgets off.')
        if args.compare_weighted_exact or args.lexicographic_cost or args.include_threshold:
            raise ValueError('Normalized slack sweep supports --compare-exact, not weighted-exact, lexicographic or threshold modes.')
        if any(getattr(args,key) is not None for key in ('epsilon_g','epsilon_line','epsilon_unit','epsilon_g_grid','epsilon_line_grid','epsilon_unit_grid','reference_cost')) or args.line_allowance or args.unit_allowance:
            raise ValueError('Cost-budget epsilons/reference costs/allowances do not apply to --normalized-slack-sweep')
    if args.min_eta_fraction:
        if args.thermal_mode != 'soft' or args.budget_increase is not None or eta_requested or args.no_generation_budget or args.thermal_cost_budgets != 'on':
            raise ValueError('--min-eta-fraction requires soft mode and replaces budget/direct-eta-cap options. Remove --budget-increase, --no-generation-budget, --thermal-cost-budgets off, and --eta-... caps.')
        if args.compare_weighted_exact or args.lexicographic_cost or args.include_threshold:
            raise ValueError('Minimum-fraction diagnostic supports --compare-exact; weighted/lexicographic/threshold modes do not apply.')
        if any(getattr(args,key) is not None for key in ('epsilon_g','epsilon_line','epsilon_unit','epsilon_g_grid','epsilon_line_grid','epsilon_unit_grid','reference_cost')) or args.line_allowance or args.unit_allowance:
            raise ValueError('Cost-budget epsilons, reference costs and allowances do not apply to --min-eta-fraction')
    if any(value is not None and (not np.isfinite(value) or value < 0) for value in eta_values):
        raise ValueError('Direct eta limits must be finite and nonnegative')
    if eta_requested and ((args.budget_increase is None and not standalone_slack) or args.thermal_mode != 'soft'):
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
    metadata = dict(implementation_version=IMPLEMENTATION_VERSION,
                    arguments=vars(args), case_path=str(path), python=sys.version, platform=platform.platform(),
                    numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,
                    input_preprocessing_seconds=preprocess, demand_MW=case['D']*case['base'],
                    transformer_phase_shift_assumption='zero; input branch phase-shift data are not used',
                    flow_convention='f=K@(B@g-d); no transformer phase-shift offsets; symmetric thermal limits',
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
    print('Transformer phase shifts: assumed zero; branch phase-shift data ignored (Model 1).')
    if not args.min_eta_fraction and not standalone_slack:
        print(f'Candidate soft-limit threshold B={B:.8g}; tested W={weights}')
        print('NOTE: The proposed sufficient-weight argument concerns only SOFT limits with unbounded slacks and NO cost budget.')
    if np.any(case['gen'].c2.to_numpy(float)!=0): print('NOTE: c2 ignored; using linear c1 generation costs.')
    if any(r['capacity_shortage_MW']>args.feas_tol_mw for r in screen):
        print('INFEASIBLE: at least one outage leaves insufficient total surviving capacity. See capacity_screen.csv; thermal slacks cannot replace generation.')
        pd.DataFrame([dict(model='capacity_screen',status='infeasible_capacity')]).to_csv(output/'summary.csv',index=False)
        return 2
    if args.min_eta_fraction:
        return run_minimum_eta_fraction(case,outages,args,fixed,output,metadata,preprocess)
    if args.weight_bisection:
        return run_weight_bisection(case,outages,args,fixed,output,metadata,preprocess)
    if args.normalized_slack_sweep or args.normalized_slack_only:
        return run_normalized_slack_sweep(case,outages,args,fixed,output,metadata,preprocess)
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



def comparison_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version', action='version', version='%(prog)s '+IMPLEMENTATION_VERSION)
    p.add_argument('--case', required=True)
    p.add_argument('--case-dir', default='../excel_outputs')
    p.add_argument('--output-dir')
    p.add_argument('--solver', choices=['auto','gurobi','highs'], default='gurobi')
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--demand-scale', type=float, default=1.0)
    p.add_argument('--penalty', type=float, default=1500.0, help='Original objective thermal penalty per aggregate pu of overload.')
    p.add_argument('--islanding', choices=['error','skip'], default='skip')
    p.add_argument('--generator-contingencies', default='all')
    p.add_argument('--no-line-contingencies', action='store_true')
    p.add_argument('--weight-start', type=float, default=1.0)
    p.add_argument('--weight-tol', type=float, default=0.001)
    p.add_argument('--weight-rtol', type=float, default=0.0)
    p.add_argument('--weight-max', type=float, default=1e6)
    p.add_argument('--weight-max-solves', type=int, default=40)
    p.add_argument('--lp-time-limit', type=float, default=300.0, help='Seconds per LP solve; the whole bisection can take longer.')
    p.add_argument('--mip-time-limit', type=float, default=3600.0, help='Seconds per cold/warm MIP optimize call, excluding construction.')
    p.add_argument('--mip-gap', type=float, default=1e-7)
    p.add_argument('--apr-tol-mw', type=float, default=1e-4)
    p.add_argument('--feas-tol-mw', type=float, default=1e-4)
    p.add_argument('--with-lp-start', action='store_true', help='Also run a fresh Gurobi MIP using the verified LP candidate as a full MIP start.')
    p.add_argument('--tee', action='store_true', help='Print solver logs, including whether Gurobi accepts the MIP start.')
    return p


def evaluate_common(case, outages, g, y, z, args):
    """Evaluate ORIGINAL cost and direct APR without repairing either dispatch."""
    metrics = operating_metrics(case,outages,g,y,not args.no_line_contingencies,args.penalty)
    apr = max((float(np.abs(y[s]-physical(case,g,z[s],k)).sum())*case['base']
               for s,k in enumerate(outages)),default=0.0)
    signal_error = max(max0(-z),max0(z-1.0))
    row = dict(original_objective=metrics['operating_objective'],
               generation_cost=metrics['generation_cost'],
               thermal_penalty=metrics['line_thermal_penalty']+metrics['unit_thermal_penalty'],
               thermal_slack_sum_pu=metrics['sum_overload_MW']/case['base'],
               total_overload_MW=metrics['sum_overload_MW'],
               line_overload_MW=metrics['line_sum_overload_MW'],
               generator_outage_overload_MW=metrics['unit_sum_overload_MW'],
               max_overload_MW=metrics['max_overload_MW'],
               max_APR_L1_error_MW=apr,balance_error_MW=metrics['balance_error_MW'],
               generator_bound_error_MW=metrics['generator_bound_error_MW'],
               signal_domain_error=signal_error)
    row['original_soft_model_feasible'] = bool(np.isfinite(g).all() and np.isfinite(y).all()
        and np.isfinite(z).all() and apr<=args.apr_tol_mw
        and metrics['balance_error_MW']<=args.feas_tol_mw
        and metrics['generator_bound_error_MW']<=args.feas_tol_mw and signal_error<=1e-7)
    return row


def load_lp_candidate(case, outages, output):
    base=case['base'];folder=output/'lp_search'
    tab=pd.read_csv(folder/'best_dispatch.csv')
    lookup={id_key(r.gen_ID):float(r.generation_MW)/base for r in tab.itertuples()}
    g=np.array([lookup[id_key(v)] for v in case['gen'].gen_ID])
    tab=pd.read_csv(folder/'best_apr_checks.csv')
    lookup={id_key(r.gen_ID):float(r.signal) for r in tab.itertuples()}
    z=np.array([lookup[id_key(case['gen'].iloc[k].gen_ID)] for k in outages])
    tab=pd.read_csv(folder/'best_apr_dispatch.csv')
    lookup={(id_key(r.outage_gen_ID),id_key(r.gen_ID)):float(r.provisional_MW)/base for r in tab.itertuples()}
    y=np.array([[lookup[(id_key(case['gen'].iloc[k].gen_ID),id_key(v))]
                 for v in case['gen'].gen_ID] for k in outages])
    return g,y,z


def make_mip_start(case, outages, model, candidate, args):
    g,y,z=candidate
    point=np.zeros(len(model.c0))
    point[model.g]=g;point[model.y]=y;point[model.z]=z
    H,offset,_,_=network_blocks(case,False)
    f0=H@g+offset
    point[model.nominal_flow]=f0
    for (s,i),index in model.x.items():
        point[index]=float(g[i]+case['a'][i]*z[s]<=case['hi'][i])
    generators={int(k):s for s,k in enumerate(outages)}
    previous=None;flows=None
    for index,family,state,j in model.eta_records:
        key=(family,state)
        if key!=previous:
            flows=(H@y[generators[state]]+offset if family=='unit' else
                   f0 if state==-1 else f0+case['L'][:,state]*f0[state])
            previous=key
        point[index]=max(0.0,abs(flows[j])-case['rates'][j])
    residual=model.A@point
    power_violation=max(max0(model.lower-residual),max0(residual-model.upper),
                        max0(model.lb-point),max0(point-model.ub))*case['base']
    integer_error=max0(np.abs(point[model.integer>0]-np.rint(point[model.integer>0])))
    if power_violation>args.feas_tol_mw or integer_error>1e-6:
        raise ValueError(f'LP candidate failed full MIP-start validation: {power_violation:g} MW')
    return point


def save_mip_details(case,outages,g,y,z,directory):
    write_search_csv(directory/'dispatch.csv',[
        dict(gen_ID=id_key(v),generation_MW=float(g[i])*case['base'])
        for i,v in enumerate(case['gen'].gen_ID)])
    write_search_csv(directory/'apr_checks.csv',[
        dict(outage_gen_ID=id_key(case['gen'].iloc[k].gen_ID),signal=float(z[s]),
             apr_L1_error_MW=float(np.abs(y[s]-physical(case,g,z[s],k)).sum())*case['base'])
        for s,k in enumerate(outages)])
    write_search_csv(directory/'apr_dispatch.csv',[
        dict(outage_gen_ID=id_key(case['gen'].iloc[k].gen_ID),gen_ID=id_key(v),
             generation_MW=float(y[s,i])*case['base'])
        for s,k in enumerate(outages) for i,v in enumerate(case['gen'].gen_ID)])


def run_comparison_mip(case,outages,args,root,method,candidate=None):
    """Separate model construction and optimize for each independent MIP run."""
    directory=root/method;directory.mkdir()
    begun=perf_counter()
    row=dict(method=method,status='error',original_soft_model_feasible=False,
             mip_start_submitted=False,original_objective=None)
    try:
        print(f'Building {method}: ORIGINAL generation cost + thermal penalty...',flush=True)
        t=perf_counter();model=build_model(case,outages,args,True,None)
        row['python_model_build_seconds']=perf_counter()-t
        mip_args=copy.copy(args);mip_args.time_limit=args.mip_time_limit
        if args.backend=='gurobi':
            with GurobiSession(model,mip_args) as session:
                row['native_model_build_seconds']=session.build_seconds
                if candidate is not None:
                    t=perf_counter();start=make_mip_start(case,outages,model,candidate,args)
                    session.v.setAttr(session.gp.GRB.Attr.Start,start)
                    row.update(mip_start_submitted=True,mip_start_validated=True,
                               mip_start_preparation_seconds=perf_counter()-t)
                    print('Verified full LP-derived MIP start submitted; enable --tee to inspect solver acceptance.',flush=True)
                    del start
                result,point=solve_model(model,0.0,mip_args,gurobi_session=session)
        else:
            if candidate is not None:
                raise ValueError('SciPy/HiGHS does not expose MIP starts; use Gurobi for --with-lp-start')
            result,point=solve_model(model,0.0,mip_args)
        row.update(status=result['status'],solve_seconds=result['total_seconds'],
                   solver_runtime_seconds=result.get('solver_runtime_seconds'),
                   mip_lower_bound=result.get('solver_best_bound_penalized'),
                   mip_relative_gap=result.get('mip_relative_gap'),
                   mip_node_count=result.get('mip_node_count'),
                   solver_incumbent_objective=result.get('solver_penalized_objective'),
                   solver_point_verified=result.get('solver_point_verified',False))
        if point is not None:
            g,y,z=point[model.g],point[model.y],point[model.z]
            row.update(evaluate_common(case,outages,g,y,z,args))
            row['original_soft_model_feasible'] &= bool(result.get('solver_point_verified'))
            row['canonical_cost_minus_solver_cost']=row['original_objective']-result['solver_penalized_objective']
            save_mip_details(case,outages,g,y,z,directory)
        row['optimal_within_solver_tolerances']=bool(row['status']=='optimal' and row['original_soft_model_feasible'])
        write_search_json(directory/'solver_result.json',result)
    except Exception as exc:
        row.update(status='error',error=str(exc),original_soft_model_feasible=False)
        (directory/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print(f'{method} failed: {exc}',file=sys.stderr,flush=True)
    row['method_stage_seconds']=perf_counter()-begun
    write_search_json(directory/'result.json',row)
    return row


def comparison_report(rows, metadata, root):
    """Do not label an incumbent difference as an optimum gap after a timeout."""
    feasible_mips=[r for r in rows if r['method'].startswith('mip_')
                   and r.get('original_soft_model_feasible') and r.get('original_objective') is not None]
    reference=min(feasible_mips,key=lambda r:r['original_objective']) if feasible_mips else None
    optimal_mips=[r for r in feasible_mips if r.get('optimal_within_solver_tolerances')]
    optimum_reference=min(optimal_mips,key=lambda r:r['original_objective']) if optimal_mips else None
    bounds=[r['mip_lower_bound'] for r in rows if r['method'].startswith('mip_')
            and r.get('mip_lower_bound') is not None and np.isfinite(r['mip_lower_bound'])]
    lower_bound=max(bounds) if bounds else None
    for row in rows:
        row.update(reference_mip_method=None if reference is None else reference['method'],
                   difference_to_best_mip_incumbent_pct=None,
                   gap_to_mip_optimum_pct=None,
                   original_objective_lower_bound=lower_bound,
                   relative_gap_to_lower_bound_pct=None)
        if row.get('original_soft_model_feasible') and row.get('original_objective') is not None:
            cost=row['original_objective']
            if reference is not None:
                row['difference_to_best_mip_incumbent_pct']=100*(cost-reference['original_objective'])/max(abs(reference['original_objective']),1e-12)
            if optimum_reference is not None:
                row['gap_to_mip_optimum_pct']=100*(cost-optimum_reference['original_objective'])/max(abs(optimum_reference['original_objective']),1e-12)
            if lower_bound is not None:
                row['relative_gap_to_lower_bound_pct']=max(0.0,100*(cost-lower_bound)/max(abs(cost),1e-12))
    write_search_csv(root/'comparison.csv',rows)
    write_search_json(root/'comparison.json',dict(settings=metadata,results=rows,
        best_mip_incumbent=None if reference is None else reference['original_objective'],
        mip_optimum_available_within_solver_tolerances=bool(optimum_reference),
        original_objective_lower_bound=lower_bound,
        gap_note='gap_to_mip_optimum_pct is populated only after a feasible MIP reports optimal within solver tolerances; otherwise only incumbent differences and lower-bound gaps are reported.',
        timing_note='Each method pays common preprocessing. LP pays the entire bisection and validation. LP-started MIP pays the entire LP pipeline plus its own MIP stage. Cold MIP receives no LP start. No speedup claim is made for a time-limited run.'))


def comparison_main():
    options=comparison_parser().parse_args()
    for key in ('demand_scale','penalty','weight_start','weight_tol','weight_max',
                'lp_time_limit','mip_time_limit','apr_tol_mw','feas_tol_mw'):
        if not np.isfinite(getattr(options,key)) or getattr(options,key)<=0:
            raise ValueError(key+' must be finite and positive')
    if not np.isfinite(options.gamma) or not 0<=options.gamma<=1:
        raise ValueError('gamma must lie in [0,1]')
    if options.threads<1 or options.weight_max_solves<2 or options.weight_max<options.weight_start:
        raise ValueError('Invalid thread count or weight search limits')
    if not np.isfinite(options.weight_rtol) or options.weight_rtol<0 or not np.isfinite(options.mip_gap) or options.mip_gap<0:
        raise ValueError('Relative weight tolerance and MIP gap must be finite and nonnegative')
    args=parser().parse_args(['--case',options.case])
    vars(args).update(vars(options))
    args.time_limit=options.lp_time_limit
    args.weight_bisection=True;args.slack_normalization='none';args.thermal_mode='soft'
    args.backend=args.solver if args.solver!='auto' else ('gurobi' if importlib.util.find_spec('gurobipy') else 'highs')
    if args.with_lp_start and args.backend!='gurobi':
        raise ValueError('--with-lp-start requires --solver gurobi')
    begun=perf_counter();path=resolve_case(args.case,args.case_dir)
    case=load_case(path,args.demand_scale);case['a']=args.gamma*(case['hi']-case['lo']);args.base=case['base']
    outages=select_outages(case,args.generator_contingencies)
    if len(case['lo'])<2 or not outages:
        raise ValueError('Comparison requires at least two active generators and one generator outage')
    if len(case['excluded']) and not args.no_line_contingencies and args.islanding=='error':
        raise ValueError('Islanding line outages found; use --islanding skip for both methods')
    if any(case['hi'].sum()-case['hi'][k]<case['D']-args.feas_tol_mw/case['base'] for k in outages):
        raise ValueError('At least one generator outage has insufficient surviving capacity')
    root=Path(args.output_dir or 'output_apr_comparison_'+path.stem)
    if any((root/name).exists() for name in ('lp_search','mip_cold','mip_lp_start')):
        raise ValueError('Comparison output already exists; choose a fresh --output-dir')
    root.mkdir(parents=True,exist_ok=True)
    import hashlib
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    metadata=dict(implementation_version=IMPLEMENTATION_VERSION,arguments=vars(args).copy(),
        case_path=str(path),case_sha256=digest.hexdigest(),python=sys.version,numpy=np.__version__,
        scipy=scipy.__version__,pandas=pd.__version__,platform=platform.platform(),
        thermal_mode='soft',phase_shifts='assumed zero',gamma=args.gamma,baseMVA=case['base'],
        demand_MW=case['D']*case['base'],generator_outages=len(outages),
        generator_outage_IDs=[id_key(case['gen'].iloc[k].gen_ID) for k in outages],
        line_outages=0 if args.no_line_contingencies else len(case['eligible']),
        excluded_islanding=0 if args.no_line_contingencies else len(case['excluded']),
        objective='sum(c1*generation_MW+c0) + penalty*sum(required thermal overload_pu)',
        penalty_per_pu=args.penalty,c2_ignored=True,no_cost_budgets=True,
        candidate_selection='Smallest verified passing weight found; no cost-based selection or dispatch repair',
        note='A positive thermal slack is permitted; verified feasibility refers to the soft-limit model.')
    if args.backend=='gurobi':
        import gurobipy as gp
        metadata['gurobi_version']=list(gp.gurobi.version())
    common=perf_counter()-begun;metadata['common_preprocessing_seconds']=common
    write_search_json(root/'settings.json',metadata)
    case['branch'].iloc[case['excluded']].to_csv(root/'excluded_line_outages.csv',index=False)
    print(f'Comparison version: {IMPLEMENTATION_VERSION}\nCase: {path.name}; generator outages={len(outages)}; line outages={metadata["line_outages"]}',flush=True)
    print('Common evaluation: generation cost + penalty * REQUIRED overload_pu. Linear generation costs; zero phase shifts.',flush=True)
    if np.any(case['gen'].c2.to_numpy(float)!=0):
        print('NOTE: c2 is ignored consistently in BOTH models, matching the original linear-cost MIP.',flush=True)
    rows=[];candidate=None
    lpdir=root/'lp_search';lpdir.mkdir()
    t=perf_counter()
    lp=dict(method='lp_bisection',status='error',original_soft_model_feasible=False,original_objective=None)
    try:
        rc=run_weight_bisection(case,outages,args,None,lpdir,dict(metadata),0.0)
        search=json.loads((lpdir/'weight_search_summary.json').read_text())
        lp.update(status=search['termination'],search_completed=search['search_completed'],
                  weight=search.get('best_passing_weight'),lp_solve_count=search['solve_count'],
                  solve_seconds=search['total_solve_seconds'],
                  weight_interval_width=search.get('interval_width'))
        if search.get('best_passing_weight') is not None:
            candidate=load_lp_candidate(case,outages,root)
            lp.update(evaluate_common(case,outages,*candidate,args))
            best=pd.read_csv(lpdir/'best_weight.csv').iloc[0]
            lp['both_APR_slacks_positive_pairs']=int(best['both_APR_slacks_positive_pairs'])
            lp['original_soft_model_feasible'] &= bool(lp['both_APR_slacks_positive_pairs']==0)
            if not lp['original_soft_model_feasible']:candidate=None
    except Exception as exc:
        lp.update(status='error',error=str(exc))
        (lpdir/'comparison_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print('LP search failed: '+str(exc),file=sys.stderr,flush=True)
    lp['method_stage_seconds']=perf_counter()-t
    lp['pipeline_seconds']=common+lp['method_stage_seconds']
    rows.append(lp);comparison_report(rows,metadata,root)
    cold=run_comparison_mip(case,outages,args,root,'mip_cold')
    cold['pipeline_seconds']=common+cold['method_stage_seconds']
    rows.append(cold);comparison_report(rows,metadata,root)
    if args.with_lp_start:
        if candidate is None:
            warm=dict(method='mip_lp_start',status='skipped_no_verified_LP_candidate',original_soft_model_feasible=False,original_objective=None)
        else:
            warm=run_comparison_mip(case,outages,args,root,'mip_lp_start',candidate)
            warm['pipeline_seconds']=lp['pipeline_seconds']+warm['method_stage_seconds']
            warm['includes_LP_search_seconds']=lp['pipeline_seconds']
        rows.append(warm);comparison_report(rows,metadata,root)
    columns=['method','status','original_soft_model_feasible','original_objective','generation_cost',
             'thermal_slack_sum_pu','max_APR_L1_error_MW','pipeline_seconds','mip_relative_gap',
             'gap_to_mip_optimum_pct','difference_to_best_mip_incumbent_pct']
    table=pd.DataFrame(rows)
    print('\nCOMMON-OBJECTIVE COMPARISON\n'+table[[c for c in columns if c in table]].to_string(index=False))
    print('Saved: '+str(root.resolve()))
    return 0 if all(r.get('original_soft_model_feasible') for r in rows) else 2


if __name__=='__main__':
    try:
        sys.exit(comparison_main())
    except Exception as exc:
        print('ERROR: '+str(exc),file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
