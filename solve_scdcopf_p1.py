#!/usr/bin/env python3
"""Extensive preventive SCDCOPF (P1), adapted from SCDCOPF new solve.ipynb.

Excel sheets: baseMVA, bus, gen, gencost, branch (PGLib spreadsheet schema).
Generator outages and optional non-islanding line outages; Pyomo + Gurobi.
Generation costs are LINEAR (c1), matching the notebook, not quadratic c2.
Thermal limits are soft, generator limits and APR are hard.

Usage:
  python solve_scdcopf_p1.py --case pglib_opf_case3_lmbd.xlsx --gamma 1.0
  python solve_scdcopf_p1.py --case pglib_opf_case3_lmbd.xlsx --gamma 1.0 --line-contingencies
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

EXCEL_DIR = Path(r"M:\projects\excel_outputs")

def load_case(path, gamma=0.05, m_eta=1500.0):
    sheets = pd.read_excel(path, sheet_name=None)
    base = float(sheets['baseMVA'].iloc[0, 0])
    if base <= 0:
        raise ValueError('baseMVA must be positive')
    buses = sheets['bus'].copy()
    gens = sheets['gen'].copy()
    costs = sheets['gencost'].copy()
    lines = sheets['branch'].copy()
    for name, df in [('bus', buses), ('gen', gens), ('gencost', costs), ('branch', lines)]:
        if df.empty:
            raise ValueError(f'{name} sheet is empty')
    # Join costs by generator identity rather than relying on row positions.
    if gens['gen_ID'].duplicated().any() or costs['gen_ID'].duplicated().any():
        raise ValueError('gen_ID must uniquely identify generators')
    gens = gens.merge(costs[['gen_ID', 'c1', 'c0']], on='gen_ID', how='left', validate='one_to_one')
    if gens['c1'].isna().any():
        raise ValueError('Missing generator cost records')
    # Consistent with the notebook: exclude zero-capacity units and dispatchable loads.
    gens = gens[(gens['status'] > 0) & (gens['Pmax'] > 0) & (gens['Pmin'] >= 0)].reset_index(drop=True)
    lines = lines[lines['status'] > 0].reset_index(drop=True)
    if gens.empty or lines.empty:
        raise ValueError('No active generators or branches')
    bus_ids = buses['bus_i'].tolist()
    if len(set(bus_ids)) != len(bus_ids):
        raise ValueError('Duplicate bus IDs')
    bus_idx = {bus: i for i, bus in enumerate(bus_ids)}
    if not set(gens['bus_i']).issubset(bus_idx) or not set(lines['bus_i']).issubset(bus_idx) or not set(lines['bus_j']).issubset(bus_idx):
        raise ValueError('Generator/branch references an unknown bus')
    refs = buses.loc[buses['type'] == 3, 'bus_i'].tolist()
    if len(refs) != 1:
        raise ValueError('Expected exactly one reference bus')
    return dict(base=base, buses=buses, gens=gens, lines=lines, bus_ids=bus_ids,
                bus_idx=bus_idx, ref=bus_idx[refs[0]], gamma=float(gamma), m_eta=float(m_eta))

def build_matrices(case):
    """DC PTDF, and LODF for non-islanding line outages.

    Uses x and off-nominal transformer tap ratios. Nonzero phase shifts are
    rejected because the affine shift-flow term is not modeled here.
    """
    lines = case['lines']
    n, m = len(case['bus_ids']), len(lines)
    A = np.zeros((m, n))
    b = np.empty(m)
    for j, r in lines.iterrows():
        if abs(float(r['angle'])) > 1e-10:
            raise ValueError('Nonzero transformer phase shifts require an affine DC flow offset; unsupported here')
        x = float(r['x'])
        tap = float(r['ratio'])
        tap = 1.0 if tap == 0 else tap
        if x == 0 or tap <= 0:
            raise ValueError('Branch reactance must be nonzero and tap positive')
        A[j, case['bus_idx'][r['bus_i']]] = 1
        A[j, case['bus_idx'][r['bus_j']]] = -1
        b[j] = 1.0 / (x * tap)
    bus_b = A.T @ (b[:, None] * A)
    keep = [i for i in range(n) if i != case['ref']]
    inv = np.zeros((n, n))
    try:
        inv[np.ix_(keep, keep)] = np.linalg.inv(bus_b[np.ix_(keep, keep)])
    except np.linalg.LinAlgError as exc:
        raise ValueError('Network is disconnected; PTDF cannot be formed') from exc
    K = b[:, None] * (A @ inv)
    L = np.zeros((m, m))
    eligible = []
    for k in range(m):
        h = K @ A[k, :]
        denom = 1.0 - h[k]
        if abs(denom) < 1e-8:
            continue  # islanding outage, excluded from Ke
        L[:, k] = h / denom
        L[k, k] = -1.0
        eligible.append(k)
    B = np.zeros((n, len(case['gens'])))
    for i, row in case['gens'].iterrows():
        B[case['bus_idx'][row['bus_i']], i] = 1
    return K, L, B, eligible

def build_p1(case, include_lines=False):
    K, L, B, eligible = build_matrices(case)

    n = len(case['buses'])
    ng = len(case['gens'])
    nl = len(case['lines'])

    Kg = list(range(ng))
    Ke = eligible if include_lines else []

    base = case['base']
    d = case['buses']['Pd'].to_numpy(dtype=float) / base
    lo = case['gens']['Pmin'].to_numpy(dtype=float) / base
    hi = case['gens']['Pmax'].to_numpy(dtype=float) / base
    c1 = case['gens']['c1'].to_numpy(dtype=float) * base
    c0 = case['gens']['c0'].to_numpy(dtype=float)
    rates = case['lines']['rateA'].to_numpy(dtype=float) / base

    gamma, penalty = case['gamma'], case['m_eta']

    if gamma < 0:
        raise ValueError('gamma must be nonnegative')

    mdl = pyo.ConcreteModel(name='Extensive_SCDCOPF_P1')

    # ============================================================
    # SETS AND INDICES
    # ============================================================

    mdl.G = pyo.RangeSet(0, ng - 1)
    mdl.N = pyo.RangeSet(0, n - 1)
    mdl.BR = pyo.RangeSet(0, nl - 1)

    mdl.Kg = pyo.Set(initialize=Kg)
    mdl.Ke = pyo.Set(initialize=Ke)

    mdl.GK = pyo.Set(
        dimen=2,
        initialize=[(k, i) for k in Kg for i in range(ng)]
    )

    mdl.XK = pyo.Set(
        dimen=2,
        initialize=[
            (k, i) for k in Kg
            for i in range(ng) if i != k
        ]
    )

    mdl.EKG = pyo.Set(
        dimen=2,
        initialize=[
            (k, l) for k in Kg for l in range(nl)
        ]
    )

    mdl.EKE = pyo.Set(
        dimen=2,
        initialize=[
            (k, l) for k in Ke
            for l in range(nl) if l != k
        ]
    )

    # ============================================================
    # DECISION VARIABLES
    # ============================================================

    # (1d): Nominal generator bounds
    mdl.g = pyo.Var(
        mdl.G,
        bounds=lambda _, i: (float(lo[i]), float(hi[i]))
    )

    # (1g)-(1h): Post-contingency generation variables
    mdl.gk = pyo.Var(
        mdl.GK,
        domain=pyo.NonNegativeReals
    )

    # (1p): Binary APR variables
    mdl.xk = pyo.Var(
        mdl.XK,
        domain=pyo.Binary
    )

    # (1o): Global primary-response signal
    mdl.zk = pyo.Var(
        mdl.Kg,
        bounds=(0, 1)
    )

    # (1n): Nonnegative thermal-limit slack variables
    mdl.eta0 = pyo.Var(
        mdl.BR,
        domain=pyo.NonNegativeReals
    )

    mdl.eta_kg = pyo.Var(
        mdl.EKG,
        domain=pyo.NonNegativeReals
    )

    mdl.eta_ke = pyo.Var(
        mdl.EKE,
        domain=pyo.NonNegativeReals
    )

    # ============================================================
    # (1a): OBJECTIVE FUNCTION
    # Minimize nominal generation cost and thermal violations
    # ============================================================

    mdl.obj = pyo.Objective(
        expr=
        sum(
            c1[i] * mdl.g[i] + c0[i]
            for i in mdl.G
        )
        + penalty * (
            sum(mdl.eta0[l] for l in mdl.BR)
            + sum(
                mdl.eta_kg[k, l]
                for k, l in mdl.EKG
            )
            + sum(
                mdl.eta_ke[k, l]
                for k, l in mdl.EKE
            )
        ),
        sense=pyo.minimize
    )

    demand = float(d.sum())

    # ============================================================
    # (1b): NOMINAL POWER BALANCE
    # 1^T g = 1^T (d + omega)
    # ============================================================

    mdl.balance = pyo.Constraint(
        expr=sum(mdl.g[i] for i in mdl.G) == demand
    )

    # ============================================================
    # (1c): NOMINAL TRANSMISSION FLOW LIMITS
    # f = K(d + omega - Bg)
    # f_lower - eta0 <= f <= f_upper + eta0
    # ============================================================

    mdl.flow = pyo.Expression(
        mdl.BR,
        rule=lambda mm, l:
        sum(
            float(K[l, j]) * (
                float(d[j])
                - sum(
                    float(B[j, i]) * mm.g[i]
                    for i in mm.G
                )
            )
            for j in mm.N
        )
    )

    mdl.base_upper = pyo.Constraint(
        mdl.BR,
        rule=lambda mm, l:
        pyo.Constraint.Skip
        if rates[l] <= 0
        else mm.flow[l] <= float(rates[l]) + mm.eta0[l]
    )

    mdl.base_lower = pyo.Constraint(
        mdl.BR,
        rule=lambda mm, l:
        pyo.Constraint.Skip
        if rates[l] <= 0
        else mm.flow[l] >= -float(rates[l]) - mm.eta0[l]
    )

    # ============================================================
    # (1d): NOMINAL GENERATOR BOUNDS
    # g_lower <= g <= g_upper
    # ============================================================
    # Enforced directly through mdl.g variable bounds.

    # ============================================================
    # (1e): GENERATOR-CONTINGENCY POWER BALANCE
    # 1^T g_k = 1^T (d + omega), for all k in Kg
    # ============================================================

    mdl.balance_k = pyo.Constraint(
        mdl.Kg,
        rule=lambda mm, k:
        sum(mm.gk[k, i] for i in mm.G) == demand
    )

    # ============================================================
    # (1f): GENERATOR-CONTINGENCY TRANSMISSION LIMITS
    # f_k = K(d + omega - Bg_k)
    # f_lower - eta_k <= f_k <= f_upper + eta_k
    # ============================================================

    mdl.flow_k = pyo.Expression(
        mdl.Kg,
        mdl.BR,
        rule=lambda mm, k, l:
        sum(
            float(K[l, j]) * (
                float(d[j])
                - sum(
                    float(B[j, i]) * mm.gk[k, i]
                    for i in mm.G
                )
            )
            for j in mm.N
        )
    )

    mdl.k_upper = pyo.Constraint(
        mdl.EKG,
        rule=lambda mm, k, l:
        pyo.Constraint.Skip
        if rates[l] <= 0
        else mm.flow_k[k, l] <= float(rates[l]) + mm.eta_kg[k, l]
    )

    mdl.k_lower = pyo.Constraint(
        mdl.EKG,
        rule=lambda mm, k, l:
        pyo.Constraint.Skip
        if rates[l] <= 0
        else mm.flow_k[k, l] >= -float(rates[l]) - mm.eta_kg[k, l]
    )

    # ============================================================
    # (1g): POST-CONTINGENCY GENERATOR BOUNDS
    # g_lower_i <= g_ki <= g_upper_i, for i != k
    # ============================================================

    mdl.gen_upper_k = pyo.Constraint(
        mdl.GK,
        rule=lambda mm, k, i:
        mm.gk[k, i] == 0
        if i == k
        else mm.gk[k, i] <= float(hi[i])
    )

    mdl.gen_lower_k = pyo.Constraint(
        mdl.XK,
        rule=lambda mm, k, i:
        mm.gk[k, i] >= float(lo[i])
    )

    # ============================================================
    # (1h): OUTAGED GENERATOR OUTPUT
    # g_kk = 0, for all k in Kg
    # ============================================================
    # Enforced by mdl.gen_upper_k when i == k.

    # ============================================================
    # APR AUXILIARY EXPRESSION
    # Delta_ki = g_ki - g_i - z_k gamma_i (gmax_i - gmin_i)
    # ============================================================

    def apr_delta(mm, k, i):
        return (
            mm.gk[k, i]
            - mm.g[i]
            - mm.zk[k] * gamma * float(hi[i] - lo[i])
        )

    # ============================================================
    # (1i): APR UPPER BIG-M CONSTRAINT
    # Delta_ki <= gmax_i (1 - x_ki)
    # ============================================================

    mdl.apr1 = pyo.Constraint(
        mdl.XK,
        rule=lambda mm, k, i:
        apr_delta(mm, k, i)
        <= float(hi[i]) * (1 - mm.xk[k, i])
    )

    # ============================================================
    # (1j): APR LOWER BIG-M CONSTRAINT
    # -gmax_i (1 - x_ki) <= Delta_ki
    # ============================================================

    mdl.apr2 = pyo.Constraint(
        mdl.XK,
        rule=lambda mm, k, i:
        apr_delta(mm, k, i)
        >= -float(hi[i]) * (1 - mm.xk[k, i])
    )

    # ============================================================
    # (1k): APR SATURATION CONDITION
    # g_i + z_k gamma_i (gmax_i - gmin_i)
    # >= gmax_i (1 - x_ki)
    # ============================================================

    mdl.apr3 = pyo.Constraint(
        mdl.XK,
        rule=lambda mm, k, i:
        mm.g[i]
        + mm.zk[k] * gamma * float(hi[i] - lo[i])
        >= float(hi[i]) * (1 - mm.xk[k, i])
    )

    # ============================================================
    # (1l): APR SATURATED GENERATOR OUTPUT
    # g_ki >= gmax_i (1 - x_ki)
    # ============================================================

    mdl.apr4 = pyo.Constraint(
        mdl.XK,
        rule=lambda mm, k, i:
        mm.gk[k, i]
        >= float(hi[i]) * (1 - mm.xk[k, i])
    )

    # ============================================================
    # (1m): LINE-CONTINGENCY TRANSMISSION LIMITS
    # f_lower - eta_k <= f + f_k L_k <= f_upper + eta_k
    # ============================================================

    mdl.line_upper = pyo.Constraint(
        mdl.EKE,
        rule=lambda mm, k, l:
        pyo.Constraint.Skip
        if rates[l] <= 0
        else (
            mm.flow[l]
            + float(L[l, k]) * mm.flow[k]
            <= float(rates[l]) + mm.eta_ke[k, l]
        )
    )

    mdl.line_lower = pyo.Constraint(
        mdl.EKE,
        rule=lambda mm, k, l:
        pyo.Constraint.Skip
        if rates[l] <= 0
        else (
            mm.flow[l]
            + float(L[l, k]) * mm.flow[k]
            >= -float(rates[l]) - mm.eta_ke[k, l]
        )
    )

    # ============================================================
    # (1n): NONNEGATIVE THERMAL SLACK VARIABLES
    # eta_k >= 0
    # ============================================================
    # Enforced by NonNegativeReals for eta0, eta_kg, eta_ke.

    # ============================================================
    # (1o): PRIMARY RESPONSE SIGNAL BOUNDS
    # 0 <= z_k <= 1
    # ============================================================
    # Enforced directly by mdl.zk bounds=(0, 1).

    # ============================================================
    # (1p): BINARY APR DECISION VARIABLES
    # x_ki in {0, 1}
    # ============================================================
    # Enforced directly by mdl.xk domain=pyo.Binary.

    return mdl, dict(
        K=K,
        L=L,
        B=B,
        Kg=Kg,
        Ke=Ke,
        base=base,
        d=d,
        hi=hi,
        lo=lo
    )

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--case', required=True, help='Path to PGLib-style Excel workbook')
    parser.add_argument('--gamma', type=float, default=0.05, help='APR coefficient (default: 0.05)')
    parser.add_argument('--m-eta', type=float, default=1500.0, help='Thermal slack penalty (default: 1500)')
    parser.add_argument('--line-contingencies',action=argparse.BooleanOptionalAction,default=True,help='Enable non-islanding line contingencies (default: enabled)' )
    parser.add_argument('--time-limit', type=float, default=None)
    parser.add_argument('--mip-gap', type=float, default=1e-4)
    parser.add_argument('--output-dir', type=str, default='output', help='Directory for saving optimization results' )
    parser.add_argument('--tee', action='store_true', help='Print Gurobi log')
    args = parser.parse_args()
    case_path = Path(args.case)
    # Search the default directory if only a filename is provided
    if not case_path.is_absolute() and not case_path.exists():
        case_path = EXCEL_DIR / case_path
    # Automatically append .xlsx if missing
    if not case_path.suffix:
        case_path = case_path.with_suffix(".xlsx")
    if not case_path.is_file():
        parser.error(f"Excel file not found: {case_path}")
    case = load_case(case_path, args.gamma, args.m_eta)
        
    # Output directory relative to this Python script
    output_dir = Path(__file__).resolve().parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    # Example: pglib_opf_case3_lmbd.xlsx -> p1_solution_pglib_opf_case3_lmbd.xlsx
    case_name = case_path.stem
    output_path = output_dir / f"p1_solution_{case_name}.xlsx"

    mdl, data = build_p1(case, args.line_contingencies)
    print(f"Buses={len(case['buses'])}, generators={len(case['gens'])}, branches={len(case['lines'])}, "
          f"generator outages={len(data['Kg'])}, line outages={len(data['Ke'])}")
    solver = pyo.SolverFactory('gurobi')
    if not solver.available(exception_flag=False):
        raise RuntimeError('Gurobi is not available to Pyomo; install gurobipy and configure a valid license')
    if args.time_limit is not None:
        solver.options['TimeLimit'] = args.time_limit
    solver.options['MIPGap'] = args.mip_gap
    result = solver.solve(mdl, tee=args.tee, load_solutions=False)
    term = result.solver.termination_condition
    print(f'Solver status: {result.solver.status}; termination: {term}')
    if len(result.solution) == 0:
        print('No feasible incumbent returned. Check APR capacity, contingency feasibility, and solver log.')
        return 2
    mdl.solutions.load_from(result)
    print(f'Objective: {pyo.value(mdl.obj):.6f}')
    print(f'Base dispatch (MW): {[round(pyo.value(mdl.g[i])*data["base"],6) for i in mdl.G]}')
    gens = case['gens']
    gen_df = pd.DataFrame([dict(gen_ID=gens.iloc[i]['gen_ID'], bus_i=gens.iloc[i]['bus_i'],
        Pg_MW=pyo.value(mdl.g[i])*data['base']) for i in mdl.G])
    cont_df = pd.DataFrame([dict(outage_gen_ID=gens.iloc[k]['gen_ID'],
        gen_ID=gens.iloc[i]['gen_ID'], Pg_MW=pyo.value(mdl.gk[k,i])*data['base'],
        z=pyo.value(mdl.zk[k]), x=(int(round(pyo.value(mdl.xk[k,i]))) if i != k else None))
        for k in mdl.Kg for i in mdl.G])
    slack_df = pd.DataFrame([dict(type='base', outage=None, line_ID=case['lines'].iloc[l]['line_ID'],
        slack_MW=pyo.value(mdl.eta0[l])*data['base']) for l in mdl.BR] +
        [dict(type='generator', outage=gens.iloc[k]['gen_ID'], line_ID=case['lines'].iloc[l]['line_ID'],
        slack_MW=pyo.value(mdl.eta_kg[k,l])*data['base']) for k,l in mdl.EKG] +
        [dict(type='line', outage=case['lines'].iloc[k]['line_ID'], line_ID=case['lines'].iloc[l]['line_ID'],
        slack_MW=pyo.value(mdl.eta_ke[k,l])*data['base']) for k,l in mdl.EKE])
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        gen_df.to_excel(writer, sheet_name='nominal_dispatch', index=False)
        cont_df.to_excel(writer, sheet_name='generator_contingencies', index=False)
        slack_df.to_excel(writer, sheet_name='thermal_slacks', index=False)
        pd.DataFrame([dict(termination=str(term), objective=pyo.value(mdl.obj), gamma=args.gamma,
            line_contingencies=args.line_contingencies)]).to_excel(writer, sheet_name='summary', index=False)
    print(f'Saved {output_path}')
    if term != TerminationCondition.optimal:
        print('WARNING: Solution is an incumbent, not a certified optimum.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
