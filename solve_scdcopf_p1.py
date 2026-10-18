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
from apr_binary_search import apr_binary_search
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

    # Nominal demand
    d_nominal = (
        case['buses']['Pd'].to_numpy(dtype=float) / base
    )

    # Selected uncertainty realization (per unit)
    omega = np.asarray(
            case.get('omega', np.zeros_like(d_nominal)),
            dtype=float
        )

    if omega.shape != d_nominal.shape:
        raise ValueError("Uncertainty dimension does not match buses")

    # Realized demand
    d = d_nominal + omega

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
    # mutable Pyomo parameters
    # ============================================================
    mdl.d = pyo.Param(
        mdl.N,
        initialize={
            j: float(d[j])
            for j in range(n)
        },
        mutable=True
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

    # ============================================================
    # (1b): NOMINAL POWER BALANCE
    # 1^T g = 1^T (d + omega)
    # ============================================================

    mdl.balance = pyo.Constraint(
                    expr=sum(mdl.g[i] for i in mdl.G)
                    == sum(mdl.d[j] for j in mdl.N)
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
                mm.d[j]
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
                    sum(mm.gk[k, i] for i in mm.G)
                    == sum(mm.d[j] for j in mm.N)
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
                mm.d[j]
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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--case', required=True)
    # change gamma! if needed
    parser.add_argument('--gamma', type=float, default=0.05)
    parser.add_argument('--m-eta', type=float, default=1500.0)

    parser.add_argument(
        '--line-contingencies',
        action=argparse.BooleanOptionalAction,
        default=True
    )

    parser.add_argument('--time-limit', type=float, default=None)
    parser.add_argument('--mip-gap', type=float, default=1e-4)
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--tee', action='store_true')

    # Uncertainty arguments
    parser.add_argument(
        '--sigma-scaling',
        type=float,
        default=None,
        help='Enable uncertainty when specified'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=19,
        help='Seed identifying the uncertainty dataset'
    )

    parser.add_argument(
        '--scenario-id',
        type=int,
        default=None,
        help='Solve one scenario; default is all scenarios'
    )

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
    
    # ============================================================
    # LOAD UNCERTAINTY SCENARIOS
    # ============================================================

    if args.sigma_scaling is None:
        # No uncertainty: solve the nominal case only
        scenarios = [(None, np.zeros(len(case["buses"])))]
    else:
        sigma_tag = f"{args.sigma_scaling:g}"

        uncertainty_path = (
            Path(__file__).resolve().parent
            / "uncertainty"
            / f"omega_{case_path.stem}_sigma{sigma_tag}_seed{args.seed}.npz"
        )


        if not uncertainty_path.is_file():
            parser.error(f"Uncertainty file not found: {uncertainty_path}")

        with np.load(uncertainty_path) as uncertainty_data:
            omega_all = uncertainty_data["omega"]
            saved_bus_ids = uncertainty_data["bus_ids"]

            if "sigma_scaling" in uncertainty_data:
                saved_sigma = float(uncertainty_data["sigma_scaling"])
                if not np.isclose(saved_sigma, args.sigma_scaling):
                    raise ValueError("Sigma scaling mismatch")

            if "seed" in uncertainty_data:
                if int(uncertainty_data["seed"]) != args.seed:
                    raise ValueError("Seed mismatch")

        current_bus_ids = case["buses"]["bus_i"].to_numpy()

        if not np.array_equal(saved_bus_ids, current_bus_ids):
            raise ValueError("Bus ordering mismatch")

        if omega_all.ndim != 2 or omega_all.shape[1] != len(current_bus_ids):
            raise ValueError("Invalid uncertainty matrix dimensions")

        if args.scenario_id is None:
            scenarios = list(enumerate(omega_all))
        else:
            if not 0 <= args.scenario_id < len(omega_all):
                parser.error("Invalid scenario ID")

            scenarios = [(args.scenario_id, omega_all[args.scenario_id])]

    if args.sigma_scaling is None:
        print("Mode: Nominal SCDCOPF (omega = 0)")
    else:
        print(f"Uncertainty file: {uncertainty_path.name}")
        print(f"Seed: {args.seed}")
        print(f"Sigma scaling: {args.sigma_scaling}")

    print(f"Number of scenarios to solve: {len(scenarios)}")
    
    # ============================================================
    # OUTPUT DIRECTORY FOR EACH TEST CASE
    # ============================================================

    case_name = case_path.stem

    output_dir = (
        Path(__file__).resolve().parent
        / args.output_dir
        / case_name
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # SOLVE ALL SELECTED SCENARIOS
    # ============================================================

    solver = pyo.SolverFactory("gurobi")

    if not solver.available(exception_flag=False):
        raise RuntimeError("Gurobi is not available")

    if args.time_limit is not None:
        solver.options["TimeLimit"] = args.time_limit

    solver.options["MIPGap"] = args.mip_gap

    # ============================================================
    # BUILD P1 ONCE
    # ============================================================

    # Nominal demand in per unit
    d_nominal = (
        case["buses"]["Pd"].to_numpy(dtype=float)
        / case["base"]
    )

    # Initialize the model with nominal demand
    case["omega"] = np.zeros_like(d_nominal)

    mdl, data = build_p1(
        case,
        include_lines=args.line_contingencies
    )

    # ============================================================
    # SOLVE ALL SELECTED SCENARIOS
    # ============================================================

    summary_records = []

    for scenario_id, omega in scenarios:

        # Realized demand for this scenario
        d_realized = d_nominal + np.asarray(
            omega, dtype=float
        )

        # Update mutable demand parameters
        for j in mdl.N:
            mdl.d[j].set_value(float(d_realized[j]))

        print("\n" + "=" * 65)
        print(
            f"Solving scenario: "
            f"{scenario_id if scenario_id is not None else 'nominal'}"
        )
        print("=" * 65)

        # Solve the same Pyomo model with updated demand
        result = solver.solve(
            mdl,
            tee=args.tee,
            load_solutions=False
        )

        
        term = result.solver.termination_condition
        print(f"Termination: {term}")

        if len(result.solution) == 0:
            summary_records.append({
                "scenario_id": scenario_id,
                "termination": str(term),
                "objective": np.nan
            })
            continue

        mdl.solutions.load_from(result)

        # ============================================================
        # APR BINARY-SEARCH VALIDATION
        # ============================================================

        g_star = np.array([
            pyo.value(mdl.g[i])
            for i in mdl.G
        ])

        demand = sum(
            pyo.value(mdl.d[j])
            for j in mdl.N
        )

        gmin = data["lo"]
        gmax = data["hi"]

        print("\nAPR BINARY-SEARCH VALIDATION")

        for k in mdl.Kg:

            # Gurobi post-contingency dispatch
            gk_star = np.array([
                pyo.value(mdl.gk[k, i])
                for i in mdl.G
            ])

            # Reconstruct using APR binary search
            gk_bs, zk_bs, xk_bs, feasible = apr_binary_search(
                g=g_star,
                gmin=gmin,
                gmax=gmax,
                gamma=args.gamma,
                k=k,
                demand=demand
            )

            if not feasible:
                print(f"Contingency {k}: APR reconstruction failed")
                continue

            # Compare reconstructed dispatch against Gurobi
            error_MW = np.max(
                np.abs(gk_bs - gk_star)
            ) * data["base"]

            print(
                f"Contingency {k}: "
                f"z_BS={zk_bs:.8f}, "
                f"dispatch_error={error_MW:.8e} MW"
            )

        # ============================================================
        # EXISTING RESULT EXTRACTION
        # ============================================================

        objective = pyo.value(mdl.obj)
        base = data["base"]
        gens = case["gens"]
        lines = case["lines"]


        # Nominal dispatch
        gen_df = pd.DataFrame([
            {
                "gen_ID": gens.iloc[i]["gen_ID"],
                "bus_i": gens.iloc[i]["bus_i"],
                "Pg_MW": pyo.value(mdl.g[i]) * base
            }
            for i in mdl.G
        ])

        # Generator-contingency dispatch
        cont_df = pd.DataFrame([
            {
                "outage_gen_ID": gens.iloc[k]["gen_ID"],
                "gen_ID": gens.iloc[i]["gen_ID"],
                "Pg_MW": pyo.value(mdl.gk[k, i]) * base,
                "z": pyo.value(mdl.zk[k]),
                "x": (
                    int(round(pyo.value(mdl.xk[k, i])))
                    if i != k else None
                )
            }
            for k in mdl.Kg for i in mdl.G
        ])

        # Thermal slacks
        slack_df = pd.DataFrame(
            [
                {
                    "type": "base",
                    "outage": None,
                    "line_ID": lines.iloc[l]["line_ID"],
                    "slack_MW": pyo.value(mdl.eta0[l]) * base
                }
                for l in mdl.BR
            ]
            +
            [
                {
                    "type": "generator",
                    "outage": gens.iloc[k]["gen_ID"],
                    "line_ID": lines.iloc[l]["line_ID"],
                    "slack_MW": pyo.value(mdl.eta_kg[k, l]) * base
                }
                for k, l in mdl.EKG
            ]
            +
            [
                {
                    "type": "line",
                    "outage": lines.iloc[k]["line_ID"],
                    "line_ID": lines.iloc[l]["line_ID"],
                    "slack_MW": pyo.value(mdl.eta_ke[k, l]) * base
                }
                for k, l in mdl.EKE
            ]
        )


        # ============================================================
        # CENTRALIZED CONTINUOUS-RELAXATION EXPERIMENT
        # ============================================================

        relaxed = mdl.clone()

        # Change only the binary domains: {0, 1} --> [0, 1].
        for k, i in relaxed.XK:
            relaxed.xk[k, i].domain = pyo.UnitInterval
            relaxed.xk[k, i].setlb(0.0)
            relaxed.xk[k, i].setub(1.0)

        lp_result = solver.solve(
            relaxed,
            tee=args.tee,
            load_solutions=False
        )

        print("\nCONTINUOUS-RELAXATION EXPERIMENT")
        print("LP termination:", lp_result.solver.termination_condition)

        if lp_result.solver.termination_condition == TerminationCondition.optimal:
            relaxed.solutions.load_from(lp_result)

            milp_objective = pyo.value(mdl.obj)
            lp_objective = pyo.value(relaxed.obj)

            print(f"MILP objective: {milp_objective:.8f}")
            print(f"LP objective:   {lp_objective:.8f}")
            print(f"Objective difference: {milp_objective - lp_objective:.8f}")

            g_lp = np.array([
                pyo.value(relaxed.g[i]) for i in relaxed.G
            ])

            response = args.gamma * (data["hi"] - data["lo"])

            for k in relaxed.Kg:
                y_k = np.array([
                    pyo.value(relaxed.gk[k, i]) for i in relaxed.G
                ])
                z_k = pyo.value(relaxed.zk[k])

                # Physical APR outputs at the CURRENT relaxed signal.
                apr_at_current_signal = np.minimum(
                    g_lp + z_k * response,
                    data["hi"]
                )
                apr_at_current_signal[k] = 0.0

                delta_MW = (
                    np.abs(y_k - apr_at_current_signal).sum()
                    * data["base"]
                )

                x_values = [
                    pyo.value(relaxed.xk[k, i])
                    for i in relaxed.G if i != k
                ]

                print(
                    f"Contingency {k}: "
                    f"z_LP={z_k:.8f}, "
                    f"x_LP={x_values}, "
                    f"APR inconsistency={delta_MW:.8e} MW"
                )

                # Recover a BALANCING physical response at the LP dispatch.
                demand_lp = sum(
                    pyo.value(relaxed.d[j]) for j in relaxed.N
                )

                p_k, z_recovered, _, feasible = apr_binary_search(
                    g=g_lp,
                    gmin=data["lo"],
                    gmax=data["hi"],
                    gamma=args.gamma,
                    k=k,
                    demand=demand_lp
                )

                if not feasible:
                    print(
                        f"Contingency {k}: physical recovery failed "
                        "at the LP preventive dispatch"
                    )
                    continue

                recovery_change_MW = (
                    np.abs(p_k - y_k).sum() * data["base"]
                )

                # Centralized experiment: dispatch copies do not exist,
                # so r_k = 0 and the theoretical bound is 2 * delta_k.
                bound_MW = 2.0 * delta_MW

                # Account for numerical balance residuals.
                numerical_allowance_MW = (
                    abs(y_k.sum() - demand_lp)
                    + abs(p_k.sum() - demand_lp)
                ) * data["base"]

                print("  LP outputs [MW]:",
                      np.round(y_k * data["base"], 8))
                print("  APR at LP signal [MW]:",
                      np.round(apr_at_current_signal * data["base"], 8))
                print("  Recovered outputs [MW]:",
                      np.round(p_k * data["base"], 8))
                print(f"  Recovered signal: {z_recovered:.8f}")
                print(f"  Actual recovery change: "
                      f"{recovery_change_MW:.8e} MW")
                print(f"  Theoretical bound 2*delta: "
                      f"{bound_MW:.8e} MW")
                print("  Bound satisfied within numerical allowance:",
                      recovery_change_MW
                      <= bound_MW + numerical_allowance_MW + 1e-7)
        else:
            print("An optimal LP solution was not obtained; comparison skipped.")

        # Save each scenario independently
        if scenario_id is None:
            filename = f"p1_solution_{case_name}_nominal.xlsx"
        else:
            filename = (
                f"p1_solution_{case_name}_"
                f"sigma{sigma_tag}_seed{args.seed}_"
                f"scenario{scenario_id}.xlsx"
            )

        output_path = output_dir / filename

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            gen_df.to_excel(
                writer, sheet_name="nominal_dispatch", index=False
            )
            cont_df.to_excel(
                writer, sheet_name="generator_contingencies", index=False
            )
            slack_df.to_excel(
                writer, sheet_name="thermal_slacks", index=False
            )
            pd.DataFrame([{
                "scenario_id": scenario_id,
                "termination": str(term),
                "objective": objective,
                "gamma": args.gamma,
                "line_contingencies": args.line_contingencies
            }]).to_excel(
                writer, sheet_name="summary", index=False
            )

        summary_records.append({
            "scenario_id": scenario_id,
            "termination": str(term),
            "objective": objective
        })

        print(f"Objective: {objective:.6f}")
        print(f"Saved: {output_path.name}")

    # Save combined summary for all scenarios
    if args.sigma_scaling is None:
        summary_name = f"p1_summary_{case_name}_nominal.xlsx"
    else:
        summary_name = (
            f"p1_summary_{case_name}_"
            f"sigma{sigma_tag}_seed{args.seed}.xlsx"
        )

    summary_path = output_dir / summary_name

    pd.DataFrame(summary_records).to_excel(
        summary_path,
        index=False
    )

    print(f"\nCompleted {len(scenarios)} scenario attempts.")
    print(f"Summary saved: {summary_path}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
