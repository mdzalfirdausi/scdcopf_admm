#!/usr/bin/env python3
"""Compare saved original SCOPF MILP and budget-LP operating costs.

No optimization is performed. Both CSVs must concern the same input case,
demand, gamma, cost convention, and contingency sets. The operating objective
is generation cost + base/line thermal penalties + generator-outage thermal
penalties; the optimized APR-slack objective is reported separately.

Example:
 python compare_scdcopf_costs.py --mip-summary output_apr_mip_compare118/summary.csv \
   --slack-summary output_apr_205_case118/summary.csv --epsilon-unit 2.075

Install dependency: python -m pip install pandas numpy
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mip-summary', required=True)
    p.add_argument('--slack-summary', required=True)
    p.add_argument('--epsilon-unit', type=float, help='Select the saved epsilon value: 2.075 means a 207.5 percent increase.')
    p.add_argument('--output-dir', default='output_cost_comparison')
    args = p.parse_args()
    exact = pd.read_csv(args.mip_summary)
    candidates = pd.read_csv(args.slack_summary)
    columns = ['generation_cost', 'line_thermal_penalty', 'unit_thermal_penalty', 'operating_objective']
    for name, table in [('MILP', exact), ('slack', candidates)]:
        missing = set(['model', 'status']+columns)-set(table.columns)
        if missing: raise ValueError(f'{name} summary missing columns: {sorted(missing)}')
    reference = exact[(exact.model == 'exact_APR') & (exact.status == 'optimal')]
    if 'weight' in reference: reference = reference[np.isclose(reference.weight, 0.)]
    if len(reference) != 1:
        raise ValueError('MILP file must contain exactly one optimal exact_APR row with W=0, the ORIGINAL operating objective.')
    mip = reference.iloc[0]
    if 'provisional_original_APR_feasible' in mip and not truth(mip['provisional_original_APR_feasible']):
        raise ValueError('Saved MILP point failed the original APR feasibility check')
    candidates = candidates[candidates.model == 'budget_LP'].copy()
    if args.epsilon_unit is not None:
        if 'epsilon_unit' not in candidates: raise ValueError('Slack file has no epsilon_unit column')
        candidates = candidates[np.isclose(candidates.epsilon_unit, args.epsilon_unit, atol=1e-10, rtol=0)]
    if candidates.empty: raise ValueError('No matching budget_LP result found')
    for key in ['gamma', 'demand_scale', 'thermal_mode']:
        if key in mip and key in candidates:
            if any(not same(mip[key], value) for value in candidates[key]):
                raise ValueError(f'Incompatible saved assumptions: {key}')
    print('Compare only runs using the SAME workbook, contingencies, cost coefficients, and demand.')
    print('The APR-slack objective is separate from the operating-cost comparison below.')
    records = []
    mapping = [('Generation C_g', 'generation_cost', 'generation'),
               ('Base/line thermal C_l', 'line_thermal_penalty', 'line'),
               ('Generator-outage thermal C_u', 'unit_thermal_penalty', 'unit'),
               ('TOTAL operating cost', 'operating_objective', None)]
    for number, (_, row) in enumerate(candidates.iterrows(), 1):
        if row.status != 'optimal':
            print(f'Note: candidate status={row.status}; this is not a certified optimum of its LP.')
        label = row.get('budget_point', number)
        epsilon = row.get('epsilon_unit', np.nan)
        verified = truth(row.get('provisional_APR_and_budget_feasible', False))
        print(f'\nCandidate {label}; epsilon_unit={epsilon}; APR and budget verified={verified}')
        print(f'APR mismatch [MW]: {row.get("max_APR_L1_error_MW", np.nan):.10g}')
        print(f'Optimized objective ({row.get("objective_kind", "unspecified")}): {row.get("optimized_budget_objective", np.nan):.10g}')
        for source in [mip, row]:
            parts = [float(source[k]) for k in columns[:3]]
            total = float(source.operating_objective)
            if not np.isfinite(parts+[total]).all() or not np.isclose(sum(parts), total, atol=1e-5, rtol=1e-9):
                raise ValueError('A cost row is nonfinite or its components do not sum to the operating objective')
        table = []
        for name, field, family in mapping:
            a, b = float(mip[field]), float(row[field])
            cap = row.get(f'provisional_{family}_budget_cap', np.nan) if family else np.nan
            record = dict(budget_point=label, epsilon_unit=epsilon, component=name,
                          MILP=a, slack_model=b, difference=b-a,
                          percentage_difference=100*(b-a)/abs(a) if a != 0 else np.nan,
                          candidate_APR_and_budget_verified=verified, candidate_budget_cap=cap)
            if family and np.isfinite(cap):
                tolerance = max(1e-4, 1e-8*max(1., abs(cap)))
                record['known_MILP_within_component_cap'] = bool(a <= cap+tolerance)
                record['cap_increase_needed_to_admit_known_MILP'] = max(0., a-cap)
                # For automatically referenced split mode, the saved envelope LP
                # has the same original-cost optimum as the envelope reference.
                # It can have a DIFFERENT optimal component allocation, so do
                # not use its allocation to infer the actual reference epsilons.
                # Instead reconstruct the actual reference from its saved cap.
                eps_key = 'epsilon_generation' if family == 'generation' else 'epsilon_'+family
                allowance_key = 'absolute_'+family+'_allowance'
                allowance = float(row.get(allowance_key, 0.)) if family != 'generation' else 0.
                ev = row.get(eps_key, np.nan)
                if np.isfinite(ev):
                    ref = (float(cap)-allowance)/(1.+float(ev))
                    record['actual_reference_component'] = ref
                    record['relative_increase_pct_needed_to_admit_known_MILP'] = 100*max(0., (a-allowance)/ref-1.) if ref > 0 else (0. if a <= allowance+tolerance else np.inf)
            records.append(record)
            table.append(record)
        print(pd.DataFrame(table)[['component','MILP','slack_model','difference','percentage_difference']].to_string(index=False, float_format=lambda x:f'{x:,.2f}'))
        excluded = [r['component'] for r in table if r.get('known_MILP_within_component_cap') is False]
        print('Known MILP solution excluded by these component caps: '+(', '.join(excluded) if excluded else 'none identified'))
        if not verified: print('The cost difference above is for an APR-inconsistent or unverified raw LP point; it is not a feasible-SCOPF optimality gap.')
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out/'cost_comparison.csv', index=False)
    print(f'\nSaved: {(out/"cost_comparison.csv").resolve()}')
    print('Admitting the known MILP point does not guarantee that minimizing APR slacks selects that point.')
    return 0


def truth(value):
    if isinstance(value, str): return value.strip().lower() == 'true'
    return False if pd.isna(value) else bool(value)


def same(a, b):
    try: return bool(np.isclose(float(a), float(b), atol=1e-10, rtol=0))
    except (ValueError, TypeError): return str(a) == str(b)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, FileNotFoundError) as exc:
        raise SystemExit(f'ERROR: {exc}')
