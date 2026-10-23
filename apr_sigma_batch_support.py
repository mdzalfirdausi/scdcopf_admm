#!/usr/bin/env python3
"""Demand-sensitivity study: positive-weight LP search + original MIP LP relaxation.
No mixed-integer optimization is performed. See README.md for units and timing.
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback

import numpy as np
import apr_sigma_batch_core as core

VERSION = '2026-10-23-apr-sigma-study-npz-v2'


def write_json(path, value):
    """Atomic, strict JSON; no NaN/Infinity or pickle required."""
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [clean(v) for v in x]
        if isinstance(x, np.ndarray):
            return clean(x.tolist())
        if isinstance(x, np.generic):
            return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_json(path, fallback=None):
    return json.loads(path.read_text()) if path.exists() else fallback


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)
    def flush(self):
        for stream in self.streams:
            stream.flush()


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', help='Excel case, with or without .xlsx suffix')
    p.add_argument('--case-dir', default='../excel_outputs')
    p.add_argument('--output-dir', default='output_apr_sigma_study')
    p.add_argument('--samples', type=int, default=20, help='R samples PER sigma scale')
    p.add_argument('--sigma-scales', nargs='+', type=float, default=[.01,.02,.03,.04,.05])
    p.add_argument('--seed', type=int, default=19)
    p.add_argument('--uncertainty-dir', help='Read saved omega_CASE_sigmaALPHA_seedSEED.npz files (omega in pu)')
    p.add_argument('--solver', choices=['gurobi', 'highs'], default='gurobi')
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--penalty-per-mw', type=float, default=1500.,
                   help='Thermal penalty per MW; multiplied by case baseMVA exactly once')
    p.add_argument('--weight-start', type=float, help='Default: M_eta = penalty-per-mw * baseMVA')
    p.add_argument('--weight-tol', type=float, default=.001)
    p.add_argument('--weight-max', type=float, default=1e9)
    p.add_argument('--weight-max-solves', type=int, default=40)
    p.add_argument('--gamma', type=float, default=1.)
    p.add_argument('--demand-scale', type=float, default=1.)
    p.add_argument('--lp-time-limit', type=float, default=300., help='Seconds PER weight test')
    p.add_argument('--relaxation-time-limit', type=float, default=600.)
    p.add_argument('--apr-tol-mw', type=float, default=1e-4)
    p.add_argument('--feas-tol-mw', type=float, default=1e-4)
    p.add_argument('--full-eta-diagnostics', action='store_true', help='Large per-line/state CSVs')
    p.add_argument('--tee', action='store_true', help='Show solver logs')
    p.add_argument('--resume', action='store_true', help='Resume same settings; retain finished samples')
    p.add_argument('--summarize-only', action='store_true', help='Rebuild tables/plots from saved result.json files')
    p.add_argument('--version', action='version', version=VERSION)
    return p.parse_args()


def solver_args(cli, case_path, base):
    args = core.parser().parse_args(['--case', str(case_path)])
    args.backend = args.solver = cli.solver
    args.base = base
    args.penalty = cli.penalty_per_mw * base
    args.gamma = cli.gamma
    args.demand_scale = cli.demand_scale
    args.threads = cli.threads
    args.tee = cli.tee
    args.time_limit = cli.lp_time_limit
    args.apr_tol_mw = cli.apr_tol_mw
    args.feas_tol_mw = cli.feas_tol_mw
    args.thermal_mode = 'soft'
    args.islanding = 'skip'
    args.weight_bisection = True
    args.slack_normalization = 'none'
    args.weight_start = cli.weight_start if cli.weight_start is not None else args.penalty
    args.weight_max = cli.weight_max
    args.weight_tol = cli.weight_tol
    args.weight_rtol = 0.
    args.weight_max_solves = cli.weight_max_solves
    args.eta_summary_only = not cli.full_eta_diagnostics
    args.save_best_lp_vector = True
    args.reference_time_limit = cli.relaxation_time_limit
    return args


def capacity_error(case, outages, tol):
    if case['D'] < case['lo'].sum()-tol or case['D'] > case['hi'].sum()+tol:
        return 'Nominal demand is outside aggregate generation bounds'
    for k in outages:
        if case['D'] > case['hi'].sum()-case['hi'][k]+tol:
            return f'Insufficient surviving capacity after generator index {k} outage'
        if case['D'] < case['lo'].sum()-case['lo'][k]-tol:
            return f'Demand below surviving minimum output after generator index {k} outage'
    return None


def run_sample(baseline, outages, args, root, alpha, sample, seed, shared_preprocess, saved_omega=None):
    started = perf_counter()
    directory = root/'samples'/f'alpha_{alpha:.12g}'/f'sample_{sample:04d}'
    directory.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (directory/f'attempt_{attempt:03d}').exists():
        attempt += 1
    work = directory/f'attempt_{attempt:03d}'
    work.mkdir()
    lpdir = work/'lp_search'
    lpdir.mkdir()
    result = dict(alpha=alpha, sample=sample, initial_weight=args.weight_start,
                  apr_feasible=False, constructed_ip_verified=False, search_completed=False,
                  selected_weight=None, original_objective_U=None, weighted_lp_objective=None,
                  lp_relaxation_bound_L=None, bound_gap_pct=None, lp_weight_tests=0,
                  total_lp_time_s=shared_preprocess, reference_lp_time_s=0.,
                  search_status='not_started', reference_status='not_started',
                  work_directory=str(work.relative_to(root)), finished=False)
    # Same sample index uses the same standardized z across scales (paired experiment).
    # Independent draws across buses and sample indices; no clipping/resampling.
    demand0 = baseline['d']*baseline['base']
    if saved_omega is None:
        rng = np.random.default_rng(np.random.SeedSequence([seed, sample]))
        z = rng.standard_normal(len(baseline['d']))
        omega = alpha*np.abs(demand0)*z
    else:
        # The user's generator saved omega IN PU. Convert it to MW once.
        omega_pu = saved_omega[sample-1]
        omega = omega_pu*baseline['base']
        denominator = alpha*np.abs(baseline['d'])
        z = np.divide(omega_pu,denominator,out=np.zeros_like(omega_pu),where=denominator>0)
        result['uncertainty_row_zero_based'] = sample-1
    demand = demand0 + omega
    np.savez_compressed(work/'demand_sample.npz', z=z, omega_MW=omega, demand_MW=demand,
                        baseline_demand_MW=demand0, bus_ID=baseline['bus'].bus_i.to_numpy())
    write_csv(work/'demand_sample.csv', [dict(bus_ID=core.id_key(bus), baseline_MW=float(d0),
              standardized_z=float(zi), omega_MW=float(w), sampled_MW=float(di))
              for bus,d0,zi,w,di in zip(baseline['bus'].bus_i, demand0,z,omega,demand)])
    case = dict(baseline)
    case.pop('_network_blocks_cache', None)
    case['d'] = demand/case['base']
    case['D'] = float(case['d'].sum())
    case['bus'] = baseline['bus'].copy()
    case['bus']['Pd'] = demand
    result['total_demand_MW'] = float(demand.sum())
    candidate = None
    with (work/'experiment.log').open('w') as log, redirect_stdout(Tee(sys.stdout,log)), redirect_stderr(Tee(sys.stderr,log)):
        try:
            issue = capacity_error(case, outages, args.feas_tol_mw/case['base'])
            if issue:
                result.update(search_status='capacity_infeasible', reference_status='not_run_capacity_infeasible', message=issue)
            else:
                core.run_weight_bisection(case,outages,args,None,lpdir,
                    dict(study_version=VERSION, alpha=alpha, sample=sample,
                         penalty_per_pu=args.penalty, demand_file='../demand_sample.npz'),0.)
                search = read_json(lpdir/'weight_search_summary.json', {})
                result.update(search_status=search.get('termination','unknown'),
                              search_completed=bool(search.get('search_completed')),
                              lp_weight_tests=search.get('solve_count',0),
                              weight_interval_width=search.get('interval_width'))
                if search.get('best_passing_weight') is not None:
                    candidate = core.load_lp_candidate(case,outages,work)
                    common = core.evaluate_common(case,outages,*candidate,args)
                    result.update(common)
                    best = read_json(lpdir/'metadata.json', {}).get('best_result') or {}
                    verified = bool(common['original_soft_model_feasible'] and
                                    best.get('both_APR_slacks_positive_pairs') == 0)
                    result.update(apr_feasible=verified,
                                  selected_weight=search['best_passing_weight'] if verified else None,
                                  weighted_lp_objective=best.get('optimized_objective'),
                                  original_objective_U=common['original_objective'] if verified else None)
                    if verified:
                        np.savez_compressed(work/'candidate_dispatch.npz', g=candidate[0], y=candidate[1],
                                            z=candidate[2], outages=np.asarray(outages), baseMVA=case['base'])
                    else:
                        candidate = None
                result['total_lp_time_s'] += perf_counter()-started
                # Build the ORIGINAL disjunctive model, then relax integrality only.
                t = perf_counter()
                exact = core.build_model(case,outages,args,True,None)
                result['reference_model_build_s'] = perf_counter()-t
                if candidate is not None:
                    t = perf_counter()
                    point = core.make_mip_start(case,outages,exact,candidate,args)
                    np.savez_compressed(work/'constructed_ip_solution.npz', vector=point,
                        binary_indices=np.asarray(list(exact.x.values()),dtype=int),
                        binary_pairs=np.asarray(list(exact.x.keys()),dtype=int),
                        g=exact.g,y=exact.y,z=exact.z,nominal_flow=exact.nominal_flow,
                        eta_line=exact.eta_line,eta_unit=exact.eta_unit)
                    ip_cost = float(exact.c0@point + exact.constant)
                    if abs(ip_cost-result['original_objective_U']) > 1e-7*max(1.,abs(ip_cost)):
                        raise ValueError('Constructed-IP cost does not match independent original objective')
                    result['constructed_ip_verified'] = True
                    result['ip_construction_validation_s'] = perf_counter()-t
                    result['total_lp_time_s'] += result['ip_construction_validation_s']
                    del point
                relaxed = replace(exact, integer=np.zeros_like(exact.integer))
                del exact
                assert not relaxed.integer.any(), 'This study must never optimize a MIP'
                bound_args = argparse.Namespace(**vars(args))
                bound_args.time_limit = args.reference_time_limit
                t = perf_counter()
                reference, solution = core.solve_model(relaxed,0.,bound_args)
                result['reference_lp_time_s'] = result['reference_model_build_s'] + perf_counter()-t
                result['reference_status'] = reference['status']
                write_json(work/'reference_lp_result.json', reference)
                if reference['status']=='optimal' and reference.get('solver_point_verified'):
                    lower = float(reference['solver_penalized_objective'])
                    result['lp_relaxation_bound_L'] = lower
                    if solution is not None:
                        np.savez_compressed(work/'reference_lp_solution.npz', vector=solution)
                    if result['constructed_ip_verified']:
                        upper = result['original_objective_U']
                        if lower > upper + 1e-7*max(1.,abs(upper)):
                            result['reference_status'] = 'invalid_bound_above_candidate'
                            result['lp_relaxation_bound_L'] = None
                        elif abs(upper)>1e-12:
                            result['bound_gap_pct'] = 100.*(upper-lower)/abs(upper)
                            # Suppress only a numerically negligible negative roundoff.
                            if -1e-5 < result['bound_gap_pct'] < 0:
                                result['bound_gap_pct'] = 0.
                del relaxed, solution
        except Exception as error:
            result.update(error_type=type(error).__name__, error_message=str(error))
            (work/'error.txt').write_text(traceback.format_exc())
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            search = read_json(lpdir/'weight_search_summary.json',{})
            result['lp_weight_tests'] = search.get('solve_count',result['lp_weight_tests'])
            result['search_status'] = search.get('termination', 'error')
            if result['search_status']=='running':
                result['search_status']='error'
            # Keep LP time separate when an error occurred after reference construction.
            if result['total_lp_time_s']==shared_preprocess:
                result['total_lp_time_s'] += perf_counter()-started
        finally:
            if result['total_lp_time_s']==shared_preprocess:
                result['total_lp_time_s'] += perf_counter()-started
            result['scenario_wall_time_s'] = shared_preprocess + perf_counter()-started
            result['finished'] = True
            write_json(directory/'result.json',result)
    return result


def stats(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) is not None and np.isfinite(row[field])]
    return [float(fn(values)) if values else None for fn in (np.min,np.mean,np.max)]


def summarize(root, settings, rows):
    summaries=[]
    for alpha in settings['sigma_scales']:
        subset = [r for r in rows if r['alpha']==alpha]
        feasible = [r for r in subset if r.get('constructed_ip_verified')]
        gaps = [r for r in feasible if r.get('bound_gap_pct') is not None]
        n = settings['samples']
        record = dict(alpha=alpha, R=n, finished_samples=len(subset), initial_weight=settings['initial_weight'],
                      apr_feasible_pct=100.*sum(bool(r.get('apr_feasible')) for r in subset)/n,
                      completed_search_pct=100.*sum(bool(r.get('search_completed')) for r in subset)/n,
                      verified_ip_count=len(feasible), valid_gap_count=len(gaps))
        for label, data, key in [('weight',feasible,'selected_weight'),('gap_pct',gaps,'bound_gap_pct'),
                                 ('lp_tests',subset,'lp_weight_tests'),('time_s',subset,'total_lp_time_s')]:
            record.update(zip([label+'_min',label+'_avg',label+'_max'],stats(data,key)))
        summaries.append(record)
    write_csv(root/'scenario_results.csv',rows)
    write_csv(root/'table_B.csv',summaries)
    latex_table(root, summaries, settings)
    plots(root,rows,settings)
    return summaries


def latex_table(root, rows, settings):
    def number(x):
        if x is None:
            return '--'
        return f'{x:,.4g}'
    lines = [r'% Requires: \usepackage{booktabs,graphicx}',r'\begin{table}[htbp]',r'\centering',
        r'\caption{Demand sensitivity of the proposed LP approach on '+settings['case_name'].replace('_',r'\_')+r'.}',
        r'\label{tab:lp_sigma_sensitivity}',r'\scriptsize',r'\setlength{\tabcolsep}{4pt}',
        r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{rrrrr rrr rrr rrr rrr}',r'\toprule',
        r'\multicolumn{5}{c}{Scenario and search settings} & \multicolumn{3}{c}{Selected weight $W_f$} & \multicolumn{3}{c}{Gap to LP bound (\%)} & \multicolumn{3}{c}{LP weight tests} & \multicolumn{3}{c}{Total LP time (s)} \\',
        r'\cmidrule(lr){1-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(lr){12-14}\cmidrule(lr){15-17}',
        r'$\alpha$ & $R$ & $W_0$ & \shortstack{APR-feasible\\(\%)} & \shortstack{Completed\\(\%)} & Min. & Avg. & Max. & Min. & Avg. & Max. & Min. & Avg. & Max. & Min. & Avg. & Max. \\',r'\midrule']
    for r in rows:
        values = [r['alpha'],r['R'],r['initial_weight'],r['apr_feasible_pct'],r['completed_search_pct']]
        values += [r[p+'_'+s] for p in ('weight','gap_pct','lp_tests','time_s') for s in ('min','avg','max')]
        lines.append(' & '.join(number(v) for v in values)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}}',r'\vspace{2mm}',r'\begin{minipage}{\textwidth}\footnotesize',
        r'Demand scenarios satisfy $d_i^{(r)}=d_i+\omega_i^{(r)}$, with $\omega_i^{(r)}\sim\mathcal{N}(0,\alpha^2|d_i|^2)$. Supplied uncertainty files are used unchanged after matching bus IDs; no clipping or resampling is applied. The seed and file hashes are retained in the study settings.',
        r'$W_f$ is the smallest verified passing weight tested, not a certified globally minimum weight. $W>0$ throughout the search. Gap is $100(U-L)/|U|$, where $U$ is the original objective of the verified constructed integer-feasible solution and $L$ is the optimal objective of the original MIP with binary variables relaxed to $[0,1]$. No MIP is solved.',
        r'Weight statistics require a verified constructed integer-feasible solution; gap statistics additionally require a valid reference bound. Counts are saved in table\_B.csv. LP-test and time statistics include all finished attempted scenarios, including unsuccessful searches. Percentages use the planned $R$ samples as denominator. Partial tables also record the number finished.',
        r'Total LP time includes an equal share of common preprocessing, sample preparation, weight-search construction and solves, candidate construction, validation, and search checkpoint writing. Reference-model construction and reference-LP solving time are recorded separately. Time limits apply per solve.',
        r'\end{minipage}',r'\end{table}']
    (root/'table_B.tex').write_text('\n'.join(lines)+'\n')


def plots(root, rows, settings):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=root/'plots'
    folder.mkdir(exist_ok=True)
    definitions=[('selected_weight','Selected weight $W_f$',True),
                 ('bound_gap_pct','Gap to original LP bound (%)',True),
                 ('total_lp_time_s','Total LP time (s)',False)]
    for key,label,needs_verified in definitions:
        groups=[]; labels=[]
        for alpha in settings['sigma_scales']:
            data=[float(r[key]) for r in rows if r['alpha']==alpha and r.get(key) is not None
                  and (not needs_verified or r.get('constructed_ip_verified')) and np.isfinite(r[key])]
            if data:
                groups.append(data); labels.append(f'{alpha:g}\n(n={len(data)})')
        for kind in ('box','violin'):
            fig,ax=plt.subplots(figsize=(6.2,3.8),layout='constrained')
            if groups:
                for pos,data in enumerate(groups,1):
                    if kind=='violin' and len(data)>1 and np.ptp(data)>1e-12:
                        ax.violinplot([data],positions=[pos],showmedians=True)
                    else:
                        ax.boxplot([data],positions=[pos],widths=.45)
                    # Deterministic jitter; every scenario remains visible.
                    ax.scatter(pos+np.linspace(-.08,.08,len(data)),data,s=13,alpha=.65,zorder=3)
                ax.set_xticks(range(1,len(groups)+1),labels)
            else:
                ax.text(.5,.5,'No verified data available',ha='center',transform=ax.transAxes)
                ax.set_xticks([])
            ax.set_xlabel('Sigma scaling $\\alpha$')
            ax.set_ylabel(label)
            ax.grid(axis='y',alpha=.25)
            for ext in ('png','pdf'):
                fig.savefig(folder/f'{key}_{kind}.{ext}',dpi=180)
            plt.close(fig)


def load_uncertainty(cli, baseline, case_path):
    """Validate and reorder saved pu noise to the case bus order before solving."""
    if cli.uncertainty_dir is None:
        return {}, []
    if cli.demand_scale != 1.:
        raise ValueError('Saved omega was generated around the Excel demand; use --demand-scale 1')
    arrays, sources = {}, []
    target = [core.id_key(v) for v in baseline['bus'].bus_i.to_numpy()]
    if len(set(target)) != len(target):
        raise ValueError('Excel case has duplicate bus IDs')
    for alpha in cli.sigma_scales:
        path = Path(cli.uncertainty_dir)/f'omega_{case_path.stem}_sigma{alpha}_seed{cli.seed}.npz'
        if not path.is_file():
            raise FileNotFoundError(f'Uncertainty file missing: {path}')
        with np.load(path,allow_pickle=False) as data:
            required = {'omega','bus_ids','baseMVA','sigma_scaling','seed'}
            if not required.issubset(data.files):
                raise ValueError(f'{path.name}: missing fields {required-set(data.files)}')
            if not np.isclose(float(data['baseMVA']),baseline['base'],rtol=0,atol=1e-12):
                raise ValueError(f'{path.name}: baseMVA mismatch')
            if not np.isclose(float(data['sigma_scaling']),alpha,rtol=1e-12,atol=1e-15):
                raise ValueError(f'{path.name}: sigma scaling mismatch')
            if int(data['seed']) != cli.seed:
                raise ValueError(f'{path.name}: seed mismatch')
            omega = np.asarray(data['omega'],dtype=float)
            ids_array = np.asarray(data['bus_ids'])
            if ids_array.ndim != 1:
                raise ValueError(f'{path.name}: bus_ids must be one-dimensional')
            ids = [core.id_key(v) for v in ids_array]
            if omega.ndim!=2 or omega.shape[1]!=len(ids) or omega.shape[0]<cli.samples:
                raise ValueError(f'{path.name}: expected at least {cli.samples} rows and one column per bus')
            if len(set(ids))!=len(ids) or set(ids)!=set(target):
                raise ValueError(f'{path.name}: duplicate or mismatched bus IDs')
            lookup = {value:i for i,value in enumerate(ids)}
            omega = omega[:cli.samples, [lookup[value] for value in target]].copy()
            if not np.isfinite(omega).all():
                raise ValueError(f'{path.name}: nonfinite omega values')
            zero = np.abs(baseline['d']) == 0
            if np.any(np.abs(omega[:,zero])>1e-12):
                raise ValueError(f'{path.name}: nonzero noise at a zero-demand bus')
            if alpha==0 and np.any(omega!=0):
                raise ValueError(f'{path.name}: nonzero noise with sigma=0')
            arrays[alpha] = omega
            sources.append(dict(alpha=alpha,path=str(path.resolve()),name=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                available_scenarios=int(data['omega'].shape[0]),used_scenarios=cli.samples))
    return arrays,sources


def main():
    cli=arguments()
    root=Path(cli.output_dir).resolve()
    if cli.summarize_only:
        settings=read_json(root/'study_settings.json')
        if settings is None:
            raise ValueError('No study_settings.json in output directory')
        rows=[read_json(p) for p in sorted((root/'samples').glob('alpha_*/sample_*/result.json'))]
        summarize(root,settings,rows)
        print('Rebuilt table_B.csv, table_B.tex and plots in '+str(root))
        return 0
    if not cli.case:
        raise ValueError('--case is required')
    for name in ('samples','threads','penalty_per_mw','gamma','demand_scale','lp_time_limit',
                 'relaxation_time_limit','weight_tol','weight_max','apr_tol_mw','feas_tol_mw'):
        value=getattr(cli,name)
        if not np.isfinite(value) or value<=0:
            raise ValueError(name+' must be positive and finite')
    if cli.seed<0 or cli.weight_max_solves<2 or not cli.sigma_scales or any(not np.isfinite(a) or a<0 for a in cli.sigma_scales):
        raise ValueError('Invalid seed, solve count or sigma scales')
    if len(set(cli.sigma_scales))!=len(cli.sigma_scales):
        raise ValueError('Sigma scales must be distinct')
    case_path=core.resolve_case(cli.case,cli.case_dir).resolve()
    t=perf_counter()
    baseline=core.load_case(case_path,cli.demand_scale)
    baseline['a']=cli.gamma*(baseline['hi']-baseline['lo'])
    outages=core.select_outages(baseline,'all')
    saved_omega, uncertainty_sources=load_uncertainty(cli,baseline,case_path)
    preprocess=perf_counter()-t
    args=solver_args(cli,case_path,baseline['base'])
    if not np.isfinite(args.weight_start) or args.weight_start<=0 or args.weight_start>args.weight_max:
        raise ValueError('Require 0 < weight-start <= weight-max')
    if cli.solver=='gurobi':
        import gurobipy  # fail early before creating study directories if missing
    settings=vars(cli).copy()
    for key in ('case_dir','output_dir','resume','summarize_only','case'):
        settings.pop(key,None)
    settings.update(version=VERSION,case_name=case_path.name,
                    uncertainty_source='saved_npz' if saved_omega else 'internal_rng',
                    uncertainty_files=uncertainty_sources,
                    input_sha256=hashlib.sha256(case_path.read_bytes()).hexdigest(),
                    core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
                    baseMVA=baseline['base'],penalty_per_pu=args.penalty,initial_weight=args.weight_start,
                    N=len(baseline['bus']),B=len(baseline['branch']),G=len(baseline['gen']),
                    generator_outages=len(outages),line_outages=len(baseline['eligible']),
                    excluded_line_outages=len(baseline['excluded']))
    signature=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
    existing=read_json(root/'study_settings.json')
    if root.exists() and any(root.iterdir()):
        if not cli.resume or not existing or existing.get('signature')!=signature:
            raise ValueError('Output directory is not empty. Use a NEW directory, or --resume with identical settings.')
        shared=existing['shared_preprocessing_per_sample_s']
        settings=existing
    else:
        root.mkdir(parents=True,exist_ok=True)
        shared=preprocess/(cli.samples*len(cli.sigma_scales))
        settings.update(signature=signature,shared_preprocessing_s=preprocess,
                        shared_preprocessing_per_sample_s=shared,
                        python_version=sys.version,
                        packages={name:importlib.metadata.version(name) for name in
                                  ['numpy','scipy','pandas','openpyxl','matplotlib']+(['gurobipy'] if cli.solver=='gurobi' else [])})
        write_json(root/'study_settings.json',settings)
        shutil.copy2(case_path,root/'input_case.xlsx')
        shutil.copy2(Path(__file__),root/'apr_sigma_study.py')
        shutil.copy2(core.__file__,root/'apr_sigma_core.py')
        if uncertainty_sources:
            (root/'input_uncertainty').mkdir()
            for source in uncertainty_sources:
                shutil.copy2(source['path'],root/'input_uncertainty'/source['name'])
    print(f'{VERSION}: case={case_path.name}; R={cli.samples} per scale; {cli.samples*len(cli.sigma_scales)} scenarios',flush=True)
    print('Uncertainty source: '+settings['uncertainty_source'],flush=True)
    print(f'M_eta={args.penalty:g} per pu; W0={args.weight_start:g}; W>0; no MIP solves',flush=True)
    rows=[]
    for alpha in cli.sigma_scales:
        for sample in range(1,cli.samples+1):
            result_path=root/'samples'/f'alpha_{alpha:.12g}'/f'sample_{sample:04d}'/'result.json'
            result=read_json(result_path) if cli.resume else None
            if not result or not result.get('finished'):
                print(f'\nSCENARIO alpha={alpha:g}; sample={sample}/{cli.samples}',flush=True)
                result=run_sample(baseline,outages,args,root,alpha,sample,cli.seed,shared,saved_omega.get(alpha))
            rows.append(result)
            # Small incremental tables; plots are generated once at the end.
            write_csv(root/'scenario_results.csv',rows)
            print(f"SCENARIO RESULT: status={result['search_status']}; IP verified={result['constructed_ip_verified']}; "
                  f"W={result['selected_weight']}; gap={result['bound_gap_pct']}",flush=True)
    summarize(root,settings,rows)
    print('\nSaved table_B.csv, table_B.tex, scenario_results.csv and plots: '+str(root),flush=True)
    if any(r.get('error_message') for r in rows):
        print('Some scenarios raised errors; inspect scenario_results.csv and per-sample error.txt.',file=sys.stderr)
        return 2
    return 0


if __name__=='__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Interrupted; resume using identical settings and --resume.',file=sys.stderr)
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
