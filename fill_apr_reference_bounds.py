#!/usr/bin/env python3
"""Fill Table A lower bounds from saved LP-only runs; never repeat weight searches."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import traceback

import numpy as np
import apr_bound_core as core

VERSION = '2026-10-23-reference-lp-only-v1'


def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',nargs='+',required=True,help='Existing run directories containing settings.json and lp_search/')
    p.add_argument('--case-dir',default='../excel_outputs')
    p.add_argument('--output-dir',default='output_apr_reference_bounds')
    p.add_argument('--solver',choices=['gurobi','highs'],default='gurobi')
    p.add_argument('--threads',type=int,default=16)
    p.add_argument('--time-limit',type=float,default=3600.,help='Seconds per reference LP solve')
    p.add_argument('--tee',action='store_true')
    p.add_argument('--version',action='version',version=VERSION)
    return p.parse_args()


def read(path):
    return json.loads(path.read_text())


def write(path,data):
    core.write_search_json(path,data)


def find_case(metadata,cli):
    name=Path(metadata.get('case_path') or metadata['arguments']['case']).name
    choices=[Path(cli.case_dir)/name,Path(metadata.get('case_path','')),Path(metadata['arguments']['case'])]
    path=next((p for p in choices if p.is_file()),None)
    if path is None:
        raise FileNotFoundError(f'Case {name} missing; supply --case-dir with the original Excel cases')
    expected=metadata.get('case_sha256')
    if not expected:
        raise ValueError('Saved case_sha256 missing; cannot establish the matching original instance')
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=expected:
        raise ValueError(f'Excel SHA256 mismatch for {path}; use the unchanged original input file')
    return path.resolve(),actual


def run_one(folder,cli,destination):
    begun=perf_counter()
    row=dict(result_directory=str(folder),case=None,selected_weight=None,
             original_objective_U=None,lp_relaxation_bound_L=None,bound_gap_pct=None,
             candidate_verified=False,reference_status='not_started',reference_time_s=None)
    destination.mkdir()
    try:
        metadata_path=folder/'settings.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f'{metadata_path} missing. Supply the full original result directory, not logs alone.')
        metadata=read(metadata_path)
        path,digest=find_case(metadata,cli)
        row.update(case=path.stem,case_sha256=digest)
        original=metadata.get('arguments',{})
        required=['gamma','demand_scale','penalty','generator_contingencies','no_line_contingencies','islanding','thermal_mode','apr_tol_mw','feas_tol_mw']
        missing=[key for key in required if key not in original]
        if missing:
            raise ValueError('Saved model settings missing: '+', '.join(missing))
        if original['thermal_mode']!='soft':
            raise ValueError('This Table A reference script expects the original soft-thermal-limit comparison formulation')
        args=core.parser().parse_args(['--case',str(path)])
        vars(args).update(original)
        args.backend=args.solver=cli.solver
        args.threads=cli.threads
        args.time_limit=cli.time_limit
        args.tee=cli.tee
        case=core.load_case(path,args.demand_scale)
        args.base=case['base']
        case['a']=args.gamma*(case['hi']-case['lo'])
        if metadata.get('baseMVA') is not None and case['base']!=metadata['baseMVA']:
            raise ValueError('Base-MVA mismatch')
        if metadata.get('demand_MW') is not None and not np.isclose(case['D']*case['base'],metadata['demand_MW'],rtol=1e-12,atol=1e-8):
            raise ValueError('Total-demand mismatch; sampled-demand runs require their saved actual demand vectors')
        outages=core.select_outages(case,args.generator_contingencies)
        if len(case['excluded']) and not args.no_line_contingencies and args.islanding=='error':
            raise ValueError('Original settings reject the islanding line outages in this case')
        ids=[core.id_key(case['gen'].iloc[k].gen_ID) for k in outages]
        if metadata.get('generator_outage_IDs') is not None and ids!=metadata['generator_outage_IDs']:
            raise ValueError('Generator contingency IDs mismatch')
        row.update(N=len(case['bus']),B=len(case['branch']),G=len(case['gen']),
                   generator_outages=len(outages),line_outages=0 if args.no_line_contingencies else len(case['eligible']),
                   excluded_islanding=0 if args.no_line_contingencies else len(case['excluded']),
                   penalty_per_pu=args.penalty,gamma=args.gamma,demand_scale=args.demand_scale)
        search_path=folder/'lp_search'/'weight_search_summary.json'
        if search_path.exists():
            search=read(search_path)
            row.update(selected_weight=search.get('best_passing_weight'),search_status=search.get('termination'),
                       lp_weight_tests=search.get('solve_count'))
        t=perf_counter()
        # Keep original objective, all disjunctive rows and binary [0,1] bounds.
        exact=core.apply_eta_limits(core.build_model(case,outages,args,True,None),case,args)
        row['reference_model_build_s']=perf_counter()-t
        if any((folder/'lp_search'/name).exists() for name in ['best_dispatch.csv','best_apr_checks.csv','best_apr_dispatch.csv']):
            try:
                candidate=core.load_lp_candidate(case,outages,folder)
                metrics=core.evaluate_common(case,outages,*candidate,args)
                if not metrics['original_soft_model_feasible']:
                    raise ValueError('Saved candidate failed independent original-APR/balance/bounds checks')
                # Construct/check the IP point. This is not an optimization solve.
                point=core.make_mip_start(case,outages,exact,candidate,args)
                upper=float(exact.c0@point+exact.constant)
                if not np.isclose(upper,metrics['original_objective'],rtol=1e-7,atol=1e-6):
                    raise ValueError('Constructed-IP cost mismatch')
                saved_meta=folder/'lp_search'/'metadata.json'
                if saved_meta.exists():
                    best=read(saved_meta).get('best_result') or {}
                    if best.get('operating_objective') is not None and not np.isclose(upper,best['operating_objective'],rtol=1e-7,atol=1e-6):
                        raise ValueError('Recomputed U differs from the saved candidate operating objective')
                row.update(candidate_verified=True,original_objective_U=upper,
                           max_APR_L1_error_MW=metrics['max_APR_L1_error_MW'])
                np.savez_compressed(destination/'verified_existing_ip.npz',vector=point,
                                    binary_pairs=np.asarray(list(exact.x.keys())),
                                    binary_indices=np.asarray(list(exact.x.values())),g=exact.g,y=exact.y,z=exact.z)
                del point
            except Exception as error:
                row['candidate_error']=str(error)
                print('Candidate unavailable/unverified; L will still be solved: '+str(error),flush=True)
        else:
            row['candidate_error']='No saved candidate dispatch files; L can be reported, gap cannot'
        relaxed=replace(exact,integer=np.zeros_like(exact.integer))
        del exact
        assert not relaxed.integer.any(),'Reference must be continuous'
        t=perf_counter()
        # The zero argument disables an unrelated signal penalty, not an APR W test.
        result,vector=core.solve_model(relaxed,0.,args)
        row['reference_solve_s']=perf_counter()-t
        row['reference_time_s']=row['reference_model_build_s']+row['reference_solve_s']
        row['reference_status']=result['status']
        write(destination/'solver_result.json',result)
        if result['status']=='optimal' and result.get('solver_point_verified'):
            lower=float(result['solver_penalized_objective'])
            row['lp_relaxation_bound_L']=lower
            if vector is not None:
                np.savez_compressed(destination/'reference_lp_solution.npz',vector=vector)
            if row['candidate_verified']:
                upper=row['original_objective_U']
                if lower > upper+1e-7*max(1.,abs(upper)):
                    row.update(reference_status='invalid_bound_above_candidate',lp_relaxation_bound_L=None)
                elif abs(upper)>1e-12:
                    gap=100.*(upper-lower)/abs(upper)
                    row['bound_gap_pct']=max(0.,gap) if gap>-1e-5 else gap
        elif result['status']=='optimal':
            row['reference_status']='validation_failed'
        write(destination/'matching_settings.json',dict(original_settings=metadata,
              reference_script_version=VERSION,reference_solver=cli.solver,reference_threads=cli.threads,
              reference_time_limit=cli.time_limit,core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()))
    except Exception as error:
        row.update(reference_status='error',error_type=type(error).__name__,error_message=str(error))
        (destination/'error.txt').write_text(traceback.format_exc())
        print(traceback.format_exc(),file=sys.stderr)
    row['total_postprocessing_s']=perf_counter()-begun
    write(destination/'reference_bound.json',row)
    return row


def main():
    cli=parse()
    if cli.threads<1 or not np.isfinite(cli.time_limit) or cli.time_limit<=0:
        raise ValueError('Threads and time limit must be positive')
    folders=[Path(p).resolve() for p in cli.results]
    if len(set(folders))!=len(folders):
        raise ValueError('Repeated result directory')
    if any(not p.is_dir() for p in folders):
        raise ValueError('Every --results value must be an existing result directory')
    root=Path(cli.output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('Choose a new/empty --output-dir; existing results are never overwritten')
    root.mkdir(parents=True,exist_ok=True)
    print(VERSION+': reference LP only; saved weight searches are not rerun',flush=True)
    rows=[]
    for index,folder in enumerate(folders,1):
        print(f'\n[{index}/{len(folders)}] {folder.name}',flush=True)
        row=run_one(folder,cli,root/f'{index:03d}_{folder.name}')
        rows.append(row)
        core.write_search_csv(root/'table_A_bounds.csv',rows)
        print(f"L={row['lp_relaxation_bound_L']}; U={row['original_objective_U']}; gap={row['bound_gap_pct']}; status={row['reference_status']}",flush=True)
    print('\nSaved '+str(root/'table_A_bounds.csv'),flush=True)
    return 0 if all(r['reference_status']=='optimal' and r['candidate_verified'] for r in rows) else 2


if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
