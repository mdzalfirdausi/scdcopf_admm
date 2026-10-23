#!/usr/bin/env python3
"""Plan, run and collect disjoint batches of original saved demand scenarios."""
import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import shlex
import shutil
import sys
from time import perf_counter
import traceback

import numpy as np
import apr_sigma_batch_support as study

VERSION='2026-10-23-apr-scenario-arrays-v1'
DEFAULT_CASES='118_ieee 162_ieee_dtc 179_goc 200_activ 240_pserc 300_ieee 500_goc 588_sdet 793_goc 1888_rte 1951_rte 2000_goc'.split()
# Match the original core byte-for-byte to permit verified reuse of prior v2 runs.
CORE_HASH=hashlib.sha256(Path(study.core.__file__).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    plan=sub.add_parser('plan',help='Snapshot inputs, identify reusable results, write task manifests')
    plan.add_argument('--cases',nargs='+',default=['pglib_opf_case'+v for v in DEFAULT_CASES])
    plan.add_argument('--case-dir',default='../excel_outputs')
    plan.add_argument('--uncertainty-dir',default='uncertainty')
    plan.add_argument('--output-dir',required=True,help='NEW plan/result directory; never overwrites an old study')
    plan.add_argument('--samples',type=int,default=100,help='R per sigma, using original NPZ row numbers')
    plan.add_argument('--sigma-scales',type=float,nargs='+',default=[.01,.02,.03,.04,.05])
    plan.add_argument('--batch-size',type=int,default=10)
    plan.add_argument('--seed',type=int,default=19)
    plan.add_argument('--solver',choices=['gurobi','highs'],default='gurobi')
    plan.add_argument('--threads',type=int,default=16)
    plan.add_argument('--penalty-per-mw',type=float,default=1500.)
    plan.add_argument('--weight-start',type=float,default=150000.)
    plan.add_argument('--weight-tol',type=float,default=.001)
    plan.add_argument('--weight-max',type=float,default=1e9)
    plan.add_argument('--weight-max-solves',type=int,default=40)
    plan.add_argument('--gamma',type=float,default=1.)
    plan.add_argument('--lp-time-limit',type=float,default=300.)
    plan.add_argument('--relaxation-time-limit',type=float,default=600.)
    plan.add_argument('--apr-tol-mw',type=float,default=1e-4)
    plan.add_argument('--feas-tol-mw',type=float,default=1e-4)
    plan.add_argument('--case-lp-time-limit',nargs='*',default=[],help='Optional CASE=SECONDS overrides; only after a pilot')
    plan.add_argument('--reuse-roots',nargs='*',default=[],help='Explicit existing KFUPM study roots; shell wildcards allowed')
    plan.add_argument('--reuse-inconclusive',action='store_true',help='Also retain verified feasible trials with valid bounds whose weight search stopped early; preserve their incomplete status')
    plan.add_argument('--concurrent-small',type=int,default=4)
    plan.add_argument('--concurrent-large',type=int,default=4)
    plan.add_argument('--small-mem',default='64G')
    plan.add_argument('--large-mem',default='128G')
    plan.add_argument('--wall-time',default='24:00:00')
    plan.add_argument('--tee',action='store_true')
    run=sub.add_parser('run',help='Compute worker, normally called by Slurm')
    run.add_argument('--plan',required=True)
    run.add_argument('--group',choices=['small','large'],required=True)
    run.add_argument('--task-id',type=int,required=True)
    collect=sub.add_parser('collect',help='Check coverage and generate Table B for each case')
    collect.add_argument('--plan',required=True)
    collect.add_argument('--allow-partial',action='store_true')
    status=sub.add_parser('status')
    status.add_argument('--plan',required=True)
    return p


def protocol(cli,case_name,base,input_hash,uncertainty):
    values={key:getattr(cli,key) for key in ('samples','sigma_scales','seed','solver','threads',
        'penalty_per_mw','weight_start','weight_tol','weight_max','weight_max_solves','gamma',
        'lp_time_limit','relaxation_time_limit','apr_tol_mw','feas_tol_mw','tee')}
    for override in cli.case_lp_time_limit:
        name,seconds=override.split('=',1)
        if Path(name).stem==Path(case_name).stem:
            values['lp_time_limit']=float(seconds)
    values.update(version=VERSION,case_name=case_name,baseMVA=base,input_sha256=input_hash,
        core_sha256=CORE_HASH,uncertainty_source='saved_npz',uncertainty_files=uncertainty,
        initial_weight=values['weight_start'],penalty_per_pu=values['penalty_per_mw']*base,
        demand_scale=1.,full_eta_diagnostics=False)
    return values


def compatible(old,new,alpha):
    keys=('case_name','baseMVA','input_sha256','core_sha256','uncertainty_source','seed','solver',
          'threads','penalty_per_mw','initial_weight','weight_tol','weight_max','weight_max_solves',
          'gamma','demand_scale','lp_time_limit','relaxation_time_limit','apr_tol_mw','feas_tol_mw',
          'full_eta_diagnostics')
    if any(old.get(k)!=new.get(k) for k in keys):
        return False
    before={float(u['alpha']):u['sha256'] for u in old.get('uncertainty_files',[])}
    after={float(u['alpha']):u['sha256'] for u in new['uncertainty_files']}
    return before.get(alpha) is not None and before.get(alpha)==after.get(alpha)


def reusable(row, include_inconclusive=False):
    return bool(row.get('finished') and not row.get('error_message') and (
        row.get('search_status')=='capacity_infeasible' or
        ((row.get('search_completed') or include_inconclusive) and row.get('constructed_ip_verified') and
         row.get('reference_status')=='optimal' and row.get('lp_relaxation_bound_L') is not None)))


def plan(cli):
    for name in ('samples','batch_size','threads','concurrent_small','concurrent_large','weight_max_solves'):
        if getattr(cli,name)<1:
            raise ValueError(name+' must be positive')
    for name in ('penalty_per_mw','weight_start','weight_tol','weight_max','gamma','lp_time_limit',
                 'relaxation_time_limit','apr_tol_mw','feas_tol_mw'):
        if not np.isfinite(getattr(cli,name)) or getattr(cli,name)<=0:
            raise ValueError(name+' must be positive and finite')
    if cli.seed<0 or cli.weight_max<cli.weight_start or cli.weight_max_solves<2:
        raise ValueError('Invalid seed or weight search limits')
    if not cli.sigma_scales or any(not np.isfinite(a) or a<0 for a in cli.sigma_scales) or len(set(cli.sigma_scales))!=len(cli.sigma_scales):
        raise ValueError('Sigma scales must be distinct nonnegative finite numbers')
    names=[Path(name).stem for name in cli.cases]
    if len(set(names))!=len(names):
        raise ValueError('Duplicate cases')
    for entry in cli.case_lp_time_limit:
        name,seconds=entry.split('=',1)
        if Path(name).stem not in names or not np.isfinite(float(seconds)) or float(seconds)<=0:
            raise ValueError('Invalid case LP-time override: '+entry)
    root=Path(cli.output_dir).resolve()
    if root.exists():
        raise ValueError('Plan directory must be NEW, to protect results')
    # Resolve every input before creating the plan. No SCOPF models are built here.
    inputs=[]
    for supplied_name,name in zip(cli.cases,names):
        case_path=study.core.resolve_case(supplied_name,cli.case_dir).resolve()
        uncertainty=[]
        base=None
        for alpha in cli.sigma_scales:
            f=Path(cli.uncertainty_dir)/f'omega_{name}_sigma{alpha}_seed{cli.seed}.npz'
            with np.load(f,allow_pickle=False) as data:
                required={'omega','bus_ids','baseMVA','sigma_scaling','seed'}
                if not required.issubset(data.files):
                    raise ValueError(f'{f}: missing saved-generator fields')
                if data['omega'].ndim!=2 or data['omega'].shape[0]<cli.samples:
                    raise ValueError(f'{f}: fewer than {cli.samples} scenarios or invalid dimensions')
                if int(data['seed'])!=cli.seed or not np.isclose(float(data['sigma_scaling']),alpha,rtol=1e-12,atol=1e-15):
                    raise ValueError(f'{f}: seed/sigma mismatch')
                b=float(data['baseMVA'])
                if not np.isfinite(b) or b<=0 or (base is not None and b!=base):
                    raise ValueError(f'{f}: inconsistent baseMVA')
                base=b
            uncertainty.append(dict(alpha=alpha,name=f.name,path=str(f.resolve()),sha256=digest(f)))
        inputs.append((case_path,base,uncertainty))
    root.mkdir(parents=True)
    (root/'logs').mkdir()
    # Archive code once, so later edits to the project cannot change this plan's workers.
    code=root/'code';code.mkdir()
    here=Path(__file__).resolve().parent
    for name in ('apr_sigma_parallel.py','apr_sigma_batch_support.py','apr_sigma_batch_core.py','hpc_run_apr_sigma_batches.sh'):
        shutil.copy2(here/name,code/name)
    candidates=[]
    for oldroot in cli.reuse_roots:
        p=Path(oldroot).resolve()
        if not (p/'study_settings.json').is_file():
            print('Ignoring reuse root without study_settings.json: '+str(p))
            continue
        candidates.append((p,read(p/'study_settings.json')))
    data=dict(version=VERSION,root=str(root),cases={},tasks={'small':[],'large':[]},
              code_sha256={p.name:digest(p) for p in code.iterdir() if p.is_file()},
              reused=[],duplicates_ignored=[],notes=['Reused timings are retained, not remeasured. New batches amortize preprocessing over only their newly computed scenarios.'])
    for case_path,base,uncertainty in inputs:
        name=case_path.stem
        config=protocol(cli,case_path.name,base,digest(case_path),uncertainty)
        folder=root/'inputs'/name;folder.mkdir(parents=True)
        shutil.copy2(case_path,folder/case_path.name)
        for u in uncertainty:
            shutil.copy2(u['path'],folder/u['name'])
        config.update(case_path=str(folder/case_path.name),uncertainty_dir=str(folder))
        data['cases'][name]=config
        reused={}
        for oldroot,old in candidates:
            if old.get('case_name')!=case_path.name:
                continue
            for p in sorted((oldroot/'samples').glob('alpha_*/sample_*/result.json')):
                r=read(p)
                alpha=float(r.get('alpha',-1));sample=int(r.get('sample',0))
                if alpha not in cli.sigma_scales or not 1<=sample<=cli.samples or not reusable(r,cli.reuse_inconclusive) or not compatible(old,config,alpha):
                    continue
                key=(alpha,sample)
                if key in reused:
                    data['duplicates_ignored'].append(dict(case=name,alpha=alpha,sample=sample,unused=str(p)))
                    continue
                # Freeze the small record; retain pointers to the bulk original solutions.
                reused[key]=dict(case=name,alpha=alpha,sample=sample,result=r,
                    source_result=str(p),source_result_sha256=digest(p),source_root=str(oldroot))
        data['reused'].extend(reused.values())
        match=re.search(r'case(\d+)',name)
        group='large' if match and int(match.group(1))>=700 else 'small'
        for alpha in cli.sigma_scales:
            for start in range(1,cli.samples+1,cli.batch_size):
                ids=[r for r in range(start,min(cli.samples+1,start+cli.batch_size)) if (alpha,r) not in reused]
                if ids:
                    data['tasks'][group].append(dict(case=name,alpha=alpha,sample_ids=ids))
        print(f'{name}: {len(reused)}/{cli.samples*len(cli.sigma_scales)} reusable; group={group}')
    study.write_json(root/'plan.json',data)
    commands=['#!/bin/bash','set -euo pipefail']
    for group,concurrency,mem in [('small',cli.concurrent_small,cli.small_mem),('large',cli.concurrent_large,cli.large_mem)]:
        n=len(data['tasks'][group])
        if n:
            words=['sbatch',f'--array=0-{n-1}%{concurrency}',f'--cpus-per-task={cli.threads}',f'--mem={mem}',
                   f'--time={cli.wall_time}',f'--output={root}/logs/{group}_%A_%a.out',
                   f'--error={root}/logs/{group}_%A_%a.err',str(code/'hpc_run_apr_sigma_batches.sh'),str(root),group]
            commands.append(' '.join(shlex.quote(w) for w in words))
    (root/'submit_arrays.sh').write_text('\n'.join(commands)+'\n')
    print(f"Plan saved: {root}; tasks: small={len(data['tasks']['small'])}, large={len(data['tasks']['large'])}")
    print('Stop overlapping old runs before submitting. Then: bash '+shlex.quote(str(root/'submit_arrays.sh')))


def load_plan(path):
    root=Path(path).resolve()
    data=read(root/'plan.json')
    if data['root']!=str(root):
        raise ValueError('Do not move the plan directory; archived paths and source pointers are absolute')
    for name,sha in data['code_sha256'].items():
        if digest(root/'code'/name)!=sha:
            raise ValueError('Archived source changed: '+name)
    return root,data


def worker(cli):
    import fcntl
    root,data=load_plan(cli.plan)
    tasks=data['tasks'][cli.group]
    if not 0<=cli.task_id<len(tasks):
        raise ValueError('Task ID outside manifest')
    spec=tasks[cli.task_id];config=data['cases'][spec['case']]
    work=root/'tasks'/cli.group/f'task_{cli.task_id:05d}'
    work.mkdir(parents=True,exist_ok=True)
    with (work/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if 'SLURM_CPUS_PER_TASK' in __import__('os').environ:
            allocated=int(__import__('os').environ['SLURM_CPUS_PER_TASK'])
            if allocated<config['threads']:
                raise ValueError('Slurm CPU allocation smaller than planned solver threads')
        for u in config['uncertainty_files']:
            if digest(Path(config['uncertainty_dir'])/u['name'])!=u['sha256']:
                raise ValueError('Archived uncertainty changed')
        if digest(config['case_path'])!=config['input_sha256']:
            raise ValueError('Archived Excel changed')
        started=perf_counter()
        # Same model defaults as the v2 serial study. All sample IDs are GLOBAL NPZ row IDs.
        options=argparse.Namespace(**config)
        options.sigma_scales=[spec['alpha']]
        baseline=study.core.load_case(Path(config['case_path']),1.)
        baseline['a']=options.gamma*(baseline['hi']-baseline['lo'])
        omega,_=study.load_uncertainty(options,baseline,Path(config['case_path']))
        outages=study.core.select_outages(baseline,'all')
        args=study.solver_args(options,Path(config['case_path']),baseline['base'])
        if args.backend=='gurobi':
            import gurobipy
        remaining=[sample for sample in spec['sample_ids'] if not (work/'samples'/f"alpha_{spec['alpha']:.12g}"/f'sample_{sample:04d}'/'result.json').exists()]
        shared=(perf_counter()-started)/max(1,len(remaining))
        settings=dict(config,active_alpha=spec['alpha'],sample_ids=spec['sample_ids'],
                      actual_hostname=__import__('socket').gethostname(),slurm_job_id=__import__('os').environ.get('SLURM_JOB_ID'),
                      shared_preprocessing_s=perf_counter()-started,shared_preprocessing_per_sample_s=shared,
                      task_group=cli.group,task_id=cli.task_id)
        study.write_json(work/'study_settings.json',settings)
        print(f"TASK {cli.group}/{cli.task_id}: {spec['case']}; alpha={spec['alpha']}; original rows={spec['sample_ids']}",flush=True)
        rows=[]
        for sample in spec['sample_ids']:
            p=work/'samples'/f"alpha_{spec['alpha']:.12g}"/f'sample_{sample:04d}'/'result.json'
            r=read(p) if p.exists() else None
            if not r or not r.get('finished'):
                print(f"SCENARIO original sample={sample}/{config['samples']}",flush=True)
                r=study.run_sample(baseline,outages,args,work,spec['alpha'],sample,options.seed,shared,omega[spec['alpha']])
            rows.append(r)
            study.write_csv(work/'scenario_results.csv',rows)
            print(f"DONE sample={sample}; status={r['search_status']}; IP={r['constructed_ip_verified']}",flush=True)
        prior=read(work/'task_resources.json') if (work/'task_resources.json').exists() else {}
        peak=max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,prior.get('process_peak_rss_MiB',0.))
        study.write_json(work/'task_resources.json',dict(group=cli.group,task_id=cli.task_id,
            case=spec['case'],alpha=spec['alpha'],sample_ids=spec['sample_ids'],hostname=settings['actual_hostname'],
            slurm_job_id=settings['slurm_job_id'],process_peak_rss_MiB=peak,
            definition='Maximum Linux getrusage high-water resident memory over this task worker and any resumed attempts; not an independent per-scenario peak',
            attempt_count=prior.get('attempt_count',0)+1,
            worker_wall_time_s=prior.get('worker_wall_time_s',0.)+perf_counter()-started,threads=config['threads']))
        if any(r.get('error_message') for r in rows):
            return 2
    return 0


def inventory(root,data):
    records={name:{} for name in data['cases']}
    for item in data['reused']:
        name=item['case'];key=(item['alpha'],item['sample'])
        if key in records[name]:
            raise ValueError('Duplicate reused scenario in plan')
        row=dict(item['result'],source_root=item['source_root'],source_result=item['source_result'],reused=True)
        records[name][key]=row
    for group,tasks in data['tasks'].items():
        for index,spec in enumerate(tasks):
            work=root/'tasks'/group/f'task_{index:05d}'
            allowed={(spec['alpha'],s) for s in spec['sample_ids']}
            for p in (work/'samples').glob('alpha_*/sample_*/result.json'):
                row=read(p)
                if not row.get('finished'):
                    continue
                key=(float(row['alpha']),int(row['sample']))
                if key not in allowed:
                    raise ValueError(f'Unexpected scenario {key} in {work}')
                if key in records[spec['case']]:
                    raise ValueError(f'Duplicate scenario {key} for {spec["case"]}')
                records[spec['case']][key]=dict(row,source_root=str(work),source_result=str(p),reused=False)
    return records


def report(cli):
    root,data=load_plan(cli.plan)
    records=inventory(root,data)
    missing=[]
    for name,config in data['cases'].items():
        expected={(a,s) for a in config['sigma_scales'] for s in range(1,config['samples']+1)}
        present=set(records[name])
        if present-expected:
            raise ValueError('Out-of-range scenarios found')
        absent=sorted(expected-present)
        missing.extend(dict(case=name,alpha=a,sample=s) for a,s in absent)
        values=list(records[name].values())
        print(f"{name}: {len(values)}/{len(expected)} finished; searches complete={sum(bool(r.get('search_completed')) for r in values)}; IP verified={sum(bool(r.get('constructed_ip_verified')) for r in values)}; reused={sum(bool(r.get('reused')) for r in values)}")
    if cli.command=='status':
        return 0
    study.write_json(root/'coverage.json',dict(finished=sum(len(v) for v in records.values()),missing=missing,
        duplicates=0,complete=not missing,note='Finished includes inconclusive or unsuccessful attempts; inspect completion/feasibility rates.'))
    if missing and not cli.allow_partial:
        print(f'{len(missing)} scenarios missing. No final tables written. Use --allow-partial for provisional tables.',file=sys.stderr)
        return 2
    for name,config in data['cases'].items():
        dest=root/'tables'/name;dest.mkdir(parents=True,exist_ok=True)
        rows=[records[name][key] for key in sorted(records[name])]
        for r in rows:
            source=Path(r['source_root'])/r['work_directory']
            r['absolute_solution_directory']=str(source)
            r['case']=name
        study.write_json(dest/'study_settings.json',config)
        study.summarize(dest,config,rows)
        study.write_json(dest/'coverage.json',dict(complete=not any(m['case']==name for m in missing),
             finished=len(rows),planned=config['samples']*len(config['sigma_scales'])))
    resources=[read(p) for p in sorted((root/'tasks').glob('*/task_*/task_resources.json'))]
    if resources:
        study.write_csv(root/'task_memory_and_time.csv',resources)
    print('Tables, plots and provenance pointers: '+str(root/'tables'))
    return 0 if not missing else 2


def main():
    cli=parser().parse_args()
    if cli.command=='plan':
        plan(cli);return 0
    if cli.command=='run':
        return worker(cli)
    return report(cli)


if __name__=='__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        traceback.print_exc();sys.exit(1)
