APR LP / MIP comparison
=======================

Files
- compare_apr_lp_mip.py: standalone Python program; it does not import your old script.
- iss_run_apr_comparison.sh: ISS partition main, pytorch environment.
- hpc_run_apr_comparison.sh: cpu_x440, excludes node0032, gurobi/13.0.1 module.

Put all files in your project directory beside ../excel_outputs.

Recommended first run (ISS; 300 buses):
sbatch iss_run_apr_comparison.sh --case pglib_opf_case300_ieee --with-lp-start

1354-bus run (ISS):
sbatch --mem=128G iss_run_apr_comparison.sh --case pglib_opf_case1354_pegase --with-lp-start

Other HPC:
sbatch --mem=128G hpc_run_apr_comparison.sh --case pglib_opf_case1354_pegase --with-lp-start

--with-lp-start is optional. Without it, only LP search and cold MIP are compared.
Use --tee to inspect Gurobi's acceptance/rejection of the submitted MIP start.
The .xlsx suffix is optional. Slurm output folders include case name and job ID.
Use --output-dir PATH to override. Use a fresh directory for each run.
Both Slurm scripts default to 16 CPUs, 64 GB, and 8 hours. --mem=128G overrides
memory; sufficiency for the large case is not guaranteed.

Local Python example:
python compare_apr_lp_mip.py --case pglib_opf_case300_ieee --case-dir ../excel_outputs --solver gurobi --threads 16 --with-lp-start --mip-time-limit 3600 --output-dir output_compare_case300

Common model / quality measure
- Same demand, generator limits, gamma, contingency selection, zero phase shifts,
  line ratings, linear generation costs, and soft thermal limits for every method.
- The LP searches weights in w*sum(splus+sminus)+sum(eta_pu).
- Both MIPs minimize original generation cost + M_eta*sum(eta_pu).
- Reported original_objective ALWAYS uses generation cost + M_eta*required overload.
- Required overload is max(0,abs(flow)-rating), recomputed from dispatches.
- The response-slack weight is NOT part of the common evaluation.
- As in your current script, c2 is ignored and c0 is included. This is a linear-cost
  SC-DC-OPF comparison, not a quadratic-cost OPF comparison.
- Default gamma=1, M_eta=1500 per pu, zero phase shifts, islanding outages skipped.
- No generation-cost budgets, extra eta caps, or dispatch repair are used.
- Positive overload is permitted: feasibility means the ORIGINAL SOFT model.

Methods / time accounting
1. lp_bisection: smallest verified passing weight found within the configured search.
   pipeline_seconds includes common preprocessing, ALL LP weight tests, model
   building, validation, and LP checkpoint exports. It is not the final LP time.
2. mip_cold: a fresh exact-APR MIP with original objective and no LP start.
   pipeline_seconds includes common preprocessing and its whole MIP stage.
3. mip_lp_start: optional fresh exact-APR MIP with the LP candidate as its start.
   pipeline_seconds includes the FULL LP pipeline plus this MIP stage. Thus the
   effort spent obtaining its start is not hidden. The start is checked against
   the complete MIP constraints before submission. Submission does not guarantee
   the native solver uses the start; inspect --tee logs.
Only one large method/model is kept active at a time. The LP model is persistent
across its weight tests; cold/warm MIPs use separate fresh model instances.

Default LP time limit: 300 seconds PER weight. MIP time limit: 3600 seconds PER
MIP optimization. These are not equal whole-method wall-clock budgets. Limits,
actual pipeline times, threads, and solver statuses are all recorded. Use
--lp-time-limit, --mip-time-limit, --mip-gap, and --weight-max-solves as needed.
Do not call a time-limited runtime ratio a speedup to optimality.

Main output: comparison.csv and comparison.json
- original_objective, generation_cost, thermal_slack_sum_pu, max_overload_MW
- max_APR_L1_error_MW, balance_error_MW, generator_bound_error_MW
- original_soft_model_feasible
- pipeline_seconds, solve_seconds
- MIP incumbent, lower bound, solver relative gap, solver status
- gap_to_mip_optimum_pct only when a feasible MIP reports optimal within its
  configured solver tolerances; otherwise left blank
- difference_to_best_mip_incumbent_pct: signed comparison to a verified MIP
  incumbent, including after a time limit; this is NOT a proven optimality gap
- relative_gap_to_lower_bound_pct: (candidate cost - best bound)/abs(candidate cost)

settings.json records case SHA256, versions, all settings and contingency IDs.
lp_search/ contains every weight test and selected candidate.
mip_cold/ and mip_lp_start/ contain independent solver results and dispatches.
Outputs are checkpointed after each method. A native crash cannot be recovered
by Python; faulthandler is enabled to help diagnose it in the Slurm error log.
Exit code 0 means all requested methods produced verified feasible candidates;
it does not imply that every MIP was proven optimal. Check the status/gap fields.

Validation completed locally
- Real HiGHS LP bisection and exact-APR MILP on a synthetic 5-bus case.
- Full LP-to-MIP start feasibility, three independent method paths, resource
  lifecycle, and combined timing through a Gurobi API double backed by HiGHS.
- Timeout reporting leaves the optimum gap blank; infeasible incumbents excluded.
- Shell syntax checks for both HPC submission files.
Native Gurobi and the large benchmark still require testing on your HPC.
