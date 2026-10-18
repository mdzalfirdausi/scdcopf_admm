
from pathlib import Path
import numpy as np
import pandas as pd

EXCEL_DIR = Path(r"M:\projects\excel_outputs")
UNCERTAINTY_DIR = Path(__file__).resolve().parent / "uncertainty"


def generate_uncertainty(
    case_name,
    n_scenarios=100,
    sigma_scaling=0.03,
    seed=42
):
    case_path = EXCEL_DIR / f"{case_name}.xlsx"

    bus_df = pd.read_excel(case_path, sheet_name="bus")
    base_df = pd.read_excel(case_path, sheet_name="baseMVA")

    baseMVA = float(base_df.iloc[0, 0])

    # Nominal demand in per unit
    d = bus_df["Pd"].to_numpy(dtype=float) / baseMVA

    # Standard deviation proportional to nominal demand
    sigma = sigma_scaling * np.abs(d)

    rng = np.random.default_rng(seed)

    # Rows: scenarios; columns: buses
    omega = rng.normal(
        loc=0.0,
        scale=sigma,
        size=(n_scenarios, len(d))
    )

    UNCERTAINTY_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        UNCERTAINTY_DIR /
        f"omega_{case_name}_sigma{sigma_scaling}_seed{seed}.npz"
    )

    np.savez_compressed(
        output_path,
        omega=omega,
        bus_ids=bus_df["bus_i"].to_numpy(),
        sigma_scaling=sigma_scaling,
        seed=seed,
        baseMVA=baseMVA
    )

    print(f"Saved: {output_path}")
    print(f"Scenarios: {n_scenarios}")
    print(f"Buses: {len(d)}")

    return output_path



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Gaussian load uncertainty scenarios for SCDCOPF"
    )

    # Required arguments
    parser.add_argument(
        "--case",
        type=str,
        required=True,
        help="PGLib case name without .xlsx extension"
    )

    parser.add_argument(
        "--sigma-scaling",
        type=float,
        required=True,
        help="Standard deviation as a fraction of nominal load"
    )

    # Optional arguments
    parser.add_argument(
        "--scenarios",
        type=int,
        default=3,
        help="Number of uncertainty scenarios (default: 3)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=19,
        help="Random seed (default: 19)"
    )

    args = parser.parse_args()

    generate_uncertainty(
        case_name=args.case,
        n_scenarios=args.scenarios,
        sigma_scaling=args.sigma_scaling,
        seed=args.seed
    )
