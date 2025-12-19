import itertools
import numpy as np
import random
from collections import defaultdict
import pandas as pd

import example_02_binary as ex_bin
import example_02_real as ex_real

#METHOD = "binary"
METHOD = "real"


def get_backend(method: str):
    method = method.strip().lower()
    if method == "binary":
        return ex_bin
    if method == "real":
        return ex_real
    raise ValueError(f"Unknown METHOD={method!r}. Use 'binary' or 'real'.")


EX = get_backend(METHOD)


def fitness_to_result(ex_module, fit):

    if hasattr(ex_module, "fitness_to_true_val"):
        return ex_module.fitness_to_true_val(fit)
    if hasattr(ex_module, "fitness_to_true_value"):
        return ex_module.fitness_to_true_value(fit)
    # Ostateczny fallback: zwróć fitness wprost
    return float(fit)


def run_one(selection, crossover, mutation, seed):
    np.random.seed(seed)
    random.seed(seed)

    ga = EX.make_ga(
        parent_selection_type=selection,
        crossover_type=crossover,
        mutation_type=mutation,
        mutation_num_genes=1,
        K_tournament=3,
        log=False
    )

    ga.run()
    _, fit, _ = ga.best_solution()
    best_val = fitness_to_result(EX, fit)

    population_size = None
    num_epochs = None

    for attr in ("sol_per_pop", "population_size", "pop_size"):
        if hasattr(ga, attr):
            population_size = getattr(ga, attr)
            break

    for attr in ("num_generations", "num_epochs", "generations"):
        if hasattr(ga, attr):
            num_epochs = getattr(ga, attr)
            break

    return best_val, population_size, num_epochs


def run_grid(repeats=10, base_seed=123):
    selections = ["tournament", "rws", "random"]


    if METHOD == "binary":
        crossovers = ["single_point", "two_points", "uniform"]
        mutations = ["random", "swap"]
    else:  # real
        crossovers = ["arithmetic", "linear", "blend_alpha_beta"]  #blend_alpha, average
        mutations = ["gaussian", "uniform"]


    configs = list(itertools.product(selections, crossovers, mutations))

    total = len(configs) * repeats
    counter = 0

    agg = defaultdict(list)
    config_meta = {}

    for config_id, (sel, cr, mut) in enumerate(configs):
        for r in range(repeats):
            counter += 1
            print(f"[{METHOD}] Run {counter}/{total}: {sel}, {cr}, {mut}, rep={r}", flush=True)

            seed = base_seed + 10000 * config_id + r
            best_val, pop_size, epochs = run_one(sel, cr, mut, seed)

            agg[(sel, cr, mut)].append(best_val)

            if (sel, cr, mut) not in config_meta:
                config_meta[(sel, cr, mut)] = (pop_size, epochs)

    return agg, config_meta


def summarize_to_dataframe(agg, config_meta):
    rows = []
    for (sel, cr, mut), vals in agg.items():
        vals = np.array(vals, dtype=float)
        pop_size, epochs = config_meta.get((sel, cr, mut), (None, None))

        rows.append({
            "method": METHOD,
            "selection": sel,
            "crossover": cr,
            "mutation": mut,
            "population_size": pop_size,
            "num_epochs": epochs,
            "repeats": int(len(vals)),
            "mean_best_val": float(vals.mean()),
            "std_best_val": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        })

    df = pd.DataFrame(rows).sort_values("mean_best_val").reset_index(drop=True)
    return df


def save_results(df, out_prefix=None):
    if out_prefix is None:
        out_prefix = f"ga_{METHOD}_experiments_summary"

    csv_path = f"{out_prefix}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    agg, config_meta = run_grid(repeats=10, base_seed=123)
    df = summarize_to_dataframe(agg, config_meta)

    print(df.to_string(index=False))

    save_results(df)