import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel

RESPONSE = "elevation"
ANNUAL_PREDICTORS = ["year_offset"]
HARMONIC_PREDICTORS = [
    "sin_annual_harmonic",
    "cos_annual_harmonic",
    "sin_second_harmonic",
    "cos_second_harmonic",
    "sin_third_harmonic",
    "cos_third_harmonic",
]
# Years with fewer daily observations than this are dropped (partial years bias the annual mean by season)
MIN_DAYS_PER_YEAR = 350

# These are test parameters
GLOBAL_HYPERPARAMETERS = {
    "hyperparam_alpha_mu": 0,
    "hyperparam_alpha_sigma": 100,
    "hyperparam_beta_mu": 0,
    "hyperparam_beta_sigma": 100,
    "hyperparam_sigma": 0.25,
}
# These are the final parameters used by the models
ANNUAL_HYPERPARAMETERS = {
    "hyperparam_alpha_mu": 96.5,
    "hyperparam_alpha_sigma": 2,
    "hyperparam_beta_mu": 0.0,
    "hyperparam_beta_sigma": 0.1,
    "hyperparam_sigma": 1,
}
HARMONIC_HYPERPARAMETERS = {
    "hyperparam_alpha_mu": 0,
    "hyperparam_alpha_sigma": 1,
    "hyperparam_beta_mu": 0,
    "hyperparam_beta_sigma": 5,
    "hyperparam_sigma": 0.25,
}

# One Stan program serves both models; only the data passed in differs
STAN_MODEL = """
    data {
        int<lower=1> N;
        int<lower=1> q;
        matrix[N, q] X;
        vector[N] Y;
        real hyperparam_alpha_mu;
        real<lower=0> hyperparam_alpha_sigma;
        real hyperparam_beta_mu;
        real<lower=0> hyperparam_beta_sigma;
        real<lower=0> hyperparam_sigma;
        int<lower=0, upper=1> use_likelihood;
    }
    parameters {
        real alpha;
        vector[q] beta;
        real<lower=0> sigma;
    }
    transformed parameters {
        vector[N] mu = alpha + X * beta;
    }
    model {
        alpha ~ normal(hyperparam_alpha_mu, hyperparam_alpha_sigma);
        beta ~ normal(hyperparam_beta_mu, hyperparam_beta_sigma);
        sigma ~ exponential(hyperparam_sigma);
        if (use_likelihood) {
            Y ~ normal(mu, sigma);
        }
    }
    generated quantities {
        array[N] real Y_rep = normal_rng(mu, sigma);
    }
    """


def generate_model(stan_model_code, stan_file="model.stan"):
    with open(stan_file, "w") as f:
        f.write(stan_model_code)

    model = CmdStanModel(stan_file=stan_file)
    return model


def load_daily(data_file) -> pd.DataFrame:
    """Load raw 15-min data and return one row per day with year and harmonic features."""
    with open(data_file, "r") as f:
        df = pd.read_csv(
            f,
            sep="\t",
            skiprows=29,
            names=[
                "entity",
                "unknown",
                "datetime",
                "timezone",
                "0",
                "1",
                "elevation",
                "2",
            ],
        )
    df["date"] = pd.to_datetime(
        df["datetime"].str.split(" ").str[0]
    )  # remove time components
    df = (
        df.groupby("date").agg({"elevation": "mean"}).reset_index()
    )  # Use the daily mean elevation
    df["year"] = df["date"].dt.year
    scaled_day_of_year = (
        2 * np.pi * (((df["date"].dt.dayofyear - 1) / 365.25) % 1)
    )  # Scale the day of year beween 0 and 2pi
    # Compute the sin and cosine components for the day of year
    df["sin_annual_harmonic"] = np.sin(scaled_day_of_year)
    df["cos_annual_harmonic"] = np.cos(scaled_day_of_year)
    df["sin_second_harmonic"] = np.sin(2 * scaled_day_of_year)
    df["cos_second_harmonic"] = np.cos(2 * scaled_day_of_year)
    df["sin_third_harmonic"] = np.sin(3 * scaled_day_of_year)
    df["cos_third_harmonic"] = np.cos(3 * scaled_day_of_year)

    return df.set_index("date")


def build_annual_dataset(daily) -> tuple[pd.DataFrame, pd.Series]:
    """One row per complete year: annual mean elevation regressed on year offset."""
    days_per_year = daily.groupby("year")[RESPONSE].count()
    complete_years = days_per_year[days_per_year >= MIN_DAYS_PER_YEAR].index

    annual = (
        daily[daily["year"].isin(complete_years)]
        .groupby("year")[RESPONSE]
        .mean()
        .to_frame()
    )
    annual["year_offset"] = annual.index - annual.index.min()  # 0 = first complete year
    return annual[ANNUAL_PREDICTORS], annual[RESPONSE]


def build_harmonic_dataset(daily, Y_annual) -> tuple[pd.DataFrame, pd.Series]:
    """One row per day: elevation minus that year's annual mean, regressed on the harmonics."""
    harmonic = daily[daily["year"].isin(Y_annual.index)].copy()
    harmonic["elevation_anomaly"] = harmonic[RESPONSE] - harmonic["year"].map(Y_annual)
    return harmonic[HARMONIC_PREDICTORS], harmonic["elevation_anomaly"]


def get_model_inputs(X, Y, hyperparameters):
    data = {
        "N": len(X),
        "q": X.shape[1],
        "X": X.to_numpy(),
        "Y": Y.to_numpy(),
        **hyperparameters,
    }
    return data


def generate_prior_predictive(model, model_inputs):
    fit_prior = model.sample(
        data={**model_inputs, "use_likelihood": 0}, chains=2, seed=6300
    )
    return fit_prior


def generate_posterior_predictive(model, model_inputs):
    fit_posterior = model.sample(
        data={**model_inputs, "use_likelihood": 1}, chains=2, seed=6301
    )
    return fit_posterior


def plot_predictive_summary(
    fit,
    Y,
    predictors,
    subtitle,
    label="Prior",
    xlabel="Date",
    ylabel="Water Surface Elevation (ft)",
    n_draws=20,
    seed=42,
):
    """One-figure summary of a prior or posterior fit: predictive draws over time + parameter distributions."""
    rng = np.random.default_rng(seed)
    x_values = Y.index
    few_points = len(Y) < 100  # annual model has ~18 points; daily model has thousands
    draw_alpha, draw_size, obs_size = (0.3, 10, 30) if few_points else (0.02, 1, 2)

    # Draws from the fit
    y_rep_draws = fit.stan_variable("Y_rep")  # (n_samples, N)
    alpha_draws = fit.stan_variable("alpha")  # (n_samples,)
    beta_draws = fit.stan_variable("beta")  # (n_samples, q)
    sigma_draws = fit.stan_variable("sigma")  # (n_samples,)

    # Pointwise summaries of the predictive distribution
    y_rep_mean = y_rep_draws.mean(axis=0)
    y_rep_lower, y_rep_upper = np.percentile(y_rep_draws, [2.5, 97.5], axis=0)

    param_draws = (
        [("alpha (intercept)", alpha_draws)]
        + [(f"beta: {name}", beta_draws[:, i]) for i, name in enumerate(predictors)]
        + [("sigma", sigma_draws)]
    )

    fig = plt.figure(figsize=(15, 8), layout="constrained")
    grid = fig.add_gridspec(2, len(param_draws), height_ratios=[2, 1])
    fig.suptitle(f"{label} predictive check: {subtitle}", fontsize=14)

    # Top row: predictive distribution over time
    ax_time = fig.add_subplot(grid[0, :])
    ax_time.fill_between(
        x_values,
        y_rep_lower,
        y_rep_upper,
        color="C0",
        alpha=0.05,
        label="95% predictive interval",
    )
    for k, draw_idx in enumerate(rng.choice(len(y_rep_draws), n_draws, replace=False)):
        ax_time.scatter(
            x_values,
            y_rep_draws[draw_idx],
            s=draw_size,
            color="k",
            alpha=draw_alpha,
            rasterized=True,
            label=f"{n_draws} predictive draws" if k == 0 else None,
        )
    ax_time.plot(x_values, y_rep_mean, color="C0", linewidth=1.5, label="Predictive mean")
    if label == "Posterior":
        ax_time.scatter(
            x_values, Y.to_numpy(), s=obs_size, color="C3", alpha=0.5, label="Observed"
        )
    ax_time.set_xlabel(xlabel)
    ax_time.set_ylabel(ylabel)
    ax_time.legend(loc="upper left", markerscale=5 if not few_points else 1, frameon=False)
    ax_time.grid(alpha=0.3)

    # Bottom row: parameter distributions
    for col, (name, draws) in enumerate(param_draws):
        ax = fig.add_subplot(grid[1, col])
        ax.hist(draws, bins=50, color="C0", alpha=0.7)
        ax.axvline(draws.mean(), color="k", linestyle="--", linewidth=1)
        lower, upper = np.percentile(draws, [2.5, 97.5])
        ax.set_title(
            f"{name}\nmean {draws.mean():.2f}  [{lower:.2f}, {upper:.2f}]", fontsize=9
        )
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)

    plt.show()


def summarize_posterior(fit, predictors, interval=0.95, max_points=2000, seed=0):
    """Pair plot of all unknown parameters plus a median / equal-tailed-interval summary table."""
    # Collect posterior draws, one column per scalar parameter
    beta_draws = fit.stan_variable("beta")
    draws = pd.DataFrame({"alpha": fit.stan_variable("alpha")})
    for i, name in enumerate(predictors):
        draws[f"beta_{name}"] = beta_draws[:, i]
    draws["sigma"] = fit.stan_variable("sigma")
    params = list(draws.columns)

    # Summary table: median + equal-tailed interval
    tail_pct = (1 - interval) / 2 * 100
    lower, median, upper = np.percentile(draws, [tail_pct, 50, 100 - tail_pct], axis=0)
    summary = pd.DataFrame(
        {
            "median": median,
            f"p{round(tail_pct, 1)}": lower,
            f"p{round(100 - tail_pct, 1)}": upper,
            f"mean": draws.mean(),
            f"sd": draws.std(),
        },
        index=params,
    )

    print(summary[["mean", "sd"]].round(2).to_string())

    # Pair plot
    rng = np.random.default_rng(seed)
    thin_idx = rng.choice(len(draws), size=min(max_points, len(draws)), replace=False)
    thinned = draws.iloc[thin_idx]
    correlations = draws.corr()
    axis_limits = {p: (draws[p].min(), draws[p].max()) for p in params}

    n_params = len(params)
    fig, axes = plt.subplots(
        n_params,
        n_params,
        figsize=(2.5 * n_params, 2.5 * n_params),
        layout="constrained",
    )
    fig.suptitle(
        f"Posterior pair plot (median and {interval * 100}% interval on diagonal)",
        fontsize=13,
    )

    for row, row_param in enumerate(params):
        for col, col_param in enumerate(params):
            ax = axes[row, col]

            if row < col:  # upper triangle: correlation
                r = correlations.loc[row_param, col_param]
                ax.text(
                    0.5,
                    0.5,
                    f"r = {r:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10 + 8 * abs(r),
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            if row == col:  # diagonal: marginal posterior
                ax.hist(draws[col_param], bins=50, color="C0", alpha=0.7)
                ax.axvspan(
                    summary.loc[col_param, f"p{round(tail_pct, 1)}"],
                    summary.loc[col_param, f"p{round(100 - tail_pct, 1)}"],
                    color="C1",
                    alpha=0.15,
                )
                ax.axvline(summary.loc[col_param, "median"], color="C1", linewidth=1.5)
                ax.set_yticks([])
            else:  # lower triangle: joint posterior
                ax.scatter(
                    thinned[col_param],
                    thinned[row_param],
                    s=2,
                    alpha=0.3,
                    color="C0",
                    rasterized=True,
                )
                ax.set_ylim(axis_limits[row_param])

            ax.set_xlim(axis_limits[col_param])

            # Axis labels only on the outer edges
            if row == n_params - 1:
                ax.set_xlabel(col_param)
                ax.tick_params(axis="x", labelrotation=45)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(row_param)
            elif row != col:
                ax.set_yticklabels([])

    plt.show()


def _plot_predictive_panel(ax, x_values, rep_draws, observed, title, ylabel, rng, n_draws=20):
    """One panel in the style of plot_predictive_summary: 95% band, draws, mean, observed."""
    few_points = len(observed) < 100  # annual panel has ~18 points; daily panels have thousands
    draw_alpha, draw_size, obs_size = (0.3, 10, 30) if few_points else (0.02, 1, 2)

    rep_mean = rep_draws.mean(axis=0)
    rep_lower, rep_upper = np.percentile(rep_draws, [2.5, 97.5], axis=0)
    coverage = ((observed >= rep_lower) & (observed <= rep_upper)).mean()
    print(f"{title}")

    ax.fill_between(x_values, rep_lower, rep_upper, color="C0", alpha=0.1,
                    label="95% predictive interval")
    for k, draw_idx in enumerate(rng.choice(len(rep_draws), n_draws, replace=False)):
        ax.scatter(x_values, rep_draws[draw_idx], s=draw_size, color="k", alpha=draw_alpha,
                   rasterized=True, label=f"{n_draws} predictive draws" if k == 0 else None)
    ax.plot(x_values, rep_mean, color="C0", linewidth=1.5, label="Predictive mean")
    ax.scatter(x_values, observed, s=obs_size, color="C3", alpha=0.5, label="Observed")
    ax.set_title(f"{title}", fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def plot_combined_model(fit_annual, fit_harmonic, Y_annual, Y_harmonic, Y_observed,
                        n_draws=20, seed=42):
    """Three stacked predictive panels over the full record.

    Top:    annual model (annual mean elevation), plotted at mid-year.
    Middle: harmonic model (daily elevation minus that year's mean).
    Bottom: both combined (daily elevation), with the annual trend line.

    The two models were fit separately, so their posterior draws are independent and can be
    paired by index when combining them.
    """
    rng = np.random.default_rng(seed)

    annual_mu = fit_annual.stan_variable("mu")          # (n_samples, n_years): trend line
    annual_rep = fit_annual.stan_variable("Y_rep")      # (n_samples, n_years): trend + year noise
    seasonal_rep = fit_harmonic.stan_variable("Y_rep")  # (n_samples, n_days): cycle + daily noise
    n_samples = min(len(annual_rep), len(seasonal_rep))

    years = Y_annual.index.to_numpy()
    mid_year_dates = pd.to_datetime([f"{year}-07-01" for year in years])
    dates = Y_observed.index                       # same days, in the same order, as Y_harmonic
    year_pos = np.searchsorted(years, dates.year)  # each day's row in the annual model

    # Layout: the two individual models side by side on top, the combined model full-width below
    fig = plt.figure(figsize=(16, 11), layout="constrained")
    grid = fig.add_gridspec(2, 2)
    ax_combined = fig.add_subplot(grid[1, :])
    ax_annual = fig.add_subplot(grid[0, 0], sharex=ax_combined)
    ax_season = fig.add_subplot(grid[0, 1], sharex=ax_combined)
    fig.suptitle("Annual model, harmonic model, and both combined", fontsize=14)

    # ---- Panel 1: annual model --------------------------------------------------------------
    _plot_predictive_panel(ax_annual, mid_year_dates, annual_rep, Y_annual.to_numpy(),
                           title="Annual model", ylabel="Annual Mean Elevation (ft)",
                           rng=rng, n_draws=n_draws)
    ax_annual.legend(loc="upper right", fontsize=8, frameon=False)

    # ---- Panel 2: harmonic model ------------------------------------------------------------
    _plot_predictive_panel(ax_season, dates, seasonal_rep, Y_harmonic.to_numpy(),
                           title="Harmonic model", ylabel="Elevation Minus Annual Mean (ft)",
                           rng=rng, n_draws=n_draws)
    ax_season.legend(loc="upper right", fontsize=8, frameon=False, markerscale=5)

    # ---- Panel 3: combined model + annual trend ---------------------------------------------
    combined_rep = annual_rep[:n_samples, year_pos] + seasonal_rep[:n_samples]
    _plot_predictive_panel(ax_combined, dates, combined_rep, Y_observed.to_numpy(),
                           title="Combined model (annual + harmonic)",
                           ylabel="Water Surface Elevation (ft)", rng=rng, n_draws=n_draws)
    trend_mean = annual_mu.mean(axis=0)[year_pos]
    ax_combined.plot(dates, trend_mean, color="C1", linestyle="--", linewidth=2,
                     label="Annual trend")
    ax_combined.set_xlabel("Date")
    ax_combined.legend(loc="upper right", fontsize=8, frameon=False, markerscale=5)

    plt.show()


def generate_data_exploration_plots(X, Y):
    def _plot_regression(_X, _Y, xlabel):
        plt.scatter(
            _X,
            _Y,
            color="blue",
            label="Lake Champlain Water Surface Elevation",
            alpha=0.1,
        )
        beta, alpha = np.polyfit(_X, _Y, deg=1)
        plt.plot(
            _X,
            alpha + beta * _X,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Line of Best Fit",
        )
        plt.xlabel(xlabel)
        plt.ylabel("Water Surface Elevation (ft)")
        plt.title(f"Lake Champlain Water Surface Elevation By {xlabel}")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Plot the actual date timeseries
    plt.scatter(
        X.index,
        Y.values,
        color="blue",
        label="Lake Champlain Water Surface Elevation",
        alpha=0.1,
    )
    plt.xlabel("Date")
    plt.ylabel("Water Surface Elevation (ft)")
    plt.title("Lake Champlain Water Surface Elevation Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot regressions against each predictor
    _plot_regression(X.index.year, Y.values, "Year")  # yearly regression
    _plot_regression(
        X["sin_annual_harmonic"], Y.values, "sin(Day of Year)"
    )  # sin day of year regression
    _plot_regression(
        X["cos_annual_harmonic"], Y.values, "cos(Day of Year)"
    )  # cos day of year regression
    _plot_regression(
        X["sin_second_harmonic"], Y.values, "sin(2 * Day of Year)"
    )  # sin 2 * day of year regression
    _plot_regression(
        X["cos_second_harmonic"], Y.values, "cos(2 * Day of Year)"
    )  # cos 2 * day of year regression
    _plot_regression(
        X["sin_third_harmonic"], Y.values, "sin(3 * Day of Year)"
    )  # sin 3 * day of year regression
    _plot_regression(
        X["cos_third_harmonic"], Y.values, "cos(3 * Day of Year)"
    )  # cos 3 * day of year regression


def main(data_file, debug):
    daily = load_daily(data_file)
    X_annual, Y_annual = build_annual_dataset(daily)
    X_harmonic, Y_harmonic = build_harmonic_dataset(daily, Y_annual)
    model = generate_model(stan_model_code=STAN_MODEL)

    model_configs = {
        "annual": {
            "X": X_annual,
            "Y": Y_annual,
            "predictors": ANNUAL_PREDICTORS,
            "hyperparameters": ANNUAL_HYPERPARAMETERS,
            "xlabel": "Year",
            "ylabel": "Annual Mean Water Surface Elevation (ft)",
        },
        "harmonic": {
            "X": X_harmonic,
            "Y": Y_harmonic,
            "predictors": HARMONIC_PREDICTORS,
            "hyperparameters": HARMONIC_HYPERPARAMETERS,
            "xlabel": "Date",
            "ylabel": "Elevation Minus Annual Mean (ft)",
        },
    }

    if debug:
        generate_data_exploration_plots(daily, daily[RESPONSE])

    final_posteriors = {}  # posterior fits with the final hyperparameters, for the combined plot
    for model_name, config in model_configs.items():
        hyperparameter_sets = {f"{model_name} hyperparameters": config["hyperparameters"]}
        if debug:
            hyperparameter_sets[f"{model_name} test hyperparameters"] = GLOBAL_HYPERPARAMETERS

        for subtitle, hyperparam_set in hyperparameter_sets.items():
            model_inputs = get_model_inputs(
                X=config["X"], Y=config["Y"], hyperparameters=hyperparam_set
            )
            plot_kwargs = dict(
                Y=config["Y"],
                predictors=config["predictors"],
                subtitle=subtitle,
                xlabel=config["xlabel"],
                ylabel=config["ylabel"],
            )
            if debug:
                fit_prior = generate_prior_predictive(model, model_inputs)
                plot_predictive_summary(fit_prior, label="Prior", **plot_kwargs)
            else:
                fit_posterior = generate_posterior_predictive(model, model_inputs)
                summarize_posterior(fit_posterior, predictors=config["predictors"])
                plot_predictive_summary(fit_posterior, label="Posterior", **plot_kwargs)
                if hyperparam_set is config["hyperparameters"]:
                    final_posteriors[model_name] = fit_posterior

    if not debug:
        days = Y_harmonic.index  # complete years only, same days the harmonic model used
        plot_combined_model(
            fit_annual=final_posteriors["annual"],
            fit_harmonic=final_posteriors["harmonic"],
            Y_annual=Y_annual,
            Y_harmonic=Y_harmonic,
            Y_observed=daily.loc[days, RESPONSE],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a bayesian model")
    parser.add_argument(
        "--data-file",
        "-f",
        default=os.path.join("data", "raw.txt"),
        help="Input raw data file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Evaluate results for different priors",
    )
    args = parser.parse_args()
    main(args.data_file, args.debug)