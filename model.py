import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel

RESPONSE = "elevation"
PREDICTORS = [
    "year_offset",
    "sin_annual_harmonic",
    "cos_annual_harmonic",
    "sin_second_harmonic",
    "cos_second_harmonic",
    "sin_third_harmonic",
    "cos_third_harmonic",
]
# These are test parameters
GLOBAL_HYPERPARAMETERS = {
    "hyperparam_alpha_mu": 0,
    "hyperparam_alpha_sigma": 100,
    "hyperparam_beta_mu": 0,
    "hyperparam_beta_sigma": 100,
    "hyperparam_sigma_beta": 0.25,
}
# These are the final parameters used by the model
ALPHA_BETA_HYPERPARAMETERS = {
    "hyperparam_alpha_mu": 100,
    "hyperparam_alpha_sigma": 10,
    "hyperparam_beta_mu": 0,
    "hyperparam_beta_sigma": 5,
    "hyperparam_sigma_beta": 0.25,
}

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
        real<lower=0> hyperparam_sigma_beta;
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
        sigma ~ exponential(hyperparam_sigma_beta);
        if (use_likelihood) {
            Y ~ normal(mu, sigma);
        }
    }
    generated quantities {
        array[N] real Y_rep = normal_rng(mu, sigma);
    }
    """


def generate_model(stan_model_code):
    with open("model.stan", "w") as f:
        f.write(stan_model_code)

    model = CmdStanModel(stan_file="model.stan")
    return model


def load_and_transform(data_file) -> tuple[pd.DataFrame, pd.Series]:  # Return X, Y
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
    df["year_offset"] = df["date"].dt.year - min(
        df["date"].dt.year
    )  # Get the year offset since out start date (2007)
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

    df = df.set_index("date")
    return df[PREDICTORS], df[RESPONSE]


def get_model_inputs(X, Y, hyperparameters):
    data = {
        "N": len(X),
        "q": X.shape[1],
        "X": X,
        "Y": Y,
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


def plot_predictive_summary(fit, Y, subtitle, label="Prior", n_draws=20, seed=42):
    """One-figure summary of a prior or posterior fit: predictive draws over time + parameter distributions."""
    rng = np.random.default_rng(seed)
    dates = Y.index

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
        + [(f"beta: {name}", beta_draws[:, i]) for i, name in enumerate(PREDICTORS)]
        + [("sigma", sigma_draws)]
    )

    fig = plt.figure(figsize=(15, 8), layout="constrained")
    grid = fig.add_gridspec(2, len(param_draws), height_ratios=[2, 1])
    fig.suptitle(f"{label} predictive check: {subtitle}", fontsize=14)

    # Top row: predictive distribution over time
    ax_time = fig.add_subplot(grid[0, :])
    ax_time.fill_between(
        dates,
        y_rep_lower,
        y_rep_upper,
        color="C0",
        alpha=0.05,
        label="95% predictive interval",
    )
    for k, draw_idx in enumerate(rng.choice(len(y_rep_draws), n_draws, replace=False)):
        ax_time.scatter(
            dates,
            y_rep_draws[draw_idx],
            s=1,
            color="k",
            alpha=0.02,
            rasterized=True,
            label=f"{n_draws} predictive draws" if k == 0 else None,
        )
    ax_time.plot(dates, y_rep_mean, color="C0", linewidth=1.5, label="Predictive mean")
    if label == "Posterior":
        ax_time.scatter(
            dates, Y.to_numpy(), s=2, color="C3", alpha=0.5, label="Observed"
        )
    ax_time.set_xlabel("Date")
    ax_time.set_ylabel("Water Surface Elevation (ft)")
    ax_time.legend(loc="upper left", markerscale=5, frameon=False)
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


def summarize_posterior(fit, interval=0.95, max_points=2000, seed=0):
    """Pair plot of all unknown parameters plus a median / equal-tailed-interval summary table."""
    # Collect posterior draws, one column per scalar parameter
    beta_draws = fit.stan_variable("beta")
    draws = pd.DataFrame({"alpha": fit.stan_variable("alpha")})
    for i, name in enumerate(PREDICTORS):
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

    print(summary.round(4).to_string())

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
    X, Y = load_and_transform(data_file)
    model = generate_model(stan_model_code=STAN_MODEL)
    hyperparameters = {
        "separate alpha and beta hyperparameters": ALPHA_BETA_HYPERPARAMETERS
    }
    if debug:
        generate_data_exploration_plots(X, Y)
        hyperparameters["uniform alpha and beta hyperparameters"] = (
            GLOBAL_HYPERPARAMETERS
        )
    for k, hyperparam_set in hyperparameters.items():
        model_inputs = get_model_inputs(X=X, Y=Y, hyperparameters=hyperparam_set)
        fit_prior = generate_prior_predictive(model, model_inputs)
        if debug:
            plot_predictive_summary(fit_prior, Y, subtitle=k, label="Prior")
        else:
            fit_posterior = generate_posterior_predictive(model, model_inputs)
            summarize_posterior(fit_posterior)
            plot_predictive_summary(fit_posterior, Y, subtitle=k, label="Posterior")


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
