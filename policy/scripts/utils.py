import numpy as np
from copy import deepcopy
from scipy.io.matlab import mat_struct

def matstruct_to_dict(matobj):
    """Recursively convert mat_struct to nested dicts."""
    if isinstance(matobj, mat_struct):
        result = {}
        for field in matobj._fieldnames:
            value = getattr(matobj, field)
            if isinstance(value, mat_struct):
                result[field] = matstruct_to_dict(value)
            else:
                result[field] = value
        return result
    return matobj

def remove_bad_seeds(*scenarios):
    """
    Remove bad seed columns from any number of scenario dicts.
    Bad columns are those where nominal_gdp_quarterly contains NaN, 0, or inf.

    Usage:
        s1_clean, s2_clean, ... = remove_bad_seeds(s1, s2, ...)
    """

    # Collect nominal GDP matrices
    nominal_gdp_all = []
    for idx, scenario in enumerate(scenarios, start=1):
        if scenario:
            if "nominal_gdp_quarterly" not in scenario:
                raise ValueError(f"Scenario {idx} does not have field 'nominal_gdp_quarterly'.")
            nominal_gdp_all.append(scenario["nominal_gdp_quarterly"])

    # Compute union of bad columns across all scenarios
    if nominal_gdp_all:
        n_cols = nominal_gdp_all[0].shape[1]
        bad_columns = np.zeros(n_cols, dtype=bool)
        for gdp in nominal_gdp_all:
            bad_columns |= np.any(
                np.isnan(gdp) | (gdp == 0) | np.isinf(gdp),
                axis=0
            )
    else:
        bad_columns = np.array([], dtype=bool)

    # Clean each scenario
    cleaned = []
    for scenario in scenarios:
        if scenario:
            cleaned.append(clean_scenario(deepcopy(scenario), bad_columns))
        else:
            cleaned.append(scenario)

    return cleaned if len(cleaned) > 1 else cleaned[0]


def clean_scenario(scenario, bad_columns):
    """
    Removes columns from all numeric fields where the number of columns matches
    the nominal_gdp_quarterly field.
    """
    n_bad = len(bad_columns)

    for key, value in scenario.items():
        if isinstance(value, np.ndarray):

            # 2D numeric field
            if value.ndim == 2 and value.shape[1] == n_bad:
                scenario[key] = value[:, ~bad_columns]

            # 3D numeric field
            elif value.ndim == 3 and value.shape[1] == n_bad:
                scenario[key] = value[:, ~bad_columns, :]

    return scenario

def shaded_error_bar(ax, x, y, err, color='k', alpha=0.15, linewidth=1.25):
    """
    Plot a line with a shaded error bar.
    
    Parameters:
        ax : matplotlib.axes.Axes
            Axes object to plot on.
        x : array-like
            x values.
        y : array-like
            y values (mean line).
        err : array-like, shape (2, len(x))
            Upper and lower error values.
        color : str
            Line color.
        alpha : float
            Transparency of the shaded area.
        linewidth : float
            Width of the main line.
    
    Returns:
        line : matplotlib.lines.Line2D
            Handle to the main line.
    """
    upper = y + err[0]
    lower = y - err[1]
    
    ax.fill_between(x, lower, upper, color=color, alpha=alpha, edgecolor='none')
    line, = ax.plot(x, y, color=color, linewidth=linewidth)
    ax.plot(x, upper, '-', color=color, alpha=0.5, linewidth=0.75)
    ax.plot(x, lower, '-', color=color, alpha=0.5, linewidth=0.75)
    
    return line

def ci(data, interval=[0.9, 0.1]):
    """
    Compute confidence intervals similar to MATLAB version.
    
    Parameters:
        data : array-like, shape (n_samples, n_points)
            Input data. Each row is an observation.
        interval : list of two floats
            Upper and lower quantiles (default [0.9, 0.1]).
    
    Returns:
        err : np.ndarray, shape (2, n_points)
            Upper and lower errors for shaded error bars.
    """
    data = np.atleast_2d(data)
    if data.shape[0] > 1:
        lower = np.abs(np.quantile(data, interval[1], axis=0) - np.mean(data, axis=0))
        upper = np.abs(np.quantile(data, interval[0], axis=0) - np.mean(data, axis=0))
        return np.vstack([upper, lower])
    else:
        return np.zeros((2, data.shape[1]))
