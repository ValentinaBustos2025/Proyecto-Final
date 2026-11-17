from functools import lru_cache
from inspect import Parameter, signature
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares


@lru_cache(None)
def _firma(funcion):
    return signature(funcion)


def evaluar(funcion, **parametros):
    """Evaluate a payoff function using only the arguments it actually accepts."""
    sig = _firma(funcion)
    accepts_extra = any(p.kind == Parameter.VAR_KEYWORD
                        for p in sig.parameters.values())
    used = parametros if accepts_extra else {
        k: v for k, v in parametros.items() if k in sig.parameters
    }
    return float(funcion(**used))


def payoff(WAA, WAB, WBA, WBB, par):
    return tuple(evaluar(f, **par) for f in (WAA, WAB, WBA, WBB))


def pEstable(WAA, WAB, WBA, WBB, tol=1e-9, **kwargs):
    a, b, c, d = payoff(WAA, WAB, WBA, WBB, kwargs)
    den = a - b - c + d
    if abs(den) <= tol:
        return None
    p = (d - b) / den
    return float(p) if 0 < p < 1 else None


def classification(WAA, WAB, WBA, WBB, tol=1e-9, **kwargs):
    a, b, c, d = payoff(WAA, WAB, WBA, WBB, kwargs)
    x, y = a - c, b - d

    if abs(x) <= tol and abs(y) <= tol:
        regime = 'Neutrality'
    elif x > tol and y > tol:
        regime = 'A dominates B'
    elif x < -tol and y < -tol:
        regime = 'B dominates A'
    elif x > tol and y < -tol:
        regime = 'Bistability'
    elif x < -tol and y > tol:
        regime = 'Stable coexistence'
    else:
        regime = 'Boundary case'

    # Interior equilibrium (inlined from pEstable, avoids a second payoff call)
    den = a - b - c + d
    pstar = (d - b) / den if abs(den) > tol else None
    if pstar is not None and not (0 < pstar < 1):
        pstar = None

    ess = []
    if x > tol or (abs(x) <= tol and y > tol):
        ess.append('A')
    if y < -tol or (abs(y) <= tol and x < -tol):
        ess.append('B')
    if regime == 'Stable coexistence':
        ess.append('Mixed')

    return {
        'regime': regime,
        'p_star': pstar,
        'ESS': ', '.join(ess) if ess else 'None',
    }


def findParameters(WAA, WAB, WBA, WBB, rangos, tol=1e-9,
                   maxiter=300, seed=42, **fijos):
    """Searches for examples; 'found' only verifies each candidate, it does not guarantee existence."""
    nombres = list(rangos)
    limites = [rangos[k] for k in nombres]

    def parametros(z):
        return {**fijos, **dict(zip(nombres, map(float, z)))}

    def diferencias(z):
        a, b, c, d = payoff(WAA, WAB, WBA, WBB, parametros(z))
        return np.array([a - c, b - d])

    rows = []
    objectives = {
        'A dominates B': (1, 1),
        'B dominates A': (-1, -1),
        'Bistability': (1, -1),
        'Stable coexistence': (-1, 1),
    }

    for target, signs in objectives.items():
        def costo(z, s=np.asarray(signs, dtype=float)):
            v = diferencias(z) * s
            return np.sum(np.maximum(1e-3 - v, 0) ** 2)

        sol = differential_evolution(costo, limites, seed=seed, maxiter=maxiter)
        par = parametros(sol.x)
        res = classification(WAA, WAB, WBA, WBB, tol=tol, **par)
        rows.append({'target': target,
                     'found': res['regime'] == target,
                     **par, **res})

    sol = differential_evolution(
        lambda z: np.sum(diferencias(z) ** 2),
        limites, seed=seed, maxiter=maxiter)
    lo, hi = np.array(limites).T
    fit = least_squares(diferencias, sol.x, bounds=(lo, hi),
                        xtol=1e-13, ftol=1e-13, gtol=1e-13)
    par = parametros(fit.x)
    res = classification(WAA, WAB, WBA, WBB, tol=tol, **par)
    rows.append({'target': 'Neutrality',
                 'found': res['regime'] == 'Neutrality',
                 **par, **res})
    return pd.DataFrame(rows)


def barridoParametros(WAA, WAB, WBA, WBB, barridos,
                      archivo='Outputs/StabilityLineal.csv', **fijos):
    nombres = list(barridos)
    rows = []
    for valores in product(*(barridos[k] for k in nombres)):
        par = {**fijos, **dict(zip(nombres, map(float, valores)))}
        rows.append({**par, **classification(WAA, WAB, WBA, WBB, **par)})

    table = pd.DataFrame(rows)
    if archivo is not None:
        path = Path(archivo)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, encoding='utf-8-sig')
    return table