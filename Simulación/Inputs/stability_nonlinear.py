from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, differential_evolution, minimize_scalar

from .stability_linear import evaluar


def deltaPi(p, WAA, WAB, WBA, WBB, **kwargs):
    par = {**kwargs, 'p': float(p)}
    a, b, c, d = (evaluar(f, **par) for f in (WAA, WAB, WBA, WBB))
    return p * (a - c) + (1 - p) * (b - d)


def classificationNoLineal(WAA, WAB, WBA, WBB,
                           n_points=300, tol=1e-9, **kwargs):
    """Classify the replicator flow and return its interior equilibria."""
    def delta(p):
        return deltaPi(float(p), WAA, WAB, WBA, WBB, **kwargs)

    ps = np.linspace(0, 1, n_points + 1)
    ys = np.array([delta(p) for p in ps])
    if not np.all(np.isfinite(ys)):
        raise ValueError('Payoffs must be finite for all 0 <= p <= 1.')

    if np.max(np.abs(ys)) <= tol:
        return {'regime': 'Neutrality', 'p0': 'Neutral', 'p1': 'Neutral',
                'p_star': None, 'equilibria': [], 'n_equilibria': 0,
                'attractors': 'None'}

    roots = []
    for i in range(n_points):
        if ys[i] * ys[i + 1] < 0:
            roots.append(brentq(delta, ps[i], ps[i + 1]))
        if 0 < i < n_points and abs(ys[i]) <= tol:
            roots.append(float(ps[i]))

    # Look for local minima of |delta| to catch tangencies.
    for i in range(1, n_points):
        if abs(ys[i]) <= min(abs(ys[i - 1]), abs(ys[i + 1])):
            sol = minimize_scalar(lambda p: delta(p) ** 2,
                                  bounds=(ps[i - 1], ps[i + 1]),
                                  method='bounded', options={'xatol': 1e-12})
            if abs(delta(sol.x)) <= tol:
                roots.append(float(sol.x))

    unique = []
    for r in sorted(r for r in roots if 1e-8 < r < 1 - 1e-8):
        if not unique or abs(r - unique[-1]) > 1e-6:
            unique.append(r)

    bounds = [0.0] + unique + [1.0]
    signs = []
    for left, right in zip(bounds[:-1], bounds[1:]):
        v = delta((left + right) / 2)
        signs.append(1 if v > tol else -1 if v < -tol else 0)

    p0 = 'Stable' if signs[0] == -1 else 'Unstable' if signs[0] == 1 else 'Indeterminate'
    p1 = 'Stable' if signs[-1] == 1 else 'Unstable' if signs[-1] == -1 else 'Indeterminate'

    eq = []
    for i, root in enumerate(unique):
        s_left, s_right = signs[i], signs[i + 1]
        if (s_left, s_right) == (1, -1):
            state = 'Stable'
        elif (s_left, s_right) == (-1, 1):
            state = 'Unstable'
        elif s_left == s_right and s_left != 0:
            state = 'Semi-stable'
        else:
            state = 'Indeterminate'
        eq.append({'p': float(root), 'stability': state})

    if len(eq) > 1:
        regime = 'Multiple equilibria'
    elif len(eq) == 1 and signs == [1, -1]:
        regime = 'Stable coexistence'
    elif len(eq) == 1 and signs == [-1, 1]:
        regime = 'Bistability'
    elif len(eq) == 0 and np.all(ys[1:-1] > tol):
        regime = 'A dominates B'
    elif len(eq) == 0 and np.all(ys[1:-1] < -tol):
        regime = 'B dominates A'
    else:
        regime = 'Boundary case'

    attractors = []
    if p0 == 'Stable':
        attractors.append('p=0')
    if p1 == 'Stable':
        attractors.append('p=1')
    attractors.extend(f"p={e['p']:.6g}" for e in eq if e['stability'] == 'Stable')

    return {'regime': regime, 'p0': p0, 'p1': p1,
            'p_star': eq[0]['p'] if len(eq) == 1 else None,
            'equilibria': eq, 'n_equilibria': len(eq),
            'attractors': ', '.join(attractors) if attractors else 'None'}


def findParametersNoLineal(WAA, WAB, WBA, WBB, rangos,
                           n_points=300, maxiter=80, seed=42, **fijos):
    """Search candidates for six regimes; the result is validated afterwards."""
    nombres = list(rangos)
    limites = [rangos[k] for k in nombres]
    ps = np.linspace(0.02, 0.98, 21)
    margin = 1e-4

    def parametros(z):
        return {**fijos, **dict(zip(nombres, map(float, z)))}

    def muestras(z):
        par = parametros(z)
        return np.array([deltaPi(p, WAA, WAB, WBA, WBB, **par) for p in ps])

    objectives = ['A dominates B', 'B dominates A', 'Stable coexistence',
                  'Bistability', 'Neutrality', 'Multiple equilibria']
    rows = []
    for target in objectives:
        def costo(z, target=target):
            ds = muestras(z)
            if not np.all(np.isfinite(ds)):
                return 1e10
            if target == 'A dominates B':
                return np.sum(np.maximum(margin - ds, 0) ** 2)
            if target == 'B dominates A':
                return np.sum(np.maximum(margin + ds, 0) ** 2)
            if target == 'Stable coexistence':
                return max(0, margin - ds[0]) ** 2 + max(0, margin + ds[-1]) ** 2
            if target == 'Bistability':
                return max(0, margin + ds[0]) ** 2 + max(0, margin - ds[-1]) ** 2
            if target == 'Neutrality':
                return np.sum(ds ** 2)
            # Sign alternation at three points -> at least two roots.
            v = ds[[0, 5, 10, 15, 20]]
            return min(
                np.sum(np.maximum(margin - v[i:i + 3] * np.array(pattern), 0) ** 2)
                for i in range(3) for pattern in ([1, -1, 1], [-1, 1, -1])
            )

        sol = differential_evolution(costo, limites, seed=seed,
                                     maxiter=maxiter, polish=True)
        par = parametros(sol.x)
        res = classificationNoLineal(WAA, WAB, WBA, WBB,
                                     n_points=n_points, **par)
        rows.append({'target': target, 'found': res['regime'] == target,
                     **par, **res})
    return pd.DataFrame(rows)


def barridoParametrosNoLineal(WAA, WAB, WBA, WBB, barridos,
                              archivo='Outputs/StabilityNoLineal.csv',
                              n_points=300, **fijos):
    nombres = list(barridos)
    rows = []
    for valores in product(*(barridos[k] for k in nombres)):
        par = {**fijos, **dict(zip(nombres, map(float, valores)))}
        res = classificationNoLineal(WAA, WAB, WBA, WBB,
                                     n_points=n_points, **par)
        rows.append({**par, **res})

    table = pd.DataFrame(rows)
    if archivo is not None:
        path = Path(archivo)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, encoding='utf-8-sig')
    return table