from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import patheffects as pe
from scipy.integrate import solve_ivp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import plotly.graph_objects as go
from .stability_nonlinear import deltaPi

CARPETA = Path('Outputs')
CARPETA.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.titlesize': 12, 'axes.labelsize': 11,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 300})
REGIMENES = ['A dominates B', 'B dominates A', 'Bistability',
             'Stable coexistence', 'Neutrality']
REGIMENES_NL = REGIMENES + ['Multiple equilibria', 'Boundary case']
_ALIASES = {'A domina B': REGIMENES[0], 'B domina A': REGIMENES[1],
            'Biestabilidad': REGIMENES[2], 'Coexistencia estable': REGIMENES[3],
            'Neutralidad': REGIMENES[4], 'Multiples equilibrios': REGIMENES_NL[5],
            'Caso limite': REGIMENES_NL[6]}
_SYMBOLS = {'eta_d': r'\eta_d', 'eta_c': r'\eta_c', 'r0': r'r_0',
            'sa': 's_a', 'sb': 's_b', 'da': 'd_a', 'db': 'd_b', 'p': 'p'}
_COLORS = plt.cm.viridis([.10, .32, .55, .75, .98, .43, .90])


def guardar(fig, nombre):
    """Save as PNG in Outputs unless the name already contains a directory."""
    if nombre:
        path = Path(nombre)
        if path.parent == Path('.'):
            path = CARPETA / path
        if path.suffix.lower() not in ('.png', '.pdf', '.svg', '.jpg', '.jpeg'):
            path = Path(str(path) + '.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches='tight')


def color_p(p0):
    return plt.cm.viridis(.12 + .78*float(p0))


def estilo_ejes(ax, xlabel=None, ylabel=None, title=None, ylim=None):
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if title is not None: ax.set_title(title, pad=9)
    if ylim is not None: ax.set_ylim(*ylim)
    ax.grid(ls='--', lw=.6, alpha=.23)
    ax.set_axisbelow(True)


def _symbol(key):
    return fr'${_SYMBOLS.get(key, key)}$'


def _parameters(params):
    return ', '.join(fr'${_SYMBOLS.get(k, k)}={float(v):.3f}$'
                     for k, v in params.items() if np.isscalar(v))


def _legend(fig, handles, ncol=3):
    ncol = min(ncol, len(handles))
    if not ncol: return
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .01),
               ncol=ncol, frameon=False, fontsize=9, columnspacing=2.2,
               handletextpad=.7, borderaxespad=.4)



def _map_regimes(tabla, x, y, col, title, size, name, show,
                 regimes, xlabel=None, ylabel=None, carpeta='EqReplicador'):

    if x == y or any(k not in tabla for k in (x, y, col)):
        raise ValueError('Missing columns or identical map axes.')
    if tabla.duplicated([x, y]).any():
        raise ValueError('Each (x, y) combination must be unique.')
    M = tabla.pivot(index=y, columns=x, values=col)
    Z = np.array([[regimes.index(_ALIASES.get(v, v)) if _ALIASES.get(v, v) in regimes
                   else np.nan for v in row] for row in M.to_numpy()], dtype=float)
    cmap = ListedColormap(_COLORS[:len(regimes)]); cmap.set_bad('#F2F2F2')
    norm = BoundaryNorm(np.arange(len(regimes)+1)-.5, len(regimes))
    fig, ax = plt.subplots(figsize=size)
    ax.pcolormesh(M.columns.to_numpy(dtype=float), M.index.to_numpy(dtype=float), Z,
                  shading='nearest', cmap=cmap, norm=norm, rasterized=True)
    fixed = {k: float(tabla[k].iloc[0]) for k in _SYMBOLS if k not in (x, y)
             and k in tabla and tabla[k].nunique() == 1}
    estilo_ejes(ax, xlabel or _symbol(x), ylabel or _symbol(y),
                title + ('\n' + _parameters(fixed) if fixed else ''))
    indices = np.unique(Z[np.isfinite(Z)]).astype(int)
    handles = [Patch(facecolor=_COLORS[i], label=regimes[i]) for i in indices]
    _legend(fig, handles)
    fig.subplots_adjust(left=.12, right=.97, top=.84 if fixed else .9,
                        bottom=.17 + .055*np.ceil(max(1,len(handles))/3))
    
   
    if name:
        guardar(fig, str(CARPETA / carpeta / name))


    if show: plt.show()
    return fig, ax


def mapaRegimenes(tabla, x='r0', y='eta_d', col='regimen',
                  titulo='Evolutionary stability regimes',
                  xlabel=r'$r_0$', ylabel=r'$\eta_d$',
                  tamanio=(7.6, 5.4), nombre=None, mostrar=True,
                  carpeta='EqReplicador'):
    key = 'regime' if col not in tabla and 'regime' in tabla else col
    return _map_regimes(tabla, x, y, key, titulo, tamanio, nombre, mostrar,
                        REGIMENES, xlabel, ylabel, carpeta=carpeta)


def mapaRegimenesNoLineal(tabla, titulo='Frequency-dependent stability regimes',
                           tamanio=(7.6, 5.4), nombre=None, mostrar=True,
                           x='r0', y='eta_d', col='regimen'):
    key = 'regime' if col not in tabla and 'regime' in tabla else col
    return _map_regimes(tabla, x, y, key, titulo, tamanio, nombre, mostrar, REGIMENES_NL)


def mapaFlujoNoLineal(WAA, WAB, WBA, WBB, eta_d=np.linspace(0, 1, 150),
                       p=np.linspace(0, 1, 150),
                       titulo='Frequency-dependent evolutionary flow',
                       tamanio=(7.6, 5.4), nombre=None, mostrar=True, **kwargs):
    P, ETA = np.meshgrid(p, eta_d)
    D = np.array([[deltaPi(float(z), WAA, WAB, WBA, WBB, eta_d=float(e), **kwargs)
                   for z in p] for e in eta_d])
    if not np.isfinite(D).all():
        raise ValueError('Fitness difference contains non-finite values.')
    colors = plt.cm.viridis([.20, .75]); fig, ax = plt.subplots(figsize=tamanio)
    ax.pcolormesh(P, ETA, (D>0).astype(int), cmap=ListedColormap(colors),
                  vmin=-.5, vmax=1.5, shading='nearest', rasterized=True)
    has_zero = D.min() < 0 < D.max()
    if has_zero: ax.contour(P, ETA, D, levels=[0], colors='white', linewidths=2.1)
    style = titulo + ('\n' + _parameters(kwargs) if kwargs else '')
    estilo_ejes(ax, r'$p$', r'$\eta_d$', style)
    ax.set_xlim(min(p), max(p)); ax.set_ylim(min(eta_d), max(eta_d))
    h = [Patch(facecolor=colors[0], label=r'$\dot p < 0$ (B increases)'),
         Patch(facecolor=colors[1], label=r'$\dot p > 0$ (A increases)')]
    if has_zero: h.append(Line2D([], [], color='white', lw=2, label='Interior equilibrium'))
    _legend(fig, h)
    fig.subplots_adjust(left=.12, right=.97, top=.84 if kwargs else .90, bottom=.23)
    
    if nombre:
        guardar(fig, str(CARPETA / 'EqReplicador' / nombre))

    if mostrar: plt.show()
    return fig, ax


def _trajectories(ax, eqReplicador, pInicial, params, tFinal, nPuntos):
    t = np.linspace(0, tFinal, nPuntos)
    for p0 in pInicial:
        sol = solve_ivp(lambda t, y: [float(eqReplicador(float(y[0]), **params))],
                        (0, tFinal), [float(p0)], t_eval=t, rtol=1e-9, atol=1e-11)
        if not sol.success: raise RuntimeError(sol.message)
        ax.plot(sol.t, sol.y[0], color=color_p(p0), lw=2, label=fr'$p_0={p0:g}$')


def _equilibrium_line(ax, value, stable=True, boundary=False):
    val = float(value)
    if not 0 <= val <= 1: return
    label = (fr'$p={val:.3f}$' if boundary else fr'$p^*={val:.3f}$')
    color, ls = ('crimson', '--') if stable else ('#333333', ':')
    ax.axhline(val, color=color, ls=ls, lw=2, zorder=4)
    ax.annotate(label, (0.985, val), xycoords=('axes fraction','data'),
                xytext=(-2, 5 if val < .95 else -13), textcoords='offset points',
                ha='right', color=color, fontsize=9, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=3, foreground='white')])


def plotReplicador(eqReplicador, pInicial, parametros=None, tFinal=150,
                   nPuntos=301, usarP0EnPagos=False, equilibrios=None,
                   titulo='Replicator dynamics', tamanio=(6.8, 4.9),
                   saveName=None, carpeta='Outputs/EqReplicador', mostrar=True):
    """Plot trajectories; pass equilibria to display red attractor lines."""
    params = dict(parametros or {})
    if 'p' in params: raise ValueError('p is the dynamic variable.')
    fig, ax = plt.subplots(figsize=tamanio)
    if usarP0EnPagos:
        for p0 in pInicial: _trajectories(ax, eqReplicador, [p0], {**params,'p0':p0}, tFinal, nPuntos)
    else: _trajectories(ax, eqReplicador, pInicial, params, tFinal, nPuntos)
    if equilibrios is not None:
        for e in equilibrios:
            if isinstance(e, dict):
                value = e.get('p', e.get('p_star', e.get('pEquilibrio')))
                stable = str(e.get('stability', e.get('estabilidad','Stable'))).lower() in ('stable','estable')
            else: value, stable = e, True
            if value is not None: _equilibrium_line(ax, value, stable, float(value) in (0.,1.))
    estilo_ejes(ax, r'Time, $t$', r'Frequency of phenotype A, $p$', titulo, (0,1))
    ax.set_xlim(0, tFinal); ax.legend(loc='best', fontsize=9, frameon=False, ncol=2)
    fig.tight_layout()
    if saveName:
        path = Path(carpeta) / (str(saveName) if str(saveName).endswith('.png') else f'{saveName}.png')
        guardar(fig, str(path))
    if mostrar: plt.show()
    return fig, ax


def plotRegimenesReplicador(tabla, eqReplicador, WAA, WAB, WBA, WBB,
                            pInicial=(.05,.25,.5,.75,.95), parametros=('eta_d','r0'),
                            fijos=None, tFinal=100, nPuntos=301,
                            titulo='Replicator dynamics by stability regime',
                            nombre=None, mostrar=True):
    found_key = 'found' if 'found' in tabla else 'encontrado'
    chosen = tabla.loc[tabla[found_key].astype(str).str.lower() == 'true']
    if chosen.empty: raise ValueError('No regimes were found in the selected table.')

    rows = int(np.ceil(len(chosen)/2))
    # ↑ más alto para dar aire abajo y arriba
    fig, axes = plt.subplots(rows, 2,
                             figsize=(12.4, 4.6*rows + 1.6),
                             sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)
    any_red = any_dark = False

    for ax, (_, row) in zip(axes, chosen.iterrows()):
        params = {**(fijos or {}), **{k:float(row[k]) for k in parametros}}
        for key in (fijos or {}): params[key] = float(params[key])
        payload = dict(WAA=WAA, WAB=WAB, WBA=WBA, WBB=WBB, **params)
        _trajectories(ax, eqReplicador, pInicial, payload, tFinal, nPuntos)

        regime = _ALIASES.get(row.get('regime', row.get('regimen', '')),
                              row.get('regime', row.get('regimen', '')))
        eq = row.get('equilibria', row.get('equilibrios', []))
        if isinstance(eq, str):
            from ast import literal_eval
            eq = literal_eval(eq)
        if not isinstance(eq, (list, tuple)): eq = []

        pstar = row.get('p_star', row.get('p_estrella'))
        if pstar is not None and pd.notna(pstar) and not eq:
            eq = [{'p':float(pstar),
                   'stability':'Unstable' if regime=='Bistability' else 'Stable'}]

        if regime == 'A dominates B': eq.append({'p':1.0,'stability':'Stable'})
        elif regime == 'B dominates A': eq.append({'p':0.0,'stability':'Stable'})
        elif regime == 'Bistability':
            eq.extend([{'p':0.0,'stability':'Stable'},{'p':1.0,'stability':'Stable'}])

        for e in eq:
            value = e.get('p')
            stable = str(e.get('stability',e.get('estabilidad',''))).lower() in ('stable','estable')
            if value is not None:
                _equilibrium_line(ax, value, stable, float(value) in (0.,1.))
                any_red |= stable; any_dark |= not stable

        ax.set_title(regime + '\n' + _parameters(params), fontsize=11)
        estilo_ejes(ax, ylim=(0,1)); ax.set_xlim(0,tFinal)

    for ax in axes[len(chosen):]: ax.set_visible(False)

    fig.suptitle(titulo, y=.995, fontsize=13)
    # ↓ xlabel global más abajo para no chocar con la leyenda
    fig.supxlabel(r'Time, $step$', y=.07)
    fig.supylabel(r'Frequency of phenotype A, $p$', x=.012)

    handles = [Line2D([],[],color=color_p(q),lw=2,label=fr'$p_0={q:g}$') for q in pInicial]
    if any_red:
        handles.append(Line2D([], [], color='crimson', ls='--', lw=2,
                              label='Stable equilibrium'))
    if any_dark:
        handles.append(Line2D([], [], color='#333333', ls=':', lw=2,
                              label='Unstable equilibrium'))

    _legend(fig, handles, ncol=4)

    # ↑ más margen inferior y supxlabel por debajo de la leyenda
    fig.subplots_adjust(left=.09, right=.98, top=.89,
                        bottom=.26, hspace=.33, wspace=.14)

    if nombre:
        guardar(fig, str(CARPETA / 'EqReplicador' / nombre))

    if mostrar: plt.show()
    return fig, axes.reshape(rows,2)


def plotMatrizTransicion(M, titulo='Moran transition matrix', tamanio=(7.2,5.6),
                         nombre=None, mostrar=True):
    fig, ax = plt.subplots(figsize=tamanio)
    im = ax.imshow(np.asarray(M), origin='lower', cmap='viridis', vmin=0, vmax=1,
                   interpolation='nearest', aspect='auto')
    estilo_ejes(ax, 'Next state, $j$', 'Current state, $i$', titulo)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label='Transition probability')
    fig.tight_layout(); guardar(fig,nombre)
    if mostrar: plt.show()
    return fig, ax


def plotTiempoMarkov(resultados, titulo='Absorption dynamics (master equation)',
                      tamanio=(8,5.2), nombre=None, mostrar=True):
    steps = resultados['steps']; s = resultados['supervivencia']
    fig, ax = plt.subplots(figsize=tamanio)
    ax.plot(steps, s, lw=2.3, color=plt.cm.viridis(.45), label='Transient probability')
    crossing = resultados.get('t_convergencia_observado')
    if crossing is not None:
        ax.axvline(crossing, color='crimson', ls='--', lw=1.6,
                   label=f'Threshold reached: {crossing} steps')
    mean = resultados['tiempo_medio']
    if mean <= steps[-1]: ax.axvline(mean, color='#404040', ls=':', lw=1.7,
                                   label=f'Mean absorption: {mean:.1f} steps')
    estilo_ejes(ax, 'Markov steps (Moran events)', 'Probability not yet absorbed', titulo, (0,1))
    ax.set_xlim(0, steps[-1]); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); guardar(fig,nombre)
    if mostrar: plt.show()
    return fig, ax


def graficarEcuacionMaestra(resultados, ax=None, guardar=None, mostrar=True, **kwargs):
    P = np.asarray(resultados['distribuciones'])
    steps = np.arange(len(P)); colors = plt.cm.viridis([.30,.80])
    fig, ax = plt.subplots(figsize=(8,5.5)) if ax is None else (ax.figure,ax)
    ax.plot(steps, P[:,-1], color=colors[0],lw=2.2,label='Phenotype A fixation')
    ax.plot(steps, P[:,0], color=colors[1],lw=2.2,label='Phenotype B fixation')
    title = 'Fixation dynamics from the master equation'
    if kwargs: title += '\n' + _parameters(kwargs)
    estilo_ejes(ax, 'Markov steps (Moran events)', 'Cumulative fixation probability',title,(0,1.02))
    ax.set_xlim(0,max(1,steps[-1])); ax.legend(frameon=False)
    fig.tight_layout()
    if guardar is not None:
        ruta = Path(guardar)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta, dpi=300, bbox_inches='tight')

    if mostrar: plt.show()
    return fig, ax



def plot(tamanio, p, legendname, tittle, xlabel, ylabel, save_name,
         process_func, loc='best', zoom_x=None, zoom_y=None,
         ax=None, mostrar=True, leyenda=True, **kwargs):

    nueva = ax is None
    if nueva:
        fig, ax = plt.subplots(figsize=tamanio)
    else:
        fig = ax.figure

    for j, p0 in enumerate(np.atleast_1d(p)):
        par = dict(kwargs)
        if par.get('seed') is not None:
            par['seed'] += j

        _, history, *_ = process_func(p=float(p0), **par)

        ax.plot(
            np.arange(len(history)), history,
            color=color_p(p0), lw=1.8,
            label=fr'$p_0={p0:g}$'
        )

    estilo_ejes(ax, xlabel, ylabel, tittle, (0, 1))

    if zoom_x is not None and zoom_y is not None:
        inset = inset_axes(ax, width='38%', height='38%',
                           loc='lower right')
        for line in ax.lines:
            inset.plot(line.get_xdata(), line.get_ydata(),
                       color=line.get_color())
        inset.set(xlim=zoom_x, ylim=zoom_y)

    if leyenda:
        ax.legend(title=legendname, loc=loc, fontsize=9, frameon=False)

    if nueva:
        fig.tight_layout()

    guardar(fig, save_name)
    if mostrar:
        plt.show()

    return fig, ax





def plotFL(tamanio, legendname, tittle, xlabel, ylabel, save_name, X, Y, Z,
           zoom=False, zoom_z=None, loc='upper right'):

    fig = go.Figure(go.Surface(
        x=X, y=Y, z=np.asarray(Z, dtype=float),
        colorscale='Viridis', colorbar=dict(title=legendname)
    ))

    scene = dict(xaxis_title=xlabel, yaxis_title=ylabel, zaxis_title=legendname)
    if zoom and zoom_z is not None:
        scene['zaxis'] = {'range': list(zoom_z)}

    fig.update_layout(
        title=tittle, scene=scene,
        width=tamanio[0], height=tamanio[1]
    )

    if save_name:
        carpeta = CARPETA / 'FitnessLandscape'
        carpeta.mkdir(parents=True, exist_ok=True)

        fig.write_html(
            str(carpeta / f'{save_name}.html'),
            include_plotlyjs=True
        )
        fig.write_json(str(carpeta / f'{save_name}.json'))

    return fig






def plotHeat(tamanio, legendname, tittle, xlabel, ylabel, save_name,
             process_func, p0=0.5, N=100, T=1000,
             etas=None, eta_min=0, eta_max=1, num_eta=30,
             R=1, salto=5, ax=None, mostrar=True, **kwargs):

    etas = np.linspace(eta_min, eta_max, num_eta) if etas is None else np.asarray(etas)
    T, R, salto = int(T), int(R), int(salto)
    if T < 1 or R < 1 or salto < 1:
        raise ValueError('T, R and salto must be positive.')

    indices = np.unique(np.append(np.arange(0, T+1, salto), T))
    Z = np.zeros((len(etas), len(indices)))

    for j, eta in enumerate(etas):
        for k in range(R):
            par = dict(kwargs)
            if par.get('seed') is not None:
                par['seed'] = int(par['seed']) + j*R + k

            _, history, *_ = process_func(
                N=N, p=float(p0), T=T, eta_d=float(eta), **par
            )
            Z[j] += np.asarray(history)[indices]/R

    fig, ax = plt.subplots(figsize=tamanio) if ax is None else (ax.figure, ax)
    mesh = ax.pcolormesh(
        indices, etas, Z, cmap='viridis',
        vmin=0, vmax=1, shading='nearest', rasterized=True
    )

    estilo_ejes(ax, xlabel, ylabel, tittle)
    ax.grid(False)
    ax.set_xlim(0, T)
    fig.colorbar(mesh, ax=ax, label=legendname)
    fig.tight_layout()

    if save_name:
        guardar(fig, str(CARPETA / 'HeatPlot' / Path(save_name).name))
    if mostrar:
        plt.show()

    return fig, ax
