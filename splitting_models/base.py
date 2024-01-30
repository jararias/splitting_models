"""
ALL THE CLASSES IN THIS MODULE EXPECT PANDAS' DATAFRAMES WITH UTC TIME INDEX
"""

import abc
import warnings

import numpy as np
import pylab as pl
import pandas as pd
from matplotlib.colors import LogNorm

from loguru import logger

from gisplit.tools.filters import (
    add_site_to_index,
    drop_site_from_index
)


logger.disable(__name__)


class BaseSplittingModel(metaclass=abc.ABCMeta):

    __required__ = ['sza', 'eth', 'ghi']

    def __init__(self, max_sza=85.):
        self._data = None
        self._max_sza = float(max_sza)
        self._daytime = None
        self._K = None

    @property
    def sza(self):
        if self._data is None:
            warnings.warn('data not set', UserWarning)
            return None
        return self._data.sza

    @property
    def eth(self):
        if self._data is None:
            warnings.warn('data not set', UserWarning)
            return None
        return self._data.eth

    @property
    def ghi(self):
        if self._data is None:
            warnings.warn('data not set', UserWarning)
            return None
        return self._data.ghi

    @property
    def Kt(self):
        if self._data is None:
            warnings.warn('data not set', UserWarning)
            return None
        return self.ghi.divide(self.eth).where(self._daytime, other=np.nan)

    @property
    def K(self):
        if self._K is None:
            warnings.warn('prediction not performed yet', UserWarning)
            return None
        return self._K

    @abc.abstractmethod
    def _predict_K(self, data, **kwargs):
        """
        The implementation of this method cannot make any reference to
        the methods or attributes of this class (e.g., it cannot refers
        to self.Kt or self.ghi). The only exception is self._max_sza.
        All calculation must be performed based only on the input `data`
        """

    def predict(self, data, **kwargs):
        """
        `data` can be a multisite DataFrame. However, if so, its index must
        be multiindex, and the levels must have names 'times_utc' and 'site'
        """

        self._data = data
        if missing := set(self.__required__).difference(self._data.columns):
            raise ValueError(f'missing required columns {list(missing)}')
        self._daytime = self._data.sza <= self._max_sza

        if isinstance(self._data.index, pd.MultiIndex):
            site_values = self._data.index.get_level_values('site')
            self._K = pd.concat(
                [self._predict_K(drop_site_from_index(subset), **kwargs).pipe(add_site_to_index, site)
                 for site, subset in self._data.groupby(site_values)],
                axis=0).clip(0., 1.)

        else:
            self._K = self._predict_K(data, **kwargs)

        df = self.ghi.to_frame('ghi')
        df['sza'] = np.radians(self.sza)
        df['K'] = self._K
        df['dif'] = df.eval('''K*ghi''').clip(0., df.ghi)
        df['dir'] = df.eval('''ghi-dif''').clip(0., df.ghi)
        df['dni'] = df.eval('''dir/cos(sza)''').clip(0.).where(df['dir'].notna(), float('nan'))
        return df.get(['dif', 'dir', 'dni'])

    def diagnostics(self, data, hexbin_kwargs=None, axes=None, return_error_metrics=False,
                    return_predictions=False, model_label=None, **kwargs):
        """
        kwargs are keyword arguments for predict
        """

        def calc_error_metrics(pred_series, obs_series, as_text=False):
            residue = pred_series - obs_series
            notna = residue.notna()
            residue = residue.loc[notna]
            mobs = obs_series.loc[notna].mean()
            mbe, rmse = residue.mean(), residue.pow(2).mean()**0.5
            x = obs_series.loc[notna].to_numpy().astype('f4')
            y = pred_series.loc[notna].to_numpy().astype('f4')
            r2 = np.corrcoef(x, y)[0, 1]**2
            if as_text is True:
                return (f'MBe={mbe:+.1f}_W/m$^2$ ({mbe/mobs:+.1%})\n'
                        f'RMSe={rmse:.1f}_W/m$^2$ ({rmse/mobs:.1%})\nR2={r2:.3f}')
            return pd.Series({'mbe': mbe, 'rmbe': mbe/mobs*100, 'rmse': rmse,
                              'rrmse': rmse/mobs*100, 'r2': r2}).to_frame(self.__class__.__name__)

        def one_one_line(ax):
            vmin = min(ax.get_xlim()[0], ax.get_ylim()[0])
            vmax = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([vmin, vmax], [vmin, vmax], 'k-')
            return vmin, vmax

        # By calling self.predict, the `data` input is validated, stored in self._data
        # and the predicted dif, dir and dni, are returned in `pred`. After calling it,
        # I can use: self.ghi, self.eth, self.sza, self.Kt, self.K and self._daytime
        pred = self.predict(data, **kwargs)

        df = pd.DataFrame().assign(
            dif_pred=pred.dif,
            dir_pred=pred.dir,
            dni_pred=pred.dni,
            ghi_obs=self.ghi,
            cosz=np.cos(np.radians(self.sza))
        )

        daytime = self._daytime
        if 'dif' in self._data:
            df['dif_obs'] = self._data['dif']
            df['dir_obs'] = df.eval('''ghi_obs-dif_obs''').clip(0., df['ghi_obs'])
            df['dni_obs'] = df.eval('''dir_obs/cosz''').where(daytime & df['dir_obs'].notna(), float('nan'))
        elif 'dir' in self._data:
            df['dir_obs'] = self._data['dir']
            df['dif_obs'] = df.eval('''ghi_obs-dir_obs''').clip(0., df['ghi_obs'])
            df['dni_obs'] = df.eval('''dir_obs/cosz''').where(daytime & df['dir_obs'].notna(), float('nan'))
        elif 'dni' in self._data:
            df['dni_obs'] = self._data['dni']
            df['dir_obs'] = df.eval('''dni_obs/cosz''').where(daytime & df['dni_obs'].notna(), float('nan'))
            df['dif_obs'] = df.eval('''ghi_obs-dir_obs''').clip(0., df['ghi_obs'])
        else:
            raise ValueError('missing required `dif`, `dir` or `dni` in data')

        df['Kt'] = self.Kt
        df['K_obs'] = df.eval('''dif_obs/ghi_obs''').where(daytime, float('nan')).clip(0., 1.)
        df['K_pred'] = self.K

        df_nonan = df.dropna()

        pl.rcParams['axes.labelsize'] = 12
        pl.rcParams['axes.titlepad'] = 18

        if axes is None:
            fig, axes = pl.subplots(1, 4, figsize=(18, 4.5))
            fig.tight_layout(rect=[0.02, 0.05, 1, 0.94], w_pad=5)
        else:
            fig = axes[0].figure

        label = model_label or self.__class__.__name__
        title = f'Diagnostic plots for the {label} splitting model'
        fig.canvas.manager.set_window_title(title)

        hb_kwargs = {
            'gridsize': 200, 'mincnt': 1, 'cmap': 'jet', 'norm': LogNorm()
        } | (hexbin_kwargs or {})

        ax = axes[0]
        ax.hexbin('Kt', 'K_obs', data=df_nonan, **(hb_kwargs | {'cmap': 'copper_r'}))
        ax.hexbin('Kt', 'K_pred', data=df_nonan, **hb_kwargs)
        ax.text(0.01, 1.01, 'Background: Observations. Overlay: Predictions',
                transform=ax.transAxes, ha='left', va='bottom', fontsize=10)
        ax.grid()
        ax.set(
            xlabel='Observed Clearness Index, K$_t$',
            ylabel='Diffuse Fraction, K',
            xlim=(0, None),
            ylim=(0, None),
            title=title)

        for n_var, variable in enumerate(('dni', 'dif')):
            ax = axes[n_var+1]
            v_obs = f'{variable}_obs'
            v_pred = f'{variable}_pred'
            ax.hexbin(v_obs, v_pred, data=df_nonan, **hb_kwargs)
            vmin, vmax = one_one_line(ax)
            text = calc_error_metrics(df_nonan[v_pred], df_nonan[v_obs], as_text=True)
            ax.text(0.99, 0.005, text, fontsize=9, ha='right', va='bottom',
                    transform=ax.transAxes, bbox=dict(ec='w', fc='w', pad=1))
            ax.grid()
            ax.set(
                xlabel='Observed ' + variable.upper() + ' (W/m$^2$)',
                ylabel='Predicted ' + variable.upper() + ' (W/m$^2$)',
                xlim=(vmin, vmax),
                ylim=(vmin, vmax))

        ax = axes[3]
        q = np.linspace(0, 1, 150)
        sc_kwargs = {'marker': 'o', 's': 8}
        for variable in ('dni', 'dif'):
            x, y = variable + '_obs', variable + '_pred'
            ax.scatter(x, y, data=df_nonan[[x, y]].dropna().quantile(q),
                       label=variable, **sc_kwargs)
        vmin, vmax = one_one_line(ax)
        ax.legend()
        ax.grid()
        ax.set(
            xlabel='Observed quantiles (W/m$^2$)',
            ylabel='Predicted quantiles (W/m$^2$)',
            xlim=(vmin, vmax),
            ylim=(vmin, vmax))

        if return_error_metrics is True:
            variables = ['dni', 'dif']
            metrics = pd.concat(
                [calc_error_metrics(df_nonan[f'{variable}_pred'], df_nonan[f'{variable}_obs']).T
                 for variable in variables], axis=1, keys=variables)
            metrics.columns.names = ['variable', 'metric']

        if return_error_metrics is True and return_predictions is True:
            return fig, metrics, df
        if return_error_metrics is True:
            return fig, metrics
        if return_predictions is True:
            return fig, df
        return fig
