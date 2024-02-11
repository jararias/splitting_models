
import numpy as np
import pylab as pl
import pandas as pd
import matplotlib as mpl

import splitting_models.tests as sm_tests
import splitting_models.models as sm


def mbe_by_doy(df):
    
    def mbe(res, norm=1):
        return res.mean() / norm
    
    return (
        df.transform(lambda x, obs: x - obs, obs=df.dni)
        .divide(df.dni.mean()).mul(100)
        .reset_index().drop(columns=['site', 'dni']).set_index('times_utc')
        .pipe(lambda x: x.groupby(x.index.day_of_year).apply(mbe))
        .reindex(np.arange(1, 366))
        .rename_axis('doy')
    )


def rmse_by_doy(df):
    
    def rmse(res, norm=1):
        return (res.pow(2).mean()**0.5) / norm
    
    return (
        df.transform(lambda x, obs: x - obs, obs=df.dni)
        .divide(df.dni.mean()).mul(100)
        .reset_index().drop(columns=['site', 'dni']).set_index('times_utc')
        .pipe(lambda x: x.groupby(x.index.day_of_year).apply(rmse))
        .reindex(np.arange(1, 366))
        .rename_axis('doy')
    )


model_names = sm.available_models()
print(model_names)

data = sm_tests.load_valid_data()
deg2rad = np.radians(1)
data = data.eval('dni = (ghi-dif)/cos(sza*@deg2rad)')

for model_name in model_names:
    print(model_name)
    kwargs = {}
    predict_kwargs = {}
    if model_name == 'GISPLIT':
        kwargs.update({'engine': 'xgb'})
        predict_kwargs.update({'sky_type_or_func': lambda df: df.sky_type.values})
    data[model_name] = sm.get(model_name, **kwargs).predict(
        data.drop(columns='climate') if model_name == 'GISPLIT' else data,
        **predict_kwargs
    ).dni
data = data.get(['dni'] + model_names)

sites = data.index.get_level_values('site')

mbe = data.groupby(sites).apply(mbe_by_doy).unstack()
rmse = data.groupby(sites).apply(rmse_by_doy).unstack()

# mbe.to_parquet('mbe.parquet')
# rmse.to_parquet('rmse.parquet')

# mbe = pd.read_parquet('mbe.parquet')
# rmse = pd.read_parquet('rmse.parquet')

mbe = mbe.stack(level=0).swaplevel(0, 1, axis=0).sort_index()
rmse = rmse.stack(level=0).swaplevel(0, 1, axis=0).sort_index()

sites = ['mnm/bsrn', 'asp/bsrn', 'car/bsrn', 'bon/bsrn', 'son/bsrn']
models = ['Erbs', 'Hollands', 'DIRINT', 'Perez2', 'BRL', 'Engerer2',
          'PaulescuBlaga', 'PaulescuPaulescu1', 'PaulescuPaulescu2', 'Abreu',
          'Starke3', 'Yang4', 'Yang5', 'GISPLIT']

def get_model_label(model_name):
    return {
        'Perez2': 'P2',
        'BRL': 'BRL',
        'Engerer': 'E2',
        'PaulescuBlaga': 'PB',
        'PaulescuPaulescu1': 'M1',
        'PaulescuPaulescu2': 'M2',
        'Abreu': 'AB',
        'Starke3': 'S3',
        'Yang4': 'Y4',
        'Yang5': 'Y5',
        'GISPLIT': 'G3'
    }.get(model_name, model_name[:2])

mpl.rcParams['figure.figsize'] = (20, 8.8)
mpl.rcParams['figure.constrained_layout.use'] = True
mpl.rcParams['figure.constrained_layout.hspace'] = 0.08

fig, axes = pl.subplots(2, len(sites), dpi=150, subplot_kw={'projection': 'polar'})

for k_site, site in enumerate(sites):

    for k_metric, metric in enumerate([mbe, rmse]):
        df = metric.xs(site, axis=0, level=1).loc[models]
        theta = np.linspace(0., 2*np.pi, len(df)+1)
        doy = np.linspace(1, 365, len(df.columns)+1)

        kwargs = {
            'cmap': ['RdBu_r', 'hot_r'][k_metric],
            'vmin': df.min().min(),
            'vmax': min(100, df.max().max())
        }

        if k_metric == 0:
            kwargs['vmin'] = -kwargs['vmax']

        ax = axes[k_metric, k_site]
        pc = ax.pcolormesh(theta, doy, df.T, **kwargs)
        cb = pl.colorbar(pc, orientation='horizontal', shrink=0.7)
        cb.set_label(['DNI rMBE (%)', 'DNI rRMSE (%)'][k_metric])
        ax.set_rmax(365)
        ax.set_rticks([])
        ax.set_xticks(theta[:-1]+0.5*np.diff(theta))
        ax.set_xticklabels([get_model_label(m) for m in df.index])
        ax.set_xticks(theta, minor=True)
        ax.grid(ls='', axis='x', which='major')
        ax.grid(ls='-', axis='x', which='minor')
        for name, spine in ax.spines.items():
            spine.set_visible(False)
        if k_metric == 0:
            site_label = {
                'mnm/bsrn': 'Minamitorishima (mnm), JP\nEquatorial climate',
                'asp/bsrn': 'Alice Springs (asp), AU\nDry climate',
                'car/bsrn': 'Carpentras (car), FR\nTemperate climate',
                'bon/bsrn': 'Bondville (bon), US\nContinental climate',
                'son/bsrn': 'Sonnblick (son), AT\nPolar/snow climate'
            }.get(site, site)
            ax.set_title(site_label, y=1.17)

pl.savefig('model_benchmark.png', dpi=150)
