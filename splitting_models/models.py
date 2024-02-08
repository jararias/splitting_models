"""
ALL THE CLASSES IN THIS MODULE EXPECT PANDAS' DATAFRAMES WITH UTC TIME INDEX

El index tienen que ser times en utc o multiindex con (times_utc, site). En el segundo
caso, el BaseSplittingModel ejecuta el modelo de splitting para cada site por separado
y luego los re-ensambla para devolver el mismo multiindex. Cuando parte el data para
cada site, el index ya es un times_utc unicamente.

Con los modelos que requieren climate se busca el input argument climate a la entrada
de predict (p.ej., climate='A') y se usa como clima para ese sitio. Si no se le pasa este
argumento, se busca en data['climate'] y se coge con el primer elemento.

"""

import inspect
import warnings
from contextlib import suppress

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from loguru import logger

try:
    import sg_gisplit as gisplit
except ModuleNotFoundError:
    import gisplit

from . import models
from .base import BaseSplittingModel
from .dirint_coeffs import coeffs as DIRINT_COEFFS
from .yang5_regimes import load_yang5_regime_grid


logger.disable(__name__)


def available_models():
    def is_splitting_model(o):
        with suppress(AttributeError):
            return BaseSplittingModel in inspect.getmro(o)
        return False
    members = inspect.getmembers(models, predicate=is_splitting_model)
    return [name for name, _ in members if name != 'BaseSplittingModel']


def get(model_name, **kwargs):
    return getattr(models, model_name)(**kwargs)


class GISPLIT(BaseSplittingModel):

    '''
    If sky type is already evaluated and included in data, say in a column
    named `sky_type`, the recommended way to predict with this GISPLIT is:
    model = GISPLIT()
    pred = model.predict(data, sky_type_or_func=lambda df: df.sky_type.values)
    '''

    def __init__(self, **kwargs):
        max_sza = kwargs.pop('max_sza', 85.)
        self._gisplit_kwargs = kwargs
        super().__init__(max_sza)

    def _predict_K(self, data, **kwargs):

        # climate = kwargs.pop('climate', None)
        climate = self._gisplit_kwargs.get('climate', None)
        if (climate is None) and ('climate' in data):
            climate = data['climate'][0]

        if climate is not None:
            assert climate.upper() in 'ABCDE'

        gisplit_kwargs = self._gisplit_kwargs
        gisplit_kwargs['climate'] = climate
        gs = gisplit.GISPLIT(**gisplit_kwargs)
        pred = gs.predict(data, **kwargs)

        return pred.eval('''dif/(dir+dif)''').clip(0., 1.)


class Erbs(BaseSplittingModel):
    """
    Equation 1 In:
      Erbs, Klein and Duffie (1982) Estimation of the diffuse radiation
      fraction for hourly, daily and monthly-average global radiation,
      Solar Energy, Vol. 28(4), pp. 293-302, Equation 1
    """

    __required__ = ['sza', 'eth', 'ghi']

    def _predict_K(self, data, **kwargs):
        daytime = data['sza'] < self._max_sza
        Kt = data.ghi.divide(data.eth).where(daytime, np.nan)
        K = 0.9511 - 0.1604*Kt + 4.388*Kt**2 - 16.638*Kt**3 + 12.336*Kt**4
        K.loc[Kt <= 0.22] = 1. - 0.09*Kt.loc[Kt <= 0.22]
        K.loc[Kt > 0.8] = 0.165
        return K.clip(0., 1.)


class Hollands(BaseSplittingModel):
    """
    Equation 11 In:
      Hollands (1985) A derivation of the diffuse fraction's dependence
      on the clearness index. Solar Energy, Vol. 35, pp. 131-136.
    """
    __required__ = ['sza', 'eth', 'ghi']

    def _predict_K(self, data, **kwargs):
        daytime = data['sza'] < self._max_sza
        Kt = data.ghi.divide(data.eth).where(daytime, np.nan)
        tau_u = 0.897
        omega_l = 0.982
        a = 1. / tau_u
        b = 0.5*omega_l
        B = 1. - b
        K = (B - (B**2 - 4*a*(b**2)*Kt*(1-a*Kt)).pow(0.5)).divide(2*a*b*Kt)
        return K.clip(0., 1.)


class DISC(BaseSplittingModel):
    """
    Section 3.3 In:
      Maxwell, E. L., "A Quasi-Physical Model for Converting Hourly Global
      Horizontal to Direct Normal Insolation", Technical Report No.
      SERI/TR-215-3087, Golden, CO: Solar Energy Research Institute, 1987.

    Code adapted from pvlib:
        https://github.com/pvlib/pvlib-python/tree/master/pvlib/irradiance.py
    """
    __required__ = ['sza', 'eth', 'ghi']

    def _predict_K(self, data, **kwargs):
        daytime = data['sza'] < self._max_sza
        Kt = data.ghi.divide(data.eth).where(daytime, np.nan)
        a = 0.512 - 1.56*Kt + 2.286*Kt**2 - 2.222*Kt**3
        a = a.where(Kt <= 0.6, -5.743 + 21.77*Kt - 27.49*Kt**2 + 11.56*Kt**3)
        b = 0.37 + 0.962*Kt
        b = b.where(Kt <= 0.6, 41.4 - 118.5*Kt + 66.05*Kt**2 + 31.9*Kt**3)
        c = -0.28 + 0.932*Kt - 2.048*Kt**2
        c = c.where(Kt <= 0.6, -47.01 + 184.2*Kt - 222.0*Kt**2 + 73.81*Kt**3)

        sza = data['sza']
        cosz = np.cos(np.radians(sza))
        am = (1. / (cosz + .15 * (93.885 - sza)**(-1.253))).clip(1, 12)
        delta_Kn = a + b * np.exp(c*am)
        Knc = 0.866 - 0.122*am + 0.0121*am**2 - 0.000653*am**3 + 1.4e-05*am**4
        Kn = (Knc - delta_Kn).clip(0., 1.)
        K = 1. - Kn.divide(Kt).where(Kt > 0., np.nan)
        return K.clip(0., 1.)


class DIRINT(BaseSplittingModel):
    """
      Perez et al. (1992) Dynamic global-to-direct irradiance conversion
      models, ASHRAE Transactions, Vol. 98(1), pp. 354/369

    Code adapted from pvlib:
        https://github.com/pvlib/pvlib-python/tree/master/pvlib/irradiance.py
    """
    __required__ = ['eth', 'sza', 'ghi']

    @property
    def Kt(self):
        Kt = super().Kt
        cosz = np.cos(np.radians(self.sza))
        am = (1. / (cosz + .15 * (93.885 - self.sza)**(-1.253))).clip(1, 12)
        fc = 0.1 + 1.031*np.exp(-1.4/(0.9 + 9.4/am))
        return Kt.divide(fc).where(fc > 0., other=np.nan)

    def _predict_K(self, data, **kwargs):
        sza = data['sza']
        daytime = sza < self._max_sza
        cosz = np.cos(np.radians(sza))

        am = (1. / (cosz + .15 * (93.885 - sza)**(-1.253))).clip(1, 12)
        fc = 0.1 + 1.031*np.exp(-1.4/(0.9 + 9.4/am))
        Kt = data.ghi.divide(data.eth).where(daytime, np.nan)
        Kt = Kt.divide(fc).where(fc > 0., other=np.nan)

        Kt_next = Kt.shift(-1)
        Kt_prev = Kt.shift(+1)
        Kt_next.iloc[-1] = Kt_prev.iloc[-1]
        Kt_prev.iloc[0] = Kt_next.iloc[0]
        delta_Kt = 0.5 * (
            (Kt - Kt_next).abs().add((Kt - Kt_prev).abs(), fill_value=0)
        )

        # Create kt_prime bins
        Kt_bin = pd.Series(-1, index=data.index, dtype=np.int64)
        Kt_bounds = (0., 0.24, 0.4, 0.56, 0.7, 0.8, 1.)
        Kt_intervals = zip(Kt_bounds[:-1], Kt_bounds[1:])
        for k, (l_bound, u_bound) in enumerate(Kt_intervals):
            Kt_bin[(Kt >= l_bound) & (Kt < u_bound)] = k

        # Create solar zenith angle bins
        sza_bin = pd.Series(-1, index=data.index, dtype=np.int64)
        sza_bounds = (0., 25., 40., 55., 70., 80., 90.)
        sza_intervals = zip(sza_bounds[:-1], sza_bounds[1:])
        for k, (l_bound, u_bound) in enumerate(sza_intervals):
            sza_bin[(sza >= l_bound) & (sza < u_bound)] = k

        # Create delta_kt_prime binning.
        delta_Kt_bin = pd.Series(-1, index=data.index, dtype=np.int64)
        delta_Kt_bounds = (0., 0.015, 0.035, 0.070, 0.150, 0.300, 1.)
        delta_Kt_intervals = zip(delta_Kt_bounds[:-1], delta_Kt_bounds[1:])
        for k, (l_bound, u_bound) in enumerate(delta_Kt_intervals):
            delta_Kt_bin[(delta_Kt >= l_bound) & (delta_Kt < u_bound)] = k
        delta_Kt_bin[delta_Kt == -1] = 6

        # Create the bins for w based on dew point temperature
        w = pd.Series(-1, index=data.index, dtype=float)
        if 'w' in data:
            w = data['w']
        w_bin = pd.Series(-1, index=data.index, dtype=np.int64)
        w_bounds = (0., 1., 2., 3., 4.)
        w_intervals = zip(w_bounds[:-1], w_bounds[1:])
        for k, (l_bound, u_bound) in enumerate(w_intervals):
            w_bin[(w >= l_bound) & (w < u_bound)] = k
        w_bin[(w == -1)] = 4

        out_of_bounds = (
            (Kt_bin < 0) | (sza_bin < 0) | (delta_Kt_bin < 0) | (w_bin < 0)
        )

        disc = DISC()
        disc.predict(data)
        dirint_coeffs = np.where(
            out_of_bounds, np.nan,
            DIRINT_COEFFS[Kt_bin, sza_bin, delta_Kt_bin, w_bin])
        K = 1. - (1. - disc.K) * dirint_coeffs
        return K.clip(0., 1.)


class Perez2(BaseSplittingModel):
    """
      Perez et al. (2002) A new operational model for satellite-derived
      irradiances: description and validation. Solar Energy, Vol. 73(5)
      doi: 10.1016/S0038-092X(02)00122-6

    Code adapted from pvlib:
        https://github.com/pvlib/pvlib-python/tree/master/pvlib/irradiance.py
    """

    __required__ = ['sza', 'eth', 'ghi', 'ghics', 'difcs']

    def _predict_K(self, data, **kwargs):
        sza = data['sza']
        daytime = sza < self._max_sza

        ghics = data['ghics']
        difcs = data['difcs']
        dircs = ghics.sub(difcs).clip(0., ghics)

        clearsky = data[['eth', 'sza', 'ghics']].rename(columns={'ghics': 'ghi'})

        dirint = DIRINT()
        dirint_dircs = dirint.predict(clearsky)['dir']
        Fcs = dircs.divide(dirint_dircs).where(daytime, np.nan)

        dirint = DIRINT()
        dirint.predict(data)
        K = 1. - (1. - dirint.K)*Fcs
        return K.clip(0., 1.)


class BRL(BaseSplittingModel):
    """
    Equation 10 In:
      Ridley et al. (2010) Modelling of diffuse solar fraction with
      multiple predictors. Solar Energy, doi: 10.1016/j.renene.2009.07.018
    """
    __required__ = ['longitude', 'sza', 'eth', 'ghi']

    @staticmethod
    def apparent_solar_time(data):
        times_utc = data.index
        longitude = data['longitude']

        # eq. of time
        doy = times_utc.day_of_year.astype(float)
        angle = np.radians((360./365.242) * (doy - 1.))
        eot = (0.258 * np.cos(angle) - 7.416 * np.sin(angle)
               - 3.648 * np.cos(2*angle) - 9.228 * np.sin(2*angle))

        # local solar noon
        lsn = 12. - (longitude/15.) - (eot/60.)

        # hour angle
        dtime = times_utc.hour + times_utc.minute/60. + times_utc.second/3600.
        hourangle = (dtime - lsn)*15
        hourangle[hourangle >= 180] = hourangle[hourangle >= 180.] - 360.
        hourangle[hourangle <= -180] = 360. - hourangle[hourangle <= -180.]

        # apparent solar time
        ast = hourangle/15 + 12.
        ast[ast < 0] = abs(ast[ast < 0])
        return ast

    def _predict_K(self, data, **kwargs):
        def upscale(s, dt):
            return s.resample(dt).mean().reindex(s.index, method='ffill')

        sza = data['sza']
        daytime = sza < self._max_sza
        Kt = data.ghi.divide(data.eth).where(daytime, np.nan)

        ghi_daily = upscale(data.ghi, 'D')
        eth_daily = upscale(data.eth, 'D')
        Kt_daily = ghi_daily.divide(eth_daily).clip(0., 1.)

        ast = self.apparent_solar_time(data)
        psi = 0.5 * (Kt.shift(-1) + Kt.shift(+1))
        p = [-5.38, 6.63, 6e-3, -7e-3, 1.75, 1.31]

        argument = (
            p[0] + p[1]*Kt + p[2]*ast + p[3]*(90. - sza)
            + p[4]*Kt_daily + p[5]*psi)
        K = 1. / (1. + np.exp(argument))

        return K.clip(0., 1.)


class Engerer2(BaseSplittingModel):
    """
    Equation 33 In:
      Engerer (2015) Minute resolution estimates of the diffuse
      fraction of global irradiance for southeastern Australia
      Solar Energy, doi: 10.1016/j.solener.2015.04.012

    Parameters revisited In:
      Bright and Engerer (2019) Engerer2: Global re-parameterisation,
      update, and validation of an irradiance separation model at
      different temporal resolutions. J. Renew. Sustain. Energy.
      doi: 10.1063/1.5097014

    Code adapted from: https://github.com/JamieMBright/
        Engerer2-separation-model/blob/master/Engerer2Separation.py
    """
    __required__ = ['longitude', 'sza', 'eth', 'ghi', 'ghics']

    def __init__(self, parameters='bright_engerer_2019', max_sza=85.):
        super().__init__(max_sza)
        assert parameters in ('bright_2015', 'bright_engerer_2019')
        self._parameters = parameters

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        assert parameters in ('bright_2015', 'bright_engerer_2019')
        self._parameters = parameters

    @staticmethod
    def apparent_solar_time(data):
        times_utc = data.index
        longitude = data['longitude']

        # eq. of time
        doy = times_utc.day_of_year.astype(float)
        angle = np.radians((360./365.242) * (doy - 1.))
        eot = (0.258 * np.cos(angle) - 7.416 * np.sin(angle)
               - 3.648 * np.cos(2*angle) - 9.228 * np.sin(2*angle))

        # local solar noon
        lsn = 12. - (longitude/15.) - (eot/60.)

        # hour angle
        dtime = times_utc.hour + times_utc.minute/60. + times_utc.second/3600.
        hourangle = (dtime - lsn)*15
        hourangle[hourangle >= 180] = hourangle[hourangle >= 180.] - 360.
        hourangle[hourangle <= -180] = 360. - hourangle[hourangle <= -180.]

        # apparent solar time
        ast = hourangle/15 + 12.
        ast[ast < 0] = abs(ast[ast < 0])
        return ast

    def _predict_K(self, data, **kwargs):
        sza = data['sza']
        daytime = sza < self._max_sza
        Kt = data.ghi.divide(data.eth).where(daytime, np.nan)

        Kde = (data.ghi.sub(data.ghics).divide(data.ghi).clip(0.)  # Eq 32, Engerer, 2015
               .where(daytime, other=np.nan))
        Ktc = data.ghics.divide(data.eth).where(daytime, np.nan).clip(0.)
        dKtc = Ktc - Kt

        p = {
            'engerer_2015':  [  # for 1-min data
                4.2336e-2, -3.7912, 7.5479, -1.0036e-2, 3.1480e-3, -5.3146, 1.7073],
            'bright_engerer_2019': [  # for 1-min data
                1.0562e-1, -4.1332, 8.2578,  1.0087e-2, 8.8801e-4, -4.9302, 4.4378e-1]
        }[self.parameters]

        ast = self.apparent_solar_time(data)
        argument = p[1] + p[2]*Kt + p[3]*ast + p[4]*sza + p[5]*dKtc
        K = p[0] + (1 - p[0]) / (1 + np.exp(argument)) + p[6]*Kde

        return K.clip(0., 1.)


class Abreu(BaseSplittingModel):
    """
    Equation 3 In:
      Abreu, Canhoto and Joao Costa (2019) Prediction of diffuse
      horizontal irradiance using a new climate zone model
      Renew Sustain Energy Rev, doi: 10.1016/j.rser.2019.04.055
    """

    def _predict_K(self, data, **kwargs):
        """"
        climate: Koeppen-Geiger climate classification
            A: Tropical, B: Arid, C: Temperate, D: Continental, E: Polar
        """

        climate = kwargs.pop('climate', None)

        if (climate is None) and ('climate' in data):
            climate = data['climate'][0]

        if climate is None:
            climate = 'C'
            warnings.warn('climate input missing. Assumed climate `C`', UserWarning)

        assert climate.upper() in 'ABCDE'

        index = 'ABCDE'.index(climate.upper())
        A = [11.59, 11.39, 10.79, 10.79, 7.83][index]
        B = [-6.14, -6.25, -5.87, -5.87, -4.59][index]
        n = [1.87, 1.86, 2.24, 2.24, 3.25][index]

        sza = data['sza']
        daytime = sza < self._max_sza
        Kt = data.ghi.divide(data.eth).where(daytime, np.nan)

        Z = 1. + B*(Kt - 0.5) + A*(Kt - 0.5)**2
        K = (1. + Z**(-n))**(-1./n)
        return K.clip(0., 1.)


class PaulescuBlaga(BaseSplittingModel):
    """
    Equation 14 In:
      Paulescu and Blaga (2019) A simple and reliable empirical
      model with two predictors for estimating 1-minute diffuse
      fraction, Solar Energy, doi: 10.1016/j.solener.2019.01.029
    """

    def _predict_K(self, data, **kwargs):

        def upscale(s, dt):
            return s.resample(dt).mean().reindex(s.index, method='ffill')

        sza = data['sza']
        daytime = sza < self._max_sza
        Kt = data.ghi.divide(data.eth).where(daytime, np.nan)

        ghi_daily = upscale(data.ghi, 'D')
        eth_daily = upscale(data.eth, 'D')
        Kt_daily = ghi_daily.divide(eth_daily).clip(0., 1.)

        def Heaviside(x, x0):
            return 0.5*(1 + np.sign(x-x0))

        K = (
            1.0119 - 0.0316*Kt - 0.0294*Kt_daily
            - 1.6567*(Kt - 0.367) * Heaviside(Kt, 0.367)
            + 1.8982*(Kt - 0.734) * Heaviside(Kt, 0.734)
            - 0.8548*(Kt_daily - 0.462) * Heaviside(Kt_daily, 0.462)
        )
        return K.clip(0., 1.)


class Starke3(BaseSplittingModel):
    """
    This model is also known as BRL-minute as stated in its original
    publication (Starke et al, 2018). This class implements the last
    model version as presented in Starke et al (2021), but following
    the implementation process described by Yang (2022).

      Starke et al (2018) Resolution of the cloud enhancement problem
      for one-minute diffuse radiation prediction. Renewable Energy
      doi: 10.1016/j.renene.2018.02.107

      Starke et al (2021) Assessing one-minute diffuse fraction models
      based on worldwide climate features. Renewable Energy
      doi: 10.1016/j.renene.2021.05.108

      Yang (2022) Estimating 1-min beam and diffuse irradiance from the
      global irradiance: A review and an extensive worldwide comparison
      of latest separation models at 126 stations. Renewable and
      Sustainable Energy Reviews, doi: 10.1016/j.rser.2022.112195
    """

    __required__ = ['longitude', 'sza', 'eth', 'ghi', 'ghics']

    @staticmethod
    def apparent_solar_time(data):
        times_utc = data.index
        longitude = data['longitude']

        # eq. of time
        doy = times_utc.day_of_year.astype(float)
        angle = np.radians((360./365.242) * (doy - 1.))
        eot = (0.258 * np.cos(angle) - 7.416 * np.sin(angle)
               - 3.648 * np.cos(2*angle) - 9.228 * np.sin(2*angle))

        # local solar noon
        lsn = 12. - (longitude/15.) - (eot/60.)

        # hour angle
        dtime = times_utc.hour + times_utc.minute/60. + times_utc.second/3600.
        hourangle = (dtime - lsn)*15
        hourangle[hourangle >= 180] = hourangle[hourangle >= 180.] - 360.
        hourangle[hourangle <= -180] = 360. - hourangle[hourangle <= -180.]

        # apparent solar time
        ast = hourangle/15 + 12.
        ast[ast < 0] = abs(ast[ast < 0])
        return ast

    def _predict_K(self, data, **kwargs):
        """"
        climate: Koeppen-Geiger climate classification
            A: Tropical, B: Arid, C: Temperate, D: Continental, E: Polar
        """

        climate = kwargs.pop('climate', None)

        if (climate is None) and ('climate' in data):
            climate = data['climate'][0]

        if climate is None:
            climate = 'C'
            warnings.warn('climate input missing. Assumed climate `C`', UserWarning)

        assert climate.upper() in 'ABCDE'

        def upscale(s, dt):
            return s.resample(dt).mean().reindex(s.index, method='ffill')

        sza = data['sza']
        daytime = sza < self._max_sza
        Kt = data.ghi.divide(data.eth).where(daytime, np.nan)

        ghi_hourly = upscale(data.ghi, 'H')
        eth_hourly = upscale(data.eth, 'H')
        Kt_hourly = ghi_hourly.divide(eth_hourly).clip(0., 1.)

        ghi_daily = upscale(data.ghi, 'D')
        eth_daily = upscale(data.eth, 'D')
        Kt_daily = ghi_daily.divide(eth_daily).clip(0., 1.)

        Kcs = data.ghi.divide(data.ghics).where(daytime, np.nan).clip(0.)

        ast = self.apparent_solar_time(data)
        psi = 0.5 * (Kt.shift(-1) + Kt.shift(+1))

        index = 'ABCDE'.index(climate.upper())
        p = [
            [0.29566, -3.64571, -0.00353, -0.01721, 1.7119, 0.79448,
             0.00271, 1.38097, -7.00586, 6.35348, -0.00087, 0.00308,
             2.89595, 1.13655, -0.0013, 2.75815],  # climate A
            [-1.7463, -2.20055, 0.01182, -0.03489, 2.46116, 0.70287,
             0.00329, 2.30316, -6.53133, 6.63995, 0.01318, -0.01043,
             1.73562, 0.85521, -0.0003, 2.63141],  # climate B
            [-0.0830, -3.14711, 0.00176, -0.03354, 1.40264, 0.81353,
             0.00343, 1.95109, -7.28853, 7.15225, 0.00384, 0.02535,
             2.35926, 0.83439, -0.00327, 3.19723],  # climate C
            [0.67867, -3.79515, -0.00176, -0.03487, 1.33611, 0.76322,
             0.00353, 1.82346, -7.90856, 7.63779, 0.00145, 0.10784,
             2.00908, 1.12723, -0.00889, 3.72947],  # climate D
            [0.51643, -5.32887, -0.00196, -0.07346, 1.6064, 0.74681,
             0.00543, 3.53205, -11.70755, 10.8476, 0.00759, 0.53397,
             1.76082, 0.41495, -0.03513, 6.04835]  # climate E
        ][index]

        argument1 = (
            p[0] + p[1]*Kt + p[2]*ast + p[3]*(90. - sza)
            + p[4]*Kt_daily + p[5]*psi + p[6]*data.ghics + p[7]*Kt_hourly)

        argument2 = (
            p[8] + p[9]*Kt + p[10]*ast + p[11]*(90. - sza)
            + p[12]*Kt_daily + p[13]*psi + p[14]*data.ghics + p[15]*Kt_hourly)

        K = 1. / (1. + np.exp(argument1))
        K = K.where((Kcs >= 1.05) & (Kt > 0.75), 1. / (1. + np.exp(argument2)))

        return K.clip(0., 1.)


class Yang4(BaseSplittingModel):
    """
    Equation 9 In:
      Yang, D. (2021) Temporal-resolution cascade model for separation
      of 1-min beam and diffuse irradiance, J Renew Sustain Energy
      doi: 10.1063/5.0067997
    """
    __required__ = ['longitude', 'sza', 'eth', 'ghi', 'ghics']

    @staticmethod
    def apparent_solar_time(data):
        times_utc = data.index
        longitude = data['longitude']

        # eq. of time
        doy = times_utc.day_of_year.astype(float)
        angle = np.radians((360./365.242) * (doy - 1.))
        eot = (0.258 * np.cos(angle) - 7.416 * np.sin(angle)
               - 3.648 * np.cos(2*angle) - 9.228 * np.sin(2*angle))

        # local solar noon
        lsn = 12. - (longitude/15.) - (eot/60.)

        # hour angle
        dtime = times_utc.hour + times_utc.minute/60. + times_utc.second/3600.
        hourangle = (dtime - lsn)*15
        hourangle[hourangle >= 180] = hourangle[hourangle >= 180.] - 360.
        hourangle[hourangle <= -180] = 360. - hourangle[hourangle <= -180.]

        # apparent solar time
        ast = hourangle/15 + 12.
        ast[ast < 0] = abs(ast[ast < 0])
        return ast

    def _predict_K(self, data, **kwargs):

        def upscale(s, dt):
            return s.rolling(dt, min_periods=1, center=True).mean()
            # # with resample there appear artifacts at sunrise and sunset
            # return s.resample(dt).mean().reindex(s.index, method='ffill')

        daytime = data['sza'] < self._max_sza
        variates = pd.DataFrame(index=data.index)
        variates['sza'] = data['sza']
        variates['Kt'] = data.eval('''ghi/eth''').where(daytime, float('nan'))
        variates['Kde'] = data.eval('''(ghi-ghics)/ghi''').clip(0.).where(daytime, float('nan'))  # Eq 32, Engerer, 2015
        variates['dKtc'] = data.eval('''ghics/eth''').clip(0.).where(daytime, float('nan')) - variates['Kt']
        variates['ast'] = self.apparent_solar_time(data)

        eng = Engerer2()
        eng.predict(upscale(data[eng.__required__], 'h'))
        variates['Kds'] = eng.K

        p = [.0361, -.5744, 4.3184, -.0011, .0004, -4.7952, 1.4414, -2.8396]
        A = (p[1] + p[2]*variates['Kt'] + p[3]*variates['ast'] +
             p[4]*variates['sza'] + p[5]*variates['dKtc'] + p[7]*variates['Kds'])
        K = p[0] + (1 - p[0]) / (1 + np.exp(A)) + p[6]*variates['Kde']

        return K.clip(0., 1.)


class Yang5(Yang4):
    """
    Yang et al. (2023) Regime-dependent 1-min irradiance separation model with
    climatology clustering. Renewable and Sustainable Energy Reviews, Vol. 189,
    113992, doi: 10.1016/j.rser.113992
    """

    __required__ = ['longitude', 'eth', 'sza', 'ghi', 'ghics']

    @staticmethod
    def get_regime(lon, lat):
        regime_grid = load_yang5_regime_grid()
        lons = regime_grid.index.get_level_values('lon')
        lats = regime_grid.index.get_level_values('lat')
        return int(griddata((lons, lats), regime_grid, (lon, lat), method='nearest'))

    @staticmethod
    def parameters(regime):
        p = {
            1: [0.13105, -4.26740, 7.68051, 0.00540, 0.01748, 0.91590, 0.52176, -1.68819],
            2: [-0.01614, -3.33038, 5.72307, 0.01296, 0.01230, -0.96483, 0.94204, -1.68332],
            3: [-0.27475, 0.36085, 0.39869, 0.00479, 0.00039, -10.20264, 2.12475, -1.78455],
            4: [-0.01095, -0.92129, 3.65015, 0.00767, 0.00494, -3.76465, 1.36482, -2.11867],
            5: [0.04297, -1.64437, 4.71808, 0.01462, 0.00745, -3.35223, 1.25192, -2.36477]
        }.get(int(regime), None)
        if p is None:
            raise ValueError(f'unknown Yang5\'s regime {int(regime)}')
        return p

    def _predict_K(self, data, **kwargs):

        def upscale(s, dt):
            return s.resample(dt).mean().reindex(s.index, method='ffill')

        daytime = data['sza'] < self._max_sza
        variates = pd.DataFrame(index=data.index)
        variates['sza'] = data['sza']
        variates['Kt'] = data.eval('''ghi/eth''').where(daytime, float('nan'))
        variates['Kde'] = data.eval('''(ghi-ghics)/ghi''').clip(0.).where(daytime, float('nan'))  # Eq 32, Engerer, 2015
        variates['dKtc'] = data.eval('''ghics/eth''').clip(0.).where(daytime, float('nan')) - variates['Kt']
        variates['ast'] = self.apparent_solar_time(data)

        eng = Engerer2()
        eng.predict(upscale(data[eng.__required__], 'h'))
        variates['Kds'] = eng.K

        def calculate_K(df, p=None):
            if p is None:
                lon = df.lon.unique().item()
                lat = df.lat.unique().item()
                regime = Yang5.get_regime(lon, lat)
                p = Yang5.parameters(regime)

            A = (p[1] + p[2]*df['Kt'] + p[3]*df['ast'] +
                    p[4]*df['sza'] + p[5]*df['dKtc'] + p[7]*df['Kds'])

            return (p[0] + (1 - p[0]) / (1 + np.exp(A)) + p[6]*df['Kde']).clip(0., 1.)

        if (regime := kwargs.pop('regime', None)) is None:
            required = list(Yang5.__required__) + ['latitude']
            if missing := set(required).difference(data.columns):
                raise ValueError(f'missing required columns {list(missing)}')
            variates['lon'] = data['longitude']
            variates['lat'] = data['latitude']
            K = variates.groupby([variates.lon, variates.lat]).apply(calculate_K).T
            K.columns = pd.Index(['K'])
        else:
            K = calculate_K(variates, Yang5.parameters(regime))

        return K
