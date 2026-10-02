"""All learned features fit exclusively on the current fitting partition."""
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler
from .data import RAW, ENV


def physics(x, cfg):
    f = x[RAW].copy()
    # Ratios are empirical proxies; np semantics are unresolved, never called shaft power.
    f['np_ng_ratio_proxy'] = x.np / np.maximum(np.abs(x.ng), 1e-6)
    f['np_ng_difference'] = x.np - x.ng
    f['torque_ng_ratio_proxy'] = x.trq_measured / np.maximum(np.abs(x.ng), 1e-6)
    f['torque_mgt_ratio_proxy'] = x.trq_measured / np.maximum(np.abs(x.mgt), 1e-6)
    f['oat_pa_interaction'] = x.oat * x.pa
    f['torque_ng_interaction'] = x.trq_measured * x.ng
    f['ias_squared_proxy'] = x.ias ** 2
    f['hover_indicator'] = (x.ias == 0).astype(float)
    if cfg['temperature_unit'] == 'C':
        ambient = x.oat + 273.15
        gas = x.mgt + 273.15
        if (ambient <= 0).any() or (gas <= 0).any():
            raise ValueError('Nonphysical absolute temperatures')
        theta = ambient / 288.15
        f['theta_ambient'] = theta
        f['gas_ambient_temperature_ratio_proxy'] = gas / ambient
        f['gas_ambient_temperature_difference'] = gas - ambient
        # Uses ambient static temperature rather than unavailable inlet total temperature.
        f['ng_corrected_ambient_proxy'] = x.ng / np.sqrt(theta)
        f['thermal_loading_proxy'] = (gas - ambient) / np.maximum(np.abs(x.ng), 1e-6)
        if cfg['pa_meaning'] == 'pressure_altitude':
            # Explicit unit hypothesis; negative pressure altitudes are allowed.
            h = x.pa * (0.3048 if cfg['pa_unit'] == 'ft' else 1.)
            if ((h < -2000) | (h > 11000)).any():
                raise ValueError('Altitude outside implemented tropospheric approximation')
            t_isa = 288.15 - .0065 * h
            delta = (t_isa / 288.15) ** 5.25588
            rho = delta / theta
            f['isa_temperature_K'] = t_isa
            f['isa_temperature_deviation_K'] = ambient - t_isa
            f['pressure_ratio_isa_hypothesis'] = delta
            f['density_ratio_isa_hypothesis'] = rho
            f['density_altitude_m_hypothesis'] = (1 - rho ** (1 / 4.25588)) * 288.15 / .0065
            f['torque_density_normalized_proxy'] = x.trq_measured / rho
            if cfg['ias_unit'] == 'knots':
                # Low-Mach IAS~EAS approximation, not actual airspeed measurement.
                v_eas = x.ias * .514444
                v_tas = v_eas / np.sqrt(rho)
                f['tas_mps_low_mach_proxy'] = v_tas
                f['mach_low_mach_proxy'] = v_tas / np.sqrt(1.4 * 287.05 * ambient)
                f['dynamic_pressure_IAS_proxy_Pa'] = .5 * 1.225 * v_eas ** 2
    return f


class FeatureBuilder:
    def __init__(self, mode, cfg, seed=42):
        self.mode, self.cfg, self.seed = mode, cfg, seed

    def fit(self, x):
        if self.mode in ['statistical', 'combined']:
            a = x[RAW].to_numpy()
            self.median = np.median(a, axis=0)
            self.mad = np.maximum(np.median(np.abs(a - self.median), axis=0), 1e-6)
            self.q01, self.q99 = np.quantile(a, [.01, .99], axis=0)
            self.ecdf = {c: np.sort(x[c].to_numpy()) for c in RAW}
            self.scaler = RobustScaler().fit(x[ENV])
            self.cluster = MiniBatchKMeans(n_clusters=self.cfg['regimes'], n_init=3,
                                           random_state=self.seed, batch_size=2048)
            self.cluster.fit(self.scaler.transform(x[ENV]))
            g = self.cluster.predict(self.scaler.transform(x[ENV]))
            self.regime_medians = x[RAW].assign(regime=g).groupby('regime').median()
        return self

    def transform(self, x):
        f = physics(x, self.cfg) if self.mode in ['physics', 'combined'] else x[RAW].copy()
        if self.mode in ['statistical', 'combined']:
            a = x[RAW].to_numpy()
            z = (a - self.median) / (1.4826 * self.mad)
            for i, c in enumerate(RAW):
                f[f'{c}_robust_z'] = z[:, i]
                f[f'{c}_train_ecdf'] = np.searchsorted(self.ecdf[c], a[:, i], side='right') / len(self.ecdf[c])
                f[f'{c}_outside_train_98pct'] = ((a[:, i] < self.q01[i]) | (a[:, i] > self.q99[i])).astype(float)
            distances = self.cluster.transform(self.scaler.transform(x[ENV]))
            g = distances.argmin(axis=1)
            for k in range(distances.shape[1]):
                f[f'regime_distance_{k}'] = distances[:, k]
                f[f'regime_member_{k}'] = (g == k).astype(float)
            for c in ['trq_measured', 'mgt', 'ng', 'np']:
                centers = self.regime_medians[c].reindex(range(self.cfg['regimes'])).fillna(self.median[RAW.index(c)]).to_numpy()
                f[f'{c}_regime_deviation'] = x[c].to_numpy() - centers[g]
        if not np.isfinite(f.to_numpy()).all():
            raise ValueError('Nonfinite engineered features')
        return f.astype('float32')

    def fit_transform(self, x):
        return self.fit(x).transform(x)
