#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""QAC: quick, catalogue-based astrometry for FITS images and image stacks.

Author: Andrea Rossi, with the assistance of ChatGPT.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
import re
import sys
import tempfile
import warnings

import numpy as np
import requests
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS, NoConvergence
from astropy.wcs.utils import fit_wcs_from_points
from erfa import ErfaWarning
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder

VERSION = "0.5.11"
MIN_MATCHES = 8
DEFAULT_BRIGHT_MAG = 12.0
GAIA_DEFAULT_FAINT_MAG = 20.0
# The ESA Gaia synchronous TAP endpoint returns at most 2000 rows.
GAIA_ROW_LIMIT = 2000
GAIA_COLUMNS = {'G': 'phot_g_mean_mag', 'BP': 'phot_bp_mean_mag', 'RP': 'phot_rp_mean_mag'}
TWOMASS_COLUMNS = {'J': 'Jmag', 'H': 'Hmag', 'KS': 'Kmag'}
TWOMASS_DEFAULT_LIMITS = {'J': 15.8, 'H': 15.1, 'KS': 14.3}
PS1_BANDS = ('g', 'r', 'i', 'z', 'y')
PS1_API = 'https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/mean.csv'


@dataclass(frozen=True)
class InstrumentPreset:
    scale: float  # arcsec per pixel at reference binning
    field: float  # nominal full field width in arcmin
    binning: int = 1
    flip: str = ''  # default virtual orientation; explicit -f takes precedence


INSTRUMENTS = {
    'fors': InstrumentPreset(.25, 6.8, 2),
    'mods': InstrumentPreset(.123, 6.0),
    'luci': InstrumentPreset(.120, 4.0),  # seeing-limited N3.75 imaging
    'hawki': InstrumentPreset(.106, 7.5),  # full four-detector mosaic
    'bfosc': InstrumentPreset(.57, 13.0, flip='y'),  # Loiano imaging
    'lbc': InstrumentPreset(.225, 25.0),  # full LBC mosaic; each CCD is smaller
    'gmos': InstrumentPreset(.16, 5.5, 2),  # GMOS-N/S imaging, 2x2 binning
    'sifapsoft': InstrumentPreset(.067, 10.0),
    'nics': InstrumentPreset(.25, 4.3),  # TNG NICS imaging
}


class QACError(Exception):
    """A recoverable failure with a useful message for the user."""


class CatalogUnavailable(QACError):
    """A wider search cannot fix a service or catalogue access failure."""


@dataclass(frozen=True)
class Frame:
    hdu: int
    plane: int | None
    name: str

    @property
    def label(self) -> str:
        part = f"HDU {self.hdu}"
        if self.name:
            part += f" ({self.name})"
        if self.plane is not None:
            part += f", plane {self.plane + 1}"
        return part


@dataclass
class ReferenceCatalog:
    sky: SkyCoord
    mag: np.ndarray
    label: str


class CatalogProvider:
    def query(self, center: SkyCoord, radius: u.Quantity, obstime: Time | None) -> ReferenceCatalog:
        raise NotImplementedError


class GaiaDR3(CatalogProvider):
    def __init__(self, band: str = 'RP', mag_limit: float | None = None,
                 min_mag: float = DEFAULT_BRIGHT_MAG, refresh: bool = False):
        self.band = band
        self.mag_limit = mag_limit
        self.min_mag = min_mag
        self.refresh = refresh
        self._memory_cache: dict[str, Table] = {}

    def _fetch(self, ra: float, dec: float, radius: float, lower: float, upper: float) -> Table:
        mag_col = GAIA_COLUMNS[self.band]
        adql = f"""SELECT TOP {GAIA_ROW_LIMIT} ra, dec, pmra, pmdec, {mag_col}
            FROM gaiadr3.gaia_source_lite
            WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra:.10f},{dec:.10f},{radius:.10f}))
              AND {mag_col} >= {lower:.3f} AND {mag_col} < {upper:.3f}"""
        key = hashlib.sha256(adql.encode('utf-8')).hexdigest()
        if key in self._memory_cache:
            return self._memory_cache[key]
        cache_root = Path(os.environ.get('XDG_CACHE_HOME', Path.home()/'.cache')) / 'qac' / 'gaia-dr3-v1'
        cache_file = cache_root / f'{key}.ecsv'
        if not self.refresh and cache_file.is_file():
            try:
                table = Table.read(cache_file, format='ascii.ecsv')
                print(f"  Gaia DR3 cache: {len(table)} stars", flush=True)
                self._memory_cache[key] = table
                return table
            except (OSError, ValueError):
                pass  # A damaged cache entry is replaced by a fresh query.
        from astroquery.gaia import Gaia
        print(f"  Querying Gaia DR3 ({lower:g} <= {self.band} < {upper:g}, "
              f"radius {radius*60:.1f} arcmin) ...", flush=True)
        try:
            table = Gaia.launch_job(adql, verbose=False).get_results()
        except Exception as exc:
            raise CatalogUnavailable(f"Gaia DR3 query failed: {exc}") from exc
        self._memory_cache[key] = table
        try:
            cache_root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=cache_root, suffix='.ecsv', delete=False) as tmp:
                tmp_name = Path(tmp.name)
            try:
                table.write(tmp_name, format='ascii.ecsv', overwrite=True)
                os.replace(tmp_name, cache_file)
            finally:
                tmp_name.unlink(missing_ok=True)
        except OSError:
            pass  # A read-only cache does not prevent astrometry.
        return table

    def query(self, center: SkyCoord, radius: u.Quantity, obstime: Time | None,
              *, deep: bool = False) -> ReferenceCatalog:
        ra, dec, r = center.ra.deg, center.dec.deg, radius.to_value(u.deg)
        maximum = self.mag_limit if self.mag_limit is not None else GAIA_DEFAULT_FAINT_MAG
        upper = maximum if deep or self.mag_limit is not None else min(18.5, maximum)
        table = self._fetch(ra, dec, r, self.min_mag, upper)
        if self.mag_limit is None and not deep and len(table) < 20 and upper < maximum:
            upper = maximum
            table = self._fetch(ra, dec, r, self.min_mag, upper)
        capped = len(table) >= GAIA_ROW_LIMIT
        if self.mag_limit is None:
            for brighter in (18.5, 17.0, 15.5, 14.0, 12.5):
                if len(table) < GAIA_ROW_LIMIT:
                    break
                if self.min_mag < brighter < upper:
                    upper = brighter
                    table = self._fetch(ra, dec, r, self.min_mag, upper)
        if len(table) >= GAIA_ROW_LIMIT:
            raise CatalogUnavailable("Gaia query reached its 2,000-row limit; choose a narrower -m range or a smaller field")
        if len(table) == 0:
            raise QACError(f"Gaia DR3 returned no {self.band}-selected reference stars")
        self.last_upper = upper
        self.last_capped = capped
        ra = np.asarray(table['ra'], dtype=float)
        dec = np.asarray(table['dec'], dtype=float)
        mag = np.ma.filled(np.ma.asarray(table[GAIA_COLUMNS[self.band]], dtype=float), np.nan)
        pmra = np.ma.filled(np.ma.asarray(table['pmra'], dtype=float), np.nan)
        pmdec = np.ma.filled(np.ma.asarray(table['pmdec'], dtype=float), np.nan)
        valid = (np.isfinite(ra) & np.isfinite(dec) & np.isfinite(mag) &
                 (mag >= self.min_mag) & (mag < upper))
        sky = SkyCoord(ra[valid] * u.deg, dec[valid] * u.deg, frame='icrs')
        if obstime is not None:
            moving = np.isfinite(pmra[valid]) & np.isfinite(pmdec[valid])
            if moving.any():
                old = SkyCoord(ra[valid][moving] * u.deg, dec[valid][moving] * u.deg,
                               pm_ra_cosdec=pmra[valid][moving] * u.mas/u.yr,
                               pm_dec=pmdec[valid][moving] * u.mas/u.yr,
                               obstime=Time(2016.0, format='jyear'), frame='icrs')
                with warnings.catch_warnings():
                    # No parallax/distance is needed for angular propagation.
                    warnings.filterwarnings('ignore', message='.*distance overridden.*', category=ErfaWarning)
                    propagated = old.apply_space_motion(new_obstime=obstime)
                rr, dd = sky.ra.deg.copy(), sky.dec.deg.copy()
                rr[moving], dd[moving] = propagated.ra.deg, propagated.dec.deg
                sky = SkyCoord(rr*u.deg, dd*u.deg, frame='icrs')
        return ReferenceCatalog(sky, mag[valid], f'Gaia DR3 ({self.band})')


class TwoMASS(CatalogProvider):
    """2MASS point sources from VizieR II/246/out; no proper motions available."""
    def __init__(self, band: str = 'J', mag_limit: float | None = None,
                 min_mag: float = DEFAULT_BRIGHT_MAG):
        self.band = band
        self.mag_limit = mag_limit
        self.min_mag = min_mag
        self._memory_cache = {}

    def query(self, center: SkyCoord, radius: u.Quantity, obstime: Time | None) -> ReferenceCatalog:
        from astroquery.vizier import Vizier

        column = TWOMASS_COLUMNS[self.band]
        limit = self.mag_limit if self.mag_limit is not None else TWOMASS_DEFAULT_LIMITS[self.band]
        key = (center.ra.deg, center.dec.deg, radius.to_value(u.deg),
               self.band, self.min_mag, limit)
        if key in self._memory_cache:
            return self._memory_cache[key]
        band_label = 'Ks' if self.band == 'KS' else self.band
        print(f"  Querying 2MASS ({self.min_mag:g} <= {band_label} < {limit:g}, "
              f"radius {radius.to_value(u.arcmin):.1f} arcmin) ...", flush=True)
        vizier = Vizier(columns=['_RAJ2000', '_DEJ2000', column],
                        column_filters={column: f'{self.min_mag:g}..{limit:g}'}, row_limit=10000)
        try:
            tables = vizier.query_region(center, radius=radius, catalog='II/246/out')
        except Exception as exc:
            raise CatalogUnavailable(f"2MASS query failed: {exc}") from exc
        if not tables:
            raise QACError(f"2MASS returned no {self.band}-selected reference stars")
        table = tables[0]
        if len(table) >= 10000:
            raise CatalogUnavailable("2MASS query reached its 10,000-row limit; choose a narrower -m range or a smaller field")
        ra_key = '_RAJ2000' if '_RAJ2000' in table.colnames else 'RAJ2000'
        dec_key = '_DEJ2000' if '_DEJ2000' in table.colnames else 'DEJ2000'
        if not {ra_key, dec_key, column}.issubset(table.colnames):
            raise QACError(f"2MASS response lacks RA, Dec, or {column} columns")
        try:
            ra_values = np.ma.filled(np.ma.asarray(table[ra_key], dtype=float), np.nan)
            dec_values = np.ma.filled(np.ma.asarray(table[dec_key], dtype=float), np.nan)
        except ValueError:
            positions = SkyCoord(table[ra_key], table[dec_key], unit=(u.hourangle, u.deg))
            ra_values, dec_values = positions.ra.deg, positions.dec.deg
        mag_values = np.ma.filled(np.ma.asarray(table[column], dtype=float), np.nan)
        valid = (np.isfinite(ra_values) & np.isfinite(dec_values) & np.isfinite(mag_values) &
                 (mag_values >= self.min_mag) & (mag_values < limit))
        if not valid.any():
            raise QACError(f"2MASS returned no usable {self.band}-band positions and magnitudes")
        result = ReferenceCatalog(SkyCoord(ra_values[valid]*u.deg, dec_values[valid]*u.deg),
                                  mag_values[valid], f'2MASS ({band_label})')
        self._memory_cache[key] = result
        return result


class PanSTARRS1(CatalogProvider):
    """Gaia EDR3-corrected PS1 DR2 single-epoch mean astrometry from MAST."""

    PAGE_SIZE = 5000
    MAX_ROWS = 50000

    def __init__(self, band: str = 'r', mag_limit: float = 21.0,
                 min_mag: float = 15.0):
        self.band = band
        self.mag_limit = mag_limit
        self.min_mag = min_mag
        self._memory_cache: dict[tuple, list[dict[str, str]]] = {}

    @staticmethod
    def _number(row: dict[str, str], key: str) -> float:
        try:
            value = float(row[key])
        except (KeyError, ValueError, TypeError):
            return np.nan
        return value if np.isfinite(value) and value != -999 else np.nan

    @classmethod
    def _flag(cls, row: dict[str, str], key: str) -> int:
        value = cls._number(row, key)
        return int(value) if np.isfinite(value) and value >= 0 else 0

    def _fetch(self, center: SkyCoord, radius: u.Quantity) -> list[dict[str, str]]:
        mag_key = f'{self.band}MeanPSFMag'
        params = {'ra': center.ra.deg, 'dec': center.dec.deg,
                  'radius': radius.to_value(u.deg),
                  'nDetections.gte': 3,
                  f'{mag_key}.gte': self.min_mag,
                  f'{mag_key}.lt': self.mag_limit,
                  'pagesize': self.PAGE_SIZE, 'sort_by': 'objID.asc'}
        needed = {'objID', 'raMean', 'decMean', 'raMeanErr', 'decMeanErr',
                  'pmra', 'pmdec', 'pmraErr', 'pmdecErr', 'epochMean',
                  'astrometryCorrectionFlag', 'qualityFlag', 'objInfoFlag',
                  'nDetections', f'n{self.band}', mag_key,
                  f'{self.band}MeanPSFMagErr', f'{self.band}QfPerfect',
                  f'{self.band}Flags'}
        records = []
        print(f'  Querying PS1 DR2 mean ({self.min_mag:g} <= {self.band} '
              f'< {self.mag_limit:g}, radius {radius.to_value(u.arcmin):.1f} arcmin) ...',
              flush=True)
        for page in range(1, self.MAX_ROWS // self.PAGE_SIZE + 2):
            try:
                response = requests.get(PS1_API, params={**params, 'page': page},
                                        timeout=(10, 60))
                response.raise_for_status()
            except requests.RequestException as exc:
                raise CatalogUnavailable(f'PS1 DR2 query failed: {exc}') from exc
            reader = csv.DictReader(io.StringIO(response.text))
            if not needed.issubset(reader.fieldnames or ()):
                missing = ', '.join(sorted(needed - set(reader.fieldnames or ())))
                raise CatalogUnavailable(f'PS1 DR2 response lacks columns: {missing}')
            rows = list(reader)
            records.extend({key: row[key] for key in needed} for row in rows)
            if len(rows) < self.PAGE_SIZE:
                break
            if len(records) >= self.MAX_ROWS:
                raise CatalogUnavailable('PS1 DR2 query exceeds 50,000 rows; '
                                         'choose a narrower -m range or smaller field')
        return records

    def query(self, center: SkyCoord, radius: u.Quantity,
              obstime: Time | None) -> ReferenceCatalog:
        key = (center.ra.deg, center.dec.deg, radius.to_value(u.deg),
               self.band, self.min_mag, self.mag_limit)
        if key not in self._memory_cache:
            self._memory_cache[key] = self._fetch(center, radius)
        records = self._memory_cache[key]
        mag_key = f'{self.band}MeanPSFMag'
        good = []
        seen = set()
        for row in records:
            objid = row['objID']
            if objid in seen:
                continue
            seen.add(objid)
            ra, dec = self._number(row, 'raMean'), self._number(row, 'decMean')
            mag = self._number(row, mag_key)
            mag_err = self._number(row, f'{self.band}MeanPSFMagErr')
            ra_err = self._number(row, 'raMeanErr')
            dec_err = self._number(row, 'decMeanErr')
            qf = self._number(row, f'{self.band}QfPerfect')
            quality = self._flag(row, 'qualityFlag')
            info = self._flag(row, 'objInfoFlag')
            corrected = self._flag(row, 'astrometryCorrectionFlag')
            band_flags = self._flag(row, f'{self.band}Flags')
            # ObjectQualityFlags: GOOD=4, EXT=1|2. ObjectInfoFlags:
            # transient=256, solar-system=512|1024, no mean=524288,
            # stack-for-mean=1048576. Band EXT=0x01000000.
            if (not np.isfinite(ra) or not np.isfinite(dec) or
                not 0 <= ra < 360 or not -90 <= dec <= 90 or
                not np.isfinite(mag) or not self.min_mag <= mag < self.mag_limit or
                not np.isfinite(mag_err) or not 0 < mag_err <= .3 or
                not np.isfinite(ra_err) or not 0 <= ra_err <= .3 or
                not np.isfinite(dec_err) or not 0 <= dec_err <= .3 or
                not np.isfinite(qf) or qf < .85 or
                not np.isfinite(self._number(row, 'nDetections')) or
                self._number(row, 'nDetections') < 3 or
                not np.isfinite(self._number(row, f'n{self.band}')) or
                self._number(row, f'n{self.band}') < 1 or
                not corrected & 1 or not quality & 4 or quality & 3 or
                info & (256 | 512 | 1024 | 524288 | 1048576) or
                band_flags & 0x01000000):
                continue
            good.append(row)
        if not good:
            raise QACError(f'PS1 DR2 returned no usable corrected {self.band}-band mean stars')

        ra = np.array([self._number(row, 'raMean') for row in good])
        dec = np.array([self._number(row, 'decMean') for row in good])
        mag = np.array([self._number(row, mag_key) for row in good])
        sky = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
        moved = 0
        if obstime is not None:
            pmra = np.array([self._number(row, 'pmra') for row in good])
            pmdec = np.array([self._number(row, 'pmdec') for row in good])
            pmra_err = np.array([self._number(row, 'pmraErr') for row in good])
            pmdec_err = np.array([self._number(row, 'pmdecErr') for row in good])
            epoch = np.array([self._number(row, 'epochMean') for row in good])
            pm_flag = np.array([bool(self._flag(row, 'astrometryCorrectionFlag') & 4)
                                for row in good])
            bad_pm = np.array([bool(self._flag(row, 'objInfoFlag') & 4194304)
                               for row in good])
            dt = np.abs(obstime.mjd - epoch) / 365.25
            reliable = (pm_flag & ~bad_pm & np.isfinite(pmra) & np.isfinite(pmdec) &
                        np.isfinite(pmra_err) & np.isfinite(pmdec_err) &
                        np.isfinite(epoch) & (epoch >= 50000) & (epoch <= 70000) &
                        (pmra_err >= 0) & (pmra_err <= 2) &
                        (pmdec_err >= 0) & (pmdec_err <= 2) &
                        (np.hypot(pmra_err, pmdec_err)*dt <= 100))
            moved = int(reliable.sum())
            if moved:
                # PS1 pmra is mas/yr in the RA tangent direction (mu_alpha*cos(dec)),
                # as checked against Gaia at high declination. epochMean is MJD.
                old = SkyCoord(ra[reliable]*u.deg, dec[reliable]*u.deg,
                               pm_ra_cosdec=pmra[reliable]*u.mas/u.yr,
                               pm_dec=pmdec[reliable]*u.mas/u.yr,
                               obstime=Time(epoch[reliable], format='mjd'), frame='icrs')
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*distance overridden.*',
                                            category=ErfaWarning)
                    current = old.apply_space_motion(new_obstime=obstime)
                ra[reliable], dec[reliable] = current.ra.deg, current.dec.deg
                sky = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
        print(f'  PS1 DR2: {len(records)} mean rows; {len(good)} usable Gaia EDR3-corrected '
              f'point sources; {moved} positions propagated to observation date', flush=True)
        return ReferenceCatalog(sky, mag, f'PS1 DR2 ({self.band}, corrected mean)')


class LocalCatalog(CatalogProvider):
    def __init__(self, path: Path):
        self.path = path
        self._loaded: tuple[SkyCoord, np.ndarray] | None = None

    def query(self, center: SkyCoord, radius: u.Quantity, obstime: Time | None) -> ReferenceCatalog:
        if self._loaded is None:
            positions, mags = [], []
            try:
                with self.path.open(encoding='utf-8-sig') as handle:
                    for lineno, line in enumerate(handle, 1):
                        line = line.partition('#')[0].strip()
                        if not line:
                            continue
                        fields = re.split(r'[,\s]+', line)
                        if len(fields) < 2:
                            raise QACError(f"{self.path}:{lineno}: expected RA DEC [MAG]")
                        if fields[0].upper() == 'RA' and fields[1].upper() == 'DEC':
                            continue
                        try:
                            positions.append(parse_coordinates(fields[0], fields[1]))
                            mags.append(float(fields[2]) if len(fields) > 2 else np.nan)
                        except (ValueError, TypeError) as exc:
                            raise QACError(f"{self.path}:{lineno}: invalid coordinates or magnitude: {exc}") from exc
            except OSError as exc:
                raise CatalogUnavailable(f"Cannot read catalogue {self.path}: {exc}") from exc
            if not positions:
                raise QACError(f"No stars in catalogue {self.path}")
            self._loaded = SkyCoord(positions), np.asarray(mags)
        sky, mags = self._loaded
        mask = sky.separation(center) < radius
        return ReferenceCatalog(sky[mask], mags[mask], self.path.name)


def parse_coordinates(ra: str, dec: str) -> SkyCoord:
    """Two decimal numbers mean degrees; sexagesimal RA means hours."""
    if dec.startswith('QAC_NEG_DEC_'):
        dec = '-' + dec[len('QAC_NEG_DEC_'):]
    try:
        return SkyCoord(float(ra)*u.deg, float(dec)*u.deg, frame='icrs')
    except ValueError:
        return SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs')


def frames_in(hdul: fits.HDUList) -> list[Frame]:
    frames = []
    for index, hdu in enumerate(hdul):
        if not isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)):
            continue
        shape = hdu.shape
        if shape is None or len(shape) < 2:
            continue
        if len(shape) == 2:
            frames.append(Frame(index, None, hdu.name if index else ''))
        elif len(shape) == 3:
            frames.extend(Frame(index, j, hdu.name if index else '') for j in range(shape[0]))
        else:
            raise QACError(f"HDU {index} has {len(shape)} axes; only 2-D images and 3-D stacks are supported")
    return frames


def select_frames(frames: list[Frame], choice: str | None) -> list[Frame]:
    if not frames:
        raise QACError("No 2-D image frames found")
    if choice is None:
        if len(frames) > 1:
            raise QACError(f"Multiple image frames detected ({len(frames)}); use -q/--cube to solve all, or -q FRAME")
        return frames
    if choice == 'all':
        return frames
    if choice.isdecimal():
        n = int(choice)
        if not 1 <= n <= len(frames):
            raise QACError(f"Frame {n} is out of range (1..{len(frames)})")
        return [frames[n-1]]
    selected = [f for f in frames if f.name.upper() == choice.upper()]
    if len(selected) != 1:
        raise QACError(f"EXTNAME {choice!r} identifies {len(selected)} frames; choose a frame number (1..{len(frames)})")
    return selected


def image_of(hdul: fits.HDUList, frame: Frame) -> np.ndarray:
    data = hdul[frame.hdu].data
    return np.asarray(data if frame.plane is None else data[frame.plane], dtype=float)


def exposure_time(hdul: fits.HDUList, frame: Frame) -> Time | None:
    for header in (hdul[frame.hdu].header, hdul[0].header):
        for key in ('DATE-OBS', 'MJD-OBS'):
            if key in header:
                try:
                    return Time(header[key], format='mjd' if key == 'MJD-OBS' else None)
                except (ValueError, TypeError):
                    raise QACError(f"Invalid {key} in {frame.label}: {header[key]!r}")
    return None


def instrument_scale(hdul: fits.HDUList, frame: Frame, name: str) -> float:
    """Adjust a preset scale to the image binning when available."""
    preset = INSTRUMENTS[name]
    pair_keys = (('ESO DET WIN1 BINX', 'ESO DET WIN1 BINY'),
                 ('XBINNING', 'YBINNING'), ('CCDXBIN', 'CCDYBIN'),
                 ('BINX', 'BINY'))
    binning = None
    for header in (hdul[frame.hdu].header, hdul[0].header):
        for xkey, ykey in pair_keys:
            if xkey in header and ykey in header:
                binning = (header[xkey], header[ykey])
                break
        if binning is None and 'CCDSUM' in header:
            binning = str(header['CCDSUM']).replace('x', ' ').split()
        if binning is not None:
            break
    if binning is None:
        factor = 1
        print(f"  {name}: assuming standard binning ({preset.binning}x{preset.binning})", flush=True)
    else:
        try:
            bx, by = (int(value) for value in binning)
        except (ValueError, TypeError) as exc:
            raise QACError(f"{frame.label}: invalid detector binning {binning!r}") from exc
        if bx <= 0 or by <= 0 or bx != by:
            raise QACError(f"{frame.label}: instrument preset needs equal positive X/Y binning; "
                           "supply -s for asymmetric binning")
        factor = bx / preset.binning
    scale = preset.scale * factor
    print(f"  {name}: initial scale {scale:.4f} arcsec/pixel", flush=True)
    return scale


def initial_wcs(hdul: fits.HDUList, frame: Frame, shape: tuple[int, int],
                coords: SkyCoord | None, scale: float | None,
                instrument: str | None = None) -> tuple[WCS, bool]:
    header = hdul[frame.hdu].header.copy()
    # A primary HDU may hold pointing keywords for an image extension.
    for key in ('RA', 'DEC', 'OBJRA', 'OBJDEC', 'PIXSCALE', 'SECPIX'):
        if key not in header and key in hdul[0].header:
            header[key] = hdul[0].header[key]
    pointing = None
    pointing_keys = None
    for ra_key, dec_key in (('RA', 'DEC'), ('OBJRA', 'OBJDEC')):
        if ra_key in header and dec_key in header:
            try:
                pointing = parse_coordinates(str(header[ra_key]), str(header[dec_key]))
                pointing_keys = f'{ra_key}/{dec_key}'
                break
            except ValueError:
                pass
    wcs = None
    try:
        candidate = WCS(header, naxis=2, relax=True)
        if candidate.has_celestial and candidate.wcs.lngtyp == 'RA' and candidate.wcs.lattyp == 'DEC':
            wcs = candidate.celestial
            if not np.all(np.isfinite(wcs.pixel_scale_matrix)):
                wcs = None
    except (ValueError, TypeError, KeyError):
        pass
    if wcs is not None:
        if coords is not None:
            wcs.wcs.crpix = [(shape[1]+1)/2, (shape[0]+1)/2]
            wcs.wcs.crval = [coords.ra.deg, coords.dec.deg]
        elif pointing is not None:
            center = wcs.pixel_to_world((shape[1]-1)/2, (shape[0]-1)/2)
            corners = wcs.pixel_to_world([0, shape[1]-1, 0, shape[1]-1],
                                         [0, 0, shape[0]-1, shape[0]-1])
            radius = max(center.separation(corners))
            separation = center.separation(pointing)
            if separation > max(1*u.arcmin, radius*0.25):
                print(f'  WARNING: input WCS center is {separation.to_value(u.arcmin):.2f} '
                      f'arcmin from {pointing_keys}; check the pointing or use -c RA DEC '
                      'to override the input WCS', flush=True)
        if scale is not None:
            matrix = wcs.pixel_scale_matrix.copy()
            current = np.sqrt(abs(np.linalg.det(matrix))) * 3600
            if not np.isfinite(current) or current <= 0:
                raise QACError(f"Invalid pixel scale in {frame.label}")
            factor = scale / current
            if wcs.wcs.has_pc():
                wcs.wcs.pc = wcs.wcs.pc * factor
            elif wcs.wcs.has_cd():
                wcs.wcs.cd = wcs.wcs.cd * factor
            else:
                wcs.wcs.cdelt = wcs.wcs.cdelt * factor
        return wcs, False
    if coords is None:
        coords = pointing
    if coords is None:
        raise QACError(f"{frame.label}: approximate position required; give -c RA DEC")
    if scale is None:
        for key in ('PIXSCALE', 'SECPIX'):
            if key in header:
                scale = float(header[key]); break
    if scale is None and instrument is not None:
        scale = instrument_scale(hdul, frame, instrument)
    assumed_scale = scale is None
    if assumed_scale:
        scale = 0.2
        print('\n' + '!'*78)
        print('!!!! WARNING: NO PIXEL SCALE IN THE HEADER OR COMMAND LINE.           !!!!')
        print('!!!! ASSUMING 0.2 ARCSEC/PIXEL. CHECK THE FITTED SCALE CAREFULLY.      !!!!')
        print('!!!! FOR BEST RESULTS, GIVE THE REAL SCALE WITH -s.                    !!!!')
        print('!'*78 + '\n', flush=True)
    if not np.isfinite(scale) or scale <= 0:
        raise QACError(f"{frame.label}: invalid pixel scale; give -s ARCSEC_PER_PIXEL")
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [(shape[1]+1)/2, (shape[0]+1)/2]
    wcs.wcs.cd = [[-scale/3600, 0], [0, scale/3600]]
    wcs.wcs.crval = [coords.ra.deg, coords.dec.deg]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.cunit = ['deg', 'deg']
    return wcs, assumed_scale


def detect_sources(image: np.ndarray) -> np.ndarray:
    valid = np.isfinite(image)
    if min(image.shape) < 24 or valid.sum() < 100:
        raise QACError("Image too small or contains too few finite pixels")
    try:
        bkg = Background2D(image, box_size=min(64, max(16, min(image.shape)//4)),
                           filter_size=(3, 3), mask=~valid, bkg_estimator=MedianBackground())
        data = image - bkg.background
        _, _, sigma = sigma_clipped_stats(data[valid], sigma=3)
    except (ValueError, TypeError):
        _, med, sigma = sigma_clipped_stats(image[valid], sigma=3)
        data = image - med
    if not np.isfinite(sigma) or sigma <= 0:
        raise QACError("Cannot estimate positive image noise")
    finder = DAOStarFinder(fwhm=3.0, threshold=5*sigma, exclude_border=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        sources = finder(np.where(valid, data, 0.0))
    if sources is None or len(sources) < MIN_MATCHES:
        raise QACError(f"Only {0 if sources is None else len(sources)} sources detected; need at least {MIN_MATCHES}")
    peak = np.asarray(sources['peak'], dtype=float)
    flux = np.asarray(sources['flux'], dtype=float)
    # Photutils 3 renamed these columns; 2.x uses xcentroid/ycentroid.
    xname = 'x_centroid' if 'x_centroid' in sources.colnames else 'xcentroid'
    yname = 'y_centroid' if 'y_centroid' in sources.colnames else 'ycentroid'
    xy = np.column_stack((sources[xname], sources[yname])).astype(float)
    good = np.isfinite(xy).all(axis=1) & np.isfinite(peak) & np.isfinite(flux) & (flux > 0)
    good &= (xy[:,0] > 6) & (xy[:,0] < image.shape[1]-7)
    good &= (xy[:,1] > 6) & (xy[:,1] < image.shape[0]-7)
    finite = image[valid]
    good &= peak < np.percentile(finite, 99.99) * 1.1 if len(finite) > 10000 else True
    return xy[good][np.argsort(flux[good])[::-1]][:200]


def catalog_pixels(catalog: ReferenceCatalog, wcs: WCS, shape: tuple[int,int]) -> tuple[np.ndarray, SkyCoord]:
    x, y = wcs.world_to_pixel(catalog.sky)
    xy = np.column_stack((x, y))
    margin = max(shape) * 0.3
    good = np.isfinite(xy).all(axis=1) & (x > -margin) & (x < shape[1]+margin)
    good &= (y > -margin) & (y < shape[0]+margin)
    xy, sky, mag = xy[good], catalog.sky[good], catalog.mag[good]
    order = np.argsort(np.where(np.isfinite(mag), mag, np.inf))
    return xy[order], sky[order]


def magnitude_seed_groups(count: int) -> list[tuple[int, int]]:
    """Try bright stars first, then wider and successively fainter groups."""
    groups = []
    for end in (min(80, count), min(180, count)):
        if end and (0, end) not in groups:
            groups.append((0, end))
    for start in range(80, count, 180):
        end = min(start + 180, count)
        if end - start >= MIN_MATCHES and (start, end) not in groups:
            groups.append((start, end))
    return groups


def unique_matches(transformed: np.ndarray, refs: np.ndarray, radius: float) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    dist, idx = cKDTree(refs).query(transformed, distance_upper_bound=radius)
    source = np.flatnonzero(np.isfinite(dist))
    used = set(); pairs = []
    for i in source[np.argsort(dist[source])]:
        j = int(idx[i])
        if j not in used:
            used.add(j); pairs.append((int(i), j, float(dist[i])))
    if not pairs:
        return np.array([], int), np.array([], int), np.array([])
    return tuple(np.asarray(v, dtype=(float if k == 2 else int)) for k, v in
                 enumerate(zip(*pairs)))


def edges(xy: np.ndarray, count: int, neighbors: int = 5) -> list[tuple[int,int]]:
    xy = xy[:count]
    distances, indices = cKDTree(xy).query(xy, k=min(neighbors+1, len(xy)))
    result = []
    for i in range(len(xy)):
        for d, j in zip(np.atleast_1d(distances[i])[1:], np.atleast_1d(indices[i])[1:]):
            if d > 12:
                result.append((i, int(j)))
    return result


def coarse_match(sources: np.ndarray, refs: np.ndarray,
                 scale_range: tuple[float,float] = (0.75, 1.25),
                 seed_indices: np.ndarray | None = None,
                 source_limit: int = 80) -> tuple[np.ndarray,np.ndarray]:
    seed_refs = refs if seed_indices is None else refs[seed_indices]
    if len(seed_refs) < MIN_MATCHES or len(sources) < MIN_MATCHES:
        raise QACError("Fewer than eight detected or reference sources in the field")
    src = sources[:min(source_limit, len(sources))]
    ref = seed_refs[:min(180, len(seed_refs))]
    tree = cKDTree(ref)
    best_count, best_error, best_transformed = 0, np.inf, None

    def consider(transformed: np.ndarray):
        nonlocal best_count, best_error, best_transformed
        d, j = tree.query(transformed, distance_upper_bound=5.0)
        i = np.flatnonzero(np.isfinite(d))
        count = len(set(j[i]))
        if count > best_count or (count == best_count and count and np.median(d[i]) < best_error):
            best_count, best_error, best_transformed = count, np.median(d[i]), transformed.copy()

    # Translation voting quickly handles an existing nearly correct WCS.
    offsets = (ref[:100, None, :] - src[None, :min(source_limit,len(src)), :]).reshape(-1,2)
    bins = np.round(offsets/8).astype(int)
    _, inverse, counts = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
    for bucket in np.argsort(counts)[-12:]:
        shift = np.median(offsets[inverse == bucket], axis=0)
        consider(src + shift)
    if best_count < max(MIN_MATCHES, min(len(src), len(ref))//4):
        # Pair hypotheses allow a shifted and rotated/scaled initial WCS.
        src_edges = edges(src, min(35,len(src)))
        ref_edges = edges(ref, min(65,len(ref)))
        for a,b in src_edges:
            sv = src[b] - src[a]
            z = complex(*sv)
            for p,q in ref_edges:
                rv = ref[q] - ref[p]
                factor = complex(*rv)/z
                if not scale_range[0] < abs(factor) < scale_range[1]:
                    continue
                transformed = np.column_stack((
                    factor.real*(src[:,0]-src[a,0]) - factor.imag*(src[:,1]-src[a,1]) + ref[p,0],
                    factor.imag*(src[:,0]-src[a,0]) + factor.real*(src[:,1]-src[a,1]) + ref[p,1]))
                consider(transformed)
    if best_count < MIN_MATCHES or best_transformed is None:
        raise QACError("Could not find a reliable catalogue pattern; check -c, -s and the input image")
    si, ri, _ = unique_matches(best_transformed, ref, 5.0)
    # Estimate a similarity transform using the initial correspondences, then
    # rematch all sources and all projected catalogue entries.
    left = src[si]; right = ref[ri]
    a = np.column_stack((left[:,0], -left[:,1], np.ones(len(left)), np.zeros(len(left))))
    b = np.column_stack((left[:,1], left[:,0], np.zeros(len(left)), np.ones(len(left))))
    matrix = np.vstack((a,b))
    target = np.concatenate((right[:,0],right[:,1]))
    p = np.linalg.lstsq(matrix, target, rcond=None)[0]
    transformed = np.column_stack((p[0]*sources[:,0]-p[1]*sources[:,1]+p[2],
                                    p[1]*sources[:,0]+p[0]*sources[:,1]+p[3]))
    si, ri, _ = unique_matches(transformed, refs, 5.0)
    return si, ri


def fit_solution(sources: np.ndarray, refs: np.ndarray, sky: SkyCoord,
                 shape: tuple[int,int], initial: WCS,
                 scale_range: tuple[float,float] = (0.75, 1.25),
                 seed_indices: np.ndarray | None = None,
                 source_limit: int = 80) -> tuple[WCS, int, float, np.ndarray, np.ndarray]:
    si, ri = coarse_match(sources, refs, scale_range, seed_indices, source_limit)
    if len(si) < MIN_MATCHES:
        raise QACError("Too few unique catalogue matches")
    for iteration in range(4):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            fitted = fit_wcs_from_points((sources[si,0], sources[si,1]), sky[ri], projection='TAN')
        px, py = fitted.world_to_pixel(sky)
        predicted = np.column_stack((px,py))
        mask = np.isfinite(predicted).all(axis=1)
        rindices = np.flatnonzero(mask)
        snew, rnew, _ = unique_matches(sources, predicted[mask], 3.0 if iteration else 5.0)
        si, ri = snew, rindices[rnew]
        if len(si) < MIN_MATCHES:
            raise QACError("WCS fit lost too many matches")
        residual = sky[ri].separation(fitted.pixel_to_world(sources[si,0], sources[si,1])).arcsec
        median = np.median(residual)
        mad = 1.4826*np.median(np.abs(residual-median))
        keep = residual <= max(0.4, median+3*mad)
        si, ri = si[keep], ri[keep]
        if len(si) < MIN_MATCHES:
            raise QACError("Too few matches after residual clipping")
    fitted = fit_wcs_from_points((sources[si,0], sources[si,1]), sky[ri], projection='TAN')
    residual = sky[ri].separation(fitted.pixel_to_world(sources[si,0], sources[si,1])).arcsec
    rms = float(np.sqrt(np.mean(residual**2)))
    scale = np.sqrt(abs(np.linalg.det(fitted.pixel_scale_matrix)))*3600
    initial_scale = np.sqrt(abs(np.linalg.det(initial.pixel_scale_matrix)))*3600
    spread = np.ptp(sources[si], axis=0)
    if (not np.isfinite(rms) or rms > max(1.5, 3*scale) or
        not scale_range[0]*0.95 < scale/initial_scale < scale_range[1]*1.05 or
        np.linalg.norm(spread) < min(shape)*0.15):
        raise QACError(f"Unreliable solution ({len(si)} matches, RMS {rms:.2f}\", scale {scale:.3f}\"/px)")
    return fitted, len(si), rms, si, ri


def validate_linear_solution(sources: np.ndarray, sky: SkyCoord,
                             source_indices: np.ndarray, reference_indices: np.ndarray,
                             linear_wcs: WCS) -> float:
    """Check TAN predictions for spatially distributed matches omitted from a refit."""
    xy = sources[source_indices]
    spatial_order = np.argsort(xy[:, 0] + 0.7*xy[:, 1])
    held_out = spatial_order[::5]
    training = np.setdiff1d(np.arange(len(xy)), held_out)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            trial = fit_wcs_from_points((xy[training, 0], xy[training, 1]),
                                        sky[reference_indices[training]], projection='TAN')
        training_error = sky[reference_indices[training]].separation(
            trial.pixel_to_world(xy[training, 0], xy[training, 1])).arcsec
        held_out_error = sky[reference_indices[held_out]].separation(
            trial.pixel_to_world(xy[held_out, 0], xy[held_out, 1])).arcsec
        training_rms = float(np.sqrt(np.mean(training_error**2)))
        held_out_rms = float(np.sqrt(np.mean(held_out_error**2)))
    except (ValueError, TypeError, RuntimeError) as exc:
        raise QACError(f'Held-out TAN validation failed: {exc}') from exc
    scale = np.sqrt(abs(np.linalg.det(linear_wcs.pixel_scale_matrix))) * 3600
    limit = max(0.75, 3*scale, 3*training_rms)
    if not np.isfinite(held_out_rms) or held_out_rms > limit:
        raise QACError(f'Held-out TAN RMS {held_out_rms:.3f} arcsec exceeds '
                       f'{limit:.3f} arcsec; check the field or use a distortion fit')
    return held_out_rms


def fit_sip_distortion(sources: np.ndarray, sky: SkyCoord,
                       source_indices: np.ndarray, reference_indices: np.ndarray,
                       shape: tuple[int,int], degree: int,
                       linear_wcs: WCS) -> tuple[WCS, int, float]:
    """Fit and validate SIP in original detector pixels after TAN matching."""
    minimum = {2: 25, 3: 40}[degree]
    if len(source_indices) < minimum:
        raise QACError(f"SIP degree {degree} needs at least {minimum} matched stars "
                       f"(found {len(source_indices)}); rerun without -d")
    xy = sources[source_indices]
    spread = np.ptp(xy, axis=0)
    min_coverage = 0.5 if degree == 2 else 0.6
    quadrants = ((xy[:,0] >= (shape[1]-1)/2).astype(int) +
                 2*(xy[:,1] >= (shape[0]-1)/2).astype(int))
    if (spread[0] < min_coverage*shape[1] or
        spread[1] < min_coverage*shape[0] or
        len(np.unique(quadrants)) < (3 if degree == 2 else 4)):
        raise QACError(f"SIP degree {degree} needs matches spread across both image axes; "
                       "rerun without -d")

    # Reserve stars across the detector for an independent distortion check.
    spatial_order = np.argsort(xy[:,0] + 0.7*xy[:,1])
    held_out = spatial_order[::5]
    training = np.setdiff1d(np.arange(len(xy)), held_out)

    def fit_subset(indices: np.ndarray, sip_degree: int | None) -> WCS:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            return fit_wcs_from_points((xy[indices,0], xy[indices,1]),
                                       sky[reference_indices[indices]],
                                       projection='TAN', sip_degree=sip_degree)

    def residuals(wcs: WCS, indices: np.ndarray) -> np.ndarray:
        return sky[reference_indices[indices]].separation(
            wcs.pixel_to_world(xy[indices,0], xy[indices,1])).arcsec

    try:
        baseline = fit_subset(training, None)
        trial = fit_subset(training, degree)
        linear_validation = float(np.sqrt(np.mean(residuals(baseline, held_out)**2)))
        sip_validation = float(np.sqrt(np.mean(residuals(trial, held_out)**2)))
        if (not np.isfinite(sip_validation) or
            sip_validation > max(linear_validation*1.15, linear_validation+0.02)):
            raise QACError(f"SIP degree {degree} failed the held-out-star check "
                           f"({sip_validation:.3f} vs {linear_validation:.3f} arcsec); "
                           "rerun without -d")
        fitted = fit_subset(np.arange(len(xy)), degree)
        residual = residuals(fitted, np.arange(len(xy)))
        rms = float(np.sqrt(np.mean(residual**2)))
        linear_residual = residuals(linear_wcs, np.arange(len(xy)))
        linear_rms = float(np.sqrt(np.mean(linear_residual**2)))
        if not np.isfinite(rms) or rms > max(linear_rms*1.05, linear_rms+0.02):
            raise QACError(f"SIP degree {degree} has an unreliable full-field RMS; "
                           "rerun without -d")
    except QACError:
        raise
    except (ValueError, TypeError, RuntimeError) as exc:
        raise QACError(f"SIP degree {degree} fit failed: {exc}; rerun without -d") from exc
    return fitted, len(xy), rms


def flip_points(xy: np.ndarray, shape: tuple[int,int], mode: str) -> np.ndarray:
    transformed = xy.copy()
    if 'x' in mode:
        transformed[:,0] = shape[1] - 1 - transformed[:,0]
    if 'y' in mode:
        transformed[:,1] = shape[0] - 1 - transformed[:,1]
    return transformed


def unflip_wcs(fitted: WCS, shape: tuple[int,int], mode: str) -> WCS:
    """Express a WCS fitted to virtual flipped pixels in original FITS pixels."""
    if not mode:
        return fitted
    result = fitted.deepcopy()
    signs = np.array([-1 if 'x' in mode else 1, -1 if 'y' in mode else 1])
    crpix = result.wcs.crpix.copy()
    if 'x' in mode:
        crpix[0] = shape[1] + 1 - crpix[0]
    if 'y' in mode:
        crpix[1] = shape[0] + 1 - crpix[1]
    result.wcs.crpix = crpix
    result.wcs.cd = fitted.pixel_scale_matrix @ np.diag(signs)
    return result


_WCS_KEYS = re.compile(
    r'^(?:WCSAXES[A-Z]?|WCSNAME[A-Z]?|'
    r'WCSDIM|WAT\d+_\d+|'
    r'(?:CTYPE|CRVAL|CRPIX|CUNIT|CDELT|CROTA|CNAME|CRDER|CSYER)\d+[A-Z]?|'
    r'(?:CD|PC|PV|PS)\d+_\d+[A-Z]?|'
    r'(?:LONPOLE|LATPOLE|RADESYS|RADECSYS|EQUINOX)[A-Z]?|'
    r'(?:[AB]P?_(?:ORDER|DMAX|\d+_\d+))|'
    r'(?:CPERR|CQERR|DPERR|DQERR|D2IMERR)\d+|'
    r'(?:AXISCORR|D2IMFILE|D2IMEXT|D2IMERR|NPOLFILE|SIPNAME)|'
    r'(?:CPDIS|CQDIS|D2IMDIS|D2IM|DET2IM|DP|DQ)\d+'
    r'(?:\.[A-Z0-9_.]+)?)$')


def write_wcs(header: fits.Header, wcs: WCS, *, resampled: bool = False) -> None:
    remove_wcs(header)
    fitted_header = wcs.to_header(relax=True)
    # Astropy often writes PC with CDELT=1. Store the equivalent full CD
    # matrix instead so there is one unambiguous, directly readable scale.
    for key in list(fitted_header):
        if re.fullmatch(r'(?:CD|PC)\d+_\d+|CDELT\d+|CROTA\d+', key):
            del fitted_header[key]
    matrix = wcs.pixel_scale_matrix
    for i in range(2):
        for j in range(2):
            fitted_header[f'CD{i+1}_{j+1}'] = float(matrix[i, j])
    header.update(fitted_header)
    scale = float(np.sqrt(abs(np.linalg.det(matrix))) * 3600)
    for key in ('PIXSCALE', 'SECPIX'):
        if key in header:
            header[key] = (scale, 'Fitted arcsec/pixel at CRPIX')
    if any(key in header for key in ('RA', 'DEC', 'OBJRA', 'OBJDEC')):
        header.add_history('Original pointing keywords retained; fitted sky WCS is in CRVAL1/2')
    if resampled:
        header.add_history(f"QAC {VERSION}: image resampled to TAN with bilinear interpolation")
        header.add_history("Pixel-area correction applied; blank/outside pixels are NaN")
    else:
        model = f"SIP order {wcs.sip.a_order}" if wcs.sip is not None else "linear TAN"
        header.add_history(f"QAC {VERSION}: {model} astrometry; pixels unchanged")


def remove_wcs(header: fits.Header) -> None:
    for key in list(header):
        if _WCS_KEYS.fullmatch(key):
            del header[key]


def remove_physical_coordinates(header: fits.Header) -> None:
    """Old IRAF detector-pixel mappings no longer describe rectified pixels."""
    for key in list(header):
        if re.fullmatch(r'LTV\d+|LTM\d+_\d+', key):
            del header[key]


def resample_image(image: np.ndarray, solution: WCS) -> tuple[np.ndarray, WCS]:
    """Rectify SIP pixels onto the solution's undistorted, same-size TAN grid.

    Flux per input pixel is corrected by the local SIP pixel-area ratio.
    The WCS linear terms and CRPIX remain fixed, so the grid agrees with the
    original near the reference pixel and has no SIP terms in its header.
    """
    if solution.sip is None:
        raise QACError("Resampling requires a fitted SIP distortion (-d 2 or -d 3)")
    target = solution.deepcopy()
    target.sip = None
    target.wcs.ctype = [ctype.replace('-SIP', '') for ctype in target.wcs.ctype]
    height, width = image.shape
    dtype = np.float64 if image.dtype == np.float64 else np.float32
    result = np.full(image.shape, np.nan, dtype=dtype)
    finite = np.isfinite(image)
    values = np.where(finite, image, 0.0).astype(np.float64, copy=False)
    weights = finite.astype(np.float32)
    sip = solution.sip
    da_dx = np.polynomial.polynomial.polyder(sip.a, axis=0)
    da_dy = np.polynomial.polynomial.polyder(sip.a, axis=1)
    db_dx = np.polynomial.polynomial.polyder(sip.b, axis=0)
    db_dy = np.polynomial.polynomial.polyder(sip.b, axis=1)
    # Limit temporary coordinate arrays for large camera frames.
    rows_per_chunk = max(1, 1_000_000 // width)
    for row in range(0, height, rows_per_chunk):
        stop = min(height, row + rows_per_chunk)
        tx, ty = np.meshgrid(np.arange(width, dtype=float),
                             np.arange(row, stop, dtype=float))
        ra, dec = target.pixel_to_world_values(tx, ty)
        try:
            sx, sy = solution.world_to_pixel_values(ra, dec)
        except NoConvergence as exc:
            raise QACError("SIP inverse did not converge during resampling") from exc
        inside = (np.isfinite(sx) & np.isfinite(sy) &
                  (sx >= 0) & (sx <= width-1) & (sy >= 0) & (sy <= height-1))
        coords = [np.where(inside, sy, 0), np.where(inside, sx, 0)]
        sampled = map_coordinates(values, coords, order=1, mode='constant', cval=0.0)
        weight = map_coordinates(weights, coords, order=1, mode='constant', cval=0.0)
        u = sx - (sip.crpix[0] - 1)
        v = sy - (sip.crpix[1] - 1)
        with np.errstate(invalid='ignore', over='ignore'):
            jacobian = ((1 + np.polynomial.polynomial.polyval2d(u, v, da_dx)) *
                        (1 + np.polynomial.polynomial.polyval2d(u, v, db_dy)) -
                        np.polynomial.polynomial.polyval2d(u, v, da_dy) *
                        np.polynomial.polynomial.polyval2d(u, v, db_dx))
        if np.any(inside & (~np.isfinite(jacobian) | (jacobian <= 0))):
            raise QACError("SIP distortion folds or is invalid within the image")
        good = inside & (weight > 1e-6) & np.isfinite(jacobian) & (jacobian > 0)
        np.divide(sampled, weight * jacobian, out=result[row:stop], where=good)
    return result, target


def output_hdus(hdul: fits.HDUList, selected: list[Frame], solutions: dict[Frame,WCS],
                *, resample: bool = False, extract_selected: bool = False) -> fits.HDUList:
    # A specifically selected frame is a standalone 2-D FITS image, even if
    # it came from an image extension. An all-frame 3-D stack is split into
    # separate 2-D extensions with independent WCS headers.
    def pixels_and_wcs(frame: Frame) -> tuple[np.ndarray, WCS]:
        data = hdul[frame.hdu].data
        plane = data if frame.plane is None else data[frame.plane]
        if resample:
            print(f"  Resampling {frame.label} onto a TAN grid ...", flush=True)
            return resample_image(np.asarray(plane), solutions[frame])
        return plane, solutions[frame]

    def resampled_header(header: fits.Header) -> None:
        if resample:
            for key in ('BLANK', 'BSCALE', 'BZERO', 'DATAMIN', 'DATAMAX'):
                header.pop(key, None)
            remove_physical_coordinates(header)

    if len(selected) == 1 and (extract_selected or selected[0].plane is not None):
        f = selected[0]
        header = hdul[f.hdu].header.copy()
        for key in ('NAXIS3', 'XTENSION', 'PCOUNT', 'GCOUNT',
                    'EXTNAME', 'EXTVER', 'EXTLEVEL'):
            header.pop(key, None)
        if f.plane is not None:
            remove_physical_coordinates(header)
        data, wcs = pixels_and_wcs(f)
        resampled_header(header)
        write_wcs(header, wcs, resampled=resample)
        if f.plane is not None:
            header.add_history(f"Extracted plane {f.plane+1} from HDU {f.hdu}")
        else:
            header.add_history(f"Extracted image from HDU {f.hdu}")
        return fits.HDUList([fits.PrimaryHDU(data=data, header=header)])
    out = []
    for hdu_num, hdu in enumerate(hdul):
        fset = [f for f in selected if f.hdu == hdu_num]
        if fset and fset[0].plane is not None:
            if hdu_num == 0:
                out.append(fits.PrimaryHDU())
            for f in fset:
                header = hdu.header.copy()
                header.pop('NAXIS3', None)
                remove_physical_coordinates(header)
                header['EXTNAME'] = f'{hdu.name if hdu_num else "FRAME"}_{f.plane+1}'
                header['QACPLANE'] = (f.plane+1, 'Original 1-based plane number')
                data, wcs = pixels_and_wcs(f)
                resampled_header(header)
                write_wcs(header, wcs, resampled=resample)
                out.append(fits.ImageHDU(data=data, header=header))
        else:
            copied = hdu.copy()
            if fset:
                data, wcs = pixels_and_wcs(fset[0])
                if resample:
                    resampled_header(copied.header)
                    copied.data = data
                write_wcs(copied.header, wcs, resampled=resample)
            out.append(copied)
    return fits.HDUList(out)


def output_path(path: Path, *, resampled: bool = False, frame_number: int | None = None) -> Path:
    name = path.name
    frame_suffix = f'_q{frame_number}' if frame_number is not None else ''
    ending = frame_suffix + ('_resampled.fits' if resampled else '_astro.fits')
    for suffix in ('.fits.gz', '.fit.gz', '.fits', '.fit', '.fts'):
        if name.lower().endswith(suffix):
            return path.with_name(name[:-len(suffix)] + ending)
    return path.with_name(name + ending)


def match_in_magnitude_order(sources: np.ndarray, refs: np.ndarray, sky: SkyCoord,
                             shape: tuple[int,int], initial: WCS,
                             scale_range: tuple[float,float], modes: tuple[str,...],
                             distortion: int | None,
                             source_limit: int = 80) -> tuple[WCS, int, float, str]:
    if len(refs) < MIN_MATCHES:
        raise QACError(f"Only {len(refs)} catalogue stars project near the image; need at least {MIN_MATCHES}")
    failures = []
    best = None
    strong_match = max(25, min(40, len(sources)//2), {None: 0, 2: 25, 3: 40}[distortion])
    for start, stop in magnitude_seed_groups(len(refs)):
        print(f"  Matching catalogue stars ranked {start+1}-{stop} by brightness ...", flush=True)
        seeds = np.arange(start, stop)
        candidates = []
        for mode in modes:
            if len(modes) > 1:
                print(f"  Trying orientation: {mode or 'normal'}", flush=True)
            try:
                oriented_sources = flip_points(sources, shape, mode)
                fitted, count, rms, si, ri = fit_solution(
                    oriented_sources, refs, sky, shape, initial,
                    scale_range, seeds, source_limit)
                validation_rms = (validate_linear_solution(oriented_sources, sky, si, ri, fitted)
                                  if distortion is None else None)
            except QACError as exc:
                failures.append(f"{mode or 'normal'}: {exc}")
                continue
            candidates.append((count, -rms, mode, unflip_wcs(fitted, shape, mode),
                               rms, si, ri, validation_rms))
        for count, _, mode, fitted, rms, si, ri, validation_rms in sorted(candidates, reverse=True,
                                                           key=lambda row: (row[0], row[1])):
            if distortion is not None:
                try:
                    fitted, count, rms = fit_sip_distortion(
                        sources, sky, si, ri, shape, distortion, fitted)
                except QACError as exc:
                    failures.append(f"SIP {distortion}: {exc}")
                    continue
            if best is None or (count, -rms) > (best[1], -best[2]):
                best = (fitted, count, rms, mode, validation_rms)
            if count >= strong_match:
                if validation_rms is not None:
                    print(f'  Held-out TAN RMS: {validation_rms:.3f} arcsec', flush=True)
                return fitted, count, rms, mode
    if best is not None:
        if best[4] is not None:
            print(f'  Held-out TAN RMS: {best[4]:.3f} arcsec', flush=True)
        return best[:4]
    raise QACError('No reliable catalogue match: ' + '; '.join(failures[-4:]))


def solve(args: argparse.Namespace) -> Path:
    path = Path(args.image)
    if not path.is_file():
        raise QACError(f"Input file does not exist: {path}")
    coords = parse_coordinates(*args.coordinates) if args.coordinates else None
    catalog_name = args.catalog.lower()
    band = args.band.upper() if args.band else None
    mag_min = args.mag[0] if args.mag is not None and len(args.mag) == 2 else DEFAULT_BRIGHT_MAG
    mag_max = args.mag[-1] if args.mag is not None else None
    if band == 'K':
        band = 'KS'
    if catalog_name == 'gaia':
        band = band or 'RP'
        if band not in GAIA_COLUMNS:
            raise QACError("Gaia --band must be G, BP, or RP")
        provider = GaiaDR3(band=band, mag_limit=mag_max, min_mag=mag_min,
                           refresh=args.refresh_catalog)
    elif catalog_name == '2mass':
        band = band or 'J'
        if band not in TWOMASS_COLUMNS:
            raise QACError("2MASS --band must be J, H, or Ks")
        provider = TwoMASS(band=band, mag_limit=mag_max, min_mag=mag_min)
    elif catalog_name == 'ps1':
        ps1_band = (args.band or 'r').lower()
        if ps1_band not in PS1_BANDS:
            raise QACError("PS1 --band must be g, r, i, z, or y")
        ps1_min = args.mag[0] if args.mag is not None and len(args.mag) == 2 else 15.0
        provider = PanSTARRS1(band=ps1_band, min_mag=ps1_min,
                              mag_limit=mag_max if mag_max is not None else 21.0)
    else:
        if band is not None or args.mag is not None:
            raise QACError("--band and --mag require --catalog gaia, 2mass, or ps1")
        provider = LocalCatalog(Path(args.catalog))
    with fits.open(path, memmap=False) as hdul:
        frames = frames_in(hdul)
        choice = args.cube
        if choice is None and args.instrument == 'mods' and len(frames) > 1:
            merged = [f for f in frames if f.plane is None and f.name.upper() == 'MERGED']
            if len(merged) == 1:
                choice = str(frames.index(merged[0]) + 1)
                print(f"MODS multi-extension file: selecting MERGED "
                      f"(image frame {choice}, HDU {merged[0].hdu})", flush=True)
        selected = select_frames(frames, choice)
        frame_number = (frames.index(selected[0]) + 1 if choice not in (None, 'all') else None)
        target = (Path(args.output) if args.output else
                  output_path(path, resampled=args.resample, frame_number=frame_number))
        if target.resolve() == path.resolve():
            raise QACError("Output path must differ from input")
        if target.exists() and not args.overwrite:
            raise QACError(f"Output already exists: {target} (use --overwrite)")
        solutions: dict[Frame, WCS] = {}
        for frame in selected:
            print(f"Solving {frame.label} ...", flush=True)
            image = image_of(hdul, frame)
            source = detect_sources(image)
            # BFOSC is a shorthand for an explicit scale and Y flip. As with
            # -s, its scale supersedes a stale header WCS or PIXSCALE value.
            scale = args.scale
            if scale is None and args.instrument == 'bfosc':
                scale = instrument_scale(hdul, frame, 'bfosc')
            initial, assumed_scale = initial_wcs(
                hdul, frame, image.shape, coords, scale, args.instrument)
            center = initial.pixel_to_world((image.shape[1]-1)/2, (image.shape[0]-1)/2)
            corners = initial.pixel_to_world([0, image.shape[1]-1, 0, image.shape[1]-1],
                                             [0, 0, image.shape[0]-1, image.shape[0]-1])
            corner_radius = max(center.separation(corners))
            wide_radius = corner_radius * (1.8 if assumed_scale else 1.3) + 1*u.arcmin
            radius = (corner_radius * 1.1 + .3*u.arcmin
                      if args.instrument and not assumed_scale else wide_radius)
            if args.instrument:
                field = INSTRUMENTS[args.instrument].field
                width = image.shape[1] * np.linalg.norm(initial.pixel_scale_matrix[:, 0]) * 60
                height = image.shape[0] * np.linalg.norm(initial.pixel_scale_matrix[:, 1]) * 60
                if max(width, height) > field * 1.4:
                    print(f"  WARNING: image footprint {width:.1f} x {height:.1f} arcmin "
                          f"exceeds the nominal {args.instrument} field ({field:g} arcmin); "
                          "check the scale or binning", flush=True)
            obstime = exposure_time(hdul, frame)
            flip = (args.flip if args.flip is not None else
                    INSTRUMENTS[args.instrument].flip if args.instrument else '')
            modes = ('', 'x', 'y', 'xy') if flip == 'auto' else (flip,)
            distortion = args.distortion if args.distortion is not None else (2 if args.resample else None)
            def attempt(reference: ReferenceCatalog) -> tuple[WCS, int, float, str, int]:
                refxy, sky = catalog_pixels(reference, initial, image.shape)
                fitted, count, rms, mode = match_in_magnitude_order(
                    source, refxy, sky, image.shape, initial,
                    (0.5, 1.8) if assumed_scale else (0.75, 1.25), modes, distortion,
                    200 if isinstance(provider, PanSTARRS1) else 80)
                return fitted, count, rms, mode, len(refxy)

            def solve_at(query_radius: u.Quantity) -> tuple[tuple[WCS, int, float, str, int], ReferenceCatalog]:
                catalog = provider.query(center, query_radius, obstime)
                first_error = None
                try:
                    result = attempt(catalog)
                except QACError as exc:
                    first_error = exc
                    result = None
                weak_limit = min(20, max(12, len(source)//3))
                deepen = (isinstance(provider, GaiaDR3) and args.mag is None and
                          not provider.last_capped and
                          provider.last_upper < GAIA_DEFAULT_FAINT_MAG and
                          (result is None or result[1] < weak_limit))
                if deepen:
                    print("  Extending the Gaia faint limit to 20 ...", flush=True)
                    try:
                        deeper_catalog = provider.query(center, query_radius, obstime, deep=True)
                        deeper_result = attempt(deeper_catalog)
                    except CatalogUnavailable as exc:
                        if result is None:
                            raise
                        print(f"  Deeper Gaia query failed; retaining the initial match: {exc}", flush=True)
                    except QACError as exc:
                        if result is None:
                            raise QACError(f"Gaia matching failed at both depths: {first_error}; {exc}") from exc
                        print(f"  Deeper Gaia retry failed; retaining the initial match: {exc}", flush=True)
                    else:
                        if result is None or (deeper_result[1], -deeper_result[2]) > (result[1], -result[2]):
                            result, catalog = deeper_result, deeper_catalog
                if result is None:
                    raise first_error
                return result, catalog

            try:
                result, catalog = solve_at(radius)
            except CatalogUnavailable:
                raise
            except QACError as first_error:
                if radius is wide_radius:
                    raise
                print("  Retrying with a wider catalogue search ...", flush=True)
                try:
                    result, catalog = solve_at(wide_radius)
                except QACError as exc:
                    raise QACError(f"Instrument search and wider retry failed: {first_error}; {exc}") from exc
            fitted, count, rms, mode, reference_count = result
            fitted_scale = np.sqrt(abs(np.linalg.det(fitted.pixel_scale_matrix))) * 3600
            print(f"  {len(source)} detected, {reference_count} {catalog.label} references; "
                  f"{count} matches; RMS {rms:.3f} arcsec; scale {fitted_scale:.4f} arcsec/pixel; "
                  f"orientation {mode or 'normal'}; WCS "
                  f"{'SIP ' + str(distortion) if distortion is not None else 'TAN'}")
            solutions[frame] = fitted
        output_hdus(hdul, selected, solutions, resample=args.resample,
                    extract_selected=frame_number is not None).writeto(
            target, overwrite=args.overwrite, checksum=True)
    return target


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # argparse treats a negative sexagesimal Dec as another command option.
    # Replace it before parsing, then restore its sign in parse_coordinates.
    for i, arg in enumerate(argv[:-2]):
        if arg in ('-c', '--coordinates') and argv[i+2].startswith('-') and ':' in argv[i+2]:
            argv[i+2] = 'QAC_NEG_DEC_' + argv[i+2][1:]
    parser = argparse.ArgumentParser(description="Quick Gaia DR3, PS1 DR2, 2MASS, or local-catalogue astrometry for FITS images")
    parser.add_argument('image', help='FITS image or multi-frame file')
    parser.add_argument('-c', '--coordinates', nargs=2, metavar=('RA','DEC'), help='approximate field centre (degrees or RA hours:minutes:seconds, Dec degrees:minutes:seconds)')
    parser.add_argument('-s', '--scale', type=float, metavar='ARCSEC_PIXEL', help='approximate image scale in arcsec/pixel')
    parser.add_argument('-i', '--instrument', choices=tuple(INSTRUMENTS), type=str.lower,
                        metavar='NAME', help='imaging preset: fors, mods, luci, hawki, bfosc, lbc, gmos, sifapsoft, nics; BFOSC defaults to Y flip and supersedes header scale')
    parser.add_argument('-q', '--cube', nargs='?', const='all', metavar='FRAME', help='solve all frames, or only 1-based frame number / EXTNAME')
    parser.add_argument('-f', '--flip', nargs='?', const='auto', choices=('auto','x','y','xy'), metavar='X|Y|XY', help='try x, y, or xy flip; bare -f tries all orientations')
    parser.add_argument('-d', '--distorsion', '--distortion', dest='distortion', nargs='?',
                        const=2, type=int, choices=(2,3), metavar='2|3',
                        help='fit SIP distortion of degree 2 (bare -d) or 3; default linear TAN')
    parser.add_argument('-r', '--resample', action='store_true',
                        help='rectify image pixels using SIP onto a plain TAN grid; implies -d 2')
    parser.add_argument('--catalog', default='gaia', metavar='GAIA|PS1|2MASS|FILE', help='Gaia DR3 (default), PS1 DR2, 2MASS, or local RA DEC [MAG] text file')
    parser.add_argument('-b', '--band', metavar='BAND', help='Gaia G/BP/RP (default RP), PS1 g/r/i/z/y (default r), or 2MASS J/H/Ks (default J)')
    parser.add_argument('-m', '--mag', type=float, nargs='+', metavar='MAG',
                        help='faint limit MAG (bright limit 12, or 15 for PS1), or bright/faint range: -m 15 22')
    parser.add_argument('--refresh-catalog', action='store_true', help='query Gaia again instead of reading the local cache')
    parser.add_argument('-o', '--output', help='output FITS path (default: INPUT[_qN]_astro.fits, or INPUT[_qN]_resampled.fits with -r)')
    parser.add_argument('--overwrite', action='store_true', help='replace existing output file')
    parser.add_argument('--version', action='version', version=f'qac {VERSION}')
    args = parser.parse_args(argv)
    try:
        if args.scale is not None and (not np.isfinite(args.scale) or args.scale <= 0):
            raise QACError("-s/--scale must be positive")
        if args.mag is not None:
            if len(args.mag) not in (1, 2) or not np.all(np.isfinite(args.mag)):
                raise QACError("-m/--mag requires one or two finite magnitudes")
            lower = (args.mag[0] if len(args.mag) == 2 else
                     15.0 if args.catalog.lower() == 'ps1' else DEFAULT_BRIGHT_MAG)
            if lower >= args.mag[-1]:
                raise QACError("-m/--mag requires a faint limit above the bright limit; "
                               "use -m BRIGHT FAINT to change the bright limit")
        out = solve(args)
    except (QACError, ValueError, OSError) as exc:
        parser.exit(2, f"qac: error: {exc}\n")
    print(f"ASTROMETRY OK: {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
