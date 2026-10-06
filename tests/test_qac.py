"""End-to-end FITS tests using a deterministic offline reference catalogue."""
import csv
import io
import sys
import types

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS, Sip
from scipy.ndimage import gaussian_filter

import qac


def sample(tmp_path):
    rng = np.random.default_rng(123)
    points = rng.uniform(22, 278, (65, 2))
    true = WCS(naxis=2)
    true.wcs.crpix = [150.5, 150.5]
    true.wcs.crval = [182.4, 32.1]
    true.wcs.cd = [[-.25/3600, 0], [0, .25/3600]]
    true.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    sky = true.pixel_to_world(points[:, 0], points[:, 1])
    catalog = tmp_path/'stars.txt'
    catalog.write_text('RA DEC\n' + ''.join(f'{s.ra.deg:.10f} {s.dec.deg:.10f}\n' for s in sky))
    canvas = np.zeros((300, 300))
    for x, y in points:
        canvas[int(round(y)), int(round(x))] += rng.uniform(400, 1000)
    image = (gaussian_filter(canvas, 1.25) + rng.normal(100, 2, canvas.shape)).astype('f4')
    guess = WCS(naxis=2)
    guess.wcs.crpix = true.wcs.crpix
    guess.wcs.crval = [182.402, 32.098]
    angle = np.deg2rad(110)
    s = .25/3600
    guess.wcs.cd = [[-s*np.cos(angle), -s*np.sin(angle)],
                    [-s*np.sin(angle), s*np.cos(angle)]]
    guess.wcs.ctype = true.wcs.ctype
    return image, true, guess, catalog


def ps1_row(objid, ra, dec, mag=18.0, **changes):
    row = {'objID': str(objid), 'raMean': str(ra), 'decMean': str(dec),
           'raStack': '-999', 'decStack': '-999',
           'raMeanErr': '.02', 'decMeanErr': '.02',
           'pmra': '100', 'pmdec': '20', 'pmraErr': '.5', 'pmdecErr': '.5',
           'epochMean': '56000', 'astrometryCorrectionFlag': '7',
           'qualityFlag': '52', 'objInfoFlag': '0', 'nDetections': '5',
           'nr': '3', 'rMeanPSFMag': str(mag), 'rMeanPSFMagErr': '.05',
           'rQfPerfect': '.99', 'rFlags': '0'}
    row.update({key: str(value) for key, value in changes.items()})
    return row


def fake_ps1_api(monkeypatch, rows, calls):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=ps1_row(0, 0, 0).keys())
    writer.writeheader()
    writer.writerows(rows)

    class Response:
        text = output.getvalue()

        def raise_for_status(self):
            pass

    def get(url, *, params, timeout):
        calls.append((url, params))
        return Response()

    monkeypatch.setattr(qac.requests, 'get', get)


def test_single_and_cube(tmp_path):
    image, true, guess, catalog = sample(tmp_path)
    single = tmp_path/'single.fits'
    old_header = guess.to_header()
    old_header['PIXSCALE'] = .57
    old_header['SECPIX'] = .57
    old_header['RA'] = '12:09:36'  # original telescope pointing is provenance
    fits.PrimaryHDU(image, header=old_header).writeto(single)
    assert qac.main([str(single), '--catalog', str(catalog)]) == 0
    with fits.open(tmp_path/'single_astro.fits', checksum=True) as solved:
        assert np.array_equal(solved[0].data, image)
        result = WCS(solved[0].header)
        assert result.pixel_to_world(150, 150).separation(true.pixel_to_world(150, 150)).arcsec < .15
        assert all(f'CD{i}_{j}' in solved[0].header for i in (1,2) for j in (1,2))
        assert not any(key.startswith(('PC1_', 'PC2_', 'CDELT', 'CROTA')) for key in solved[0].header)
        assert solved[0].header['PIXSCALE'] == pytest.approx(.25, abs=.005)
        assert solved[0].header['SECPIX'] == pytest.approx(.25, abs=.005)
        assert solved[0].header['RA'] == '12:09:36'
    mef = tmp_path/'mef.fits'
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(image, header=guess.to_header(), name='SCI'),
                  fits.ImageHDU(image, header=guess.to_header(), name='OTHER')]).writeto(mef)
    try:
        qac.main([str(mef), '--catalog', str(catalog)])
        assert False, 'multiple images must require -q'
    except SystemExit as exc:
        assert exc.code == 2
    assert qac.main([str(mef), '-q', 'SCI', '--catalog', str(catalog)]) == 0
    with fits.open(tmp_path/'mef_q1_astro.fits') as solved:
        assert len(solved) == 1
        assert solved[0].data.shape == image.shape
        assert solved[0].header['CTYPE1'] == 'RA---TAN'
    assert qac.main([str(mef), '-q', '2', '--catalog', str(catalog)]) == 0
    with fits.open(tmp_path/'mef_q2_astro.fits') as solved:
        assert len(solved) == 1
        assert np.array_equal(solved[0].data, image)
    assert qac.main([str(mef), '-q', '--catalog', str(catalog)]) == 0
    with fits.open(tmp_path/'mef_astro.fits') as solved:
        assert len(solved) == 3
        assert solved[1].header['CTYPE1'] == 'RA---TAN'
        assert solved[2].header['CTYPE1'] == 'RA---TAN'
    stack = tmp_path/'stack.fits'
    fits.PrimaryHDU(np.stack((image, image))).writeto(stack)
    assert qac.main([str(stack), '-q', '--catalog', str(catalog), '-c', '182.4', '32.1', '-s', '.25']) == 0
    with fits.open(tmp_path/'stack_astro.fits') as solved:
        assert [h.name for h in solved] == ['PRIMARY', 'FRAME_1', 'FRAME_2']
        assert np.array_equal(solved[2].data, image)
        assert solved[2].header['QACPLANE'] == 2
    only = tmp_path/'only.fits'
    assert qac.main([str(stack), '-q', '2', '-o', str(only), '--catalog', str(catalog),
                     '-c', '182.4', '32.1', '-s', '.25']) == 0
    with fits.open(only) as solved:
        assert len(solved) == 1 and solved[0].data.shape == image.shape
    five = tmp_path/'five.fits'
    fits.PrimaryHDU(np.stack([image]*5)).writeto(five)
    assert qac.main([str(five), '-q', '5', '--catalog', str(catalog),
                     '-c', '182.4', '32.1', '-s', '.25']) == 0
    with fits.open(tmp_path/'five_q5_astro.fits') as solved:
        assert len(solved) == 1 and solved[0].data.shape == image.shape


def test_mods_selects_unique_merged_image(tmp_path, capsys):
    image, _, guess, catalog = sample(tmp_path)
    path = tmp_path/'mods_archon.fits'
    hdus = [fits.PrimaryHDU()]
    hdus.append(fits.ImageHDU(image + 10, header=guess.to_header(), name='IM1'))
    hdus.extend(fits.ImageHDU(np.zeros((24, 24), dtype='f4'), name=f'IM{i}')
                for i in (2, 3, 4))
    hdus.append(fits.BinTableHDU.from_columns([], name='CONPARS'))
    hdus.append(fits.ImageHDU(image, header=guess.to_header(), name='MERGED'))
    fits.HDUList(hdus).writeto(path)

    assert qac.main([str(path), '-i', 'mods', '--catalog', str(catalog)]) == 0
    assert 'selecting MERGED (image frame 5, HDU 6)' in capsys.readouterr().out
    with fits.open(tmp_path/'mods_archon_q5_astro.fits') as solved:
        assert len(solved) == 1
        assert np.array_equal(solved[0].data, image)
        assert solved[0].header['CTYPE1'] == 'RA---TAN'

    assert qac.main([str(path), '-i', 'mods', '-q', '1',
                     '--catalog', str(catalog)]) == 0
    with fits.open(tmp_path/'mods_archon_q1_astro.fits') as solved:
        assert np.array_equal(solved[0].data, image + 10)

    without_merged = tmp_path/'mods_other.fits'
    fits.HDUList(hdus[:-1]).writeto(without_merged)
    with pytest.raises(SystemExit) as exc:
        qac.main([str(without_merged), '-i', 'mods', '--catalog', str(catalog)])
    assert exc.value.code == 2


def test_gaia_motion_and_negative_dec(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path))
    result = Table({'ra':[10.,11.], 'dec':[-23., -22.],
                    'pmra':[1000., np.nan], 'pmdec':[0., np.nan],
                    'phot_rp_mean_mag':[15., 16.]})
    queries = []
    class FakeGaia:
        @staticmethod
        def launch_job(query, verbose=False):
            assert 'gaiadr3.gaia_source_lite' in query and 'phot_rp_mean_mag' in query
            assert 'ORDER BY' not in query
            queries.append(query)
            return types.SimpleNamespace(get_results=lambda: result)
    monkeypatch.setitem(sys.modules, 'astroquery.gaia', types.SimpleNamespace(Gaia=FakeGaia))
    cat = qac.GaiaDR3().query(SkyCoord(10*u.deg, -23*u.deg), 2*u.deg,
                              Time(2026, format='jyear'))
    assert len(queries) == 2  # two stars trigger the deeper RP query
    assert 'phot_rp_mean_mag >= 12.000' in queries[0]
    assert 'phot_rp_mean_mag < 20.000' in queries[1]
    qac.GaiaDR3().query(SkyCoord(10*u.deg, -23*u.deg), 2*u.deg,
                        Time(2026, format='jyear'))
    assert len(queries) == 2  # reused from disk across runs
    assert cat.sky[0].ra.deg > 10.002
    assert cat.sky[1].ra.deg == 11.0
    assert qac.parse_coordinates('00:40:00', 'QAC_NEG_DEC_23:00:00').dec.deg == -23.


def test_gaia_once_for_two_frames(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path/'cache'))
    image, _, guess, catalog = sample(tmp_path)
    positions = np.loadtxt(catalog, skiprows=1)
    result = Table({'ra':positions[:,0], 'dec':positions[:,1],
                    'pmra':np.full(len(positions), np.nan),
                    'pmdec':np.full(len(positions), np.nan),
                    'phot_rp_mean_mag':np.linspace(12, 18, len(positions))})
    calls = []
    class FakeGaia:
        @staticmethod
        def launch_job(query, verbose=False):
            calls.append(query)
            return types.SimpleNamespace(get_results=lambda: result)
    monkeypatch.setitem(sys.modules, 'astroquery.gaia', types.SimpleNamespace(Gaia=FakeGaia))
    path = tmp_path/'two.fits'
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(image, header=guess.to_header(), name='A'),
                  fits.ImageHDU(image, header=guess.to_header(), name='B')]).writeto(path)
    assert qac.main([str(path), '-q', '-m', '13', '19']) == 0
    assert len(calls) == 1
    assert 'phot_rp_mean_mag >= 13.000' in calls[0]
    assert 'phot_rp_mean_mag < 19.000' in calls[0]


def test_gaia_brightens_when_sync_result_hits_cap(monkeypatch):
    calls = []
    def table(n):
        return Table({'ra':np.linspace(10,10.1,n), 'dec':np.zeros(n),
                      'pmra':np.full(n,np.nan), 'pmdec':np.full(n,np.nan),
                      'phot_rp_mean_mag':np.full(n,16.)})
    provider = qac.GaiaDR3()
    def fetch(ra, dec, radius, lower, rp_limit):
        calls.append((lower, rp_limit))
        return table(qac.GAIA_ROW_LIMIT if rp_limit == 18.5 else 50)
    monkeypatch.setattr(provider, '_fetch', fetch)
    cat = provider.query(SkyCoord(10*u.deg, 0*u.deg), .1*u.deg, None)
    assert calls == [(12., 18.5), (12., 17.0)]
    assert len(cat.sky) == 50


def test_flip_and_default_scale(tmp_path, capsys):
    image, true, _, catalog = sample(tmp_path)
    flipped = tmp_path/'flipped.fits'
    fits.PrimaryHDU(np.fliplr(image)).writeto(flipped)
    assert qac.main([str(flipped), '-c', '182.4', '32.1', '-s', '.25',
                     '-f', '--catalog', str(catalog)]) == 0
    with fits.open(tmp_path/'flipped_astro.fits') as h:
        recovered = WCS(h[0].header)
        assert recovered.pixel_to_world(30,40).separation(
            true.pixel_to_world(299-30,40)).arcsec < .15
        assert np.array_equal(h[0].data, np.fliplr(image))
    assert 'orientation x' in capsys.readouterr().out

    for mode, data, px, py in (
        ('x', np.fliplr(image), 299-30, 40),
        ('y', np.flipud(image), 30, 299-40),
        ('xy', np.flipud(np.fliplr(image)), 299-30, 299-40),
    ):
        input_path = tmp_path/f'explicit_{mode}.fits'
        fits.PrimaryHDU(data).writeto(input_path)
        assert qac.main([str(input_path), '-c', '182.4', '32.1', '-s', '.25',
                         '-f', mode, '--catalog', str(catalog)]) == 0
        with fits.open(tmp_path/f'explicit_{mode}_astro.fits') as h:
            assert WCS(h[0].header).pixel_to_world(30,40).separation(
                true.pixel_to_world(px,py)).arcsec < .15

    # Test both ends of the user's typical 0.12–0.30 arcsec/pixel range.
    old_sky = np.loadtxt(catalog, skiprows=1)
    x, y = true.world_to_pixel(SkyCoord(old_sky[:,0]*u.deg, old_sky[:,1]*u.deg))
    plain = tmp_path/'plain.fits'
    fits.PrimaryHDU(image).writeto(plain)
    for scale in (.12, .30):
        w = WCS(naxis=2)
        w.wcs.crpix = [150.5,150.5]; w.wcs.crval = [182.4,32.1]
        w.wcs.cd = [[-scale/3600,0],[0,scale/3600]]
        w.wcs.ctype = ['RA---TAN','DEC--TAN']
        sky = w.pixel_to_world(x,y)
        stars = tmp_path/f'{scale}.txt'
        stars.write_text('RA DEC\n' + ''.join(f'{s.ra.deg:.10f} {s.dec.deg:.10f}\n' for s in sky))
        output = tmp_path/f'{scale}.fits'
        assert qac.main([str(plain), '-c', '182.4', '32.1',
                         '--catalog', str(stars), '-o', str(output)]) == 0
        with fits.open(output) as solved:
            fitted_scale = np.sqrt(abs(np.linalg.det(WCS(solved[0].header).pixel_scale_matrix)))*3600
            assert abs(fitted_scale-scale) < .005
        assert '!!!! WARNING: NO PIXEL SCALE' in capsys.readouterr().out


def test_instrument_scales_binning_and_precedence(capsys):
    header = fits.Header()
    hdul = fits.HDUList([fits.PrimaryHDU(np.zeros((100, 100)), header=header)])
    frame = qac.Frame(0, None, '')
    center = SkyCoord(182.4*u.deg, 32.1*u.deg)

    def starting_scale(name, scale=None):
        wcs, assumed = qac.initial_wcs(hdul, frame, (100, 100), center, scale, name)
        assert not assumed
        return np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600

    assert starting_scale('fors') == pytest.approx(.25)
    assert starting_scale('mods') == pytest.approx(.123)
    assert starting_scale('luci') == pytest.approx(.120)
    assert starting_scale('hawki') == pytest.approx(.106)
    assert starting_scale('bfosc') == pytest.approx(.57)
    assert starting_scale('lbc') == pytest.approx(.225)
    assert starting_scale('gmos') == pytest.approx(.16)
    assert starting_scale('sifapsoft') == pytest.approx(.067)
    assert starting_scale('nics') == pytest.approx(.25)
    assert '!!!! WARNING' not in capsys.readouterr().out

    hdul[0].header['CCDSUM'] = '2 2'
    assert starting_scale('mods') == pytest.approx(.246)
    assert starting_scale('bfosc') == pytest.approx(1.14)
    assert starting_scale('lbc') == pytest.approx(.45)
    assert starting_scale('gmos') == pytest.approx(.16)
    assert starting_scale('sifapsoft') == pytest.approx(.134)
    assert starting_scale('nics') == pytest.approx(.5)
    assert starting_scale('fors') == pytest.approx(.25)
    hdul[0].header['CCDSUM'] = '1 1'
    assert starting_scale('fors') == pytest.approx(.125)
    assert starting_scale('gmos') == pytest.approx(.08)
    hdul[0].header['HIERARCH ESO DET WIN1 BINX'] = 2
    hdul[0].header['HIERARCH ESO DET WIN1 BINY'] = 2
    assert starting_scale('fors') == pytest.approx(.25)
    hdul[0].header['PIXSCALE'] = .22
    assert starting_scale('mods') == pytest.approx(.22)
    assert starting_scale('mods', .19) == pytest.approx(.19)
    hdul[0].header.pop('PIXSCALE')
    existing = qac.WCS(naxis=2)
    existing.wcs.crpix = [50., 50.]
    existing.wcs.crval = [182.4, 32.1]
    existing.wcs.cd = [[-.3/3600, 0], [0, .3/3600]]
    existing.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    hdul[0].header.update(existing.to_header())
    assert starting_scale('luci') == pytest.approx(.3)
    assert starting_scale('luci', .2) == pytest.approx(.2)


def test_input_wcs_warns_when_pointing_disagrees(capsys):
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [150.5, 150.5]
    wcs.wcs.crval = [182.4, 32.1]
    wcs.wcs.cd = [[-.25/3600, 0], [0, .25/3600]]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    primary = fits.PrimaryHDU()
    primary.header['OBJRA'] = '12:09:36.0'
    primary.header['OBJDEC'] = '+32:06:00'
    image = fits.ImageHDU(np.zeros((300, 300)), header=wcs.to_header(), name='SCI')
    hdul = fits.HDUList([primary, image])
    frame = qac.Frame(1, None, 'SCI')

    qac.initial_wcs(hdul, frame, image.shape, None, None)
    assert 'WARNING: input WCS center' not in capsys.readouterr().out

    primary.header['OBJRA'] = '12:10:00.0'
    qac.initial_wcs(hdul, frame, image.shape, None, None)
    warning = capsys.readouterr().out
    assert 'WARNING: input WCS center' in warning
    assert 'OBJRA/OBJDEC' in warning and 'use -c RA DEC' in warning

    qac.initial_wcs(hdul, frame, image.shape,
                    SkyCoord(182.4*u.deg, 32.1*u.deg), None)
    assert 'WARNING: input WCS center' not in capsys.readouterr().out


def test_held_out_tan_validation_detects_bad_predictions():
    rng = np.random.default_rng(45)
    xy = rng.uniform(20, 280, (40, 2))
    true = WCS(naxis=2)
    true.wcs.crpix = [150.5, 150.5]
    true.wcs.crval = [182.4, 32.1]
    true.wcs.cd = [[-.25/3600, 0], [0, .25/3600]]
    true.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    sky = true.pixel_to_world(xy[:, 0], xy[:, 1])
    indices = np.arange(len(xy))
    assert qac.validate_linear_solution(xy, sky, indices, indices, true) < .001

    order = np.argsort(xy[:, 0] + .7*xy[:, 1])
    wrong_ra = sky.ra.deg.copy()
    wrong_ra[order[::5]] += 4/3600/np.cos(np.deg2rad(sky.dec.deg[order[::5]]))
    wrong_sky = SkyCoord(wrong_ra*u.deg, sky.dec.deg*u.deg)
    with pytest.raises(qac.QACError, match='Held-out TAN RMS'):
        qac.validate_linear_solution(xy, wrong_sky, indices, indices, true)


def test_bfosc_defaults_to_y_flip_but_explicit_flip_wins(tmp_path, capsys):
    image, true, _, catalog = sample(tmp_path)
    positions = np.loadtxt(catalog, skiprows=1)
    x, y = true.world_to_pixel(SkyCoord(positions[:, 0]*u.deg, positions[:, 1]*u.deg))
    actual = true.deepcopy()
    actual.wcs.cd = [[-.57/3600, 0], [0, .57/3600]]
    sky = actual.pixel_to_world(x, y)
    references = tmp_path/'bfosc_stars.txt'
    references.write_text('RA DEC\n' + ''.join(
        f'{s.ra.deg:.10f} {s.dec.deg:.10f}\n' for s in sky))
    path = tmp_path/'bfosc.fits'
    fits.PrimaryHDU(np.flipud(image)).writeto(path)
    base = [str(path), '-c', '182.4', '32.1', '--catalog', str(references)]

    assert qac.main(base + ['-i', 'bfosc']) == 0
    assert 'orientation y' in capsys.readouterr().out
    assert qac.main(base + ['-s', '.57', '-f', 'y', '-o',
                            str(tmp_path/'explicit.fits')]) == 0
    with fits.open(tmp_path/'bfosc_astro.fits') as preset, fits.open(tmp_path/'explicit.fits') as explicit:
        fitted = WCS(preset[0].header)
        assert fitted.pixel_to_world(30, 40).separation(
            actual.pixel_to_world(30, 299-40)).arcsec < .15
        for px, py in ((30, 40), (150, 150), (240, 210)):
            assert fitted.pixel_to_world(px, py).separation(
                WCS(explicit[0].header).pixel_to_world(px, py)).arcsec < .01

    stale = true.to_header()
    stale['PIXSCALE'] = .25
    fits.PrimaryHDU(np.flipud(image), header=stale).writeto(tmp_path/'bfosc_stale.fits')
    assert qac.main([str(tmp_path/'bfosc_stale.fits'), '-i', 'bfosc',
                     '--catalog', str(references)]) == 0
    with fits.open(tmp_path/'bfosc_stale_astro.fits') as solved:
        assert WCS(solved[0].header).pixel_to_world(30, 40).separation(
            actual.pixel_to_world(30, 299-40)).arcsec < .15

    fits.PrimaryHDU(np.fliplr(image)).writeto(tmp_path/'bfosc_x.fits')
    assert qac.main([str(tmp_path/'bfosc_x.fits'), '-i', 'bfosc', '-f', 'x',
                     '-c', '182.4', '32.1', '--catalog', str(references)]) == 0
    assert 'orientation x' in capsys.readouterr().out


@pytest.mark.parametrize(('instrument', 'scale'), [('luci', .120), ('hawki', .106),
                                                    ('lbc', .225), ('gmos', .16),
                                                    ('sifapsoft', .067), ('nics', .25)])
def test_instrument_solves_without_input_wcs(tmp_path, capsys, instrument, scale):
    image, true, _, catalog = sample(tmp_path)
    positions = np.loadtxt(catalog, skiprows=1)
    x, y = true.world_to_pixel(SkyCoord(positions[:, 0]*u.deg, positions[:, 1]*u.deg))
    actual = true.deepcopy()
    actual.wcs.cd = [[-scale/3600, 0], [0, scale/3600]]
    sky = actual.pixel_to_world(x, y)
    references = tmp_path/f'{instrument}_stars.txt'
    references.write_text('RA DEC\n' + ''.join(
        f'{s.ra.deg:.10f} {s.dec.deg:.10f}\n' for s in sky))
    path = tmp_path/f'{instrument}.fits'
    fits.PrimaryHDU(image).writeto(path)
    assert qac.main([str(path), '-c', '182.4', '32.1', '-i', instrument,
                     '--catalog', str(references)]) == 0
    output = capsys.readouterr().out
    assert '!!!! WARNING: NO PIXEL SCALE' not in output
    assert 'Held-out TAN RMS:' in output
    with fits.open(tmp_path/f'{instrument}_astro.fits') as result:
        restored = WCS(result[0].header)
        assert restored.pixel_to_world(150, 150).separation(
            actual.pixel_to_world(150, 150)).arcsec < .15


def test_instrument_narrow_query_retries_wider(tmp_path, monkeypatch, capsys):
    image, _, guess, catalog = sample(tmp_path)
    path = tmp_path/'instrument.fits'
    fits.PrimaryHDU(image, header=guess.to_header()).writeto(path)
    query = qac.LocalCatalog.query
    radii = []

    def first_cone_has_no_matches(self, center, radius, obstime):
        radii.append(radius.to_value(u.arcmin))
        if len(radii) == 1:
            raise qac.QACError('no matching stars in initial cone')
        return query(self, center, radius, obstime)

    monkeypatch.setattr(qac.LocalCatalog, 'query', first_cone_has_no_matches)
    assert qac.main([str(path), '--catalog', str(catalog), '-i', 'MODS']) == 0
    assert len(radii) == 2 and radii[0] < radii[1]
    assert 'Retrying with a wider catalogue search' in capsys.readouterr().out
    with fits.open(tmp_path/'instrument_astro.fits') as result:
        assert 'CD1_1' in result[0].header


def test_ps1_catalog_selection_and_magnitude_limits(tmp_path, monkeypatch):
    image, true, guess, catalog = sample(tmp_path)
    path = tmp_path/'ps1.fits'
    fits.PrimaryHDU(image, header=guess.to_header()).writeto(path)
    positions = np.loadtxt(catalog, skiprows=1)
    rows = [ps1_row(i, ra, dec, 15.5 + i/20, pmra='', pmdec='')
            for i, (ra, dec) in enumerate(positions)]
    calls = []
    fake_ps1_api(monkeypatch, rows, calls)

    assert qac.main([str(path), '--catalog', 'ps1', '-b', 'r', '-m', '15', '22']) == 0
    assert calls[0][0] == qac.PS1_API
    assert calls[0][1]['rMeanPSFMag.gte'] == 15
    assert calls[0][1]['rMeanPSFMag.lt'] == 22
    assert calls[0][1]['nDetections.gte'] == 3
    assert 'columns' not in calls[0][1]  # MAST omits new PM fields in mixed selections.
    with fits.open(tmp_path/'ps1_astro.fits') as result:
        assert WCS(result[0].header).pixel_to_world(150, 150).separation(
            true.pixel_to_world(150, 150)).arcsec < .15


def test_ps1_motion_uses_mjd_and_ra_cosdec(monkeypatch):
    calls = []
    fake_ps1_api(monkeypatch, [ps1_row(1, 10, 60)], calls)
    provider = qac.PanSTARRS1(mag_limit=22)
    center = SkyCoord(10*u.deg, 60*u.deg)
    date = Time(56000 + 3652.5, format='mjd')
    cat = provider.query(center, .1*u.deg, date)
    assert len(cat.sky) == 1
    assert (cat.sky[0].ra.deg - 10)*3600 == pytest.approx(2.0, abs=.01)
    assert (cat.sky[0].dec.deg - 60)*3600 == pytest.approx(.2, abs=.01)
    assert provider.query(center, .1*u.deg, None).sky[0].ra.deg == pytest.approx(10)
    assert len(calls) == 1  # raw query reused for a different observation date


def test_ps1_missing_or_unreliable_motion_stays_at_mean_epoch(monkeypatch):
    rows = [ps1_row(1, 10, 60, pmra=''),
            ps1_row(2, 10.01, 60, epochMean=-999),
            ps1_row(3, 10.02, 60, pmraErr=5),
            ps1_row(4, 10.03, 60, astrometryCorrectionFlag=3),
            ps1_row(5, 10.04, 60, objInfoFlag=4194304)]
    fake_ps1_api(monkeypatch, rows, [])
    cat = qac.PanSTARRS1().query(SkyCoord(10*u.deg, 60*u.deg),
                                  .1*u.deg, Time(60000, format='mjd'))
    assert np.allclose(cat.sky.ra.deg, [10, 10.01, 10.02, 10.03, 10.04])


def test_ps1_faint_quality_excludes_stack_positions(monkeypatch):
    rows = [ps1_row(1, 10, 20, 21.9),
            ps1_row(2, -999, 20, 21.5, raStack=10.001, decStack=20),
            ps1_row(3, 10.002, 20, 21.5, astrometryCorrectionFlag=0),
            ps1_row(4, 10.003, 20, 21.5, qualityFlag=53),
            ps1_row(5, 10.004, 20, 21.5, rQfPerfect=.4),
            ps1_row(6, 10.005, 20, 21.5, rMeanPSFMagErr=.5),
            ps1_row(7, 10.006, 20, 21.5, raMeanErr=.6),
            ps1_row(8, 10.007, 20, 22.0),
            ps1_row(9, 10.008, 20, 14.9)]
    fake_ps1_api(monkeypatch, rows, [])
    cat = qac.PanSTARRS1(min_mag=15, mag_limit=22).query(
        SkyCoord(10*u.deg, 20*u.deg), .1*u.deg, None)
    assert len(cat.sky) == 1
    assert cat.mag[0] == pytest.approx(21.9)


def test_ps1_match_can_use_fainter_image_detections():
    rng = np.random.default_rng(903)
    sources = rng.uniform(100, 1900, (200, 2))
    shift = np.array([35., -20.])
    refs = np.vstack((sources[100:125] + shift + rng.normal(0, .1, (25, 2)),
                      rng.uniform(100, 1900, (55, 2))))
    si, ri = qac.coarse_match(sources, refs, source_limit=200)
    assert len(si) >= 24
    assert np.count_nonzero((si >= 100) & (si < 125) & (ri < 25)) >= 24


def test_2mass_band_and_mag(tmp_path, monkeypatch):
    image, _, guess, catalog = sample(tmp_path)
    positions = np.loadtxt(catalog, skiprows=1)
    requests = []
    class FakeVizier:
        def __init__(self, *, columns, column_filters, row_limit):
            requests.append((columns, column_filters, row_limit))
        def query_region(self, center, *, radius, catalog):
            assert catalog == 'II/246/out'
            col = next(iter(requests[-1][1]))
            return [Table({'_RAJ2000':positions[:,0], '_DEJ2000':positions[:,1],
                           col:np.linspace(12, 14, len(positions))})]
    monkeypatch.setitem(sys.modules, 'astroquery.vizier', types.SimpleNamespace(Vizier=FakeVizier))
    path = tmp_path/'nir.fits'
    fits.PrimaryHDU(image, header=guess.to_header()).writeto(path)
    assert qac.main([str(path), '--catalog', '2mass', '-b', 'Ks', '-m', '12.5', '14.1']) == 0
    assert requests[0][1] == {'Kmag':'12.5..14.1'}
    with fits.open(tmp_path/'nir_astro.fits') as solved:
        assert solved[0].header['CTYPE1'] == 'RA---TAN'
    single_limit = tmp_path/'nir_single_limit.fits'
    assert qac.main([str(path), '--catalog', '2mass', '-b', 'Ks', '-m', '14.1',
                     '-o', str(single_limit)]) == 0
    assert requests[1][1] == {'Kmag':'12..14.1'}


def test_gaia_selected_band_and_fixed_mag(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path))
    calls = []
    result = Table({'ra':[10.], 'dec':[0.], 'pmra':[np.nan], 'pmdec':[np.nan],
                    'phot_g_mean_mag':[17.]})
    class FakeGaia:
        @staticmethod
        def launch_job(query, verbose=False):
            calls.append(query)
            return types.SimpleNamespace(get_results=lambda: result)
    monkeypatch.setitem(sys.modules, 'astroquery.gaia', types.SimpleNamespace(Gaia=FakeGaia))
    cat = qac.GaiaDR3(band='G', mag_limit=19.1).query(
        SkyCoord(10*u.deg, 0*u.deg), .01*u.deg, None)
    assert len(calls) == 1
    assert 'phot_g_mean_mag >= 12.000' in calls[0]
    assert 'phot_g_mean_mag < 19.100' in calls[0]
    assert cat.label == 'Gaia DR3 (G)'


def test_gaia_magnitude_range_filters_catalogue(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path))
    calls = []
    result = Table({'ra':[10.,10.01,10.02,10.03], 'dec':[0.,0.,0.,0.],
                    'pmra':[np.nan]*4, 'pmdec':[np.nan]*4,
                    'phot_rp_mean_mag':[11.,15.,18.,19.]})
    class FakeGaia:
        @staticmethod
        def launch_job(query, verbose=False):
            calls.append(query)
            return types.SimpleNamespace(get_results=lambda: result)
    monkeypatch.setitem(sys.modules, 'astroquery.gaia', types.SimpleNamespace(Gaia=FakeGaia))
    catalog = qac.GaiaDR3(min_mag=15., mag_limit=19.).query(
        SkyCoord(10*u.deg, 0*u.deg), .1*u.deg, None)
    assert len(calls) == 1
    assert 'phot_rp_mean_mag >= 15.000' in calls[0]
    assert 'phot_rp_mean_mag < 19.000' in calls[0]
    assert np.array_equal(catalog.mag, [15.,18.])


@pytest.mark.parametrize('limits', [('15','15'), ('19','15'), ('11',),
                                     ('15','19','20'), ('nan','20')])
def test_invalid_magnitude_range_is_rejected(limits):
    with pytest.raises(SystemExit) as exc:
        qac.main(['missing.fits', '-m', *limits])
    assert exc.value.code == 2


def test_match_moves_past_bright_unmatched_stars(tmp_path, capsys):
    image, base, guess, catalog = sample(tmp_path)
    positions = np.loadtxt(catalog, skiprows=1)
    rng = np.random.default_rng(900)
    decoy = guess.pixel_to_world(rng.uniform(20,280,80), rng.uniform(20,280,80))
    sky = SkyCoord(np.r_[decoy.ra.deg, positions[:,0]]*u.deg,
                   np.r_[decoy.dec.deg, positions[:,1]]*u.deg)
    mags = np.r_[np.linspace(12,14,80), np.linspace(16,18,len(positions))]
    refxy, ordered_sky = qac.catalog_pixels(
        qac.ReferenceCatalog(sky, mags, 'test'), guess, image.shape)
    fitted, count, rms, mode = qac.match_in_magnitude_order(
        qac.detect_sources(image), refxy, ordered_sky, image.shape,
        guess, (0.75,1.25), ('',), None)
    assert count >= 35 and rms < .4 and mode == ''
    assert fitted.pixel_to_world(150,150).separation(base.pixel_to_world(150,150)).arcsec < .15
    assert 'ranked 81-' in capsys.readouterr().out


def test_gaia_deepens_when_bright_group_has_no_real_matches(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path/'cache'))
    image, true, guess, catalog = sample(tmp_path)
    positions = np.loadtxt(catalog, skiprows=1)
    rng = np.random.default_rng(901)
    decoy = guess.pixel_to_world(rng.uniform(20,280,80), rng.uniform(20,280,80))
    bright = Table({'ra':decoy.ra.deg, 'dec':decoy.dec.deg,
                    'pmra':np.full(80,np.nan), 'pmdec':np.full(80,np.nan),
                    'phot_rp_mean_mag':np.linspace(12.1,14.,80)})
    faint = Table({'ra':positions[:,0], 'dec':positions[:,1],
                   'pmra':np.full(len(positions),np.nan),
                   'pmdec':np.full(len(positions),np.nan),
                   'phot_rp_mean_mag':np.full(len(positions),19.)})
    calls = []
    class FakeGaia:
        @staticmethod
        def launch_job(query, verbose=False):
            calls.append(query)
            return types.SimpleNamespace(get_results=lambda: bright if '< 18.500' in query
                                         else Table(np.concatenate((bright.as_array(), faint.as_array()))))
    monkeypatch.setitem(sys.modules, 'astroquery.gaia', types.SimpleNamespace(Gaia=FakeGaia))
    path = tmp_path/'deep.fits'
    fits.PrimaryHDU(image, header=guess.to_header()).writeto(path)
    assert qac.main([str(path)]) == 0
    assert len(calls) == 2 and '< 20.000' in calls[1]
    with fits.open(tmp_path/'deep_astro.fits') as result:
        assert WCS(result[0].header).pixel_to_world(150,150).separation(
            true.pixel_to_world(150,150)).arcsec < .15


def test_sip_orders_and_flipped_pixels(tmp_path):
    image, base, _, original_catalog = sample(tmp_path)
    positions = np.loadtxt(original_catalog, skiprows=1)
    x, y = base.world_to_pixel(SkyCoord(positions[:,0]*u.deg, positions[:,1]*u.deg))
    xx, yy = np.meshgrid(np.linspace(15,285,8), np.linspace(15,285,8))
    for degree, mode in ((2, ''), (2, 'x'), (3, '')):
        truth = base.deepcopy()
        truth.wcs.ctype = ['RA---TAN-SIP', 'DEC--TAN-SIP']
        a = np.zeros((degree+1,degree+1)); b = a.copy()
        a[2,0] = .0002; a[1,1] = .00008
        b[0,2] = -.00016; b[1,1] = -.00005
        if degree == 3:
            a[3,0] = .0000015; b[0,3] = -.0000012
        truth.sip = Sip(a,b,None,None,truth.wcs.crpix)
        sky = truth.pixel_to_world(x,y)
        catalog = tmp_path/f'sip{degree}{mode}.txt'
        catalog.write_text('RA DEC\n' + ''.join(
            f'{p.ra.deg:.10f} {p.dec.deg:.10f}\n' for p in sky))
        input_path = tmp_path/f'input{degree}{mode}.fits'
        data = np.fliplr(image) if mode else image
        fits.PrimaryHDU(data).writeto(input_path)
        args = [str(input_path), '-c', '182.4', '32.1', '-s', '.25',
                '--catalog', str(catalog), '-d' if degree == 2 else '--distorsion']
        if degree == 3:
            args.append('3')
        if mode:
            args += ['-f', mode]
        linear_path = tmp_path/f'linear{degree}{mode}.fits'
        linear_args = [str(input_path), '-c', '182.4', '32.1', '-s', '.25',
                       '--catalog', str(catalog), '-o', str(linear_path)]
        if mode:
            linear_args += ['-f', mode]
        assert qac.main(linear_args) == 0
        assert qac.main(args) == 0
        sx = 299-xx if mode == 'x' else xx
        with fits.open(linear_path) as linear_hdul:
            linear_error = truth.pixel_to_world(xx,yy).separation(
                WCS(linear_hdul[0].header).pixel_to_world(sx,yy)).arcsec
        with fits.open(tmp_path/f'input{degree}{mode}_astro.fits', checksum=True) as solved:
            header = solved[0].header
            assert np.array_equal(solved[0].data, data)
            assert header['A_ORDER'] == degree
            assert header['CTYPE1'] == 'RA---TAN-SIP'
            assert 'CD1_1' in header and 'CDELT1' not in header and 'PC1_1' not in header
            restored = WCS(header)
            error = truth.pixel_to_world(xx,yy).separation(
                restored.pixel_to_world(sx,yy)).arcsec
            assert np.sqrt(np.mean(error**2)) < .75*np.sqrt(np.mean(linear_error**2))
        if mode == 'x':
            assert qac.main(args + ['-r']) == 0
            with fits.open(tmp_path/f'input{degree}{mode}_resampled.fits') as rectified:
                assert rectified[0].header['CTYPE1'] == 'RA---TAN'
                assert 'CD1_1' in rectified[0].header and 'CDELT1' not in rectified[0].header
                assert WCS(rectified[0].header).sip is None
                assert np.isfinite(rectified[0].data[150, 150])


def test_write_wcs_clears_old_alternate_and_lookup_terms():
    header = fits.Header()
    header['CDELT1'] = -.0001
    header['PC1_1'] = 1.0
    header['CTYPE1A'] = 'RA---TAN'
    header['CDELT1A'] = -.0002
    header['WCSNAMEA'] = 'old alternative'
    header['CPDIS1'] = 'LOOKUP'
    header['DP1.AXIS.1'] = 2
    old_terms = {
        'A_DMAX': 1.2, 'B_DMAX': 1.4, 'AP_DMAX': .1,
        'CPERR1': .3, 'CQERR2': .4, 'DPERR1': .2, 'D2IMERR': .5,
        'AXISCORR': 1, 'D2IMFILE': 'old.fits', 'NPOLFILE': 'old.fits',
        'WCSDIM': 2, 'WAT1_001': 'wtype=tan', 'SIPNAME': 'old distortion',
    }
    header.update(old_terms)
    header['PIXSCALE'] = .57
    header['RA'] = 'original pointing'
    header['LTV1'] = 20.  # detector coordinates remain valid without resampling
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [100.,100.]
    wcs.wcs.crval = [182.4,32.1]
    wcs.wcs.cd = [[-.25/3600,.01/3600],[.02/3600,.25/3600]]
    wcs.wcs.ctype = ['RA---TAN','DEC--TAN']
    qac.write_wcs(header, wcs)
    assert not any(key in header for key in
                   ('CDELT1','PC1_1','CTYPE1A','CDELT1A','WCSNAMEA','CPDIS1','DP1.AXIS.1'))
    assert not old_terms.keys() & set(header)
    assert header['RA'] == 'original pointing'
    assert header['LTV1'] == 20.
    assert header['PIXSCALE'] == pytest.approx(
        np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix)))*3600)
    assert np.max(WCS(header).pixel_to_world([10,250],[20,270]).separation(
        wcs.pixel_to_world([10,250],[20,270])).arcsec) < 1e-5


def test_sip_requires_enough_stars(tmp_path):
    image, true, guess, catalog = sample(tmp_path)
    positions = np.loadtxt(catalog, skiprows=1)
    sky = SkyCoord(positions[:,0]*u.deg,positions[:,1]*u.deg)
    x,y = true.world_to_pixel(sky)
    sources = np.column_stack((x,y))
    with pytest.raises(qac.QACError, match='at least 40 matched stars'):
        qac.fit_sip_distortion(sources, sky, np.arange(30), np.arange(30),
                               (300,300), 3, true)
    limited = tmp_path/'limited.txt'
    limited.write_text('RA DEC\n' + ''.join(
        f'{ra:.10f} {dec:.10f}\n' for ra,dec in positions[:30]))
    image_path = tmp_path/'few.fits'
    fits.PrimaryHDU(image, header=guess.to_header()).writeto(image_path)
    with pytest.raises(SystemExit) as exc:
        qac.main([str(image_path), '--catalog', str(limited), '-d', '3'])
    assert exc.value.code == 2
    assert not (tmp_path/'few_astro.fits').exists()


def test_resample_known_sip_and_pixel_area():
    height = width = 200
    y, x = np.mgrid[:height, :width]
    image = (100 + 2*x + 3*y).astype('f4')
    image[100, 100] = np.nan
    sip_wcs = WCS(naxis=2)
    sip_wcs.wcs.crpix = [101., 101.]
    sip_wcs.wcs.crval = [182.4, 32.1]
    sip_wcs.wcs.cd = [[-.25/3600, 0], [0, .25/3600]]
    sip_wcs.wcs.ctype = ['RA---TAN-SIP', 'DEC--TAN-SIP']
    a = np.zeros((3, 3)); b = a.copy()
    a[2, 0] = .001
    b[0, 2] = -.0005
    sip_wcs.sip = Sip(a, b, None, None, sip_wcs.wcs.crpix)
    rectified, tan = qac.resample_image(image, sip_wcs)
    assert rectified.shape == image.shape and rectified.dtype == np.float32
    assert tan.sip is None
    assert list(tan.wcs.ctype) == ['RA---TAN', 'DEC--TAN']
    assert np.isnan(rectified[100, 100])  # masked source pixel
    assert np.isnan(rectified[100, 0])    # target pixel lies outside source
    for tx, ty in ((170, 60), (40, 120), (130, 170)):
        sky = tan.pixel_to_world(tx, ty)
        sx, sy = sip_wcs.world_to_pixel(sky)
        jac = (1 + .002*(sx-100)) * (1 - .001*(sy-100))
        expected = (100 + 2*sx + 3*sy) / jac
        assert rectified[ty, tx] == pytest.approx(expected, rel=1e-6)


def test_resample_cli_implies_sip_and_extracts_one_frame(tmp_path):
    image, base, guess, catalog = sample(tmp_path)
    points = np.loadtxt(catalog, skiprows=1)
    x, y = base.world_to_pixel(SkyCoord(points[:,0]*u.deg, points[:,1]*u.deg))
    truth = base.deepcopy()
    truth.wcs.ctype = ['RA---TAN-SIP', 'DEC--TAN-SIP']
    a = np.zeros((3,3)); b = a.copy()
    a[2,0] = .00025
    b[0,2] = -.00020
    truth.sip = Sip(a,b,None,None,truth.wcs.crpix)
    sky = truth.pixel_to_world(x,y)
    stars = tmp_path/'distorted.txt'
    stars.write_text('RA DEC\n' + ''.join(f'{s.ra.deg:.10f} {s.dec.deg:.10f}\n' for s in sky))
    path = tmp_path/'multi.fits'
    old_header = guess.to_header()
    old_header['LTV1'] = 20.
    old_header['LTM1_1'] = 1.
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(image, header=old_header, name='SCI'),
                  fits.ImageHDU(np.ones((20,20), dtype='f4'), name='OTHER')]).writeto(path)
    assert qac.main([str(path), '-q', 'SCI', '-r', '--catalog', str(stars)]) == 0
    with fits.open(tmp_path/'multi_q1_resampled.fits', checksum=True) as hdul:
        assert len(hdul) == 1
        header = hdul[0].header
        assert header['CTYPE1'] == 'RA---TAN'
        assert header['CTYPE2'] == 'DEC--TAN'
        assert 'A_ORDER' not in header and WCS(header).sip is None
        assert 'LTV1' not in header and 'LTM1_1' not in header
        assert hdul[0].data.shape == image.shape
        assert np.isfinite(hdul[0].data[150,150])
        assert not np.array_equal(hdul[0].data, image)
        assert WCS(header).pixel_to_world(150,150).separation(
            truth.pixel_to_world(150,150)).arcsec < .15
    stack = tmp_path/'stack_distorted.fits'
    fits.PrimaryHDU(np.stack([image, image]), header=guess.to_header()).writeto(stack)
    assert qac.main([str(stack), '-q', '2', '-r', '--catalog', str(stars)]) == 0
    with fits.open(tmp_path/'stack_distorted_q2_resampled.fits') as hdul:
        assert len(hdul) == 1
        assert hdul[0].data.shape == image.shape
        assert hdul[0].header['CTYPE1'] == 'RA---TAN'
        assert hdul[0].header['NAXIS'] == 2
