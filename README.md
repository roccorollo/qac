# QAC v0.5.11 — Quick Astrometry for FITS Images

Author: Andrea Rossi, with the assistance of ChatGPT.  
License: [MIT](LICENSE).

QAC refines image astrometry against Gaia DR3, Pan-STARRS1 DR2, 2MASS, or a local RA/Dec catalogue. It writes a fitted TAN WCS, optionally with SIP distortion. The original pixels are retained unless `-r` resamples them onto an undistorted TAN grid. Python 3.10+ is required.

## Install and run

```bash
conda activate co310
python -m pip install -r requirements.txt
python qac.py image.fits -c 09:59:40.64 +00:24:21.454 -s 0.23
```

The example uses a Conda environment; any Python 3.10+ environment with the listed requirements works. Gaia, PS1, and 2MASS need internet access for uncached queries. A local catalogue works offline. Run `python qac.py --help` for the complete option list.

## Examples

```bash
python qac.py image.fits                                      # use header position/scale
python qac.py image.fits -c 188.73375 31.73944 -s 0.23
python qac.py image.fits -i mods -c 188.73375 31.73944  # MODS preset
python qac.py mods_archon.fits -i mods                        # select MERGED image
python qac.py hawki.fits -i hawki -q                       # four image extensions
python qac.py bfosc.fits -i bfosc -c 188.73375 31.73944  # 0.57 arcsec/pixel, Y flip
python qac.py lbc_chip.fits -i lbc -c 188.73375 31.73944  # one LBC CCD
python qac.py gmos.fits -i gmos -c 188.73375 31.73944      # GMOS imaging, 2x2
python qac.py sifapsoft.fits -i sifapsoft -c 188.73375 31.73944
python qac.py nics.fits -i nics -c 188.73375 31.73944
python qac.py mirrored.fits -f                                 # try normal, x, y, xy
python qac.py mirrored.fits -f x -d 3                          # x flip, third-order SIP
python qac.py image.fits -r                                    # second-order SIP, rectified TAN pixels
python qac.py image.fits --catalog gaia -b G -m 15 19          # 15 <= G < 19
python qac.py image.fits --catalog ps1 -b r -m 15 22          # 15 <= rMeanPSFMag < 22
python qac.py image.fits --catalog 2mass -b Ks -m 10 14        # 10 <= Ks < 14
python qac.py mods1r.20260918.0077.fits -i mods -q MERGED --catalog ps1 -b r -m 15 22
python qac.py image.fits --catalog stars.txt                   # local RA/Dec catalogue
python qac.py stack.fits -q                                    # solve every frame
python qac.py stack.fits -q 5                                  # output only frame 5
python qac.py multi.fits -q SCI                                # output unique SCI image
```

## Position, scale, and catalogue

`-c/--coordinates RA DEC` overrides the approximate field centre. Plain numbers are degrees; sexagesimal RA uses hours and Dec uses degrees. `-s/--scale` is the initial arcsec/pixel estimate. Matching refines this scale. A header WCS or `PIXSCALE`/`SECPIX` can supply it; otherwise QAC assumes **0.2 arcsec/pixel**, prints a prominent warning, and searches a wider scale range. Give `-s` when you know the scale. An approximate centre is always needed.

`-i/--instrument NAME` selects an initial scale and field-size check:

| Name | Imaging mode | Scale (arcsec/pixel) | Nominal field |
| --- | --- | ---: | ---: |
| `fors` | FORS2 standard, 2×2 binning | 0.25 | 6.8′ × 6.8′ |
| `mods` | MODS imaging, 1×1 | 0.123 | 6′ × 6′ |
| `luci` | LUCI N3.75 seeing limited imaging | 0.120 | 4′ × 4′ |
| `hawki` | HAWK-I four-detector mosaic | 0.106 | 7.5′ × 7.5′ |
| `bfosc` | Loiano BFOSC imaging, 1×1; default Y flip | 0.57 | about 13′ × 12.6′ |
| `lbc` | LBT LBC Blue or Red, 1×1 | 0.225 | 23′ × 25′ full camera |
| `gmos` | Gemini GMOS-N or GMOS-S imaging, 2×2 | 0.160 | 5.5′ × 5.5′ |
| `sifapsoft` | SIFAPSOFT imaging, 1×1 | 0.067 | about 10′ |
| `nics` | TNG NICS imaging, 1×1 | 0.25 | about 4.3′ |

Scale overrides follow one rule: `-s` takes priority, then a valid header WCS, then `PIXSCALE`/`SECPIX`, then the instrument preset. BFOSC is the exception: `-i bfosc` acts like `-s 0.57 -f y`, so its scale also supersedes header values; explicit `-s` or `-f` still takes priority. Binning keywords adjust preset scales when used, and matching refines the starting scale. For a different camera mode or older detector, supply its measured scale with `-s`.

The preset values are based on the [ESO FORS2](https://ftp.eso.org/pub/dfs/pipelines/instruments/fors/fors-img-reflex-tutorial-2.0.pdf), [LBTO MODS](https://scienceops.lbto.org/mods/) / [LUCI](https://www.lbto.org/instruments-overview/), [ESO HAWK-I](https://eso.org/sci/facilities/paranal/instruments/hawki.html), [LBC](https://www.lbto.org/instruments-overview/), [Gemini GMOS](https://www2.gemini.edu/instrumentation/gmos/observation-preparation), and [BFOSC](https://www.oas.inaf.it/wp-content/uploads/2019/05/BFOSC_en_UM_2001.pdf) documentation. SIFAPSOFT and NICS use the scale and approximate field sizes supplied for this release. BFOSC's manual quotes 0.58″/pixel; the preset uses the requested 0.57″/pixel. The MODS preset uses one value for both channels.

When a valid input WCS and header pointing (`RA`/`DEC` or `OBJRA`/`OBJDEC`) disagree by more than 1 arcmin or a quarter of the image's half-diagonal, QAC warns with the separation. It still starts from the input WCS; use `-c RA DEC` to override its centre. Linear TAN solutions also report a held-out RMS: QAC refits using 80% of the matched stars, checks predictions for the other 20%, and rejects a solution whose held-out error is too large. SIP fits retain their separate held-out validation.

For LBC, each CCD covers about 7.8′ × 17.6′. Use the same preset for one CCD or the full camera: the image dimensions, not the nominal field in the table, determine the catalogue search. Select a CCD in a multi-extension file with `-q N` or `-q EXTNAME`. For GMOS, 1×1 binning gives a starting scale of 0.080″/pixel when recorded in the header. Use a processed GMOS image or mosaic; raw amplifier extensions contain overscan and are not independent sky frames.

With `-i`, QAC starts with a smaller catalogue cone around the footprint computed from the image dimensions and initial scale. If the match fails, it retries the wider search. The nominal field is a sanity check, not a limit on cropped, mosaicked, or offset images. HAWK-I's 7.5′ field covers all four detectors; each individual detector covers a smaller footprint, and multi-extension files need `-q`. A failed archive request or catalogue row cap is reported directly rather than repeated over a larger area.

The default catalogue is Gaia DR3 in RP with **12 ≤ RP < 20**. `--catalog ps1` uses PS1 DR2 mean astrometry and r-band mean PSF magnitude with **15 ≤ r < 21** by default; `-b` accepts g/r/i/z/y. `--catalog 2mass` uses J with **12 ≤ J < 15.8** by default; H and Ks default to faint limits of 15.1 and 14.3. `-b/--band` selects Gaia G/BP/RP, PS1 g/r/i/z/y, or 2MASS J/H/Ks. `-m FAINT` changes only the faint limit (bright limit 15 for PS1, 12 otherwise); `-m BRIGHT FAINT` changes both. Bounds are inclusive at the bright end and exclusive at the faint end.

PS1 queries the [MAST DR2 mean catalogue](https://catalogs.mast.stsci.edu/docs/panstarrs.html) using `raMean`/`decMean` positions corrected against [Gaia EDR3](https://archive.stsci.edu/contents/newsletters/july-2022/updated-astrometry-for-the-pan-starrs1-catalog). It accepts sources with the corrected-astrometry flag, at least three detections, the GOOD quality bit and no extended-source bit, band coverage, and suitable photometric and positional errors. The current limits are band `QfPerfect ≥ 0.85`, mean PSF magnitude error ≤ 0.3 mag, and each mean coordinate error ≤ 0.3 arcsec. With `DATE-OBS` or `MJD-OBS`, reliable [PS1 proper motions](https://outerspace.stsci.edu/spaces/PANSTARRS/pages/298812412/PS1%2BObjectThin%2Btable%2Bfields) are propagated from `epochMean` (MJD); `pmra` and `pmdec` are in mas/year, with `pmra` interpreted as μRA cos(Dec). QAC requires the PM flag, no bad-PM flag, errors ≤ 2 mas/year on each axis, and propagation uncertainty ≤ 0.1 arcsec. Sources lacking a usable motion or observation date retain `raMean`/`decMean`.

At r≈22, some PS1 detections exist only in the [stack catalogue](https://catalogs.mast.stsci.edu/docs/panstarrs.html). Their `raStack`/`decStack` positions do not have the same Gaia EDR3 correction and can have larger systematic errors than corrected mean positions; several faint stack rows also have missing point-source diagnostics. QAC deliberately excludes stack-only and uncorrected mean sources, so `-m 15 22` is a magnitude *ceiling*, not a promise of complete reference coverage to r=22. A future stack-only mode would need separate uncertainty weighting and validation against corrected mean stars; it cannot be pooled into the current fit as equally precise references.

Gaia initially queries to magnitude 18.5 and extends to 20 for sparse or weak matches. Dense automatic queries narrow the faint limit to avoid the archive's 2,000-row synchronous cap; an explicit range that reaches the cap produces an error. QAC matches the brightest catalogue stars first, then tries progressively fainter groups and refits using all accepted matches. PS1 matching considers up to 200 image detections during initial matching because usable optical references can rank below brighter detections in the image. Gaia results are cached in `~/.cache/qac` (or `$XDG_CACHE_HOME/qac`); `--refresh-catalog` bypasses that cache. PS1 and 2MASS are queried live, and `--refresh-catalog` does not affect them. Gaia proper motions are propagated from J2016.0 to `DATE-OBS` or `MJD-OBS` when available. 2MASS and local positions are not propagated.

A local text catalogue contains `RA DEC [MAG]` on each line, separated by spaces or commas. A `RA DEC` header and `#` comments are allowed. Coordinates may be decimal degrees or colon-separated sexagesimal (RA in hours). For example:

```text
RA DEC MAG
188.733750 31.739444 15.2
12:34:57.10 +31:44:19.0 16.1
```

## Flip, distortion, and resampling

`-f/--flip x`, `y`, or `xy` tries that virtual reversal. Bare `-f` tries normal and all three reversals, selecting the best match. The saved WCS refers to the **original pixel layout**.

For BFOSC the default Y flip is virtual: QAC writes the fitted WCS against the original image pixels.

Without `-d`, QAC fits a linear TAN WCS. Bare `-d/--distorsion` fits degree-2 SIP; `-d 3` fits degree 3. `--distortion` is also accepted. SIP is fitted in detector pixels after the initial match. Degree 2 requires at least 25 well-spread stars, degree 3 at least 40; held-out stars validate the fit. An unsuccessful explicit SIP fit stops without writing output.

`-r/--resample` implies degree-2 SIP unless `-d 3` is given. It maps pixels by bilinear interpolation onto a same-size TAN grid and applies a local pixel-area correction for count-per-pixel images. Blank or uncovered pixels become NaN; output pixels are floating point. Interpolation changes sharp features and noise, and the fixed-size grid can lose a narrow edge strip. The area correction is inappropriate for surface-brightness-per-solid-angle images: use a SIP-header output instead.

## FITS output and frames

One 2-D image is solved directly, including an image extension behind an empty primary HDU. Multiple images or a 3-D stack require `-q/--cube`, except that `-i mods` automatically selects a uniquely named `MERGED` image when no `-q` was supplied. Current Archon MODS files contain four raw image extensions (`IM1`–`IM4`), a status table, and `MERGED` in HDU 6: QAC calls `MERGED` **image frame 5** and writes only that frame. If there is no unique `MERGED` image, give `-q` explicitly; the instrument option never guesses from the extension number.

Bare `-q` solves all image frames; for a stack, it writes independent 2-D extensions. `-q N` solves **only** 1-based image frame N and writes one standalone 2-D primary image; empty and table HDUs are skipped in the count. `-q EXTNAME` selects a uniquely named image extension. Any explicit `-q` overrides MODS automatic selection. A selected plane is also written as one standalone image.

The default file is `INPUT_astro.fits`, or `INPUT_resampled.fits` with `-r`; a single selected frame inserts `_qN`, for example `INPUT_q5_astro.fits`. `-o PATH` chooses an output path and `--overwrite` permits replacing an existing output file. The input itself is never overwritten. Output is written only when all requested frames solve successfully; per-frame match counts, RMS, and fitted scales are printed.

The fitted sky mapping is stored in `CRVAL1/2`, `CRPIX1/2`, `CTYPE1/2` and **CD1_1, CD1_2, CD2_1, CD2_2**; SIP coefficients are included when `-d` is used without `-r`. Old primary and alternate WCS terms, lookup distortions, SIP metadata, and IRAF WCS descriptors are removed. Solved headers have no `PC`, `CDELT`, or `CROTA` cards. Existing `PIXSCALE` and `SECPIX` cards are updated at `CRPIX`. Pointing metadata such as `RA`/`DEC` and `OBJRA`/`OBJDEC` is kept; it is not the fitted WCS. IRAF detector coordinates (`LTV`/`LTM`) are kept when pixels are unchanged and removed for extracted stack planes or resampled images.

## Limits

The matcher needs enough stars near the approximate field centre. With a supplied scale it accepts roughly 25% initial mismatch; the 0.2 fallback searches more widely. Crowded, saturated, nebulous, or strongly distorted images can require other tools. Source detection currently uses a fixed 3-pixel FWHM and 5-sigma threshold. No diagnostic plots are generated.
