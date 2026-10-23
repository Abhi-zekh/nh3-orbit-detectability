ORBIT DETECTABILITY MAIN PIPELINE

=================================



OVERVIEW

\--------

Orbit\_detectability\_main\_pipeline.py evaluates candidate satellite orbits for

NH3 monitoring over the Netherlands. It combines orbit propagation, synthetic

instrument sampling, environmental/retrieval conditions, NH3 emission timing,

and a cell-level detectability calculation in one workflow.



The script was developed for comparative orbit-design analysis. It should be

read together with the comments and function docstrings in the Python file,

which contain the detailed implementation choices.



MAIN WORKFLOW

\-------------

For each configured orbit case, the pipeline broadly performs the following:



1\. Propagates the spacecraft trajectory with TudatPy using a perturbed dynamic

&#x20;  model.

2\. Uses a coarse-to-fine strategy: a 10 s full propagation identifies possible

&#x20;  Netherlands overpasses, followed by fine propagation within candidate

&#x20;  windows at the synthetic instrument scanline cadence.

3\. Generates synthetic cross-track pixel footprints using spherical-Earth

&#x20;  viewing geometry and assigns pixel centres to the LOTOS-EUROS analysis grid.

4\. Samples observation conditions for each retained pixel:

&#x20;    - solar zenith angle (SZA),

&#x20;    - thermal contrast (TC),

&#x20;    - total cloud cover (TCC),

&#x20;    - diurnal NH3 emission factor,

&#x20;    - seasonal NH3 emission factor.

5\. Converts these quantities into retrieval-quality and signal-effective

&#x20;  observation weights.

6\. Aggregates pixel observations by regional pass and LOTOS-EUROS grid cell and

&#x20;  calculates revisit statistics.

7\. Loads CAMS-REG NH3 emissions and remaps Netherlands emissions to the

&#x20;  LOTOS-EUROS grid using area-overlap weighting.

8\. Combines emissions, sampling, retrieval conditions and assumed retrieval

&#x20;  uncertainty to estimate NH3 detectability time for each sampled emitting

&#x20;  grid cell.

9\. Produces diagnostic plots, ablation results, summary files and optional

&#x20;  multi-orbit comparison products.



IMPORTANT MODEL CONVENTIONS

\---------------------------

\- Orbit start/end times are specified in UTC; Tudat propagation uses SPICE

&#x20; ephemeris time internally.

\- The spacecraft model uses fixed mass, drag/radiation coefficients and

&#x20; orientation-averaged reference areas. No attitude propagation is performed.

\- Earth gravity is modelled to degree/order 5, with aerodynamic drag, solar

&#x20; radiation pressure, and third-body gravity included.

\- The active synthetic instrument configuration uses 256 across-track pixels

&#x20; with a nominal 2 km nadir pixel spacing (approximately 512 km at nadir).

\- One synthetic scanline is generated at every fine-track epoch. Therefore,

&#x20; FINE\_STEP\_SECONDS is both a propagation-output cadence and an instrument

&#x20; sampling assumption.

\- Pixels are assigned to the analysis grid using their centre coordinates;

&#x20; fractional footprint/grid-cell overlap is not used for observation counting.

\- A pass\_id represents one regional fine-propagation/overpass window, not a

&#x20; globally numbered orbital revolution.

\- Environmental quantities are sampled at the pixel location and observation

&#x20; time according to the methods implemented in the corresponding functions.

\- Spatial CAMS-REG emissions use the 2022 inventory as a static baseline,

&#x20; while the orbit simulation and diurnal/seasonal temporal factors represent

&#x20; 2019.

\- Only emissions assigned to the Netherlands in the CAMS country metadata are

&#x20; retained for the detectability analysis.



OBSERVATION COUNTS

\------------------

Three related quantities are used frequently:



N\_raw

&#x20;   Raw synthetic pixel observations.



N\_eff

&#x20;   Retrieval-quality-effective observations after SZA, thermal-contrast and

&#x20;   cloud weighting.



N\_sig\_eff

&#x20;   Signal-effective observations after retrieval-quality weighting and the

&#x20;   squared combined diurnal/seasonal NH3 emission factor.



N\_eff and N\_sig\_eff are weighted sample-equivalent quantities and therefore do

not need to be integers.



DETECTABILITY

\-------------

The final detectability metric combines:

\- grid-cell NH3 emissions,

\- an assumed plume/source length scale and wind speed,

\- instrument and background column uncertainty,

\- grid-cell revisit time, and

\- signal-effective observations obtained per pass.



The final orbit-ranking formulation is the paper-style signal-effective metric (The logic followed in Jacobs et al paper for methane detection)

stored as:



&#x20;   time\_to\_detect\_days\_paper\_signalEffN



The calculation assumes independent observations so that accumulated SNR grows

with the square root of the number of effective observations. Other raw,

quality-effective and observation-rate formulations are retained primarily for

comparison and diagnostics.



CONFIGURATION BEFORE RUNNING

\----------------------------

Before using the script on another system, check the following sections/values:



\- orbit\_cases:

&#x20;   Add or uncomment the orbit case(s) to be propagated.



\- Environmental input paths:

&#x20;   HOURLY\_TC\_DIR

&#x20;   ECMWF\_TCC\_DIR

&#x20;   DIURNAL\_FACTOR\_PATH

&#x20;   SEASONAL\_FACTOR\_PATH



\- CAMS-REG emission path:

&#x20;   emis\_path inside run\_orbit\_case().



\- Instrument settings:

&#x20;   GEOM\_MODE, NADIR\_PIXEL\_M, N\_ACROSS, FINE\_STEP\_SECONDS.



\- Analysis grid settings:

&#x20;   GRID\_WEST, GRID\_SOUTH, GRID\_DLON, GRID\_DLAT, GRID\_NX, GRID\_NY.



\- Plot/output controls:

&#x20;   GENERATE\_RESULTS\_PLOTS

&#x20;   GENERATE\_SINGLE\_ORBIT\_FIGURES

&#x20;   GENERATE\_MULTI\_ORBIT\_RESULTS

&#x20;   SAVE\_RESULTS\_PNG / SAVE\_RESULTS\_PDF



The input paths in the current file point to the environment in which the

thesis analysis was performed and will normally need to be changed when running

elsewhere.



MAIN PYTHON DEPENDENCIES

\------------------------

The pipeline uses:



\- TudatPy

\- NumPy

\- pandas

\- xarray

\- netCDF4

\- Matplotlib



Parquet output also requires an appropriate pandas Parquet engine in the Python

environment.



OUTPUTS

\-------

Results are written under:



&#x20;   ./outputs/<orbit-specific-directory>/



Depending on the enabled options, outputs include:



\- daily CSO-like synthetic geometry NetCDF files;

\- fine ground-track CSV/Parquet files;

\- cell-level sampling summaries;

\- remapped NH3 emission tables;

\- cell-level observability/detectability results;

\- environmental-data and sampling diagnostics;

\- ablation-analysis CSV/Parquet files;

\- figures and diagnostic plots;

\- comparison\_cache.pkl for later multi-orbit comparison. 



\- timestamped orbit-comparison summaries.



When multiple orbit cases are run in the same execution and multi-orbit output

is enabled, final comparison products are written under:



&#x20;   ./outputs/comparison/results\_figures/



RUNNING THE SCRIPT

\------------------

After configuring the orbit case(s) and input paths:



&#x20;   python Orbit\_detectability\_main\_pipeline.py



Full-year runs can be computationally expensive because fine propagation and

pixel generation are performed around every candidate regional overpass.



NOTES FOR FUTURE USERS

\----------------------

\- The code contains both final-result calculations and additional verification,

&#x20; sensitivity and diagnostic outputs. Not every generated plot is required for

&#x20; the final detectability result.

\- Lots of diagnostic plots can also be found for re-gridding analysis. 

\- The ablation analysis reuses the already generated pixel-level dataset and

&#x20; changes the enabled quality/emission weights; it does not rerun the orbit.

\- Missing TC/TCC and temporal-factor behaviour is handled explicitly in the

&#x20; relevant sampling functions and should be checked before changing input

&#x20; datasets.

\- If modifying physical assumptions, sampling cadence, quality thresholds or

&#x20; instrument geometry, treat the resulting run as a new model configuration

&#x20; rather than directly mixing it with results produced by the original setup.



