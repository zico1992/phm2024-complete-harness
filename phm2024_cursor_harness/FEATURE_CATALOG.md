# Feature catalogue and scientific assumptions

## Observable inputs
`trq_measured`, `oat`, `mgt`, `pa`, `ias`, `np`, `ng`.
`id` is excluded. Labels `faulty` and `trq_margin` are excluded from inference inputs.

The official challenge describes PA as power available and NP as net power.
Published participant papers describe PA as pressure altitude. These are incompatible
definitions. Magnitudes alone do not establish meaning or units. The default
configuration keeps PA opaque. Celsius temperature is an explicit plausible assumption;
set temperature_unit to unknown to suppress all Kelvin-based features if unconfirmed.

## Physics-informed and empirical derived features

| Feature family | Definition | Status |
|---|---|---|
| Temperature ratio | theta=(oat+273.15)/288.15 | Assumes oat in Celsius |
| Temperature rise | mgt-oat | Temperature difference proxy, not combustor heat input |
| Gas/ambient temperature ratio | (mgt+273.15)/(oat+273.15) | Assumes both Celsius; MGT is not turbine inlet temperature |
| Ambient-corrected compressor speed | ng/sqrt(theta) | Uses ambient static rather than inlet total temperature; proxy |
| Thermal loading | (mgt-oat)/abs(ng) | Empirical operating-condition proxy |
| NP/NG ratio and difference | np/ng, np-ng | Empirical only; NP meaning unresolved |
| Torque/NG and torque/MGT | trq_measured/ng, trq_measured/mgt | Empirical loading proxies, not physical efficiencies |
| Interactions | oat*pa, torque*ng, ias squared | Empirical interactions |
| Hover indicator | ias==0 | Operating-state proxy; zero IAS does not prove hovering |
| ISA temperature | T_ISA=288.15-0.0065*h_m | Opt-in PA=pressure altitude with m/ft explicitly set |
| ISA pressure ratio | delta=(T_ISA/288.15)^5.25588 | Tropospheric approximation; PA assumption required |
| Ambient density ratio | sigma=delta/theta | Ideal gas, dry air, ISA pressure approximation |
| Density altitude | (1-sigma^(1/4.25588))*288.15/0.0065 | Derived ISA equivalent altitude, metres |
| Density-normalized torque | torque/sigma | Empirical correction, not exact performance equation |
| TAS proxy | IAS*0.514444/sqrt(sigma) | Knots required; IAS~EAS low-Mach approximation |
| Mach proxy | TAS_proxy/sqrt(1.4*287.05*T_ambient) | Low-Mach approximation, not a measured Mach number |
| Dynamic pressure proxy | 0.5*1.225*(IAS*0.514444)^2 | IAS~EAS assumption, uses sea-level reference density |

The altitude implementation accepts -2000 to 11000 metres, allows negative altitudes,
and fails rather than silently extending the tropospheric formula outside its domain.
There is no fuel flow, compressor pressure ratio, inlet total temperature, calibrated
shaft RPM or engine map. Actual thermodynamic efficiency, physical shaft power,
surge margin and remaining useful life cannot be calculated from these columns alone.

## Statistical features

| Family | Fitting source | Inference calculation |
|---|---|---|
| Robust z score | Per-feature train median and MAD | (x-median)/(1.4826*MAD) |
| Empirical percentile | Sorted fitting values | Fitting ECDF evaluated at x |
| Range flag | Fitting 1st/99th percentiles | Outside fitting central 98% |
| Operating regimes | RobustScaler + MiniBatchKMeans on oat/pa/ias | Distances and one-hot nearest regime |
| Conditional deviations | Fitting regime medians for torque/MGT/NG/NP | Value minus assigned regime median |
| Predicted margin | Group-aware cross-fitting inside model training | Fold prediction on fitting rows; frozen full-fitting model on inference rows |

No temporal rolling means, FFTs or row-difference features: observations are shuffled.
Cluster assignments are operating regimes, never asserted engine identities.
Feature selection ranks classifier permutation AUC drops on the selection partition,
keeps raw measurements, and compares the reduced feature model against its parent.
MI for both targets is reported on training rows as a complementary diagnostic.
Permutation importance on correlated derived features is not a causal explanation.

## Design-torque model

The challenge identity can be rearranged on labelled fitting data:

    design_torque = measured_torque / (1 + true_margin / 100)

The target-torque candidate predicts log(design_torque) from oat, pa and ias with a
degree-two polynomial Ridge model, then derives the predicted margin:

    predicted_margin = 100 * (measured_torque / predicted_design_torque - 1)

Positivity comes from the log link. This is a modelling hypothesis, not a recovered
manufacturer engine map. Reconstructed torque is a supervised training TARGET only.
No true margin is consumed by FeatureBuilder or inference. OOF stacking recomputes
both feature fitting and auxiliary model fitting in each duplicate-group fold.

## Primary references
- Challenge and classification scoring: https://data.phmsociety.org/phm2024-conference-data-challenge/
- Organizer forum defining columns: https://data.phmsociety.org/phmdata-forums/topic/understanding-the-training-dataset/
- Participant design-torque/altitude method: https://papers.phmsociety.org/index.php/phmconf/article/download/4191/2594
- NASA corrected parameters: https://www.grc.nasa.gov/www/k-12/airplane/wcora.html
- NASA corrected speed reference: https://ntrs.nasa.gov/api/citations/19930087116/downloads/19930087116.pdf
- NASA speed of sound: https://www.grc.nasa.gov/WWW/BGH/snddrv.html

Published participant results are context only. The harness does not import their
reported scores, asset reconstructions or fitted rules as its own validation evidence.
