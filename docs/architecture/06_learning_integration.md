# Learning

The coordinator records measured room and outdoor conditions and the curve offset. Learning can
contribute heat only when a forecast at the requested horizon has passed independent validation.
The measured temperature trends remain available without a learned forecast.

## Room forecasts

`AdaptiveThermalModel` keeps hourly observations. `thermal_forecast.py` fits a separate model for
each supported horizon. It predicts the change in room temperature from the current room, the
mean forecast outdoor temperature and the planned mean curve offset. The fitted intercept
includes the pump's native heating and unobserved gains: offset zero does not mean zero heat.
These coefficients are empirical temperature responses, not measurements of heat loss or capacity.

Only a continuous hourly suffix can teach the model. Time is measured on the absolute timeline
across clock changes. Too little elapsed coverage, gaps, non-finite observations, a flat room,
constant inputs or a rank-deficient fit leave the model untrusted.

The earlier half of the observations trains the model; the later half evaluates it. Training
origins whose target crosses the split are excluded. Each horizon needs a full day of later
validation origins. The score is the improvement in mean squared error over holding the current
room temperature. A persistence error below the sensor's resolution supplies no evidence of
useful forecast skill. The model is not refitted using its validation targets.

## When a learned forecast may request heat

`ThermalStatePredictor` needs a fresh continuous day of measured room history and a recent learned
model. The requested horizon must score at least `LEARNING_CONFIDENCE_THRESHOLD`. A high score at
another horizon does not authorize the request. The current room, forecast temperatures and
planned offsets must all stay inside the training ranges. Missing forecast hours abstain.

Without that evidence the diagnostic trajectory holds the current measured room temperature and
reports zero forecast confidence. It does not invent cooling from an unmeasured heat loss or a
configured thermal-mass setting.

Pre-heating uses only supported horizons, allows for the largest validation error, and sizes its
raise from the fitted offset response. It stays within both the observed offset range and the
existing climate-dependent caps. The normal safety, comfort, peak, immersion and compressor guards
still apply to the final offset. The model's summary confidence reports its best validated horizon;
individual forecasts carry their own scores.

The owner's recorded October–November history does not yet validate a useful learned model under
these gates. The fallback matches persistence; this is not evidence of a control benefit or of
savings. A model that predicts one measured home is not a calibrated model of every installation.

## Other learning and settings

The predictor's measured indoor and outdoor trends feed damping, peak protection, weather preheat
and airflow control. Their history-confidence measure describes observed coverage, not future
forecast accuracy. The weather-pattern learner records a daily pattern and can adjust the weather
compensation margin when its date-based comparison identifies unusual weather.

Learning preserves the owner's thermal mass and insulation settings. Thermal mass still sets the
price look-ahead and the forecast horizon. Diagnostic heat-loss, decay and response-type estimates
do not replace those settings or enter the fitted temperature equation. The unused legacy
preheat-sizing equation was removed.

## Storage

`Store(hass, 1, "effektguard_learned_data")`: saved at most hourly when something changed, the
predictor every 5 minutes, and on shutdown. Keys: `version`, `last_updated`, `last_offset`,
`thermal_model` (the configured calibration and full bounded hourly observation window),
`thermal_predictor`, `weather_patterns`, `weather_summary`, `dhw_state`.

A restart restores the adaptive history and its observation timestamp, the predictor's snapshots
and the weather patterns. Learned parameters and confidence are recomputed from the same evidence;
the owner's current thermal mass and insulation settings still take precedence. Old summary-only
stores remain readable, but cannot restore observations that were never saved.
