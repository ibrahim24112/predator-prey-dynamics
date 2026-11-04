# Predator–Prey Dynamics

A numerical study of coupled hare–lynx population dynamics. The project compares the classical Lotka–Volterra model with a logistic-prey extension and illustrates how prey self-limitation can produce damped oscillations around a coexistence equilibrium.

## Contents

```text
data/       bundled historical hare–lynx pelt observations
figures/    saved model and data visualizations
src/simulate_predator_prey.py
parameters.csv
diagnostics.csv
```

The simulation uses a transparent fourth-order Runge–Kutta solver, plots time-series and phase-space behavior, and calculates simple descriptive correlations. The historical pelt series is a proxy affected by trapping and trade; it is not a direct population census or a calibrated forecast.

## Setup and use

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
python src/simulate_predator_prey.py
```

The script writes figures and summary CSV files relative to the repository. That path was made portable in this curated copy.

## Attribution and publication note

The supplied report lists multiple team members and cites the historical data source. The original report was not copied into this draft to avoid publishing collaborator details without review. Confirm authorship, attribution, and the dataset's reuse terms before making this repository public. The source was not executed during sorting.