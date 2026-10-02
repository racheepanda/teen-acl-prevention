# Teen ACL Risk Analysis

Open-source data and code for an SSAC27 Research Paper abstract.

## What this does
Combines published baseline ACL risk for US high-school athletes with published pooled effect sizes for modifiable factors, then propagates uncertainty by Monte Carlo simulation to report absolute risk reduction (ARR), number needed to treat (NNT), team-seasons per ACL prevented, and 4-season cumulative risk.

This is a **second-order synthesis**: inputs are pooled estimates from existing meta-analyses, not a new search of primary studies.

## Run
```
pip install numpy pandas matplotlib
python src/analysis.py
```

## Limitations
- One warm-up effect is applied to all sports, so sport rankings mirror baseline risk
- Baselines are US high-school varsity seasons; year-round club play carries more exposure.
- Footwear and bracing could not be quantified. High-school football bracing studies exist but are mixed, likely biased, and mostly report knee/MCL injury rather than ACL tears; no reliable ACL benefit has been shown.
- One high-school trial (Pfeiffer et al. 2006) found no significant effect, but observed only six non-contact ACL injuries (3 vs 3), so it is underpowered.
