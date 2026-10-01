# Synthetic Hospital: assessed for this project, not used

Assessed 29 September 2026. Verdict: it does not fit the drift-attribution
experiments, so none of our results use it. The reasons are specific to what
those experiments need, not to the quality of the work.

## What it is

*Synthetic Hospital: An Open, Verifiable, Physician-Validated Longitudinal EHR
Benchmark*, Park, Chen and Dettmers (Carnegie Mellon), arXiv 2609.30027,
September 2026. Code and data: <https://github.com/sparkcpark/synthetic_hospital>
(MIT licence; release v1.3, commit `b047385`, 26 September 2026).

- 1,268 fully synthetic patients and 5,602 encounters, built from
  AI-generated USMLE-style question vignettes and an ontology-grounded
  knowledge graph (ICD-10-CM, SNOMED CT, LOINC). No real patient data.
- An Epic-like EHR simulator (FastAPI, FHIR R4, OAuth2/RBAC) and a
  reset/step RL environment for AI agents on four chart tasks: problem-list
  diagnosis, context summarisation, evidence retrieval and imaging indication.
- "Indistinguishable from real": in a blinded review, physicians told real
  from synthetic charts correctly in 53% of 100 judgments, which is not
  distinguishable from chance. This is a test of individual chart realism.

## What we checked in the release

From `patient_profiles.db` and `benchmark_v1.3.db`:

- Encounters by type: outpatient 3,079, ED 1,755, inpatient 383, ICU 296,
  other 89. Dates run 2020 to 2030; they are generated per-patient timelines,
  not calendar periods.
- Mean age 36 (range 0 to 82). The case mix follows exam topics: emergency
  medicine, obstetrics, rare endocrine and rheumatological conditions.
- One narrative note per encounter. Vital signs appear as a single snapshot
  in the text. The "labs" sections are sentences ("Her complete blood count
  is within normal limits"); numeric values exist only where a vignette gave
  one, attached to the encounter, not time-stamped within a stay.
- In the RL environment, agent actions are chart look-ups (open chart, read
  sections, submit an answer). No action changes a patient's course, and
  there is no model of outcomes, deterioration or treatment effect.

## What our experiments need

| Requirement | Why | Synthetic Hospital | MIMIC-IV (E6) |
|---|---|---|---|
| Thousands of admissions | attribution estimators and refit loops use 20,000 patients per environment | 679 inpatient or ICU encounters | on the order of 10^5 adult admissions |
| Time-stamped labs and vitals within a stay | features at a prediction time on the ward | narrative snapshot per visit | `labevents`, charted times |
| Outcome with timing (ICU transfer or death within 12 h) | the deterioration task (W3) | not represented | `icustays`, `admissions.deathtime` |
| Representative case mix and prevalence | Brier-score attribution depends on the joint distribution and on prevalence (Table 5 of the draft) | exam-topic mix, not epidemiological; the blinded test does not address population statistics | real |
| Real temporal drift between periods | the exogenous terms of the decomposition | none | `anchor_year_group` |
| Outcomes under standard care, to layer a simulated alert effect on | semi-synthetic design | charts only, no outcome process | observed outcomes |

Every row that matters for the quantitative claims fails. Using it would
replace real covariates and outcomes with education-derived vignettes. That
is weaker grounding than the MIMIC-IV run the to-do asks for (task E6), which
can be done locally once PhysioNet access is set up (`docs/NEXT_STEPS.md`).

## Where it could still be useful

- A testbed for an LLM-based monitoring or chart-review agent, if the team
  ever adds one: it is open, shareable and has verifiable rewards.
- Prototyping the logging requirements of Section 5 as FHIR resources. The
  simulator implements Observation, ServiceRequest, MedicationRequest,
  Encounter and Condition. It has no RiskAssessment, Flag, Communication or
  Task resources, which the alert exposure and acknowledgement would need.
- A citation in related work if the paper discusses synthetic evaluation
  environments. Not needed for the current claims.

## Alternatives considered

- **Knowledge-grounded trajectory generation with LLM auditing** (arXiv
  2603.06720, March 2026): trained on MIMIC-IV, about 32,000 event types;
  downstream models trained on audited synthetic data match real-data
  performance. It is derived from MIMIC-IV, so for E6 the real data is the
  stronger choice. Its licence and weights were not checked, because arxiv.org
  is blocked from this environment.
- **GHOSTS** (synthetic hospital time series, trained on MIMIC-IV and eICU;
  model, synthetic corpus and code reported as public): the closest to our
  data needs, but it models ICU time series, not ward stays. It is a possible
  open fallback if credentialed access is delayed, after checking the
  corpus licence.

## Sources

- Synthetic Hospital preprint: <https://arxiv.org/abs/2609.30027>
- Code and data: <https://github.com/sparkcpark/synthetic_hospital>
- Announcement: <https://x.com/Tim_Dettmers/status/2103499499742048292>
- Trajectory generation with auditing: <https://arxiv.org/abs/2603.06720>
- GHOSTS: <https://pubmed.ncbi.nlm.nih.gov/42155369/>,
  <https://www.medrxiv.org/content/10.1101/2024.10.29.24316332v1.full>
