# Dataset Facts

## Section 1: Public Training Dataset

| Field | Value |
|-------|-------|
| **Source** | NHANES (National Health and Nutrition Examination Survey), CDC |
| **Years** | 1999–2000 through 2017–2020 (Pre-pandemic) |
| **Total records after preprocessing** | ~50,000 |
| **Unique individuals** | ~50,000 (cross-sectional; one record per individual per survey cycle) |
| **Age range** | `[X–Y]` years |
| **Sex distribution** | `[X% Female, Y% Male]` |
| **Race/ethnicity distribution** | Nationally representative with oversampling of Hispanic, non-Hispanic Black, non-Hispanic Asian, and older adult populations |
| **Number of biomarkers used** | 42 (20 CBC + 22 CMP) |
| **Missingness rate (mean across features)** | `[X%]` |
| **Public availability** | Yes — [NHANES Data](https://wwwn.cdc.gov/nchs/nhanes/default.aspx) |

### NHANES Survey Cycles Used

| Cycle | Included |
|-------|----------|
| 1999–2000 | Yes |
| 2001–2002 | Yes |
| 2003–2004 | Yes |
| 2005–2006 | Yes |
| 2007–2008 | Yes |
| 2009–2010 | Yes |
| 2011–2012 | Yes |
| 2013–2014 | Yes |
| 2015–2016 | Yes |
| 2017–2018 | Yes |
| 2017–2020 (Pre-pandemic) | Yes |

### Biomarker Sources

Training data was constructed by merging the following NHANES data files per survey cycle:

- **Laboratory — Biochemistry Profile (BIOPRO)**: Liver enzymes, metabolic markers, proteins
- **Laboratory — Complete Blood Count (CBC)**: Hematologic parameters
- **Demographics**: Age (RIDAGEYR), sex, race/ethnicity

All files are publicly available as SAS Transport (XPT) format from the NHANES website.

---

## Section 2: Internal Validation Dataset

| Field | Value |
|-------|-------|
| **Source** | De-identified clinical blood panel records from Healome partner clinics |
| **Total records** | ~1.5M blood-test records |
| **Unique individuals** | `[X — fill in the real number]` |
| **Average records per individual** | `[X]` |
| **Time span** | `[start year – end year]` |
| **Median follow-up interval** | `[X months/years]` |
| **Geographic coverage** | `[e.g., US — specify regions if possible]` |
| **Clinical settings** | `[e.g., longevity/wellness clinics, primary care]` |
| **Biomarker overlap with NHANES model** | `[X of 42]` biomarkers available |
| **Repeated measurements available** | Yes — `[X%]` of individuals have ≥2 records |
| **Public availability** | No — patient privacy and governance constraints |
| **De-identification** | `[brief statement on method, e.g., HIPAA Safe Harbor]` |
| **Governance** | `[brief statement — IRB, BAA, HIPAA compliance as applicable]` |

### What I can share from the internal dataset

- Aggregate validation metrics (see [BENCHMARKS.md](BENCHMARKS.md))
- Summary statistics (above)
- Short-interval test-retest variance estimates (if available)

### What I cannot share

- Individual-level data
- Clinic-identifying information
- Raw biomarker distributions (protected under data governance agreements)

---

**Why this matters:** This document exists so that anyone evaluating this work can understand exactly what data was used, how it was collected, and what constraints apply to the validation claims. A researcher reading this should immediately understand what data is available, what has been done with it, and what constraints apply.
