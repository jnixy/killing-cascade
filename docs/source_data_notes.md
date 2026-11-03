# Notes on the California source data

The public URSUS files from CA DOJ have a few quirks. If you start from the raw downloads instead of `data/processed/`, you'll need to handle each of them. `analysis/scripts/study2_california/01_prepare_data.py` handles them all.

## 2022 armed status contradicts the recorded weapon

In the public 2022 civilian-officer extract, and only in that year, 99 civilians in our gunshot sample are coded `Confirmed_Armed = False` even though a specific weapon is recorded in `Confirmed_Armed_Weapon`. We confirmed this against a fresh download and against CA DOJ's own 2022 annual report. Every other year is internally consistent.

We treat the recorded weapon as authoritative. With that rule, the share of 2022 gunshot cases classified as armed is 81.4%, in line with surrounding years. Taken at face value, the flag gives 48.6%.

## Column names and labels change across reporting eras

- **Incident ID:** `incident_id` (2016–2021), `incident_code` (2022), `useofforceincidentid` (2023–2025).
- **Agency ORI:** the 2023–2024 civilian-officer files (the CIBRS era) name the column `ORINumber`. Other years use `ORI`. If you miss this, agency identifiers are blank for 2023–2024.
- **Force-type labels:** 2016 uses plain English ("Discharge of firearm (miss)", "Threat of firearm"). Other years use underscore codes (`DISCHARGE_OF_FIREARM_MISS`). Match both forms, or 2016 civilians who fired at or threatened officers will be missed.
- **Wound regions:** `received_force_location` is a multiple-response list of 11 body regions, and the spelling changes by era ("Rear upper torso/back", `rear_upper_back`, `REAR_UPPER_TORSO`). Map tokens by exact lookup after normalizing. Substring matching is unsafe here because some region names contain others (for example, "EAR" in "REAR").
- **Whitespace:** 2023–2024 categorical fields carry leading and trailing whitespace. Strip it before matching.
