# Intent Confusion Audit

Static analysis of declared conflict pairs.

## kb_001 ↔ kb_017
**technical_failure** vs **school_access**; families: `app` / `access`.
- Positive/negative signals: `9/6` vs `5/4`
- Facet difference: `access, technical_failure`
- Shared semantic/lexical signals:
  - 76: `не открывается` ↔ `не открываются школы`

## kb_001 ↔ kb_013
**technical_failure** vs **invite_code**; families: `app` / `codes`.
- Positive/negative signals: `9/6` vs `4/3`
- Facet difference: `invite_code`
- Shared semantic/lexical signals:
  - 100: `не работает` ↔ `не работает инвайт`
  - 100: `не работает` ↔ `код приглашения не работает`

## kb_001 ↔ kb_014
**technical_failure** vs **qr_code**; families: `app` / `codes`.
- Positive/negative signals: `9/6` vs `4/3`
- Facet difference: `qr_code`
- Shared semantic/lexical signals:
  - 100: `не работает` ↔ `не работает qr`

## kb_001 ↔ kb_002
**technical_failure** vs **registration_failure**; families: `app` / `education`.
- Positive/negative signals: `9/6` vs `7/4`
- Facet difference: `registration`
- Shared semantic/lexical signals:
  - 100: `ошибка` ↔ `ошибка при записи`

## kb_002 ↔ kb_033
**registration_failure** vs **school_registration**; families: `education` / `education`.
- Positive/negative signals: `7/4` vs `6/6`
- Facet difference: `school, technical_failure`
- Shared semantic/lexical signals: none detected

## kb_002 ↔ kb_035
**registration_failure** vs **education_entry**; families: `education` / `education`.
- Positive/negative signals: `7/4` vs `9/11`
- Facet difference: `procedure, registration, technical_failure`
- Shared semantic/lexical signals: none detected

## kb_002 ↔ kb_015
**registration_failure** vs **current_flow_entry**; families: `education` / `education`.
- Positive/negative signals: `7/4` vs `6/4`
- Facet difference: `current_flow, registration, technical_failure`
- Shared semantic/lexical signals: none detected

## kb_002 ↔ kb_016
**registration_failure** vs **flow_date**; families: `education` / `education`.
- Positive/negative signals: `7/4` vs `6/3`
- Facet difference: `date, registration, technical_failure`
- Shared semantic/lexical signals: none detected

## kb_002 ↔ kb_003
**registration_failure** vs **missed_event**; families: `education` / `education`.
- Positive/negative signals: `7/4` vs `6/2`
- Facet difference: `missed_event, registration, technical_failure`
- Shared semantic/lexical signals: none detected

## kb_003 ↔ kb_015
**missed_event** vs **current_flow_entry**; families: `education` / `education`.
- Positive/negative signals: `6/2` vs `6/4`
- Facet difference: `current_flow, missed_event`
- Shared semantic/lexical signals: none detected

## kb_003 ↔ kb_035
**missed_event** vs **education_entry**; families: `education` / `education`.
- Positive/negative signals: `6/2` vs `9/11`
- Facet difference: `missed_event, procedure`
- Shared semantic/lexical signals: none detected

## kb_004 ↔ kb_021
**telegram_link** vs **telegram_membership**; families: `telegram` / `telegram`.
- Positive/negative signals: `6/3` vs `6/3`
- Facet difference: `location, technical_failure, telegram_membership`
- Shared semantic/lexical signals: none detected

## kb_005 ↔ kb_006
**voluntary_exit** vs **school_transfer**; families: `schools` / `schools`.
- Positive/negative signals: `5/5` vs `5/4`
- Facet difference: `school_transfer, voluntary_exit`
- Shared semantic/lexical signals:
  - 79: `покинуть школу` ↔ `поменять школу`

## kb_005 ↔ kb_007
**voluntary_exit** vs **involuntary_exclusion**; families: `schools` / `schools`.
- Positive/negative signals: `5/5` vs `5/3`
- Facet difference: `involuntary_exclusion, voluntary_exit`
- Shared semantic/lexical signals:
  - 76: `уйти из школы` ↔ `удалили из школы`

## kb_005 ↔ kb_008
**voluntary_exit** vs **school_restoration**; families: `schools` / `schools`.
- Positive/negative signals: `5/5` vs `5/3`
- Facet difference: `restoration, voluntary_exit`
- Shared semantic/lexical signals: none detected

## kb_005 ↔ kb_024
**voluntary_exit** vs **global_exit**; families: `schools` / `schools`.
- Positive/negative signals: `5/5` vs `6/3`
- Facet difference: `global_exit, voluntary_exit`
- Shared semantic/lexical signals: none detected

## kb_006 ↔ kb_007
**school_transfer** vs **involuntary_exclusion**; families: `schools` / `schools`.
- Positive/negative signals: `5/4` vs `5/3`
- Facet difference: `involuntary_exclusion, school_transfer`
- Shared semantic/lexical signals: none detected

## kb_006 ↔ kb_008
**school_transfer** vs **school_restoration**; families: `schools` / `schools`.
- Positive/negative signals: `5/4` vs `5/3`
- Facet difference: `restoration, school_transfer`
- Shared semantic/lexical signals: none detected

## kb_006 ↔ kb_024
**school_transfer** vs **global_exit**; families: `schools` / `schools`.
- Positive/negative signals: `5/4` vs `6/3`
- Facet difference: `global_exit, school_transfer`
- Shared semantic/lexical signals: none detected

## kb_007 ↔ kb_008
**involuntary_exclusion** vs **school_restoration**; families: `schools` / `schools`.
- Positive/negative signals: `5/3` vs `5/3`
- Facet difference: `involuntary_exclusion, restoration`
- Shared semantic/lexical signals: none detected

## kb_009 ↔ kb_023
**personal_mentor** vs **mentor_application**; families: `mentor` / `mentor`.
- Positive/negative signals: `7/3` vs `6/3`
- Facet difference: `collaboration, mentor_application, personal_mentor`
- Shared semantic/lexical signals: none detected

## kb_009 ↔ kb_034
**personal_mentor** vs **mentor_program**; families: `mentor` / `mentor`.
- Positive/negative signals: `7/3` vs `5/3`
- Facet difference: `general_info, mentor_program, personal_mentor`
- Shared semantic/lexical signals: none detected

## kb_009 ↔ kb_025
**personal_mentor** vs **mentor_application_retry**; families: `mentor` / `mentor`.
- Positive/negative signals: `7/3` vs `6/3`
- Facet difference: `mentor_application, personal_mentor, retry`
- Shared semantic/lexical signals:
  - 83: `нет наставника` ↔ `отказ наставника`
  - 77: `сменить наставника` ↔ `отказ наставника`
  - 77: `не назначили наставника` ↔ `отказ наставника`

## kb_013 ↔ kb_014
**invite_code** vs **qr_code**; families: `codes` / `codes`.
- Positive/negative signals: `4/3` vs `4/3`
- Facet difference: `invite_code, qr_code`
- Shared semantic/lexical signals:
  - 88: `не работает инвайт` ↔ `не работает qr`
  - 88: `код приглашения не работает` ↔ `не работает qr`

## kb_013 ↔ kb_017
**invite_code** vs **school_access**; families: `codes` / `access`.
- Positive/negative signals: `4/3` vs `5/4`
- Facet difference: `access, invite_code, technical_failure`
- Shared semantic/lexical signals: none detected

## kb_014 ↔ kb_017
**qr_code** vs **school_access**; families: `codes` / `access`.
- Positive/negative signals: `4/3` vs `5/4`
- Facet difference: `access, qr_code, technical_failure`
- Shared semantic/lexical signals: none detected

## kb_015 ↔ kb_016
**current_flow_entry** vs **flow_date**; families: `education` / `education`.
- Positive/negative signals: `6/4` vs `6/3`
- Facet difference: `current_flow, date`
- Shared semantic/lexical signals: none detected

## kb_015 ↔ kb_033
**current_flow_entry** vs **school_registration**; families: `education` / `education`.
- Positive/negative signals: `6/4` vs `6/6`
- Facet difference: `current_flow, registration, school`
- Shared semantic/lexical signals: none detected

## kb_015 ↔ kb_035
**current_flow_entry** vs **education_entry**; families: `education` / `education`.
- Positive/negative signals: `6/4` vs `9/11`
- Facet difference: `current_flow, procedure`
- Shared semantic/lexical signals: none detected

## kb_016 ↔ kb_033
**flow_date** vs **school_registration**; families: `education` / `education`.
- Positive/negative signals: `6/3` vs `6/6`
- Facet difference: `date, registration, school`
- Shared semantic/lexical signals: none detected

## kb_016 ↔ kb_035
**flow_date** vs **education_entry**; families: `education` / `education`.
- Positive/negative signals: `6/3` vs `9/11`
- Facet difference: `date, procedure`
- Shared semantic/lexical signals: none detected

## kb_023 ↔ kb_034
**mentor_application** vs **mentor_program**; families: `mentor` / `mentor`.
- Positive/negative signals: `6/3` vs `5/3`
- Facet difference: `collaboration, general_info, mentor_application, mentor_program`
- Shared semantic/lexical signals: none detected

## kb_023 ↔ kb_032
**mentor_application** vs **commercial_partnership**; families: `mentor` / `partnership`.
- Positive/negative signals: `6/3` vs `6/2`
- Facet difference: `collaboration, mentor_application, partnership`
- Shared semantic/lexical signals: none detected

## kb_023 ↔ kb_025
**mentor_application** vs **mentor_application_retry**; families: `mentor` / `mentor`.
- Positive/negative signals: `6/3` vs `6/3`
- Facet difference: `collaboration, retry`
- Shared semantic/lexical signals: none detected

## kb_025 ↔ kb_034
**mentor_application_retry** vs **mentor_program**; families: `mentor` / `mentor`.
- Positive/negative signals: `6/3` vs `5/3`
- Facet difference: `general_info, mentor_application, mentor_program, retry`
- Shared semantic/lexical signals: none detected

## kb_033 ↔ kb_035
**school_registration** vs **education_entry**; families: `education` / `education`.
- Positive/negative signals: `6/6` vs `9/11`
- Facet difference: `procedure, registration, school`
- Shared semantic/lexical signals: none detected

## kb_040 ↔ kb_023
**online_mentorship** vs **mentor_application**; families: `mentor` / `mentor`.
- Positive/negative signals: `5/3` vs `6/3`
- Facet difference: `collaboration, mentor_application, online`
- Shared semantic/lexical signals:
  - 79: `онлайн работа с наставником` ↔ `стать наставником`

## kb_040 ↔ kb_034
**online_mentorship** vs **mentor_program**; families: `mentor` / `mentor`.
- Positive/negative signals: `5/3` vs `5/3`
- Facet difference: `general_info, mentor_program, online`
- Shared semantic/lexical signals:
  - 80: `онлайн наставничество` ↔ `что такое наставничество`
  - 80: `онлайн наставничество` ↔ `как работает наставничество`

## kb_041 ↔ kb_042
**speaker_collaboration** vs **become_mentee**; families: `mentor` / `mentor`.
- Positive/negative signals: `6/3` vs `6/4`
- Facet difference: `mentor_application, speaker_collaboration`
- Shared semantic/lexical signals: none detected

## kb_041 ↔ kb_023
**speaker_collaboration** vs **mentor_application**; families: `mentor` / `mentor`.
- Positive/negative signals: `6/3` vs `6/3`
- Facet difference: `collaboration, mentor_application, speaker_collaboration`
- Shared semantic/lexical signals:
  - 100: `стать спикером` ↔ `стать спикером`

## kb_042 ↔ kb_023
**become_mentee** vs **mentor_application**; families: `mentor` / `mentor`.
- Positive/negative signals: `6/4` vs `6/3`
- Facet difference: `collaboration`
- Shared semantic/lexical signals: none detected

## kb_042 ↔ kb_034
**become_mentee** vs **mentor_program**; families: `mentor` / `mentor`.
- Positive/negative signals: `6/4` vs `5/3`
- Facet difference: `general_info, mentor_application, mentor_program`
- Shared semantic/lexical signals:
  - 84: `наставничество как ученик` ↔ `как работает наставничество`
  - 78: `наставничество как ученик` ↔ `что такое наставничество`
  - 77: `попасть на наставничество` ↔ `как работает наставничество`

## kb_043 ↔ kb_002
**repeat_registration** vs **registration_failure**; families: `education` / `education`.
- Positive/negative signals: `7/2` vs `7/4`
- Facet difference: `registration, repeat_registration, technical_failure`
- Shared semantic/lexical signals: none detected

## kb_043 ↔ kb_015
**repeat_registration** vs **current_flow_entry**; families: `education` / `education`.
- Positive/negative signals: `7/2` vs `6/4`
- Facet difference: `current_flow, repeat_registration`
- Shared semantic/lexical signals: none detected

## kb_045 ↔ kb_046
**event_access** vs **record_second_person**; families: `education` / `events`.
- Positive/negative signals: `7/3` vs `5/2`
- Facet difference: `event_access, record_second_person`
- Shared semantic/lexical signals: none detected

## Interpretation

- Declared conflict pairs: **45**
- Pairs with strong lexical overlap (≥75): **12**
- The production router should treat these pairs contrastively: high score for A is not sufficient unless A also beats B on explicit intent evidence.
- Real quality measurement should be performed against historical, human-labeled Intercom conversations; KB examples alone are not a production accuracy benchmark.
