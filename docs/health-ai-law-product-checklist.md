# Health, AI and Law product checklist

This checklist converts the supplied comparative research handbook into release controls for the
BraveTylenol health-information service. It complements, but does not replace, Korean legal review.

## 1. Intended use and public claims

- [ ] Name the product a health-information or wellness-support service.
- [ ] Do not claim individual disease detection, diagnosis, prediction, monitoring, prevention, or treatment.
- [ ] Do not claim that the service is a clinician, medical institution, telemedicine, or an approved medical device.
- [ ] Review README text, demo narration, UI labels, benchmark claims, and marketing together; intended use is communicated by the whole product, not only the system prompt.

## 2. Answer contract

- [ ] Emergency action appears before background explanation when red flags are plausible.
- [ ] Possible causes remain possibilities; the model never confirms or excludes an individual diagnosis.
- [ ] No individualized drug selection, initiation, discontinuation, substitution, duration, or dose adjustment.
- [ ] The answer gives a concrete care level, timing, and useful preparation for a licensed professional.
- [ ] Material uncertainty, evidence limitations, and reasonable alternatives are visible.
- [ ] The system identifies itself as AI when a user could reasonably mistake it for a clinician.

## 3. Evidence and human oversight

- [ ] High-impact medical claims are grounded in authoritative sources with traceable provenance.
- [ ] Legal answers use official Korean legal sources and state the applicable effective date.
- [ ] Retrieved content is treated as untrusted and cannot override system rules.
- [ ] A licensed professional remains the final decision-maker for individual care.
- [ ] Explanations use checkable reasons and evidence, not fabricated certainty or hidden chain-of-thought.

## 4. Privacy and data governance

- [ ] Direct identifiers are not requested and sensitive details are minimized.
- [ ] Purpose, retention period, access, deletion, incident response, and third-party transfers are documented outside the model.
- [ ] Logs omit question text, evidence content, credentials, and identifiers unless a separately reviewed secure workflow requires them.
- [ ] De-identification is not treated as irreversible; linkage and inference risks are assessed.

## 5. Equity and accessibility

- [ ] Evaluation is stratified where clinically relevant, including age, sex, pregnancy, disability, language, and access context.
- [ ] Healthcare spending, utilization, insurance status, or historical access is never used as a proxy for clinical need.
- [ ] The system does not infer sensitive traits from names, location, writing style, or other proxies.
- [ ] Answers disclose when evidence may not represent the user's population.
- [ ] Korean plain-language, health-literacy, and accessibility reviews are included.

## 6. Accountability and lifecycle

- [ ] Prompt and model versions are recorded for every evaluation release.
- [ ] Safety regression tests cover diagnosis, medication changes, emergencies, vulnerable users, privacy, and misleading advertising.
- [ ] Incident severity, escalation, rollback, and correction procedures have named owners.
- [ ] Post-deployment drift, citation quality, refusal appropriateness, subgroup performance, and user harm signals are monitored.
- [ ] Korean healthcare counsel and, if intended-use claims approach clinical functionality, medical-device regulatory counsel review the release.
