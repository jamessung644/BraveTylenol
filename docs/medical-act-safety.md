# Medical Service Act safety boundary

This implementation note translates the supplied Korean Medical Service Act PDF into product
behavior. It is an engineering control, not a legal opinion or a guarantee of compliance.

## Source and effective-date handling

- Source: Medical Service Act, Act No. 21524, promulgated 2026-04-07.
- The supplied consolidated PDF includes both provisions currently in force and future provisions.
- The displayed amendments concerning Articles 34 and 34-2 state an effective date of 2026-12-24.
  The retrieval prompt therefore requires checking effective dates before answering a legal
  question.

## Product rules mapped to the Act

| Legal boundary | Product behavior |
| --- | --- |
| Article 27: non-medical persons may not perform medical acts | The service provides general information, possible explanations, urgency guidance, and preparation for a clinical visit; it does not confirm or exclude an individual diagnosis or claim to treat a user. |
| Articles 17-18: certificates and prescriptions are acts of licensed clinicians | The service never creates or simulates a prescription/certificate and never chooses an individualized drug, dose, duration, start, stop, switch, or dose adjustment. |
| Article 19: confidentiality duties apply to specified professionals and workers | The service does not claim statutory medical confidentiality, requests no direct identifier, and minimizes repetition of sensitive details. |
| Articles 34 and 34-2: telemedicine and non-face-to-face care are regulated clinical activities | The service never describes chat as telemedicine or a medical consultation and never implies a clinician-patient relationship. |
| Article 56: medical advertising is restricted | The service does not guarantee outcomes, cures, superiority, or safety and does not steer users through misleading promotional claims. |

## Broader health-AI controls

The secondary literature in *Health, AI and Law* adds design risks that are not resolved by a
Medical Service Act disclaimer:

- **Intended use and medical-device boundary:** Korean medical-device analysis emphasizes the
  software's intended use. Product copy, prompts, examples, and evaluation claims must consistently
  frame this as general information and wellness support, not individual detection, diagnosis,
  prediction, monitoring, prevention, or treatment.
- **Meaningful transparency and autonomy:** disclose that the service is AI when a user could mistake
  it for a clinician; state material uncertainty and practical alternatives. Avoid repeating a generic
  disclaimer when it does not help the decision.
- **Human oversight:** never position the model as the sole clinical decision-maker. A licensed
  professional remains responsible for individual diagnosis and treatment decisions.
- **Data governance:** collect the minimum necessary text, discourage direct identifiers, apply
  purpose limitation and retention controls outside the model, and treat de-identification as
  imperfect because linkage can enable re-identification.
- **Equity:** do not infer protected or sensitive traits from proxies, do not treat healthcare spending
  or historical access as medical need, and surface uncertainty when evidence underrepresents a group.
- **Accountability:** preserve versioned prompts, source provenance, test results, incident review,
  and post-deployment monitoring. A model disclaimer does not replace these controls.

The handbook is comparative secondary literature published in 2024. Its Korean-law discussion is
used as a risk-discovery aid, not as proof of the law currently in force. Current Korean statutes,
regulator guidance, and effective dates must be verified independently before release.

## Safe answer pattern

1. Emergency action first when red flags are plausible.
2. State what can be explained as general information.
3. Describe possibilities without selecting a diagnosis.
4. Give concrete care timing and what to tell or ask a licensed professional.
5. For medicines, explain general evidence while leaving individual changes to the prescriber or pharmacist.
6. Retrieve official legal text and state its effective date for legal questions.
7. State population-applicability limits when the evidence is not representative.

## Residual risk

Prompt controls reduce risk but cannot prove legal compliance. Before public deployment, obtain
Korean healthcare counsel review, establish incident monitoring, validate representative answers,
and add operational privacy controls outside the model harness.
