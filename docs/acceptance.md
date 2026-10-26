# Quality gates

## Initial ML milestone

- [x] Official data source and checksums recorded; all 62 labels and orientation grid visually inspected.
- [x] Stratified partitions are deterministic, exhaustive and disjoint; test examples are not used for tuning.
- [x] Owner's real-data bounded smoke run completed with finite losses and a reloadable checkpoint; hash and seeded inference verified.
- [ ] A research baseline produces recognizable requested-class prior samples across the alphabet.
- [ ] Class correctness, ambiguity and diversity are reviewed; selected failures are published alongside attractive examples.
- [ ] A composed canonical passage passes the initial visual check before polished product UI work.

Synthetic unit-test fixtures verify code only. Smoke training verifies execution only. Neither may be presented as an evaluated handwriting model.

## Provisional release targets from the revised plan

Freeze targets and rubric after the baseline and before final testing. They are not measured results:

- At least 100 fixed-seed generated samples/class; requested-class accuracy at least 99% aggregate and 97% for every class.
- Human review of at least 20 samples/class and evaluator disagreements; separately validate any classifier on real held-out characters.
- At least 10 passages of 500-1,000 characters, at most 1% incorrect/unreadable alphanumeric glyphs, no missing supported text or renderer clipping.
- Review repeated-letter grids and at least five full pages for coherent thickness, scale and slant. Character accuracy does not certify page coherence.
- Pixel-reproducible output for identical input/settings/seed/model/renderer on the pinned CPU runtime.
- Service targets, overload behavior, privacy, multi-page preview and PNG/ZIP export pass before deployment.

At 99% character correctness, a 1,000-character result can still have about ten errors. The demo must disclose this limitation. Failed gates require a scope/model decision; do not substitute a font or mock result.
