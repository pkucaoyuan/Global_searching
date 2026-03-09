# Changelog: Global Scheduling of Noise Trajectory Search

Track all modifications to maintain consistency and enable rollback.

---

## Format

```
## [Date] - [Session/Round ID]

### Changed
- [File]: [What changed] (reason)

### Added
- [File]: [What added]

### Removed
- [File]: [What removed]

### Decisions Made
- [Decision]: [Rationale]
```

---

## [2026-03-08] - Initial Creation

### Created
- `paper/main.tex`: Full paper draft with all sections
  - Introduction (sec:intro)
  - Related Work (sec:related)
  - Methodology (sec:method)
  - Experiments (sec:experiments)
  - Conclusion (sec:conclusion)
  - Algorithm 1: Offline-to-Online Budget Scheduling

### Added
- Paper state documentation:
  - `docs/paper_state/global_scheduling/overview.md`
  - `docs/paper_state/global_scheduling/symbols.md`
  - `docs/paper_state/global_scheduling/results.md`
  - `docs/paper_state/global_scheduling/insights.md`
  - `docs/paper_state/global_scheduling/figures_tables.md`
  - `docs/paper_state/global_scheduling/changelog.md`
  - `docs/paper_state/global_scheduling/framing.md`
  - `docs/paper_state/global_scheduling/cross_references.md`
  - `docs/paper_state/global_scheduling/dependencies.md`
  - `docs/paper_state/global_scheduling/abbreviations.md`
  - `docs/paper_state/global_scheduling/review_responses.md`
  - `docs/paper_state/global_scheduling/consistency_log.md`

### Decisions Made
- Use two-column format for ML venue
- Target NeurIPS/ICML/ICLR style
- Focus on Brightness and Compressibility as verifiers (reproducible, no human annotation)
- Use "Online (Ours)" vs "Naive" naming convention

### Consistency Verified
- [x] symbols.md created with all notation
- [x] results.md created with all tables
- [x] figures_tables.md created

---

## [2026-03-08] - Advisor Meeting Revision

### Changed
- `paper/main.tex`: Format changed from two-column to single-column (advisor request)
- `paper/main.tex`: Introduction - added flow model/ODE compatibility explanation
- `paper/main.tex`: Methodology - added verifier clarification ("taken as given")
- `paper/main.tex`: Local search operator - added explicit formulas for x0 prediction and K candidates

### Added
- `paper/main.tex`: New subsection sec:exp-flow (Flow-based model experiments placeholder)
- `paper/main.bib`: Added 6 new references (RL for diffusion, reward-guided sampling, flow models)
- `docs/paper_state/global_scheduling/advice/`: Created advice folder with 7 documents
  - `meeting_notes.md`: Full meeting transcript analysis
  - `01_flow_model_generalization.md`: Flow model compatibility plan
  - `02_verifier_clarification.md`: Verifier explanation plan
  - `03_technical_formulas.md`: Formula addition plan
  - `04_writing_style.md`: Hyphen/GPT style checklist
  - `05_flow_experiment.md`: Flow experiment plan
  - `06_format_and_layout.md`: Format and layout plan

### Decisions Made
- Change to single-column archive format (advisor request)
- Add noise injection explanation for ODE/flow models
- Verifier is "taken as given" - cite prior work
- Add explicit math: x0 prediction, K candidates evaluation
- Add flow-based model experiment (data TBD by user)

### TODO (User Action Required)
- [ ] Run flow-based model experiments and fill in Table 7
- [ ] Review hyphen usage throughout paper
- [ ] Optimize table layout for visual impact

### Consistency Verified
- [ ] symbols.md needs update with new symbols (D_theta, alpha, sigma)
- [ ] results.md needs update with new table
- [ ] figures_tables.md needs update with Table 7

---

## Template for Future Entries

```markdown
## [YYYY-MM-DD] - [Session ID]

### Changed
-

### Added
-

### Removed
-

### Decisions Made
-

### Consistency Verified
- [ ] symbols.md updated
- [ ] results.md updated
- [ ] figures_tables.md updated
```
