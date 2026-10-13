# Project Skills and Operating Rules

## Scope and repository boundary
- This project is the repository `super-resolution-waveforms`.
- Only access files inside this repository unless the user explicitly grants permission to use files outside it.
- Do not browse, read, modify, or rely on data from outside the repository without explicit approval.
- Keep all work scoped to the repository root and the files needed for the task.

## Domain context
- The work focuses on ultrasound data and super-resolution waveform processing.
- This project is used for a paper comparing simulations with experimental measurements.
- Experimental data were acquired using a Verasonics Vantage system.
- Treat the analysis as scientific and reproducible research code, not ad hoc scripting.

## Coding expectations
- Annotate code whenever edits are made.
- Add brief, clear comments where the logic is non-obvious or domain-specific.
- Prefer explanatory comments over verbose prose.
- Preserve MATLAB and scientific-computing conventions already used in the repository.
- Use descriptive variable names when needed, but avoid unnecessary refactoring.

## Minimal-change policy
- Change as little code as possible to solve the requested problem.
- Avoid broad cleanups, renames, or rewrites unrelated to the task.
- Prefer surgical edits that preserve existing structure and behavior.
- Do not alter unrelated files or conventions unless required by the requested change.

## Editing guidance
- Keep modifications precise and targeted.
- Match the repository's current coding style and existing file patterns.
- When adding logic, keep it consistent with the surrounding implementation.
- If a file already contains comments and section headers, extend them rather than replacing the file structure wholesale.

## Documentation and reproducibility
- Preserve traceability for data-processing steps, especially for simulation vs. experiment comparisons.
- Document any assumptions, calibration steps, or dataset-specific handling in the relevant code comments.
- Keep scientific processing steps understandable for future analysis and paper reproduction.

## Verification
- Validate the smallest relevant check for the change.
- Prefer focused validation over broad test sweeps.
- If a change affects processing logic, confirm the code still runs in the project context without introducing unrelated behavioral changes.

## Default working style
- Act as a careful scientific coding assistant.
- Favor correctness, reproducibility, and minimal intervention.
- Keep comments concise, domain-aware, and useful for paper-related workflows.
