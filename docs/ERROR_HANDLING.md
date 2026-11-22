# Error Handling Guidelines

- Fail early when required input files or columns are missing.
- Include the offending column or file path in validation errors.
- Do not silently replace invalid timestamps with current time.
- Do not continue training with an empty dataset.
- Log rejected action counts during policy evaluation.
- Preserve the original exception context when adding user-facing messages.
- Keep raw data and secrets out of logs.

These rules make experimental failures easier to diagnose and prevent silent data corruption.
