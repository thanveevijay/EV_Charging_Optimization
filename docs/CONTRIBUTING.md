# Contributing

## Development setup
1. Create a Python virtual environment.
2. Install the packages listed in `requirements.txt`.
3. Obtain the ACN dataset separately and follow its terms.
4. Run lightweight tests with `python -m unittest discover -s tests -v`.

## Pull requests
- Explain the motivation and scope.
- Include tests for new metric or scheduling behavior.
- Keep simulation constraints deterministic and testable.
- Record configuration changes in experiment documentation.
- Do not commit raw data, model checkpoints, or secrets.

## Commit messages
Use concise action-oriented messages such as `feat: add queue metric` or `fix: validate departure timestamps`.
