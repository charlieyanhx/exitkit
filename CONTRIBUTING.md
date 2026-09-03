# Contributing to exitkit

Thanks for looking. This is a small, deliberately narrow library; the most useful
contributions are bug reports with a failing case.

## Reporting a bug

Open an issue with a minimal snippet that reproduces it. If you can express it as a
failing test in `tests/`, that is ideal — every defect this package has fixed so far
became a regression test named after the defect.

## Development

```bash
git clone https://github.com/charlieyanhx/exitkit
cd exitkit
pip install -e ".[test]"
pytest -q
```

## Pull requests

- Add a test that fails before your change and passes after.
- Keep the public API surface small; new behaviour usually belongs behind an existing
  entry point rather than beside it.
- Match the surrounding style. No formatter is enforced.

## Scope

Proposals that widen the library into an orchestration framework, add a service, or
introduce a required dependency will probably be declined — not because they are bad
ideas, but because staying small is what makes this auditable.
