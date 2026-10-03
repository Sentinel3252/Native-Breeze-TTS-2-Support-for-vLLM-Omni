# Contributing

Contributions should make the native Breeze integration easier to use, verify,
or maintain. The [roadmap](docs/roadmap.md) lists current priorities. Small,
focused fixes and reproducible checkpoint failures are useful starting points.

## Development setup

Use Python 3.10+ for the standalone tooling. From the project root:

```bash
python scripts/check_project.py
python -m unittest discover -s tests -v
```

For model changes, follow [runtime setup](docs/getting-started.md) on Linux/CUDA.
Keep the native runtime and the official reference environment separate:
their Transformers requirements differ.

## Making a change

1. Describe the behavior to change and its smallest reproducer. For larger
   architecture changes, open an issue in this project's repository first.
2. Edit the native integration under `vllm-omni/`; avoid unrelated upstream
   refactors. Preserve upstream attribution and licensing headers.
3. Add a regression test when changing model, state, or transport behavior.
   Run the relevant upstream tests in a compatible runtime environment.
4. Update support status and usage docs when a user-facing contract changes.
   Add new integration files to `project-manifest.json` so source exports include them.
5. In the pull request, state the trigger, resulting behavior, exact checks run,
   and anything still unverified. Attach checkpoint/GPU evidence for capability claims.

The standalone CPU checks validate repository structure, links, source syntax,
and tooling behavior. They do not validate inference. Do not label stub-based
tests, successful imports, or registration checks as end-to-end model support.

For changes intended for vLLM-Omni upstream, also follow its
[contribution guide](vllm-omni/CONTRIBUTING.md) and lint rules. Run its hooks
from `vllm-omni/` in a prepared environment:

```bash
pre-commit run --files \
  vllm_omni/model_executor/models/breeze_tts/breeze_tts_talker.py \
  vllm_omni/model_executor/stage_input_processors/breeze_tts.py
```

Choose the actual files changed; this example is not a substitute for checks
covering an entire PR. If preparing an upstream PR from the nested checkout,
ensure the added integration files are included in its diff.

## Reporting bugs

Use this project's bug report form. Include source revision, OS, GPU, driver,
Python/runtime versions, checkpoint revision, launch command, and a minimal
request. Remove tokens, personal paths, and private audio from logs. Prefer a
small synthetic input for state-isolation problems.

Keep performance reports reproducible: include concurrency, input lengths,
chunk size, warmup, sampling parameters, and the measurement method. See
[validation](docs/validation.md) for report requirements.

## Review and licensing

Maintainers review contributions through issues and pull requests. Support
status changes require evidence; public release does not imply stable APIs.
Contributions are provided under the project's [Apache-2.0 license](LICENSE),
with existing third-party terms retained. Do not include checkpoint weights
or reference recordings in a source PR. Respect the [community guidelines](CODE_OF_CONDUCT.md).
