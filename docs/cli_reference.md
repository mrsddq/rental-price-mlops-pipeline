# CLI Reference

The command line entry point compiles the Kubeflow pipeline by default and can also run local checks before compilation.

## Common Commands

```bash
python main.py --compile-only
python main.py --validate-data --compile-only
python main.py --evaluate-local --compile-only
python main.py --predict --rooms 3 --sqft 1100 --compile-only
python main.py --validate-data --no-compile
```

## Options

| Option | Purpose |
| --- | --- |
| `--compile-only` | Compile the pipeline YAML and exit without submitting a run. |
| `--no-compile` | Run local validation, evaluation, or prediction without compiling YAML. |
| `--run` | Submit the pipeline to a Kubeflow Pipelines endpoint. |
| `--host` | Kubeflow Pipelines endpoint used with `--run`. |
| `--experiment` | Experiment name for submitted runs. |
| `--output` | Output path for the compiled pipeline YAML. |
| `--data` | Housing CSV path for validation, evaluation, and prediction. |
| `--validate-data` | Validate and summarize the dataset. |
| `--evaluate-local` | Train and evaluate the local scikit-learn model. |
| `--predict` | Fit locally and score a single rental example. |
| `--rooms` | Room count used with `--predict`. |
| `--sqft` | Square footage used with `--predict`. |
