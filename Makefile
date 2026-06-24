.PHONY: install test compile evaluate reports artifact registry serve clean

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

test:
	python -m unittest discover -s tests

compile:
	python main.py --compile-only --output rental_price_prediction_pipeline.yaml

evaluate:
	python main.py --no-compile --validate-data --evaluate-local

reports:
	python main.py --no-compile --write-reports --report-dir outputs/reports

artifact:
	python main.py --no-compile --write-artifact --artifact-path outputs/model/rental-price-model.pkl

registry:
	python main.py --no-compile --write-registry-record --registry-path outputs/model/registry-record.json --artifact-path outputs/model/rental-price-model.pkl

serve:
	uvicorn rental_mlops.serving:create_app --factory --host 0.0.0.0 --port 8000

clean:
	python -c "import shutil; from pathlib import Path; [shutil.rmtree(path, ignore_errors=True) for path in Path('.').rglob('__pycache__')]; shutil.rmtree('outputs', ignore_errors=True)"
