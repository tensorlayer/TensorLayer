default:
	@echo "Usage:"
	@echo "\tmake lint      # run pylint"
	@echo "\tmake format    # run yapf, autoflake and isort"
	@echo "\tmake install3  # install tensorlayer in current workspace with pip3"

lint:
	pylint examples/*.py
	pylint tensorlayer

test:
	python3 tests/models/test_model_core.py
	python3 tests/layers/test_layernode.py
	python3 tests/files/test_utils_saveload.py

format:
	autoflake -i examples/*.py
	autoflake -i tensorlayer/*.py
	autoflake -i tensorlayer/**/*.py

	isort -rc examples
	isort -rc tensorlayer

	yapf -i examples/*.py
	yapf -i tensorlayer/*.py
	yapf -i tensorlayer/**/*.py

install3:
	pip3 install -U . --user


TAG = tensorlayer-docs:snaphot

doc:
	docker build --rm -t $(TAG) -f docker/docs/Dockerfile .
