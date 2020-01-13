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
	autoflake -ir examples
	autoflake -ir tensorlayer
	autoflake -ir tests

	isort -rc examples
	isort -rc tensorlayer
	isort -rc tests

	yapf -ir examples
	yapf -ir tensorlayer
	yapf -ir tests

install3:
	pip3 install -U . --user


TAG = tensorlayer-docs:snaphot

doc:
	docker build --rm -t $(TAG) -f docker/docs/Dockerfile .
