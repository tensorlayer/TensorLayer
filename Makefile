default:
	@echo "Usage:"
	@echo "\tmake lint      # run pylint"
	@echo "\tmake format    # run yapf, autoflake and isort"
	@echo "\tmake install3  # install tensorlayer in current workspace with pip3"

lint:
	pylint example/*.py
	pylint tensorlayer

test:
	python3 tests/test_yapf_format.py
	# python3 tests/test_pydocstyle.py
	python3 tests/test_mnist_simple.py
	python3 tests/test_reuse_mlp.py
	python3 tests/test_layers_basic.py
	python3 tests/test_layers_convolution.py
	python3 tests/test_layers_core.py
	python3 tests/test_layers_extend.py
	python3 tests/test_layers_flow_control.py
	python3 tests/test_layers_importer.py
	python3 tests/test_layers_merge.py
	python3 tests/test_layers_normalization.py
	python3 tests/test_layers_pooling.py
	python3 tests/test_layers_recurrent.py
	python3 tests/test_layers_shape.py
	python3 tests/test_layers_spatial_transformer.py
	python3 tests/test_layers_special_activation.py
	python3 tests/test_layers_stack.py
	python3 tests/test_layers_super_resolution.py
	python3 tests/test_layers_time_distributed.py

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
