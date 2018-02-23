default:
	@echo "Usage:"
	@echo "\tmake lint      # run pylint"
	@echo "\tmake format    # run yapf, autoflake and isort"
	@echo "\tmake install3  # install tensorlayer in current workspace with pip3"

lint:
	pylint example/*.py
	pylint tensorlayer

test:
	python tests/test_yapf_format.py
	python tests/test_pydocstyle.py
	python tests/test_mnist_simple.py

format:
	yapf -i example/*.py
	yapf -i tensorlayer/*.py
	yapf -i tensorlayer/**/*.py

	autoflake -i example/*.py
	autoflake -i tensorlayer/*.py
	autoflake -i tensorlayer/**/*.py

	isort -rc example
	isort -rc tensorlayer

install3:
	pip3 install -U . --user
