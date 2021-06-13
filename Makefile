SHELL := /bin/bash
format:
	autoflake -r -i --remove-unused-variables --remove-all-unused-imports .
	isort .
	black .

style-check:
	black --diff --check .

install:
	python3 -m pip install -r requirements.txt --user
