SHELL := /bin/bash
format:
	autoflake -r -i --remove-unused-variables --remove-all-unused-imports .
	isort .
	black .
	# sort-requirements requirements.txt requirements.dev.txt

style-check:
	black --diff --check .
