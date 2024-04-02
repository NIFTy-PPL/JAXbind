#!/usr/bin/sh

set -e

FOLDER=docs/source/demos/

for file in "${FOLDER}/01_linear_function" "${FOLDER}/02_multilinear_function" "${FOLDER}/03_nonlinear_function"; do
	jupytext --to ipynb "${file}.py"
    jupyter-nbconvert --to markdown --execute --ExecutePreprocessor.timeout=None "${file}.ipynb"
done

sphinx-apidoc -e -o docs/source/mod jaxbind
sphinx-build -b html docs/source/ _site/
