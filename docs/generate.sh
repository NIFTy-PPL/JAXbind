#!/usr/bin/sh

set -e

sphinx-apidoc -e -o docs/source/mod jaxbind
# sphinx-build -b html docs/source/ docs/build/
sphinx-build -b html docs/source/ .