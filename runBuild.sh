#!/bin/bash

# install python dev headers
sudo apt-get install python3-dev

# install deps
pip install -r requirementspy.txt

cd python

python3 -m build
cd dist/
if python3 -c 'import pkgutil; exit(not pkgutil.find_loader("tool_eventdetectobjects"))'; then
    echo 'delete existing one'
    python3 -m pip uninstall tool_eventdetectobjects
fi
echo 'installing tool_eventdetectobjects...'
python3 -m pip install tool_eventdetectobjects-0.0.1-py3-none-any.whl
