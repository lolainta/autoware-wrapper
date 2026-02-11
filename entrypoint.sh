#!/bin/bash
pushd /app
source /opt/autoware/setup.bash
uv run autoware_wrapper/server.py
popd
