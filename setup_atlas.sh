#!/bin/bash
set -e
echo "Setting up Atlas..."
mkdir -p config data engine backtest dashboard storage utils data_store/cache logs
for dir in config data engine backtest dashboard storage utils; do touch ${dir}/__init__.py; done
echo "Directories created"
