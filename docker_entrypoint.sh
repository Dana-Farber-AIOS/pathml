#!/bin/bash --login
set -e

conda activate $ENV_PREFIX
exec "$@"