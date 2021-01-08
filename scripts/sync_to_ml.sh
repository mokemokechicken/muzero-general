#!/bin/sh

rsync --exclude .idea --exclude .venv --exclude __pycache__ --exclude .git -ruv . ml:muzero-general/
