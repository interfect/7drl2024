#!/usr/bin/env bash
if [[ ! -e venv ]] ; then
    python -m virtualenv venv
fi
. venv/bin/activate
pip install -r requirements.txt
python main.py