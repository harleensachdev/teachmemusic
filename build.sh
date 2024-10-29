#!/usr/bin/env bash
# exit on error
set -o errexit

pip install --upgrade pip
pip install -r requirements.txt

# Optional: Run migrations if you're using Flask-Migrate
flask db upgrade 