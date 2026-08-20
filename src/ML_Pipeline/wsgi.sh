#!/bin/bash
# 2 workers (each loads its own copy of the model), 60s timeout for cold-start inference
gunicorn -b 0.0.0.0:5001 -w 2 -t 60 wsgi:app
