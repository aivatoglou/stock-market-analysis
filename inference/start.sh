#!/bin/bash

exec python run.py &
exec streamlit run streamlit_dash.py --server.port=8501 --server.address=0.0.0.0
