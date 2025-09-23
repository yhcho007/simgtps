"""Run this module to initialize the SQLite DB used for metrics/logs."""
from app.db import init_db

if __name__ == '__main__':
    init_db()
    print('DB initialized')
