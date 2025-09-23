"""Generate daily performance report from SQLite DB and send via SMTP.
- Config via environment variables:
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, REPORT_TO
- The script queries interactions from last 24 hours and composes a summary CSV + body.
"""
import os, smtplib, csv
from email.message import EmailMessage
from datetime import datetime, timedelta
from app.db import ENGINE
import pandas as pd

DB_PATH = os.path.join(os.getcwd(), 'data', 'metrics.db')

SMTP_HOST = os.getenv('SMTP_HOST')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USER = os.getenv('SMTP_USER')
SMTP_PASS = os.getenv('SMTP_PASS')
REPORT_TO = os.getenv('REPORT_TO')


def query_last_24h():
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    since = datetime.utcnow() - timedelta(days=1)
    q = "SELECT id,timestamp,session_id,user,query,reply,response_time_ms,satisfaction,feedback FROM interactions WHERE timestamp >= ?"
    df = pd.read_sql_query(q, conn, params=(since.isoformat(),))
    conn.close()
    return df


def compose_and_send():
    df = query_last_24h()
    if df.empty:
        body = 'No interactions in the last 24 hours.'
    else:
        total = len(df)
        avg_rt = df['response_time_ms'].mean()
        sat_mean = df['satisfaction'].mean()
        low_cnt = df[df['satisfaction']<=3].shape[0]
        body = f"Daily Report:\nTotal interactions: {total}\nAvg response ms: {avg_rt:.1f}\nAvg satisfaction: {sat_mean:.2f}\nLow satisfaction (<=3): {low_cnt}\n"
    # attach CSV
    msg = EmailMessage()
    msg['Subject'] = f"Daily AI Agent Report - {datetime.utcnow().date()}"
    msg['From'] = SMTP_USER or 'noreply@example.com'
    msg['To'] = REPORT_TO or SMTP_USER
    msg.set_content(body)
    if not df.empty:
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        msg.add_attachment(csv_bytes, maintype='text', subtype='csv', filename='daily_interactions.csv')

    if not SMTP_HOST:
        print('SMTP not configured. Printing report instead:\n', body)
        return
    # send email
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)
        print('Report sent to', REPORT_TO)

if __name__ == '__main__':
    compose_and_send()
