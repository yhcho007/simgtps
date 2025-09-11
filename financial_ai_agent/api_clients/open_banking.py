import requests

BASE = "http://localhost:8001"  # mock server


def get_accounts():
    r = requests.get(f"{BASE}/accounts")
    r.raise_for_status()
    return r.json()


def get_savings():
    r = requests.get(f"{BASE}/savings")
    r.raise_for_status()
    return r.json()


def get_loans():
    r = requests.get(f"{BASE}/loans")
    r.raise_for_status()
    return r.json()


def get_subscription_score():
    r = requests.get(f"{BASE}/subscription_score")
    r.raise_for_status()
    return r.json()
