"""Mocked Bank API Adapter for demo.
- In production, this should call internal bank services over mTLS or through an internal API gateway.
- This adapter centralizes auth and audit logging for internal API calls.
"""
import random
from datetime import datetime, timedelta

class BankAPIAdapter:
    def __init__(self):
        pass

    def _get_balance(self, account_id: str):
        # return mocked balance and recent txs
        balance = random.randint(10000, 5000000)
        recent = []
        for i in range(3):
            amt = random.randint(-200000, 200000)
            ts = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d %H:%M')
            recent.append(f"{ts}: {'입금' if amt>0 else '출금'} {abs(amt)}원")
        return {'account_id': account_id, 'balance': balance, 'recent': recent}

    def call(self, api_name: str, params: dict, user: dict):
        # simple permission check (demo)
        if api_name == 'get_balance':
            return self._get_balance(params.get('account_id', user.get('account_id', '111-222-333')))
        raise NotImplementedError(api_name)
