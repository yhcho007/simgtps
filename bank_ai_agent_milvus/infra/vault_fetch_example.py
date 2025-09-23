"""Example: fetch secrets from HashiCorp Vault using hvac library.
- This is a sample to show how to integrate secrets management for tokens/keys
- In production, ensure TLS verification and proper policies (least privilege)
"""
import os
try:
    import hvac
except Exception:
    hvac = None

def fetch_secret(vault_addr, token, path):
    if not hvac:
        raise RuntimeError('hvac library not installed')
    client = hvac.Client(url=vault_addr, token=token)
    if not client.is_authenticated():
        raise RuntimeError('Vault auth failed')
    secret = client.secrets.kv.v2.read_secret_version(path=path)
    return secret['data']['data']

if __name__ == '__main__':
    print('This is an example; do not run in production without proper Vault setup.')
