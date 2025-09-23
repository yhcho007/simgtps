# mTLS & Ingress Notes (example)

- Use cert-manager to provision internal CA-signed certificates.
- Configure Envoy/NGINX ingress to require client certificate verification for internal API paths.
- Example: set `nginx.ingress.kubernetes.io/auth-tls-secret: <namespace>/client-ca` and `auth-tls-verify-client: 'on'`.
- Store private keys in Kubernetes Secrets or HashiCorp Vault.
- For pod-to-pod communication, use sidecars (Envoy) to terminate mTLS.
