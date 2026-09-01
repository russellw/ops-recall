# Edge gateway runbook

## Certificate expiry

Symptom: HTTP 526 from the CDN, `NET::ERR_CERT_DATE_INVALID` in browsers, all
TLS handshakes failing at once.

Check what is actually served, from outside the cluster -- cert-manager's own
view of the world is the thing most likely to be wrong:
`openssl s_client -connect api.example.com:443 -servername api.example.com | openssl x509 -noout -dates`

If renewal is failing, the usual cause is the DNS-01 challenge losing write
permission on the hosted zone. Check the CertificateRequest status before
touching anything else:
`kubectl describe certificaterequest -n edge`

Force a renewal once the underlying permission is restored:
`kubectl cert-manager renew wildcard-example-com -n edge`

## Rate limiting and retries

Outbound retries must use exponential backoff with full jitter. A fixed retry
interval turns a brief upstream blip into a synchronized retry storm that
exceeds the contracted rate limit and produces HTTP 429 on first attempts. The
client-side limiter is pinned 20% below the contracted ceiling.
