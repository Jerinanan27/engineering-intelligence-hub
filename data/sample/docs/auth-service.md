# Auth Service

The Auth Service (`auth-svc`) issues and validates JSON Web Tokens (JWTs) for all
internal APIs. Tokens are signed with RS256 using a private key stored in Vault at
`secret/auth/jwt`. Access tokens expire after 15 minutes; refresh tokens after 7 days.

## Validation flow
1. Gateway extracts the `Authorization: Bearer <token>` header.
2. `auth-svc` verifies the signature against the public JWKS at `/.well-known/jwks.json`.
3. Claims `exp`, `iss`, and `aud` are checked. `iss` must equal `https://auth.internal`.
4. On success the decoded claims are forwarded downstream as `X-User-Claims`.

## Rotating keys
Keys rotate every 90 days. The previous public key is kept in the JWKS for one
extra rotation window so in-flight tokens remain valid. Rotation is triggered by
the `auth-key-rotation` cron job.
