import time
import jwt
from cryptography.hazmat.primitives import serialization

ISSUER = "https://auth.internal"
ACCESS_TTL = 15 * 60


def load_private_key(path):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def issue_access_token(user_id, private_key, audience):
    now = int(time.time())
    payload = {
        "sub": user_id,
        "iss": ISSUER,
        "aud": audience,
        "iat": now,
        "exp": now + ACCESS_TTL,
    }
    return jwt.encode(payload, private_key, algorithm="RS256")


def validate_token(token, public_key, audience):
    """Verify signature and standard claims; raises on failure."""
    return jwt.decode(
        token,
        public_key,
        algorithms=["RS256"],
        audience=audience,
        issuer=ISSUER,
        options={"require": ["exp", "iss", "aud"]},
    )


class TokenError(Exception):
    pass
