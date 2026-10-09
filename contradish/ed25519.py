"""
contradish/ed25519.py -- Ed25519 signatures (RFC 8032), pure Python, no
dependencies.

Used to authenticate governing-state transitions: a pinned version is signed
by the source that issued it, and a verifier checks the signature against the
key the previous version lists for that source. Pure Python keeps contradish
and its independent checker dependency-free; it is slow (milliseconds per
operation) but signatures are made and checked once per version, not per
case. The implementation follows RFC 8032 section 6 and is tested against
the RFC's test vectors and, where installed, against the `cryptography`
package.

This module is for authenticating evidence, not for protecting secrets: it
makes no claim of constant-time execution.
"""

from __future__ import annotations

import hashlib
import os

__all__ = ["generate_seed", "public_key", "sign", "verify"]

_p = 2 ** 255 - 19
_L = 2 ** 252 + 27742317777372353535851937790883648493
_d = (-121665 * pow(121666, _p - 2, _p)) % _p
_I = pow(2, (_p - 1) // 4, _p)


def _H(m: bytes) -> bytes:
    return hashlib.sha512(m).digest()


def _xrecover(y: int) -> int:
    xx = (y * y - 1) * pow(_d * y * y + 1, _p - 2, _p)
    x = pow(xx, (_p + 3) // 8, _p)
    if (x * x - xx) % _p != 0:
        x = (x * _I) % _p
    if x % 2 != 0:
        x = _p - x
    return x


_By = (4 * pow(5, _p - 2, _p)) % _p
_Bx = _xrecover(_By)
_B = (_Bx, _By, 1, (_Bx * _By) % _p)          # extended coordinates (X, Y, Z, T)
_ZERO = (0, 1, 1, 0)


def _add(P, Q):
    A = (P[1] - P[0]) * (Q[1] - Q[0]) % _p
    B = (P[1] + P[0]) * (Q[1] + Q[0]) % _p
    C = 2 * P[3] * Q[3] * _d % _p
    D = 2 * P[2] * Q[2] % _p
    E, F, G, H = B - A, D - C, D + C, B + A
    return (E * F % _p, G * H % _p, F * G % _p, E * H % _p)


def _mul(s: int, P):
    Q = _ZERO
    while s > 0:
        if s & 1:
            Q = _add(Q, P)
        P = _add(P, P)
        s >>= 1
    return Q


def _equal(P, Q) -> bool:
    return (P[0] * Q[2] - Q[0] * P[2]) % _p == 0 and (P[1] * Q[2] - Q[1] * P[2]) % _p == 0


def _compress(P) -> bytes:
    zinv = pow(P[2], _p - 2, _p)
    x, y = P[0] * zinv % _p, P[1] * zinv % _p
    return int.to_bytes(y | ((x & 1) << 255), 32, "little")


def _decompress(s: bytes):
    if len(s) != 32:
        raise ValueError("invalid point length")
    y = int.from_bytes(s, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    if y >= _p:
        raise ValueError("invalid point")
    x = _xrecover(y)
    if (y * y - x * x - 1 - _d * x * x * y * y) % _p != 0:
        raise ValueError("point not on curve")
    if (x & 1) != sign:
        if x == 0:
            raise ValueError("invalid point")
        x = _p - x
    return (x, y, 1, x * y % _p)


def _secret_expand(seed: bytes):
    if len(seed) != 32:
        raise ValueError("seed must be 32 bytes")
    h = _H(seed)
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, h[32:]


def generate_seed() -> bytes:
    return os.urandom(32)


def public_key(seed: bytes) -> bytes:
    a, _ = _secret_expand(seed)
    return _compress(_mul(a, _B))


def sign(seed: bytes, msg: bytes) -> bytes:
    a, prefix = _secret_expand(seed)
    A = _compress(_mul(a, _B))
    r = int.from_bytes(_H(prefix + msg), "little") % _L
    R = _compress(_mul(r, _B))
    h = int.from_bytes(_H(R + A + msg), "little") % _L
    s = (r + h * a) % _L
    return R + int.to_bytes(s, 32, "little")


def verify(public: bytes, msg: bytes, signature: bytes) -> bool:
    try:
        if len(public) != 32 or len(signature) != 64:
            return False
        A = _decompress(public)
        R = _decompress(signature[:32])
        s = int.from_bytes(signature[32:], "little")
        if s >= _L:
            return False
        h = int.from_bytes(_H(signature[:32] + public + msg), "little") % _L
        return _equal(_mul(s, _B), _add(R, _mul(h, A)))
    except ValueError:
        return False
