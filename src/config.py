"""Config variables."""

import re

# feature engineering
SHORTENING_SERVICES = ("bit.ly", "tinyurl.com", "short.link", "ow.ly", "t.co", "goo.gl")
REDIRECT_KEYWORDS = ("redirect", "url", "link", "goto", "out", "away")
SUSPICIOUS_KEYWORDS = (
    "secure",
    "account",
    "update",
    "login",
    "signin",
    "bank",
    "verify",
    "confirm",
    "suspend",
    "restricted",
    "limited",
    "amazon",
    "paypal",
    "ebay",
    "apple",
    "microsoft",
    "google",
)
SUSPICIOUS_TLDS = ("tk", "ml", "ga", "cf", "top", "click", "download", "link")

SCHEME_PATTERN = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
