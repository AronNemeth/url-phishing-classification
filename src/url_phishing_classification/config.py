"""Config variables."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Feature engineering
SHORTENING_SERVICES = ("bit.ly", "tinyurl.com", "short.link", "ow.ly", "t.co", "goo.gl")
REDIRECT_KEYWORDS = ("redirect", "url", "link", "goto", "away")  #  "out",
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


# training
TARGET_COLUMN = "label"
RANDOM_STATE = 42

LENGTH_FEATURES = (
    "url_length",
    "hostname_length",
    "domain_length",
    "subdomain_length",
    "path_length",
    # "query_length", corr with has_query
    # "fragment_length", # 1.0 corr with has_fragment
)
COUNT_FEATURES = (
    # "query_parameters", corr with has_query
    "subdomain_labels",
    "host_label_count",
    "path_segment_count",
    # "dot_count", # no difference in distribution between classes
    # "hyphen_count", # no difference in distribution between classes
    "slash_count",
    "underscore_count",
    "equals_count",
    "ampersand_count",
    "non_alphanumeric_count",
    # "port", # extremly sparse feature
)
RATIO_FEATURES = (
    "digit_ratio",
    "host_digit_ratio",
    "host_letter_ratio",
    "path_digit_ratio",
    "path_letter_ratio",
    "query_digit_ratio",
    "query_letter_ratio",
)
STRUCTURAL_FLAG_FEATURES = (
    "is_http",
    "has_query",
    "has_fragment",
    "has_params",
    "no_host",
    "is_ip_literal_host",
    # "has_port", # extremly sparse feature
    # "unusual_port", # extremly sparse feature
    # "has_malformed_port", # extremly sparse feature
    "has_at",
    "has_percent_encoding",
    # "has_punycode",
    # "has_unicode",
    # "has_repeated_separator", # all urls do
    # "double_slash_in_path", # extremly sparse feature
)
RISK_SIGNAL_FEATURES = (
    "has_shortening_service",
    "has_redirect_keyword",
    "has_suspicious_keyword",
    "has_suspicious_tld",
)
ENTROPY_FEATURES = (
    "url_entropy",
    "hostname_entropy",
    "domain_entropy",
    "subdomain_entropy",
    "path_entropy",
    "query_entropy",
    # "fragment_entropy", # 1.0 corr with has_fragment
)
NORM_ENTROPY_FEATURES = (
    "url_norm_entropy",
    "hostname_norm_entropy",
    "domain_norm_entropy",
    "subdomain_norm_entropy",
    "path_norm_entropy",
    "query_norm_entropy",
    # "fragment_norm_entropy", # 1.0 corr with has_fragment
)

ALL_ENGINEERED_FEATURES = (
    LENGTH_FEATURES
    + COUNT_FEATURES
    + RATIO_FEATURES
    + STRUCTURAL_FLAG_FEATURES
    + RISK_SIGNAL_FEATURES
    + ENTROPY_FEATURES
    + NORM_ENTROPY_FEATURES
)

C_PARAM = 20.0


@dataclass(frozen=True)
class ModelSpec:
    """A finalized logistic-regression configuration."""

    artifact_name: str
    features: tuple[str, ...]
    C: float


FINAL_MODEL_SPEC = ModelSpec(
    artifact_name="all_engineered_model.joblib",
    features=ALL_ENGINEERED_FEATURES,
    C=C_PARAM,
)


# inference
OPERATING_THRESHOLD = 0.95
