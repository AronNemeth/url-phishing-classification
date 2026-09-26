"""URL cleaning and feature engineering utilities."""

from __future__ import annotations

import ipaddress
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import SplitResult, urlparse, urlsplit

import pandas as pd
import tldextract

from url_phishing_classification.config import (
    REDIRECT_KEYWORDS,
    SCHEME_PATTERN,
    SHORTENING_SERVICES,
    SUSPICIOUS_KEYWORDS,
    SUSPICIOUS_TLDS,
)

# for offline inference, disable TLD extraction cache and suffix list updates
TLD_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)


@dataclass(frozen=True)
class ParsedUrlComponents:
    """Safe, plain-Python URL components used for feature engineering."""

    scheme: str
    hostname: str
    path: str
    query: str
    fragment: str
    params: str
    domain: str
    subdomain: str
    suffix: str
    registered_domain: str
    port: int
    has_malformed_port: bool


def normalize_url(url: str) -> str:
    """Normalize case, scheme, leading ``www.`` and trailing slashes."""
    url = url.strip().lower()
    url = SCHEME_PATTERN.sub("", url, count=1)
    if url.startswith("www."):
        url = url[4:]
    return url.rstrip("/")


def parse_url(url: str, parser: Callable[[str], SplitResult] = urlsplit) -> SplitResult:
    """Parse scheme-less URLs as URLs instead of paths, handling malformed inputs."""
    url = url.strip()
    if not SCHEME_PATTERN.match(url):
        url = f"//{url}"
    try:
        return parser(url)
    except ValueError:
        return parser("")


def _coerce_url_text(value: object) -> str:
    """Return a URL string while treating missing and non-string cells as empty."""
    return value if isinstance(value, str) else ""


def _parse_url_components(raw_url: object) -> ParsedUrlComponents:
    """Parse one raw value into safe native values without exposing pandas missing values."""
    url = _coerce_url_text(raw_url)
    parsed = parse_url(url)

    parse_target = url.strip()
    if not SCHEME_PATTERN.match(parse_target):
        parse_target = f"//{parse_target}"
    try:
        parsed_with_params = urlparse(parse_target)
    except ValueError:
        parsed_with_params = urlparse("")

    hostname = (parsed.hostname or "").lower()
    extracted = TLD_EXTRACT(hostname)

    return ParsedUrlComponents(
        scheme=parsed.scheme.lower(),
        hostname=hostname,
        path=parsed.path,
        query=parsed.query,
        fragment=parsed.fragment,
        params=parsed_with_params.params,
        domain=extracted.domain,
        subdomain=extracted.subdomain,
        suffix=extracted.suffix,
        registered_domain=extracted.top_domain_under_public_suffix,
        port=_get_port(parsed),
        has_malformed_port=_has_malformed_port(parsed),
    )


def entropy(value: str) -> float:
    """Calculate Shannon entropy, returning zero for empty or uniform strings."""
    if not value:
        return 0.0
    counts = Counter(value)
    if len(counts) <= 1:
        return 0.0
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def normalized_entropy(value: str) -> float:
    """Calculate Shannon entropy normalized by the number of unique characters."""
    if not value:
        return 0.0
    counts = Counter(value)
    if len(counts) <= 1:
        return 0.0
    length = len(value)
    value_entropy = -sum((count / length) * math.log2(count / length) for count in counts.values())
    return value_entropy / math.log2(len(counts))


def deduplicate_urls(df: pd.DataFrame, label_column: str = "label") -> pd.DataFrame:
    """Remove duplicate normalized URLs and discard groups with conflicting labels.

    Requires a ``normalized_url`` column. For groups with one label, keeps the
    first occurrence; for groups with multiple labels, removes every occurrence.
    """
    if label_column not in df:
        return df.drop_duplicates(subset=["normalized_url"], keep="first").copy()

    label_counts = df.groupby("normalized_url")[label_column].nunique()
    conflicting_urls = label_counts[label_counts > 1].index
    return (
        df.loc[~df["normalized_url"].isin(conflicting_urls)]
        .drop_duplicates(subset=["normalized_url"], keep="first")
        .copy()
    )


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def _get_port(parsed_url: SplitResult) -> int:
    try:
        return parsed_url.port or 0
    except ValueError:
        return 0


def _has_malformed_port(parsed_url: SplitResult) -> bool:
    try:
        parsed_url.port
        return False
    except ValueError:
        return True


def _host_uses_service(host: str, service: str) -> bool:
    return host == service or host.endswith(f".{service}")


def _count_nonempty_parts(value: str, separator: str) -> int:
    """Count non-empty portions of a delimited URL component."""
    return sum(bool(part) for part in value.split(separator))


def _count_query_parameters(query: str) -> int:
    """Count query parameters separated by ampersands."""
    return query.count("&") + 1 if query else 0


def _digit_ratio(value: str) -> float:
    """Return the proportion of characters that are digits."""
    return sum(character.isdigit() for character in value) / max(len(value), 1)


def _has_unicode(value: str) -> bool:
    """Return whether a value contains non-ASCII characters."""
    return any(ord(character) > 127 for character in value)


def _string_features(values: list[str], prefix: str, index: pd.Index) -> pd.DataFrame:
    """Calculate entropy and length features for a URL component."""
    return pd.DataFrame(
        {
            f"{prefix}_entropy": [entropy(value) for value in values],
            f"{prefix}_norm_entropy": [normalized_entropy(value) for value in values],
            f"{prefix}_length": [len(value) for value in values],
        },
        index=index,
    )


def engineer_features(df: pd.DataFrame, url_column: str = "url") -> pd.DataFrame:
    """Add URL-derived features."""
    if url_column not in df:
        raise ValueError(f"Missing required URL column: {url_column!r}")

    result = df.copy()
    url_values = [_coerce_url_text(value) for value in result[url_column].tolist()]
    components = [_parse_url_components(value) for value in url_values]
    component_frame = pd.DataFrame(
        {
            "scheme": [component.scheme for component in components],
            "hostname": [component.hostname for component in components],
            "path": [component.path for component in components],
            "query": [component.query for component in components],
            "fragment": [component.fragment for component in components],
            "params": [component.params for component in components],
            "domain": [component.domain for component in components],
            "subdomain": [component.subdomain for component in components],
            "suffix": [component.suffix for component in components],
            "registered_domain": [component.registered_domain for component in components],
            "port": [component.port for component in components],
            "has_malformed_port": [component.has_malformed_port for component in components],
        },
        index=result.index,
    )

    urls = pd.Series(url_values, index=result.index, dtype="string")
    hostname = component_frame["hostname"]
    path = component_frame["path"]
    query = component_frame["query"]
    fragment = component_frame["fragment"]
    params = component_frame["params"]
    suffix = component_frame["suffix"]
    ports = component_frame["port"]
    url_text = urls.str.lower()
    shortening_matches = pd.DataFrame(
        {
            service: [_host_uses_service(component.hostname, service) for component in components]
            for service in SHORTENING_SERVICES
        },
        index=result.index,
    )
    redirect_matches = pd.DataFrame(
        {keyword: url_text.str.contains(keyword, regex=False) for keyword in REDIRECT_KEYWORDS},
        index=result.index,
    )
    suspicious_keyword_matches = pd.DataFrame(
        {keyword: url_text.str.contains(keyword, regex=False) for keyword in SUSPICIOUS_KEYWORDS},
        index=result.index,
    )

    result = result.assign(
        # Parsed URL components
        domain=component_frame["domain"],
        subdomain=component_frame["subdomain"],
        suffix=suffix,
        registered_domain=component_frame["registered_domain"],
        scheme=component_frame["scheme"],
        is_http=component_frame["scheme"].eq("http"),
        hostname=hostname,
        path=path,
        query=query,
        fragment=fragment,
        params=params,
        # Existing / corrected structural features
        query_parameters=[_count_query_parameters(component.query) for component in components],
        has_query=query.ne(""),
        has_fragment=fragment.ne(""),
        has_params=params.ne(""),
        subdomain_labels=[_count_nonempty_parts(component.subdomain, ".") for component in components],
        # Host and parsing features
        no_host=hostname.eq(""),
        is_ip_literal_host=[_is_ip_literal(component.hostname) for component in components],
        host_label_count=[_count_nonempty_parts(component.hostname, ".") for component in components],
        has_port=ports.gt(0),
        port=ports,
        unusual_port=ports.ne(0) & ~ports.isin([80, 443]),
        has_malformed_port=component_frame["has_malformed_port"],
        # Path / query structure
        path_segment_count=[_count_nonempty_parts(component.path, "/") for component in components],
        dot_count=urls.str.count(r"\."),
        hyphen_count=urls.str.count("-"),
        slash_count=urls.str.count("/", flags=0),
        underscore_count=urls.str.count("_", flags=0),
        equals_count=urls.str.count("=", flags=0),
        ampersand_count=urls.str.count("&", flags=0),
        non_alphanumeric_count=urls.str.count(r"[^A-Za-z0-9]"),
        # URL-wide signals
        digit_ratio=[_digit_ratio(value) for value in url_values],
        url_letter_ratio=urls.str.count(r"[A-Za-z]") / urls.str.len().clip(lower=1),
        url_non_alphanumeric_ratio=urls.str.count(r"[^A-Za-z0-9]") / urls.str.len().clip(lower=1),
        has_at=urls.str.contains("@", regex=False),
        has_percent_encoding=urls.str.contains(r"%[0-9A-Fa-f]{2}", regex=True),
        has_punycode=hostname.str.contains("xn--", regex=False),
        has_unicode=[_has_unicode(value) for value in url_values],
        has_repeated_separator=urls.str.contains(r"//|__|==|&&|\.\.|--", regex=True),
        double_slash_in_path=path.str.contains("//", regex=False),
        # Audited keyword and domain signals
        has_shortening_service=shortening_matches.any(axis=1),
        has_redirect_keyword=redirect_matches.any(axis=1),
        has_suspicious_keyword=suspicious_keyword_matches.any(axis=1),
        has_suspicious_tld=suffix.isin(SUSPICIOUS_TLDS),
        # Component-level ratios
        host_digit_ratio=hostname.str.count(r"\d") / hostname.str.len().clip(lower=1),
        host_letter_ratio=hostname.str.count(r"[A-Za-z]") / hostname.str.len().clip(lower=1),
        path_digit_ratio=path.str.count(r"\d") / path.str.len().clip(lower=1),
        path_letter_ratio=path.str.count(r"[A-Za-z]") / path.str.len().clip(lower=1),
        query_digit_ratio=query.str.count(r"\d") / query.str.len().clip(lower=1),
        query_letter_ratio=query.str.count(r"[A-Za-z]") / query.str.len().clip(lower=1),
    )

    component_values = {
        "url": url_values,
        "hostname": [component.hostname for component in components],
        "domain": [component.domain for component in components],
        "subdomain": [component.subdomain for component in components],
        "path": [component.path for component in components],
        "query": [component.query for component in components],
        "fragment": [component.fragment for component in components],
    }
    for feature, values in component_values.items():
        result = result.join(_string_features(values, feature, result.index))

    return result
