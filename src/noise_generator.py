"""
FlowScore — Transaction Description Noise Generator
=====================================================
Real bank transaction descriptions are messy. They include:
- Card processor prefixes (SQ *, DEBIT POS, CHECKCARD, ACH)
- Location suffixes (SAN DIEGO CA, #2394, STORE 847)
- Truncation (CHICK-FIL-A becomes CHICK-FI)
- Abbreviations (DOORDASH becomes DRDSH, STARBUCKS becomes SBUX)
- Date stamps (04/12, 2025-04-10)
- Reference numbers (REF#483920, *2R4T8)
- Mixed casing and formatting

This module corrupts clean merchant names to simulate real data,
creating a realistic test case for the categorizer.

Usage:
    from noise_generator import add_noise_to_transactions
    noisy_txns = add_noise_to_transactions(consumer["transactions"], rng)
"""

import re
import numpy as np
from typing import Dict, List


# ============================================================================
# NOISE PATTERNS
# ============================================================================

# Card processor / bank prefixes that get prepended to merchant names
PREFIXES = [
    "SQ *", "DEBIT POS ", "CHECKCARD ", "ACH WITHDRAWAL ",
    "POS PURCHASE ", "VISA DIRECT ", "MASTERCARD ", "DEBIT CARD ",
    "RECURRING PMT ", "AUTOPAY ", "ONLINE PMT ", "MOBILE PMT ",
    "PURCHASE ", "POS ", "DBT CRD ", "ELECTRONIC ",
    "ACH DEBIT ", "ACH CREDIT ", "DIRECT DEBIT ",
]

# Location suffixes that get appended
CITIES = [
    "SAN DIEGO CA", "LOS ANGELES CA", "NEW YORK NY", "CHICAGO IL",
    "HOUSTON TX", "PHOENIX AZ", "AUSTIN TX", "SEATTLE WA",
    "DENVER CO", "PORTLAND OR", "MIAMI FL", "ATLANTA GA",
    "BOSTON MA", "SAN FRAN CA", "DALLAS TX", "NASHVILLE TN",
]

LOCATION_FORMATS = [
    "{city}",           # SAN DIEGO CA
    "#{num}",           # #2394
    "STORE {num}",      # STORE 847
    "#{num} {city}",    # #2394 SAN DIEGO CA
    "STORE#{num}",      # STORE#847
    "LOC {num}",        # LOC 2394
    "- {city}",         # - SAN DIEGO CA
]

# Common abbreviations that real banks use
ABBREVIATIONS = {
    "STARBUCKS": ["SBUX", "STRBCKS", "STARBUCK", "STARB"],
    "MCDONALDS": ["MCD", "MCDNLDS", "MCDON"],
    "CHIPOTLE": ["CHIPOTL", "CHIPTL", "CHIPTLE"],
    "WALMART": ["WM", "WALMRT", "WAL-MART", "WAL MART"],
    "AMAZON.COM": ["AMZN MKTP US", "AMZN DIGITAL", "AMAZON COM", "AMZ*"],
    "DOORDASH ORDER": ["DRDSH", "DOORDASH*", "DD *DOORDASH"],
    "UBER EATS": ["UBEREATS", "UBER *EATS", "UBER   EATS"],
    "GRUBHUB": ["GH *GRUBHUB", "GRUBHB"],
    "NETFLIX": ["NETFLIX.COM", "NETFLIX INC", "NFLX"],
    "SPOTIFY": ["SPOTIFY USA", "SPOTIFY.COM", "SPTFY"],
    "UBER TRIP": ["UBER *TRIP", "UBER   TRIP", "UBER BV"],
    "LYFT RIDE": ["LYFT *RIDE", "LYFT   RIDE"],
    "COSTCO WHOLESALE": ["COSTCO WHSE", "COSTCO W"],
    "WHOLE FOODS MKT": ["WHOLEFDS", "WF MKT", "WHOLE FDS"],
    "HOME DEPOT": ["THE HOME DEPO", "HOME DEPT"],
    "CVS PHARMACY": ["CVS/PHARM", "CVS #", "CVS/PHARMACY"],
    "WALGREENS": ["WALGREEN", "WALGR"],
    "KROGER": ["KROGER FUEL", "KRO"],
    "DUNKIN DONUTS": ["DUNKIN", "DD #", "DUNKIN #"],
    "CHICK-FIL-A": ["CHICK-FI", "CFA RESTAURANT", "CHICKFILA"],
    "TACO BELL": ["TACO BL", "TB #"],
    "PANERA BREAD": ["PANERA", "PNRA"],
    "TARGET": ["TARGET T-", "TRGT"],
    "BEST BUY": ["BBY", "BESTBUY"],
    "NORDSTROM": ["NORDSTRM", "NRDSTRM"],
    "APPLE STORE": ["APL* APPLE", "APPLE.COM", "APPLE STO"],
    "WELLS FARGO MORTGAGE": ["WF MORTGAGE", "WELLS FARGO MTG"],
    "CHASE HOME LENDING": ["CHASE MORT", "JPMORGAN CHASE"],
    "AFFIRM PAYMENT": ["AFFIRM *", "AFFIRM INC"],
    "KLARNA PAYMENT": ["KLARNA *", "KLARNA INC"],
    "DRAFTKINGS": ["DKNG", "DRAFTKNG"],
    "FANDUEL": ["FANDL", "FD *FANDUEL"],
    "SHELL": ["SHELL OIL", "SHELL SERV"],
    "CHEVRON": ["CHEVRON STN", "CHEV"],
    "ADVANCE AMERICA": ["ADV AMERICA", "ADVNCE AMER"],
    "CHECK INTO CASH": ["CHK INTO CASH", "CHECKINTOCASH"],
}

# Date format stamps
DATE_FORMATS = [
    "{mm}/{dd}", "{mm}-{dd}", "{yyyy}-{mm}-{dd}",
    "{mm}/{dd}/{yy}", "DATE {mm}/{dd}",
]

# Reference number patterns
REF_PATTERNS = [
    "REF#{num}", "*{alpha}", "CONF#{num}", "TXN{num}",
    "ID:{num}", "#{num}", "AUTH:{num}",
]


# ============================================================================
# NOISE APPLICATION FUNCTIONS
# ============================================================================

def add_prefix(merchant: str, rng: np.random.Generator) -> str:
    """Add a card processor / bank prefix."""
    prefix = PREFIXES[int(rng.integers(0, len(PREFIXES)))]
    return prefix + merchant

def add_location(merchant: str, rng: np.random.Generator) -> str:
    """Add a location suffix."""
    city = CITIES[int(rng.integers(0, len(CITIES)))]
    fmt = LOCATION_FORMATS[int(rng.integers(0, len(LOCATION_FORMATS)))]
    num = str(int(rng.integers(100, 9999)))
    suffix = fmt.format(city=city, num=num)
    return merchant + " " + suffix

def truncate(merchant: str, rng: np.random.Generator) -> str:
    """Truncate to simulate character limits on bank statements."""
    max_len = int(rng.integers(10, 22))
    if len(merchant) > max_len:
        return merchant[:max_len]
    return merchant

def abbreviate(merchant: str, rng: np.random.Generator) -> str:
    """Replace merchant name with a known abbreviation."""
    merchant_upper = merchant.upper()
    for original, abbrevs in ABBREVIATIONS.items():
        if original in merchant_upper:
            abbrev = abbrevs[int(rng.integers(0, len(abbrevs)))]
            return merchant_upper.replace(original, abbrev)
    return merchant

def add_date_stamp(merchant: str, date_str: str, rng: np.random.Generator) -> str:
    """Add a date stamp from the transaction date."""
    parts = date_str.split("-")
    if len(parts) == 3:
        yyyy, mm, dd = parts
        yy = yyyy[2:]
        fmt = DATE_FORMATS[int(rng.integers(0, len(DATE_FORMATS)))]
        stamp = fmt.format(mm=mm, dd=dd, yy=yy, yyyy=yyyy)
        return stamp + " " + merchant
    return merchant

def add_ref_number(merchant: str, rng: np.random.Generator) -> str:
    """Add a reference number."""
    fmt = REF_PATTERNS[int(rng.integers(0, len(REF_PATTERNS)))]
    num = str(int(rng.integers(100000, 999999)))
    alpha = "".join([chr(int(rng.integers(65, 91))) for _ in range(4)])
    ref = fmt.format(num=num, alpha=alpha)
    return merchant + " " + ref

def random_case(merchant: str, rng: np.random.Generator) -> str:
    """Randomly change casing."""
    choice = int(rng.integers(0, 3))
    if choice == 0:
        return merchant.upper()
    elif choice == 1:
        return merchant.lower()
    else:
        return merchant.title()

def add_extra_spaces(merchant: str, rng: np.random.Generator) -> str:
    """Add random extra spaces (common in bank data)."""
    words = merchant.split()
    result = []
    for w in words:
        spaces = " " * int(rng.integers(1, 4))
        result.append(w)
        result.append(spaces)
    return "".join(result).strip()


# ============================================================================
# MAIN NOISE PIPELINE
# ============================================================================

# Noise levels control how much corruption is applied
NOISE_LEVELS = {
    "light": {
        "prefix_prob": 0.15,
        "location_prob": 0.20,
        "truncate_prob": 0.10,
        "abbreviate_prob": 0.15,
        "date_stamp_prob": 0.05,
        "ref_number_prob": 0.10,
        "case_change_prob": 0.20,
        "extra_spaces_prob": 0.10,
        "max_transforms": 2,
    },
    "medium": {
        "prefix_prob": 0.30,
        "location_prob": 0.35,
        "truncate_prob": 0.20,
        "abbreviate_prob": 0.30,
        "date_stamp_prob": 0.10,
        "ref_number_prob": 0.15,
        "case_change_prob": 0.30,
        "extra_spaces_prob": 0.15,
        "max_transforms": 3,
    },
    "heavy": {
        "prefix_prob": 0.45,
        "location_prob": 0.45,
        "truncate_prob": 0.30,
        "abbreviate_prob": 0.45,
        "date_stamp_prob": 0.15,
        "ref_number_prob": 0.20,
        "case_change_prob": 0.40,
        "extra_spaces_prob": 0.25,
        "max_transforms": 4,
    },
}


def corrupt_merchant(
    merchant: str,
    date_str: str,
    rng: np.random.Generator,
    noise_level: str = "medium",
) -> str:
    """
    Apply random noise transforms to a clean merchant name.

    Returns a corrupted version that simulates real bank data.
    The original merchant name should still be partially recognizable
    (just like real data — humans can usually figure it out).
    """
    config = NOISE_LEVELS[noise_level]
    result = merchant
    n_transforms = 0
    max_t = config["max_transforms"]

    # Apply transforms in a specific order that produces realistic results
    # Order matters: abbreviate first, then prefix, location, truncate last

    # 1. Abbreviation (replaces the merchant name itself)
    if n_transforms < max_t and float(rng.random()) < config["abbreviate_prob"]:
        new = abbreviate(result, rng)
        if new != result:  # only count if it actually changed
            result = new
            n_transforms += 1

    # 2. Case change
    if n_transforms < max_t and float(rng.random()) < config["case_change_prob"]:
        result = random_case(result, rng)
        n_transforms += 1

    # 3. Extra spaces
    if n_transforms < max_t and float(rng.random()) < config["extra_spaces_prob"]:
        result = add_extra_spaces(result, rng)
        n_transforms += 1

    # 4. Date stamp prefix
    if n_transforms < max_t and float(rng.random()) < config["date_stamp_prob"]:
        result = add_date_stamp(result, date_str, rng)
        n_transforms += 1

    # 5. Card processor prefix
    if n_transforms < max_t and float(rng.random()) < config["prefix_prob"]:
        result = add_prefix(result, rng)
        n_transforms += 1

    # 6. Location suffix
    if n_transforms < max_t and float(rng.random()) < config["location_prob"]:
        result = add_location(result, rng)
        n_transforms += 1

    # 7. Reference number
    if n_transforms < max_t and float(rng.random()) < config["ref_number_prob"]:
        result = add_ref_number(result, rng)
        n_transforms += 1

    # 8. Truncation (always last — it chops the end)
    if n_transforms < max_t and float(rng.random()) < config["truncate_prob"]:
        result = truncate(result, rng)
        n_transforms += 1

    return result


def add_noise_to_consumer(
    consumer: Dict,
    rng: np.random.Generator,
    noise_level: str = "medium",
) -> Dict:
    """
    Create a copy of a consumer's transactions with noisy merchant names.
    Preserves the original merchant in a 'merchant_clean' field.
    """
    noisy_txns = []
    for txn in consumer["transactions"]:
        noisy = dict(txn)
        noisy["merchant_clean"] = txn["merchant"]  # preserve original
        noisy["merchant"] = corrupt_merchant(
            txn["merchant"], txn["date"], rng, noise_level
        )
        noisy_txns.append(noisy)

    result = dict(consumer)
    result["transactions"] = noisy_txns
    return result


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == "__main__":
    import json

    rng = np.random.default_rng(42)

    # Demo: show what noise looks like
    sample_merchants = [
        ("STARBUCKS", "2024-06-15", -5.25),
        ("CHIPOTLE", "2024-06-14", -12.50),
        ("AMAZON.COM", "2024-06-13", -47.99),
        ("WELLS FARGO MORTGAGE", "2024-06-01", -1850.00),
        ("UBER EATS", "2024-06-12", -28.50),
        ("DRAFTKINGS", "2024-06-11", -25.00),
        ("ACME CORP PAYROLL", "2024-06-15", 2847.63),
        ("AFFIRM PAYMENT", "2024-06-10", -37.50),
        ("DOORDASH DASHER PAY", "2024-06-09", 142.80),
        ("CVS PHARMACY", "2024-06-08", -18.99),
        ("ADVANCE AMERICA", "2024-06-07", 500.00),
        ("COSTCO WHOLESALE", "2024-06-06", -127.43),
        ("CHICK-FIL-A", "2024-06-05", -9.85),
        ("VENMO PAYMENT RECEIVED", "2024-06-04", 75.00),
        ("OVERDRAFT FEE", "2024-06-03", -35.00),
    ]

    print("MERCHANT NOISE EXAMPLES")
    print("=" * 80)
    for level in ["light", "medium", "heavy"]:
        print(f"\n--- {level.upper()} NOISE ---")
        rng_demo = np.random.default_rng(42)
        for merchant, date, amount in sample_merchants:
            noisy = corrupt_merchant(merchant, date, rng_demo, level)
            changed = " ✓" if noisy != merchant else ""
            print(f"  {merchant:35s} → {noisy:45s}{changed}")