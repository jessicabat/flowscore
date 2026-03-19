from dotenv import load_dotenv
load_dotenv()  # loads .env into os.environ
"""
FlowScore — Transaction Categorizer (Hybrid: Rules + Claude Sonnet)
====================================================================
"""
import argparse
import json
import os
import time
import re
from typing import Dict, List, Tuple, Optional
from collections import Counter

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    print("⚠️  anthropic package not installed. Run: pip install anthropic")
    print("   LLM categorization will be skipped. Rules-only mode available.\n")


# ============================================================================
# CATEGORY TAXONOMY
# ============================================================================

CATEGORIES = [
    "payroll",
    "gig_income",
    "government_benefits",
    "rent",
    "mortgage",
    "groceries",
    "dining",
    "food_delivery",
    "subscription",
    "utilities",
    "transportation",
    "shopping",
    "healthcare",
    "travel",
    "insurance",
    "loan_payment",
    "bnpl",
    "gambling",
    "gambling_win",
    "payday_loan_deposit",
    "payday_loan_repayment",
    "fee",
    "transfer",
    "atm",
    "other",
]


# ============================================================================
# STEP 1: RULE-BASED CATEGORIZER
# ============================================================================
# Fast, free, handles the easy cases. Maps merchant name keywords to categories.
# This catches ~70-80% of transactions in our synthetic data.

# Each rule is (keyword_or_pattern, category)
# Rules are checked in order — first match wins.
RULES = [
    # ---- GIG INCOME (must come BEFORE payroll and transfers) ----
    # These merchants contain "PAY", "TRANSFER", "PAYMENT RECEIVED" etc.
    # that would otherwise match payroll or transfer rules.
    ("DRIVER PAYOUT", "gig_income"),
    ("DRIVER EARNINGS", "gig_income"),
    ("DASHER PAY", "gig_income"),
    ("INSTACART SHOPPER", "gig_income"),
    ("FIVERR", "gig_income"),
    ("UPWORK", "gig_income"),
    ("ETSY DEPOSIT", "gig_income"),
    ("SHOPIFY PAYOUT", "gig_income"),
    ("STRIPE TRANSFER", "gig_income"),
    ("SQUARE DEPOSIT", "gig_income"),
    ("TASKRABBIT", "gig_income"),
    ("ROVER SITTER", "gig_income"),
    ("VENMO PAYMENT RECEIVED", "gig_income"),
    ("ZELLE PAYMENT IN", "gig_income"),
    ("PAYPAL TRANSFER IN", "gig_income"),

    # ---- PAYROLL (after gig to avoid catching "DASHER PAY" etc.) ----
    ("PAYROLL", "payroll"),
    ("DIR DEP", "payroll"),
    ("PRESTIGE WORLDWIDE", "payroll"),
    # Employer names ending in PAY — safe now because gig rules run first
    ("INDUSTRIES PAY", "payroll"),
    ("ENTERPRISES PAY", "payroll"),
    ("DYNAMICS PAY", "payroll"),
    ("CAPITAL PAY", "payroll"),
    ("CORP PAY", "payroll"),
    ("SYSTEMS PAY", "payroll"),
    ("MIFFLIN PAY", "payroll"),
    ("COOPER PAY", "payroll"),
    ("COMPANY PAY", "payroll"),
    # Employer names ending in DEP (deposit)
    ("INC DEP", "payroll"),
    ("CORP DEP", "payroll"),

    # ---- GOVERNMENT ----
    ("TREASURY", "government_benefits"),
    ("SSA BENEFIT", "government_benefits"),
    ("STATE UI", "government_benefits"),
    ("VA BENEFIT", "government_benefits"),
    ("EDD BENEFIT", "government_benefits"),

    # ---- TRAVEL (HERTZ before housing to avoid RENTAL matching RENT) ----
    ("HERTZ", "travel"),

    # Housing
    ("RENT", "rent"),
    ("LANDLORD", "rent"),
    ("PROPERTY MGMT", "rent"),
    ("APT.*PAYMENT", "rent"),
    ("HOUSING PMT", "rent"),
    ("EQUITY RESIDENTIAL", "rent"),
    ("AVALON COMMUNIT", "rent"),
    ("GREYSTAR", "rent"),
    ("CAMDEN PROPERTY", "rent"),
    ("ESSEX PROPERTY", "rent"),
    ("AIMCO", "rent"),
    ("INVITATION HOMES", "rent"),
    ("MORTGAGE", "mortgage"),
    ("HOME LENDING", "mortgage"),
    ("QUICKEN LOANS", "mortgage"),
    ("ROCKET MORTGAGE", "mortgage"),
    ("MTG", "mortgage"),

    # Subscriptions (fixed recurring charges)
    ("NETFLIX", "subscription"),
    ("SPOTIFY", "subscription"),
    ("HULU", "subscription"),
    ("HBO MAX", "subscription"),
    ("DISNEY PLUS", "subscription"),
    ("APPLE TV", "subscription"),
    ("YOUTUBE PREMIUM", "subscription"),
    ("PARAMOUNT", "subscription"),
    ("AMAZON PRIME", "subscription"),
    ("PEACOCK", "subscription"),
    ("APPLE MUSIC", "subscription"),
    ("ADOBE", "subscription"),
    ("PLANET FITNESS", "subscription"),
    ("EQUINOX", "subscription"),
    ("CROSSFIT", "subscription"),
    ("XBOX GAME", "subscription"),

    # Food delivery (before dining, since some overlap)
    ("DOORDASH ORDER", "food_delivery"),
    ("UBER EATS", "food_delivery"),
    ("GRUBHUB", "food_delivery"),
    ("POSTMATES", "food_delivery"),
    ("INSTACART DELIVERY", "food_delivery"),

    # Dining
    ("CHIPOTLE", "dining"),
    ("MCDONALDS", "dining"),
    ("MCDONALD", "dining"),
    ("STARBUCKS", "dining"),
    ("SUBWAY", "dining"),
    ("CHICK-FIL-A", "dining"),
    ("PANERA", "dining"),
    ("TACO BELL", "dining"),
    ("WENDYS", "dining"),
    ("SHAKE SHACK", "dining"),
    ("DUNKIN", "dining"),
    ("PANDA EXPRESS", "dining"),
    ("OLIVE GARDEN", "dining"),
    ("APPLEBEES", "dining"),
    ("IHOP", "dining"),
    ("DOMINOS", "dining"),
    ("BUFFALO WILD", "dining"),
    ("FIVE GUYS", "dining"),
    ("POPEYES", "dining"),
    ("IN-N-OUT", "dining"),
    ("CHEESECAKE FACTORY", "dining"),

    # Groceries
    ("KROGER", "groceries"),
    ("SAFEWAY", "groceries"),
    ("WHOLE FOODS", "groceries"),
    ("TRADER JOE", "groceries"),
    ("WALMART GROCERY", "groceries"),
    ("COSTCO", "groceries"),
    ("ALDI", "groceries"),
    ("PUBLIX", "groceries"),
    ("HEB GROCERY", "groceries"),
    ("SPROUTS", "groceries"),
    ("FOOD LION", "groceries"),
    ("STOP AND SHOP", "groceries"),
    ("GIANT FOOD", "groceries"),
    ("WEGMANS", "groceries"),

    # Utilities
    ("ELECTRIC COMPANY", "utilities"),
    ("WATER UTILITY", "utilities"),
    ("GAS UTILITY", "utilities"),
    ("COMCAST", "utilities"),
    ("XFINITY", "utilities"),
    ("ATT WIRELESS", "utilities"),
    ("VERIZON WIRELESS", "utilities"),
    ("T-MOBILE", "utilities"),
    ("SPECTRUM", "utilities"),

    # Transportation
    ("SHELL", "transportation"),
    ("CHEVRON", "transportation"),
    ("BP GAS", "transportation"),
    ("EXXONMOBIL", "transportation"),
    ("LYFT RIDE", "transportation"),
    ("UBER TRIP", "transportation"),
    ("METRO TRANSIT", "transportation"),
    ("PARKING", "transportation"),
    ("JIFFY LUBE", "transportation"),

    # Healthcare
    ("CVS PHARMACY", "healthcare"),
    ("WALGREENS", "healthcare"),
    ("RITE AID", "healthcare"),
    ("KAISER", "healthcare"),
    ("UNITED HEALTH", "healthcare"),
    ("BLUE CROSS", "healthcare"),
    ("QUEST DIAGNOSTICS", "healthcare"),
    ("LABCORP", "healthcare"),
    ("DENTAL", "healthcare"),

    # Travel
    ("AIRLINES", "travel"),
    ("SOUTHWEST AIR", "travel"),
    ("MARRIOTT", "travel"),
    ("HILTON", "travel"),
    ("AIRBNB", "travel"),
    ("EXPEDIA", "travel"),
    ("BOOKING.COM", "travel"),

    # Insurance
    ("STATE FARM", "insurance"),
    ("GEICO", "insurance"),
    ("PROGRESSIVE INS", "insurance"),
    ("ALLSTATE", "insurance"),

    # Loan payments
    ("SOFI LOAN", "loan_payment"),
    ("LENDING CLUB", "loan_payment"),
    ("PROSPER LOAN", "loan_payment"),
    ("UPSTART LOAN", "loan_payment"),
    ("NAVIENT", "loan_payment"),
    ("FEDLOAN", "loan_payment"),
    ("GREAT LAKES STU", "loan_payment"),
    ("NELNET", "loan_payment"),
    ("CAPITAL ONE AUTO", "loan_payment"),
    ("ALLY AUTO", "loan_payment"),

    # BNPL
    ("AFFIRM PAYMENT", "bnpl"),
    ("KLARNA", "bnpl"),
    ("AFTERPAY", "bnpl"),
    ("SEZZLE", "bnpl"),
    ("ZIP PAYMENT", "bnpl"),

    # Gambling
    ("DRAFTKINGS", "gambling"),
    ("FANDUEL", "gambling"),
    ("BETMGM", "gambling"),
    ("CAESARS SPORTSBOOK", "gambling"),
    ("POKERSTARS", "gambling"),

    # Payday
    ("ADVANCE AMERICA", "payday_loan"),
    ("CHECK INTO CASH", "payday_loan"),
    ("MONEYTREE", "payday_loan"),
    ("QC HOLDINGS", "payday_loan"),
    ("SPEEDY CASH", "payday_loan"),
    ("ACE CASH EXPRESS", "payday_loan"),

    # Fees
    ("OVERDRAFT FEE", "fee"),
    ("NSF FEE", "fee"),
    ("MAINTENANCE FEE", "fee"),
    ("LATE PAYMENT FEE", "fee"),
    ("RETURNED CHECK", "fee"),

    # Transfers
    ("VENMO", "transfer"),
    ("ZELLE", "transfer"),
    ("PAYPAL TRANSFER", "transfer"),
    ("WIRE TRANSFER", "transfer"),
    ("ACH TRANSFER", "transfer"),

    # ATM
    ("ATM WITHDRAWAL", "atm"),
    ("ATM CASH", "atm"),
    ("NON-NETWORK ATM", "atm"),

    # Shopping (broad — keep near the end so more specific rules win)
    ("AMAZON.COM", "shopping"),
    ("WALMART", "shopping"),
    ("TARGET", "shopping"),
    ("BEST BUY", "shopping"),
    ("HOME DEPOT", "shopping"),
    ("LOWES", "shopping"),
    ("NORDSTROM", "shopping"),
    ("MACYS", "shopping"),
    ("TJ MAXX", "shopping"),
    ("ROSS STORES", "shopping"),
    ("MARSHALLS", "shopping"),
    ("APPLE STORE", "shopping"),
    ("NIKE", "shopping"),
    ("IKEA", "shopping"),
    ("BED BATH", "shopping"),
    ("EBAY", "shopping"),
    ("ETSY PURCHASE", "shopping"),
    ("WAYFAIR", "shopping"),
]


def rule_based_categorize(merchant: str, amount: float) -> Optional[str]:
    """
    Attempt to categorize a transaction using keyword rules.
    Returns category string if matched, None if no rule applies.
    """
    merchant_upper = merchant.upper()

    # Special handling for payday loans based on amount direction
    for pattern, category in RULES:
        if category == "payday_loan":
            if re.search(pattern, merchant_upper):
                if amount > 0:
                    return "payday_loan_deposit"
                else:
                    return "payday_loan_repayment"

    # Special handling for gambling wins (positive amounts from gambling merchants)
    for pattern, category in RULES:
        if category == "gambling":
            if re.search(pattern, merchant_upper):
                if amount > 0:
                    return "gambling_win"
                else:
                    return "gambling"

    # Special handling for income-related merchants with positive amounts
    for pattern, category in RULES:
        if re.search(pattern, merchant_upper):
            # For Venmo/Zelle/PayPal — these could be income or transfers
            if category == "transfer":
                if amount > 0:
                    return "transfer"  # incoming transfer
                else:
                    return "transfer"  # outgoing transfer
            return category

    return None  # No rule matched → send to LLM


# ============================================================================
# STEP 2: LLM CATEGORIZER (Claude Sonnet via Anthropic API)
# ============================================================================

SYSTEM_PROMPT = """You are a financial transaction categorizer for a cash flow underwriting system.

Given a batch of bank transactions, classify each one into exactly one category.

CATEGORIES:
- payroll: salary/wage deposits from employers
- gig_income: earnings from gig platforms (Uber, DoorDash, Fiverr, etc.)
- government_benefits: government payments (SSA, unemployment, tax refunds)
- rent: rent payments to landlords or property management
- mortgage: mortgage payments to banks/lenders
- groceries: grocery store purchases
- dining: restaurant and fast food purchases
- food_delivery: food delivery service orders (DoorDash, UberEats, etc.)
- subscription: recurring subscription charges (streaming, gym, software)
- utilities: electric, water, gas, internet, phone bills
- transportation: gas stations, rideshares, transit, parking, auto maintenance
- shopping: retail purchases (Amazon, Walmart, clothing, electronics, home goods)
- healthcare: pharmacy, doctor visits, medical bills, dental
- travel: airlines, hotels, rental cars, travel booking
- insurance: insurance premium payments
- loan_payment: loan repayments (student loans, personal loans, auto loans)
- bnpl: buy-now-pay-later installment payments (Affirm, Klarna, Afterpay)
- gambling: sports betting, casino, poker deposits
- gambling_win: winnings/payouts from gambling platforms
- payday_loan_deposit: incoming funds from payday lenders
- payday_loan_repayment: repayments to payday lenders
- fee: bank fees (overdraft, NSF, maintenance, late payment)
- transfer: person-to-person transfers (Venmo, Zelle, PayPal, wire transfers)
- atm: ATM cash withdrawals
- other: anything that doesn't fit above

RULES:
- Respond ONLY with a JSON array of objects, one per transaction.
- Each object must have "id" (the transaction index) and "category" (one of the categories above).
- For gambling merchants: use "gambling" for negative amounts, "gambling_win" for positive.
- For payday lenders: use "payday_loan_deposit" for positive, "payday_loan_repayment" for negative.
- No explanations, no markdown, no backticks. Just the JSON array."""


def categorize_batch_llm(
    transactions: List[Dict],
    client: "anthropic.Anthropic",
    model: str = "claude-sonnet-4-20250514",
) -> List[Dict]:
    """
    Send a batch of transactions to Claude for categorization.

    HOW THIS WORKS (step by step):
    1. We format the transactions into a simple text prompt
    2. We send it to Claude's API with our system prompt
    3. Claude returns a JSON array with categories
    4. We parse the JSON and return the results

    Parameters:
        transactions: list of dicts with "merchant" and "amount" fields
        client: initialized anthropic.Anthropic() client
        model: which Claude model to use

    Returns:
        list of dicts with "id" and "category" fields
    """

    # Format transactions for the prompt
    # We send: index, merchant name, amount, and direction (deposit/withdrawal)
    lines = []
    for i, txn in enumerate(transactions):
        direction = "deposit" if txn["amount"] > 0 else "withdrawal"
        lines.append(
            f'{i}. "{txn["merchant"]}" | ${abs(txn["amount"]):.2f} | {direction}'
        )
    prompt = "Categorize these transactions:\n\n" + "\n".join(lines)

    # ================================================================
    # THIS IS THE API CALL
    # ================================================================
    # client.messages.create() sends a request to Claude's API.
    #
    # Parameters:
    #   model: "claude-sonnet-4-20250514" — Sonnet is fast and accurate
    #   max_tokens: maximum response length (1000 is plenty for a batch)
    #   system: the system prompt (sets Claude's role/behavior)
    #   messages: the conversation history
    #     - role "user" = what the human says
    #     - role "assistant" = what Claude says (for multi-turn)
    #     - We only need one user message here.
    #
    # The response object has:
    #   response.content[0].text — Claude's text reply
    #   response.usage.input_tokens — tokens we sent (we pay for these)
    #   response.usage.output_tokens — tokens Claude generated (we pay for these)
    # ================================================================

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    # Extract the text response
    raw_text = response.content[0].text

    # Parse JSON from Claude's response
    # Sometimes Claude wraps it in ```json ... ```, so we strip that
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        results = json.loads(cleaned)
    except json.JSONDecodeError:
        # If parsing fails, return "other" for all
        print(f"  ⚠️  JSON parse failed. Raw response: {raw_text[:200]}")
        results = [{"id": i, "category": "other"} for i in range(len(transactions))]

    # Track token usage for cost estimation
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    return results, input_tokens, output_tokens


# ============================================================================
# HYBRID PIPELINE
# ============================================================================

def categorize_consumer(
    consumer: Dict,
    client: Optional["anthropic.Anthropic"] = None,
    model: str = "claude-sonnet-4-20250514",
    batch_size: int = 30,
    rules_only: bool = False,
) -> Dict:
    """
    Categorize all transactions for one consumer using the hybrid approach.

    Step 1: Try rules on every transaction
    Step 2: Batch the unmatched ones and send to Claude
    Step 3: Merge results

    Returns dict with:
      - predicted_categories: list of predicted category per transaction
      - ground_truth: list of actual category per transaction
      - rule_matched: count of transactions matched by rules
      - llm_matched: count sent to LLM
      - total_input_tokens: API tokens used (input)
      - total_output_tokens: API tokens used (output)
    """
    transactions = consumer["transactions"]
    predicted = [None] * len(transactions)
    ground_truth = [t["category"] for t in transactions]

    # Step 1: Rule-based pass
    unmatched_indices = []
    for i, txn in enumerate(transactions):
        rule_result = rule_based_categorize(txn["merchant"], txn["amount"])
        if rule_result is not None:
            predicted[i] = rule_result
        else:
            unmatched_indices.append(i)

    rule_matched = len(transactions) - len(unmatched_indices)
    total_in_tokens = 0
    total_out_tokens = 0

    # Step 2: LLM pass for unmatched transactions
    if unmatched_indices and client is not None and not rules_only:
        unmatched_txns = [transactions[i] for i in unmatched_indices]

        # Process in batches (sending too many at once can cause issues)
        for batch_start in range(0, len(unmatched_txns), batch_size):
            batch = unmatched_txns[batch_start:batch_start + batch_size]
            batch_indices = unmatched_indices[batch_start:batch_start + batch_size]

            try:
                results, in_tok, out_tok = categorize_batch_llm(
                    batch, client, model
                )
                total_in_tokens += in_tok
                total_out_tokens += out_tok

                # Map results back to original indices
                for r in results:
                    idx_in_batch = r["id"]
                    if 0 <= idx_in_batch < len(batch_indices):
                        original_idx = batch_indices[idx_in_batch]
                        cat = r["category"]
                        # Validate category
                        if cat in CATEGORIES:
                            predicted[original_idx] = cat
                        else:
                            predicted[original_idx] = "other"

            except Exception as e:
                print(f"  ⚠️  API error: {e}")
                for idx in batch_indices:
                    if predicted[idx] is None:
                        predicted[idx] = "other"

            # Rate limiting: small delay between batches
            time.sleep(0.5)

    # Fill any remaining None predictions with "other"
    for i in range(len(predicted)):
        if predicted[i] is None:
            predicted[i] = "other"

    return {
        "consumer_id": consumer["consumer_id"],
        "predicted": predicted,
        "ground_truth": ground_truth,
        "rule_matched": rule_matched,
        "llm_matched": len(unmatched_indices),
        "total_input_tokens": total_in_tokens,
        "total_output_tokens": total_out_tokens,
    }


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_results(all_results: List[Dict]) -> Dict:
    """
    Compute accuracy metrics comparing predicted vs ground truth categories.
    Returns overall accuracy plus per-category precision/recall.
    """
    all_pred = []
    all_truth = []
    total_rule = 0
    total_llm = 0
    total_in_tokens = 0
    total_out_tokens = 0

    for r in all_results:
        all_pred.extend(r["predicted"])
        all_truth.extend(r["ground_truth"])
        total_rule += r["rule_matched"]
        total_llm += r["llm_matched"]
        total_in_tokens += r["total_input_tokens"]
        total_out_tokens += r["total_output_tokens"]

    # Overall accuracy
    correct = sum(p == t for p, t in zip(all_pred, all_truth))
    total = len(all_pred)
    accuracy = correct / total if total > 0 else 0

    # Per-category precision and recall
    categories_seen = sorted(set(all_truth + all_pred))
    per_category = {}
    for cat in categories_seen:
        tp = sum(1 for p, t in zip(all_pred, all_truth) if p == cat and t == cat)
        fp = sum(1 for p, t in zip(all_pred, all_truth) if p == cat and t != cat)
        fn = sum(1 for p, t in zip(all_pred, all_truth) if p != cat and t == cat)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = sum(1 for t in all_truth if t == cat)
        per_category[cat] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    # Confusion pairs (most common misclassifications)
    confusion = Counter()
    for p, t in zip(all_pred, all_truth):
        if p != t:
            confusion[(t, p)] += 1
    top_confusions = confusion.most_common(15)

    cost_input = total_in_tokens * 3.0 / 1_000_000
    cost_output = total_out_tokens * 15.0 / 1_000_000
    total_cost = cost_input + cost_output

    return {
        "total_transactions": total,
        "overall_accuracy": round(accuracy, 4),
        "correct": correct,
        "rule_matched": total_rule,
        "llm_matched": total_llm,
        "rule_pct": round(total_rule / total, 4) if total > 0 else 0,
        "per_category": per_category,
        "top_confusions": [
            {"true": t, "predicted": p, "count": c}
            for (t, p), c in top_confusions
        ],
        "api_usage": {
            "input_tokens": total_in_tokens,
            "output_tokens": total_out_tokens,
            "estimated_cost_usd": round(total_cost, 4),
        },
    }


def print_evaluation(metrics: Dict):
    """Pretty-print evaluation results."""
    print(f"\n{'='*70}")
    print(f"CATEGORIZATION RESULTS")
    print(f"{'='*70}")
    print(f"Total transactions:  {metrics['total_transactions']:,}")
    print(f"Overall accuracy:    {metrics['overall_accuracy']:.1%}")
    print(f"Correct:             {metrics['correct']:,}")
    print(f"Rule-matched:        {metrics['rule_matched']:,} ({metrics['rule_pct']:.1%})")
    print(f"LLM-matched:         {metrics['llm_matched']:,}")

    if metrics["api_usage"]["input_tokens"] > 0:
        print(f"\nAPI Usage:")
        print(f"  Input tokens:    {metrics['api_usage']['input_tokens']:,}")
        print(f"  Output tokens:   {metrics['api_usage']['output_tokens']:,}")
        print(f"  Estimated cost:  ${metrics['api_usage']['estimated_cost_usd']:.4f}")

    print(f"\nPer-Category Performance:")
    print(f"  {'Category':<28s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Support':>8s}")
    print(f"  {'-'*54}")
    for cat, m in sorted(
        metrics["per_category"].items(),
        key=lambda x: x[1]["support"],
        reverse=True
    ):
        print(f"  {cat:<28s} {m['precision']:>6.1%} {m['recall']:>6.1%} "
              f"{m['f1']:>6.1%} {m['support']:>8,}")

    if metrics["top_confusions"]:
        print(f"\nTop Misclassifications:")
        for c in metrics["top_confusions"][:10]:
            print(f"  {c['true']:>25s} → {c['predicted']:<25s}  ({c['count']:,})")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Categorize transactions using hybrid rules + LLM approach"
    )
    parser.add_argument("--input", required=True, help="Path to dataset JSON")
    parser.add_argument("--output", required=True, help="Path to save results")
    parser.add_argument("--n_consumers", type=int, default=100,
                        help="Number of consumers to process (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for consumer selection")
    parser.add_argument("--batch_size", type=int, default=30,
                        help="Transactions per LLM batch (default: 30)")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514",
                        help="Claude model to use")
    parser.add_argument("--rules-only", action="store_true",
                        help="Skip LLM, only use rule-based categorization")
    parser.add_argument("--noise", type=str, default="none",
                        choices=["none", "light", "medium", "heavy"],
                        help="Noise level to apply to merchant strings (default: none)")
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.input}...")
    with open(args.input) as f:
        all_consumers = json.load(f)
    print(f"Loaded {len(all_consumers)} consumers.")

    # Select subset
    import numpy as np
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(all_consumers), size=args.n_consumers, replace=False)
    consumers = [all_consumers[int(i)] for i in indices]
    print(f"Selected {len(consumers)} consumers for categorization.")

    total_txns = sum(c["n_transactions"] for c in consumers)
    print(f"Total transactions to categorize: {total_txns:,}")

    # Apply noise if requested
    if args.noise != "none":
        import sys, os as _os
        sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from noise_generator import add_noise_to_consumer
        print(f"Applying {args.noise} noise to merchant strings...")
        noise_rng = np.random.default_rng(args.seed + 1000)  # separate seed for noise
        consumers = [add_noise_to_consumer(c, noise_rng, args.noise) for c in consumers]
        print(f"Noise applied. Rules will now see realistic messy merchant names.\n")
    else:
        print("No noise applied (clean merchant names).\n")

    # Initialize API client
    client = None
    if not args.rules_only:
        if not HAS_ANTHROPIC:
            print("⚠️  anthropic package not found. Falling back to rules-only mode.")
            print("   Install with: pip install anthropic\n")
            args.rules_only = True
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                print("⚠️  ANTHROPIC_API_KEY not set. Falling back to rules-only mode.")
                print("   Set with: export ANTHROPIC_API_KEY='sk-ant-...'\n")
                args.rules_only = True
            else:
                client = anthropic.Anthropic()
                print(f"Using model: {args.model}")

    if args.rules_only:
        print("Running in RULES-ONLY mode (no API calls).\n")

    # Process consumers
    all_results = []
    for i, consumer in enumerate(consumers):
        result = categorize_consumer(
            consumer, client, args.model, args.batch_size, args.rules_only
        )
        all_results.append(result)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(consumers)} consumers...")

    # Evaluate
    metrics = evaluate_results(all_results)
    print_evaluation(metrics)

    # Save results
    output_data = {
        "metrics": metrics,
        "config": {
            "n_consumers": args.n_consumers,
            "model": args.model if not args.rules_only else "rules_only",
            "batch_size": args.batch_size,
            "seed": args.seed,
            "noise_level": args.noise,
        },
        "per_consumer": [
            {
                "consumer_id": r["consumer_id"],
                "rule_matched": r["rule_matched"],
                "llm_matched": r["llm_matched"],
                "accuracy": sum(
                    p == t for p, t in zip(r["predicted"], r["ground_truth"])
                ) / len(r["predicted"]) if r["predicted"] else 0,
            }
            for r in all_results
        ],
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()