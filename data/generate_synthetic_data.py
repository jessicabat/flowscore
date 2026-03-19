"""
FlowScore — Synthetic Transaction Data Generator
==================================================
Generates realistic consumer profiles with 6-12 months of bank transaction
history and credit default labels for cash flow underwriting research.

Usage:
    python generate_synthetic_data.py --n_consumers 5000 --seed 42
"""

import argparse, json, os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np

MERCHANTS = {
    "payroll": ["ACME CORP PAYROLL","GLOBEX INC DIR DEP","INITECH PAYROLL","HOOLI INC DIR DEP","STARK INDUSTRIES PAY","WAYNE ENTERPRISES PAY","PIED PIPER INC DEP","MASSIVE DYNAMICS PAY","OSCORP PAYROLL","VEHEMENT CAPITAL PAY","UMBRELLA CORP PAY","SOYLENT CORP DIR DEP","CYBERDYNE SYSTEMS PAY","TYRELL CORP PAYROLL","WEYLAND CORP DEP","DUNDER MIFFLIN PAY","PRESTIGE WORLDWIDE","VANDELAY INDUSTRIES PAY","STERLING COOPER PAY","BLUTH COMPANY PAYROLL"],
    "gig_income": ["UBER DRIVER PAYOUT","LYFT DRIVER EARNINGS","DOORDASH DASHER PAY","INSTACART SHOPPER","FIVERR WITHDRAWAL","UPWORK FREELANCE","ETSY DEPOSIT","SHOPIFY PAYOUT","STRIPE TRANSFER","SQUARE DEPOSIT","VENMO PAYMENT RECEIVED","ZELLE PAYMENT IN","PAYPAL TRANSFER IN","TASKRABBIT PAYOUT","ROVER SITTER PAY"],
    "government": ["US TREASURY TAX REF","SSA BENEFIT PAYMENT","STATE UI BENEFIT","VA BENEFIT PAYMENT","EDD BENEFIT PMT"],
    "rent": ["EQUITY RESIDENTIAL","AVALON COMMUNITIES","GREYSTAR MGMT","CAMDEN PROPERTY","ESSEX PROPERTY TRUST","AIMCO APARTMENTS","INVITATION HOMES","RENT PAYMENT ACH","LANDLORD PMT","PROPERTY MGMT RENT","APT RENT PAYMENT","HOUSING PMT ACH"],
    "mortgage": ["WELLS FARGO MORTGAGE","CHASE HOME LENDING","QUICKEN LOANS PMT","ROCKET MORTGAGE PMT","BANK OF AMERICA MTG","US BANK MORTGAGE"],
    "groceries": ["KROGER","SAFEWAY","WHOLE FOODS MKT","TRADER JOES","WALMART GROCERY","TARGET","COSTCO WHOLESALE","ALDI","PUBLIX","HEB GROCERY","SPROUTS FARMERS","FOOD LION","STOP AND SHOP","GIANT FOOD","WEGMANS"],
    "dining": ["CHIPOTLE","MCDONALDS","STARBUCKS","SUBWAY","CHICK-FIL-A","PANERA BREAD","TACO BELL","WENDYS","SHAKE SHACK","DUNKIN DONUTS","PANDA EXPRESS","OLIVE GARDEN","APPLEBEES","IHOP","DOMINOS PIZZA","BUFFALO WILD WINGS","FIVE GUYS","POPEYES","IN-N-OUT BURGER","CHEESECAKE FACTORY"],
    "food_delivery": ["DOORDASH ORDER","UBER EATS","GRUBHUB","POSTMATES","INSTACART DELIVERY"],
    "subscriptions": [("NETFLIX",15.99),("SPOTIFY",10.99),("HULU",17.99),("HBO MAX",15.99),("DISNEY PLUS",13.99),("APPLE TV PLUS",9.99),("YOUTUBE PREMIUM",13.99),("PARAMOUNT PLUS",11.99),("AMAZON PRIME",14.99),("PEACOCK",7.99),("APPLE MUSIC",10.99),("ADOBE CC",54.99),("PLANET FITNESS",25.00),("EQUINOX GYM",200.00),("CROSSFIT GYM",150.00),("XBOX GAME PASS",16.99)],
    "utilities": [("ELECTRIC COMPANY",(80,250)),("WATER UTILITY",(30,80)),("GAS UTILITY CO",(40,150)),("COMCAST XFINITY",(60,120)),("ATT WIRELESS",(50,130)),("VERIZON WIRELESS",(50,140)),("T-MOBILE",(40,120)),("SPECTRUM INTERNET",(50,100))],
    "transportation": ["SHELL","CHEVRON","BP GAS","EXXONMOBIL","LYFT RIDE","UBER TRIP","METRO TRANSIT","PARKING METER","JIFFY LUBE"],
    "shopping": ["AMAZON.COM","WALMART","TARGET","BEST BUY","HOME DEPOT","LOWES","NORDSTROM","MACYS","TJ MAXX","ROSS STORES","MARSHALLS","APPLE STORE","NIKE","IKEA","BED BATH BEYOND","EBAY","ETSY PURCHASE","WAYFAIR"],
    "healthcare": ["CVS PHARMACY","WALGREENS","RITE AID","KAISER PERMANENTE","UNITED HEALTH","BLUE CROSS","QUEST DIAGNOSTICS","LABCORP","DENTAL OFFICE"],
    "travel": ["UNITED AIRLINES","DELTA AIRLINES","SOUTHWEST AIR","AMERICAN AIRLINES","MARRIOTT HOTEL","HILTON HOTEL","AIRBNB","EXPEDIA","BOOKING.COM","HERTZ RENTAL"],
    "insurance": [("STATE FARM INS",(80,250)),("GEICO INSURANCE",(70,200)),("PROGRESSIVE INS",(75,220)),("ALLSTATE INS",(80,230))],
    "loan_payments": ["SOFI LOAN PMT","LENDING CLUB PMT","PROSPER LOAN PMT","UPSTART LOAN PMT","NAVIENT STUDENT LOAN","FEDLOAN SERVICING","GREAT LAKES STU LOAN","NELNET STUDENT PMT","CAPITAL ONE AUTO PMT","ALLY AUTO FINANCE"],
    "bnpl": ["AFFIRM PAYMENT","KLARNA PAYMENT","AFTERPAY PMT","SEZZLE PAYMENT","ZIP PAYMENT"],
    "gambling": ["DRAFTKINGS","FANDUEL","BETMGM","CAESARS SPORTSBOOK","POKERSTARS"],
    "payday_lender": ["ADVANCE AMERICA","CHECK INTO CASH","MONEYTREE","QC HOLDINGS","SPEEDY CASH","ACE CASH EXPRESS"],
    "transfers": ["VENMO PAYMENT","ZELLE SEND","PAYPAL TRANSFER","WIRE TRANSFER OUT","ACH TRANSFER OUT"],
    "atm": ["ATM WITHDRAWAL","ATM CASH WITHDRAW","NON-NETWORK ATM"],
}

@dataclass
class Archetype:
    name: str; weight: float; income_type: str
    monthly_income_mean: Tuple[float,float]; monthly_income_std_pct: float
    pay_frequency: str; n_income_sources: Tuple[int,int]
    housing_type: str; housing_pct_income: Tuple[float,float]
    n_subscriptions: Tuple[int,int]; dining_freq_wk: Tuple[float,float]
    grocery_freq_wk: Tuple[float,float]; shopping_freq_mo: Tuple[int,int]
    avg_shop_amt: Tuple[float,float]; delivery_freq_wk: Tuple[float,float]
    n_loans: Tuple[int,int]; loan_pmt_range: Tuple[float,float]
    bnpl_prob: float; n_bnpl: Tuple[int,int]
    overdraft_mo_prob: float; gambling_prob: float
    gambling_mo_amt: Tuple[float,float]; payday_prob: float
    base_default_prob: Tuple[float,float]; history_months: Tuple[int,int]
    travel_freq_yr: Tuple[int,int]; travel_amt: Tuple[float,float]

ARCHETYPES = [
    Archetype("stable_salaried",0.30,"salaried",(3500,7000),0.02,"biweekly",(1,1),"rent",(0.25,0.35),(2,5),(2,5),(1,3),(2,6),(20,120),(0.5,2),(0,1),(150,400),0.15,(1,1),0.02,0.05,(20,100),0.0,(0.03,0.10),(9,12),(1,3),(200,800)),
    Archetype("gig_worker",0.15,"gig",(2000,5000),0.25,"irregular",(2,4),"rent",(0.30,0.45),(1,3),(3,7),(1,2),(1,4),(15,80),(1,4),(0,1),(100,300),0.35,(1,3),0.12,0.10,(30,200),0.08,(0.15,0.28),(6,12),(0,1),(100,400)),
    Archetype("high_earner_high_spender",0.15,"salaried",(7000,15000),0.03,"biweekly",(1,2),"mortgage",(0.20,0.30),(4,8),(4,8),(2,4),(4,10),(50,300),(2,5),(0,2),(200,600),0.10,(1,2),0.01,0.08,(50,500),0.0,(0.05,0.15),(9,12),(2,6),(500,2000)),
    Archetype("financially_stressed",0.15,"mixed",(1800,3500),0.15,"biweekly",(1,2),"rent",(0.35,0.50),(1,3),(2,5),(1,2),(1,3),(10,50),(1,3),(1,3),(100,350),0.50,(2,4),0.25,0.15,(50,300),0.20,(0.30,0.50),(6,10),(0,1),(100,300)),
    Archetype("thin_file_newcomer",0.12,"salaried",(2500,5000),0.04,"biweekly",(1,1),"rent",(0.28,0.40),(1,3),(1,4),(1,2),(1,4),(15,80),(0.5,2),(0,0),(0,0),0.20,(1,1),0.05,0.03,(10,50),0.0,(0.08,0.18),(3,6),(0,2),(150,600)),
    Archetype("overextended",0.13,"salaried",(3500,7000),0.05,"biweekly",(1,2),"rent",(0.28,0.38),(3,7),(3,7),(1,3),(3,8),(30,200),(2,5),(2,4),(150,500),0.60,(2,4),0.15,0.12,(30,250),0.10,(0.25,0.42),(8,12),(1,3),(200,700)),
]

class Generator:
    def __init__(self, rng):
        self.rng = rng

    def ri(self, lo, hi):
        """Random int [lo, hi] inclusive, returns Python int."""
        if lo >= hi: return lo
        return int(self.rng.integers(lo, hi + 1))

    def pick(self, seq):
        return seq[int(self.rng.integers(0, len(seq)))]

    def picks(self, seq, n):
        n = min(n, len(seq))
        idxs = self.rng.choice(len(seq), size=n, replace=False)
        return [seq[int(i)] for i in idxs]

    def txn(self, date, amount, merchant, category):
        return {"date": date.strftime("%Y-%m-%d"), "amount": round(float(amount), 2), "merchant": merchant, "category": category}

    def rand_date(self, start, end):
        span = (end - start).days
        if span <= 0: return start
        return start + timedelta(days=self.ri(0, span - 1))

    def generate(self, cid, a):
        rng = self.rng
        n_mo = self.ri(a.history_months[0], a.history_months[1])
        end = datetime(2024, 12, 31)
        start = end - timedelta(days=n_mo * 30)
        income = float(rng.uniform(*a.monthly_income_mean))

        housing_amt = 0.0
        if a.housing_type != "none":
            housing_amt = round(income * float(rng.uniform(*a.housing_pct_income)), 2)

        txns = []
        n_src = self.ri(a.n_income_sources[0], a.n_income_sources[1])
        txns += self._income(a, income, n_src, start, end)
        if housing_amt > 0:
            txns += self._housing(a, housing_amt, start, end)
        txns += self._subscriptions(self.ri(a.n_subscriptions[0], a.n_subscriptions[1]), start, end)
        txns += self._utilities(start, end)
        txns += self._variable("groceries", a.grocery_freq_wk, (25,150), start, end)
        txns += self._variable("dining", a.dining_freq_wk, (8,55), start, end)
        txns += self._variable("food_delivery", a.delivery_freq_wk, (12,60), start, end)
        txns += self._variable("transportation", (1,4), (8,70), start, end)
        txns += self._variable("atm", (0.2,1.0), (20,200), start, end)
        txns += self._shopping(a, start, end)
        txns += self._sporadic("healthcare", (0.3,1.5), (15,250), start, end)
        txns += self._insurance(start, end)

        n_loans = self.ri(a.n_loans[0], a.n_loans[1])
        txns += self._loans(n_loans, a.loan_pmt_range, start, end)

        if float(rng.random()) < a.bnpl_prob:
            txns += self._bnpl(self.ri(a.n_bnpl[0], a.n_bnpl[1]), start, end)
        if float(rng.random()) < a.gambling_prob:
            txns += self._gambling(a, start, end)

        has_payday = float(rng.random()) < a.payday_prob
        if has_payday:
            txns += self._payday(start, end)
        txns += self._fees(a, start, end)
        txns += self._transfers(start, end)
        txns += self._travel(a, start, end)

        txns.sort(key=lambda t: t["date"])

        dp = self._default_prob(a, txns, income, housing_amt, n_loans, has_payday)
        default = int(float(rng.random()) < dp)
        trad = self._trad_score(a, n_loans, has_payday, n_mo)

        return {
            "consumer_id": cid, "archetype": a.name, "history_months": n_mo,
            "monthly_income_approx": round(income, 2), "n_transactions": len(txns),
            "transactions": txns, "default_12m": default,
            "default_probability": round(dp, 4), "traditional_score": trad,
        }

    def _income(self, a, income, n_src, start, end):
        rng = self.rng; txns = []
        if a.income_type == "salaried":
            emp = self.pick(MERCHANTS["payroll"])
            pc = income / 2
            cur = start + timedelta(days=self.ri(0, 13))
            while cur <= end:
                txns.append(self.txn(cur, rng.normal(pc, pc*0.02), emp, "payroll"))
                cur += timedelta(days=14 + self.ri(-1, 1))
        elif a.income_type == "gig":
            srcs = self.picks(MERCHANTS["gig_income"], min(n_src, len(MERCHANTS["gig_income"])))
            per = income / len(srcs)
            for s in srcs:
                cur = start + timedelta(days=self.ri(0, 6))
                while cur <= end:
                    amt = abs(float(rng.normal(per/4, per/6)))
                    if amt > 10: txns.append(self.txn(cur, amt, s, "gig_income"))
                    cur += timedelta(days=self.ri(2, 10))
        elif a.income_type == "mixed":
            emp = self.pick(MERCHANTS["payroll"])
            pc = income * 0.7 / 2
            cur = start + timedelta(days=self.ri(0, 13))
            while cur <= end:
                txns.append(self.txn(cur, rng.normal(pc, pc*0.03), emp, "payroll"))
                cur += timedelta(days=14 + self.ri(-1, 1))
            gs = self.pick(MERCHANTS["gig_income"]); gp = income * 0.3 / 3
            cur = start + timedelta(days=self.ri(0, 9))
            while cur <= end:
                amt = abs(float(rng.normal(gp, gp*0.4)))
                if amt > 10: txns.append(self.txn(cur, amt, gs, "gig_income"))
                cur += timedelta(days=self.ri(5, 14))
        if a.name == "financially_stressed":
            dec = float(rng.uniform(0.02, 0.06))
            for t in txns:
                if t["amount"] > 0:
                    mo_in = (datetime.strptime(t["date"], "%Y-%m-%d") - start).days / 30
                    t["amount"] = round(max(50, t["amount"] * (1 - dec * mo_in)), 2)
        return txns

    def _housing(self, a, amt, start, end):
        txns = []; m = MERCHANTS["mortgage"] if a.housing_type == "mortgage" else MERCHANTS["rent"]
        ll = self.pick(m); cat = a.housing_type
        cur = start.replace(day=1) + timedelta(days=30)
        while cur <= end:
            d = cur.replace(day=min(self.ri(1, 3), 28))
            txns.append(self.txn(d, -(amt + float(self.rng.normal(0, amt*0.005))), ll, cat))
            cur += timedelta(days=30)
        return txns

    def _subscriptions(self, n, start, end):
        txns = []; n = min(n, len(MERCHANTS["subscriptions"]))
        for idx in self.picks(range(len(MERCHANTS["subscriptions"])), n):
            name, price = MERCHANTS["subscriptions"][idx]
            cd = self.ri(1, 27); cur = start.replace(day=1) + timedelta(days=30)
            while cur <= end:
                txns.append(self.txn(cur.replace(day=cd), -price, name, "subscription"))
                cur += timedelta(days=30)
        return txns

    def _utilities(self, start, end):
        rng = self.rng; txns = []; n = min(self.ri(3, 5), len(MERCHANTS["utilities"]))
        for idx in self.picks(range(len(MERCHANTS["utilities"])), n):
            name, (lo, hi) = MERCHANTS["utilities"][idx]
            base = float(rng.uniform(lo, hi)); cd = self.ri(5, 24)
            cur = start.replace(day=1) + timedelta(days=30)
            while cur <= end:
                txns.append(self.txn(cur.replace(day=min(cd, 28)), -float(rng.normal(base, base*0.1)), name, "utilities"))
                cur += timedelta(days=30)
        return txns

    def _variable(self, cat, freq_range, amt_range, start, end):
        rng = self.rng; txns = []; span = (end - start).days
        if span <= 0: return txns
        freq = float(rng.uniform(*freq_range)); n = int(freq * span / 7)
        for _ in range(n):
            txns.append(self.txn(self.rand_date(start, end), -float(rng.uniform(*amt_range)), self.pick(MERCHANTS[cat]), cat))
        return txns

    def _shopping(self, a, start, end):
        rng = self.rng; txns = []; span = (end - start).days
        if span <= 0: return txns
        n = int(float(rng.uniform(*a.shopping_freq_mo)) * span / 30)
        for _ in range(n):
            txns.append(self.txn(self.rand_date(start, end), -float(rng.uniform(*a.avg_shop_amt)), self.pick(MERCHANTS["shopping"]), "shopping"))
        return txns

    def _sporadic(self, cat, mo_freq, amt_range, start, end):
        rng = self.rng; txns = []; span = (end - start).days
        if span <= 0: return txns
        n = max(1, int(float(rng.uniform(*mo_freq)) * span / 30))
        for _ in range(n):
            txns.append(self.txn(self.rand_date(start, end), -float(rng.uniform(*amt_range)), self.pick(MERCHANTS[cat]), cat))
        return txns

    def _insurance(self, start, end):
        rng = self.rng; txns = []
        if float(rng.random()) < 0.6:
            idx = self.ri(0, len(MERCHANTS["insurance"]) - 1)
            name, (lo, hi) = MERCHANTS["insurance"][idx]
            amt = float(rng.uniform(lo, hi)); cd = self.ri(1, 27)
            cur = start.replace(day=1) + timedelta(days=30)
            while cur <= end:
                txns.append(self.txn(cur.replace(day=min(cd, 28)), -amt, name, "insurance"))
                cur += timedelta(days=30)
        return txns

    def _loans(self, n, pmt_range, start, end):
        rng = self.rng; txns = []
        if n == 0 or pmt_range[1] == 0: return txns
        for lender in self.picks(MERCHANTS["loan_payments"], min(n, len(MERCHANTS["loan_payments"]))):
            amt = float(rng.uniform(*pmt_range)); pd = self.ri(1, 27)
            cur = start.replace(day=1) + timedelta(days=30)
            while cur <= end:
                if float(rng.random()) > 0.03:
                    txns.append(self.txn(cur.replace(day=min(pd, 28)), -amt, lender, "loan_payment"))
                cur += timedelta(days=30)
        return txns

    def _bnpl(self, n, start, end):
        rng = self.rng; txns = []; span = (end - start).days
        for prov in self.picks(MERCHANTS["bnpl"], min(n, len(MERCHANTS["bnpl"]))):
            inst = round(float(rng.uniform(50, 400)) / 4, 2)
            off = self.ri(0, max(0, span - 60))
            first = start + timedelta(days=off)
            for i in range(4):
                d = first + timedelta(days=14 * i)
                if d <= end: txns.append(self.txn(d, -inst, prov, "bnpl"))
        return txns

    def _gambling(self, a, start, end):
        rng = self.rng; txns = []; span = (end - start).days
        if span <= 0: return txns
        mo_amt = float(rng.uniform(*a.gambling_mo_amt)); n_mo = span / 30
        n = int(n_mo * float(rng.uniform(2, 6)))
        per = mo_amt / max(1, n / n_mo)
        for _ in range(n):
            d = self.rand_date(start, end); m = self.pick(MERCHANTS["gambling"])
            amt = abs(float(rng.normal(per, per*0.5)))
            txns.append(self.txn(d, -amt, m, "gambling"))
            if float(rng.random()) < 0.30:
                txns.append(self.txn(d, amt * float(rng.uniform(0.5, 2.5)), m, "gambling_win"))
        return txns

    def _payday(self, start, end):
        rng = self.rng; txns = []; span = (end - start).days; n_mo = span / 30
        lender = self.pick(MERCHANTS["payday_lender"])
        for _ in range(self.ri(1, min(3, max(1, int(n_mo))))):
            d = start + timedelta(days=self.ri(0, max(0, span - 30)))
            amt = float(rng.uniform(200, 800))
            txns.append(self.txn(d, amt, lender, "payday_loan_deposit"))
            rd = d + timedelta(days=self.ri(14, 30))
            if rd <= end:
                txns.append(self.txn(rd, -(amt * float(rng.uniform(1.10, 1.30))), lender, "payday_loan_repayment"))
        return txns

    def _fees(self, a, start, end):
        rng = self.rng; txns = []; span = (end - start).days
        if span <= 0: return txns
        for _ in range(int(span / 30)):
            if float(rng.random()) < a.overdraft_mo_prob:
                ft = self.pick(["OVERDRAFT FEE", "NSF FEE"])
                txns.append(self.txn(self.rand_date(start, end), -(35.0 if ft == "OVERDRAFT FEE" else 30.0), ft, "fee"))
        return txns

    def _transfers(self, start, end):
        rng = self.rng; txns = []; span = (end - start).days
        if span <= 0: return txns
        for _ in range(int(float(rng.uniform(1, 4)) * span / 30)):
            d = self.rand_date(start, end); m = self.pick(MERCHANTS["transfers"])
            amt = float(rng.uniform(10, 300))
            txns.append(self.txn(d, amt if float(rng.random()) < 0.5 else -amt, m, "transfer"))
        return txns

    def _travel(self, a, start, end):
        rng = self.rng; txns = []; span = (end - start).days
        if span <= 7: return txns
        n = max(0, int(self.ri(a.travel_freq_yr[0], a.travel_freq_yr[1]) * span / 365))
        for _ in range(n):
            txns.append(self.txn(start + timedelta(days=self.ri(0, span - 7)), -float(rng.uniform(*a.travel_amt)), self.pick(MERCHANTS["travel"]), "travel"))
        return txns

    def _default_prob(self, a, txns, income, housing, n_loans, has_payday):
        rng = self.rng
        base = float(rng.uniform(*a.base_default_prob))
        tot_out = sum(abs(t["amount"]) for t in txns if t["amount"] < 0)
        tot_in = sum(t["amount"] for t in txns if t["amount"] > 0)
        n_mo = max(1, a.history_months[0])
        net = (tot_in - tot_out) / n_mo
        if net < 0: base += 0.08
        elif net < income * 0.05: base += 0.03
        base += sum(1 for t in txns if t["category"] == "fee") * 0.015
        gamb = sum(abs(t["amount"]) for t in txns if t["category"] == "gambling")
        if gamb > 500: base += 0.06
        elif gamb > 100: base += 0.03
        if has_payday: base += 0.10
        mo_oblig = housing + sum(abs(t["amount"]) for t in txns if t["category"] in ("loan_payment","bnpl")) / n_mo
        ratio = mo_oblig / max(income, 1)
        if ratio > 0.60: base += 0.08
        elif ratio > 0.45: base += 0.04
        return max(0.01, min(0.85, base + float(rng.normal(0, 0.03))))

    def _trad_score(self, a, n_loans, has_payday, n_mo):
        rng = self.rng
        ranges = {"stable_salaried":(680,780),"gig_worker":(580,680),"high_earner_high_spender":(700,800),"financially_stressed":(500,620),"thin_file_newcomer":(550,650),"overextended":(600,700)}
        lo, hi = ranges[a.name]; s = float(rng.uniform(lo, hi))
        if has_payday: s -= float(rng.uniform(30, 80))
        if n_loans > 2: s -= float(rng.uniform(10, 30))
        if n_mo < 6: s -= float(rng.uniform(20, 50))
        if a.name == "thin_file_newcomer": s -= float(rng.uniform(10, 40))
        return int(max(300, min(850, s + float(rng.normal(0, 15)))))


def generate_dataset(n_consumers=5000, seed=42, output_dir="."):
    rng = np.random.default_rng(seed)
    gen = Generator(rng)
    weights = np.array([a.weight for a in ARCHETYPES]); weights /= weights.sum()
    aidx = rng.choice(len(ARCHETYPES), size=n_consumers, p=weights)

    consumers = []; counts = {}
    print(f"Generating {n_consumers} consumer profiles...")
    for i in range(n_consumers):
        a = ARCHETYPES[int(aidx[i])]
        c = gen.generate(f"c_{i:05d}", a)
        consumers.append(c)
        counts[a.name] = counts.get(a.name, 0) + 1
        if (i+1) % 1000 == 0: print(f"  {i+1}/{n_consumers}...")

    nd = sum(c["default_12m"] for c in consumers)
    tt = sum(c["n_transactions"] for c in consumers)
    ts = [c["traditional_score"] for c in consumers]

    print(f"\n{'='*60}\nDATASET SUMMARY\n{'='*60}")
    print(f"Consumers:          {n_consumers:,}")
    print(f"Total transactions: {tt:,}")
    print(f"Avg txns/consumer:  {tt/n_consumers:.0f}")
    print(f"Default rate:       {nd/n_consumers:.1%}")
    print(f"\nArchetype breakdown:")
    for name, cnt in sorted(counts.items()):
        dr = sum(1 for c in consumers if c["archetype"]==name and c["default_12m"]==1)/max(cnt,1)
        print(f"  {name:30s} n={cnt:4d} ({cnt/n_consumers:5.1%})  default={dr:.1%}")
    print(f"\nTraditional score: mean={np.mean(ts):.0f} std={np.std(ts):.0f} range=[{np.min(ts)}, {np.max(ts)}]")

    os.makedirs(output_dir, exist_ok=True)
    fp = os.path.join(output_dir, "flowscore_dataset.json")
    with open(fp, "w") as f: json.dump(consumers, f)
    print(f"\nSaved: {fp} ({os.path.getsize(fp)/1024/1024:.1f} MB)")

    sp = os.path.join(output_dir, "flowscore_sample.json")
    with open(sp, "w") as f: json.dump(consumers[:20], f, indent=2)
    print(f"Saved: {sp}")

    stats = {"n_consumers":n_consumers,"total_transactions":tt,"avg_txns":round(tt/n_consumers,1),
             "default_rate":round(nd/n_consumers,4),
             "archetypes":{n:{"n":c,"pct":round(c/n_consumers,3)} for n,c in counts.items()},
             "trad_score":{"mean":round(float(np.mean(ts)),1),"std":round(float(np.std(ts)),1)},
             "seed":seed}
    stp = os.path.join(output_dir, "dataset_stats.json")
    with open(stp, "w") as f: json.dump(stats, f, indent=2)
    print(f"Saved: {stp}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate synthetic data for FlowScore")
    p.add_argument("--n_consumers", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default=".")
    a = p.parse_args()
    generate_dataset(a.n_consumers, a.seed, a.output_dir)