"""
Short Squeeze Finder - Backend API
Smart multi-source data fetching with intelligent rate limit management
"""

import os
import json
from flask import Flask, jsonify, render_template
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import time
import requests

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# API Keys
FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY', 'd4sq1c9r01qnb2sfb0tgd4sq1c9r01qnb2sfb0u0')
FMP_API_KEY = os.environ.get('FMP_API_KEY', 'YgihXqC8qr1xP5UMt4BrO6BWtV8JlI9f')

# Cache files
CACHE_FILE = 'stock_cache.json'
SHORT_INTEREST_CACHE_FILE = 'short_interest_cache.json'
STOCK_DATA_CACHE_FILE = 'stock_data_cache.json'

# Cache settings
STOCK_CACHE_TTL_MINUTES = 10  # How long stock data is valid

# Short interest cache (persisted to file)
short_interest_cache = {}

# Full stock data cache (persisted to file) - serves multiple users
stock_data_cache = {}


def load_short_interest_cache():
    """Load short interest data from file"""
    global short_interest_cache
    try:
        if os.path.exists(SHORT_INTEREST_CACHE_FILE):
            with open(SHORT_INTEREST_CACHE_FILE, 'r') as f:
                data = json.load(f)
                short_interest_cache = data.get('data', {})
                last_updated = data.get('last_updated', '')
                print(f"[Cache] Loaded short interest for {len(short_interest_cache)} tickers (updated: {last_updated})")
                return True
    except Exception as e:
        print(f"[Cache] Error loading short interest cache: {e}")
    return False


def save_short_interest_cache():
    """Save short interest data to file"""
    try:
        with open(SHORT_INTEREST_CACHE_FILE, 'w') as f:
            json.dump({
                'data': short_interest_cache,
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
        print(f"[Cache] Saved short interest for {len(short_interest_cache)} tickers")
    except Exception as e:
        print(f"[Cache] Error saving short interest cache: {e}")


def get_cached_short_interest(ticker: str) -> dict:
    """Get short interest from local cache"""
    if ticker in short_interest_cache:
        return short_interest_cache[ticker]
    # Also check historical data
    if ticker in SHORT_INTEREST_DATA:
        return SHORT_INTEREST_DATA[ticker]
    return None


def cache_short_interest(ticker: str, data: dict):
    """Save short interest to local cache"""
    if data and data.get('short_percent_float'):
        short_interest_cache[ticker] = {
            'short_percent_float': data['short_percent_float'],
            'shares_short': data.get('shares_short', 0),
            'float_shares': data.get('float_shares', 0),
            'days_to_cover': data.get('days_to_cover', 0),
            'cached_at': datetime.now().isoformat()
        }
        save_short_interest_cache()


# ============================================================
# FULL STOCK DATA CACHE - Serves multiple users
# ============================================================

def load_stock_data_cache():
    """Load full stock data from file"""
    global stock_data_cache
    try:
        if os.path.exists(STOCK_DATA_CACHE_FILE):
            with open(STOCK_DATA_CACHE_FILE, 'r') as f:
                stock_data_cache = json.load(f)
                print(f"[Cache] Loaded stock data for {len(stock_data_cache)} tickers")
                return True
    except Exception as e:
        print(f"[Cache] Error loading stock data cache: {e}")
    return False


def save_stock_data_cache():
    """Save full stock data to file"""
    try:
        with open(STOCK_DATA_CACHE_FILE, 'w') as f:
            json.dump(stock_data_cache, f, indent=2)
    except Exception as e:
        print(f"[Cache] Error saving stock data cache: {e}")


def is_stock_cache_fresh(ticker: str) -> bool:
    """Check if cached stock data is still fresh"""
    if ticker not in stock_data_cache:
        return False
    
    cached_at = stock_data_cache[ticker].get('cached_at')
    if not cached_at:
        return False
    
    try:
        cache_time = datetime.fromisoformat(cached_at)
        age_minutes = (datetime.now() - cache_time).total_seconds() / 60
        return age_minutes < STOCK_CACHE_TTL_MINUTES
    except:
        return False


def get_cached_stock_data(ticker: str) -> dict:
    """Get stock data from cache if fresh"""
    if is_stock_cache_fresh(ticker):
        return stock_data_cache[ticker]
    return None


def cache_stock_data(ticker: str, data: dict):
    """Save full stock data to cache"""
    if data:
        data['cached_at'] = datetime.now().isoformat()
        stock_data_cache[ticker] = data
        save_stock_data_cache()


class SmartRateLimiter:
    """
    Intelligent rate limiter that tracks API usage across multiple sources
    and decides which source to use based on available quota.
    """
    
    def __init__(self):
        # Rate limits per source (calls per time window)
        self.limits = {
            'finnhub': {'calls': 55, 'window': 60},      # 60/min, use 55 to be safe
            'yahoo': {'calls': 80, 'window': 3600},       # ~100/hour, use 80
            'fmp': {'calls': 240, 'window': 86400},       # 250/day, use 240
        }
        
        # Track API calls with timestamps
        self.call_history = defaultdict(list)
        
        # Cooldown tracking (when rate limited)
        self.cooldowns = {}
        
        # Success/failure tracking for adaptive behavior
        self.success_rate = defaultdict(lambda: {'success': 0, 'fail': 0})
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def _clean_old_calls(self, source: str):
        """Remove calls outside the time window"""
        window = self.limits[source]['window']
        cutoff = time.time() - window
        self.call_history[source] = [t for t in self.call_history[source] if t > cutoff]
    
    def get_available_calls(self, source: str) -> int:
        """Get number of available API calls for a source"""
        with self.lock:
            self._clean_old_calls(source)
            limit = self.limits[source]['calls']
            used = len(self.call_history[source])
            return max(0, limit - used)
    
    def is_in_cooldown(self, source: str) -> bool:
        """Check if source is in cooldown after rate limit error"""
        if source in self.cooldowns:
            if time.time() < self.cooldowns[source]:
                return True
            del self.cooldowns[source]
        return False
    
    def record_call(self, source: str, success: bool = True):
        """Record an API call"""
        with self.lock:
            self.call_history[source].append(time.time())
            if success:
                self.success_rate[source]['success'] += 1
            else:
                self.success_rate[source]['fail'] += 1
    
    def set_cooldown(self, source: str, seconds: int = 60):
        """Set cooldown for a source after rate limit"""
        self.cooldowns[source] = time.time() + seconds
        print(f"[RateLimiter] {source} in cooldown for {seconds}s")
    
    def get_best_source(self, need_short_interest: bool = False) -> str:
        """
        Intelligently select the best data source based on:
        - Available quota
        - Cooldown status
        - Data quality needs
        - Historical success rate
        """
        # Priority order based on data quality
        if need_short_interest:
            # For short interest: Yahoo > FMP > Finnhub
            priority = ['yahoo', 'fmp', 'finnhub']
        else:
            # For price data: Finnhub > Yahoo > FMP (Finnhub has best rate limits)
            priority = ['finnhub', 'yahoo', 'fmp']
        
        for source in priority:
            if self.is_in_cooldown(source):
                continue
            
            available = self.get_available_calls(source)
            if available > 0:
                return source
        
        # All sources exhausted - return the one with shortest cooldown
        return min(priority, key=lambda s: self.cooldowns.get(s, 0))
    
    def get_status(self) -> dict:
        """Get current status of all sources"""
        status = {}
        for source in self.limits:
            self._clean_old_calls(source)
            status[source] = {
                'available': self.get_available_calls(source),
                'limit': self.limits[source]['calls'],
                'window_minutes': self.limits[source]['window'] / 60,
                'in_cooldown': self.is_in_cooldown(source),
                'success_rate': self._get_success_rate(source)
            }
        return status
    
    def _get_success_rate(self, source: str) -> float:
        total = self.success_rate[source]['success'] + self.success_rate[source]['fail']
        if total == 0:
            return 100.0
        return round(self.success_rate[source]['success'] / total * 100, 1)


# Initialize smart rate limiter
rate_limiter = SmartRateLimiter()

# In-memory cache with TTL
class DataCache:
    def __init__(self, default_ttl: int = 1800):  # 30 min default TTL
        self.cache = {}
        self.ttl = default_ttl
        self.lock = threading.Lock()
    
    def get(self, key: str) -> dict:
        with self.lock:
            if key in self.cache:
                data, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    return data
                del self.cache[key]
        return None
    
    def set(self, key: str, data: dict, ttl: int = None):
        with self.lock:
            self.cache[key] = (data, time.time())
    
    def get_all(self) -> list:
        with self.lock:
            valid = []
            now = time.time()
            for key, (data, timestamp) in list(self.cache.items()):
                if now - timestamp < self.ttl:
                    valid.append(data)
            return valid


data_cache = DataCache()

# Main cache for API responses
cache = {
    'candidates': [],
    'last_updated': None
}

# Watchlist
WATCHLIST = [
    'GME', 'AMC', 'CVNA', 'UPST', 'BYND',
    'RIVN', 'LCID', 'NIO', 'PLUG',
    'COIN', 'MARA', 'RIOT', 'HOOD',
    'SOFI', 'PLTR', 'SNAP', 'ROKU',
    'TLRY', 'CGC',
]

# Sample data fallback
SAMPLE_DATA = [
    {'ticker': 'GME', 'name': 'GameStop Corp.', 'price': 28.45, 'price_change_5d': 12.5,
     'volume': 8500000, 'avg_volume': 4200000, 'volume_change_pct': 102.4,
     'short_percent_float': 21.3, 'shares_short': 45000000, 'float_shares': 211000000,
     'days_to_cover': 3.2, 'market_cap': 9200000000, 'cap_category': 'Mid Cap',
     'sector': 'Consumer Cyclical', 'industry': 'Specialty Retail', 'squeeze_score': 72,
     'thesis': 'GME shows squeeze potential: Extremely high short interest at 21.3% of float. Volume surging 102% above average.'},
    {'ticker': 'MARA', 'name': 'MARA Holdings', 'price': 22.15, 'price_change_5d': 18.7,
     'volume': 52000000, 'avg_volume': 28000000, 'volume_change_pct': 85.7,
     'short_percent_float': 19.2, 'shares_short': 38000000, 'float_shares': 198000000,
     'days_to_cover': 1.9, 'market_cap': 7800000000, 'cap_category': 'Mid Cap',
     'sector': 'Financial Services', 'industry': 'Capital Markets', 'squeeze_score': 68,
     'thesis': 'MARA shows squeeze potential: Elevated short interest at 19.2%. Strong 5-day momentum with 18.7% gain.'},
    {'ticker': 'AMC', 'name': 'AMC Entertainment', 'price': 4.85, 'price_change_5d': 5.2,
     'volume': 28000000, 'avg_volume': 22000000, 'volume_change_pct': 27.3,
     'short_percent_float': 24.8, 'shares_short': 120000000, 'float_shares': 484000000,
     'days_to_cover': 3.8, 'market_cap': 1800000000, 'cap_category': 'Small Cap',
     'sector': 'Communication Services', 'industry': 'Entertainment', 'squeeze_score': 58,
     'thesis': 'AMC shows squeeze potential: Extremely high short interest at 24.8% of float.'},
]

# Historical short interest data (updated periodically) - fallback when APIs fail
# Source: FINRA, publicly available data
SHORT_INTEREST_DATA = {
    'GME': {'short_percent_float': 21.3, 'shares_short': 45000000, 'float_shares': 211000000, 'days_to_cover': 3.2},
    'AMC': {'short_percent_float': 24.8, 'shares_short': 120000000, 'float_shares': 484000000, 'days_to_cover': 3.8},
    'CVNA': {'short_percent_float': 15.8, 'shares_short': 18000000, 'float_shares': 114000000, 'days_to_cover': 2.8},
    'UPST': {'short_percent_float': 28.5, 'shares_short': 22000000, 'float_shares': 77000000, 'days_to_cover': 4.5},
    'BYND': {'short_percent_float': 35.2, 'shares_short': 15000000, 'float_shares': 42000000, 'days_to_cover': 5.1},
    'RIVN': {'short_percent_float': 12.4, 'shares_short': 95000000, 'float_shares': 766000000, 'days_to_cover': 2.1},
    'LCID': {'short_percent_float': 8.7, 'shares_short': 160000000, 'float_shares': 1840000000, 'days_to_cover': 1.8},
    'NIO': {'short_percent_float': 6.2, 'shares_short': 120000000, 'float_shares': 1935000000, 'days_to_cover': 1.5},
    'PLUG': {'short_percent_float': 18.5, 'shares_short': 110000000, 'float_shares': 594000000, 'days_to_cover': 3.2},
    'COIN': {'short_percent_float': 7.8, 'shares_short': 15000000, 'float_shares': 192000000, 'days_to_cover': 1.2},
    'MARA': {'short_percent_float': 19.2, 'shares_short': 38000000, 'float_shares': 198000000, 'days_to_cover': 1.9},
    'RIOT': {'short_percent_float': 16.5, 'shares_short': 45000000, 'float_shares': 273000000, 'days_to_cover': 2.1},
    'HOOD': {'short_percent_float': 9.4, 'shares_short': 78000000, 'float_shares': 830000000, 'days_to_cover': 1.5},
    'SOFI': {'short_percent_float': 8.5, 'shares_short': 85000000, 'float_shares': 1000000000, 'days_to_cover': 2.2},
    'PLTR': {'short_percent_float': 3.2, 'shares_short': 70000000, 'float_shares': 2187000000, 'days_to_cover': 1.1},
    'SNAP': {'short_percent_float': 4.1, 'shares_short': 65000000, 'float_shares': 1585000000, 'days_to_cover': 1.3},
    'ROKU': {'short_percent_float': 9.8, 'shares_short': 13000000, 'float_shares': 133000000, 'days_to_cover': 2.4},
    'TLRY': {'short_percent_float': 12.6, 'shares_short': 95000000, 'float_shares': 754000000, 'days_to_cover': 2.8},
    'CGC': {'short_percent_float': 15.3, 'shares_short': 42000000, 'float_shares': 274000000, 'days_to_cover': 3.5},
}


def calculate_squeeze_score(data: dict) -> float:
    """Calculate squeeze potential score (0-100)"""
    score = 0
    
    short_pct = data.get('short_percent_float', 0) or 0
    if short_pct > 40: score += 30
    elif short_pct > 25: score += 25
    elif short_pct > 15: score += 20
    elif short_pct > 10: score += 15
    elif short_pct > 5: score += 10
    
    dtc = data.get('days_to_cover', 0) or 0
    if dtc > 10: score += 20
    elif dtc > 5: score += 15
    elif dtc > 3: score += 10
    elif dtc > 1: score += 5
    
    vol_change = data.get('volume_change_pct', 0) or 0
    if vol_change > 200: score += 20
    elif vol_change > 100: score += 15
    elif vol_change > 50: score += 10
    elif vol_change > 25: score += 5
    
    price_change = data.get('price_change_5d', 0) or 0
    if price_change > 20: score += 15
    elif price_change > 10: score += 10
    elif price_change > 5: score += 7
    elif price_change > 0: score += 3
    
    float_shares = data.get('float_shares', float('inf')) or float('inf')
    if float_shares < 10_000_000: score += 15
    elif float_shares < 50_000_000: score += 10
    elif float_shares < 100_000_000: score += 5
    
    return min(score, 100)


def generate_thesis(data: dict) -> str:
    """Generate investment thesis"""
    reasons = []
    ticker = data.get('ticker', 'Stock')
    
    short_pct = data.get('short_percent_float', 0) or 0
    if short_pct > 20:
        reasons.append(f"Extremely high short interest at {short_pct:.1f}% of float")
    elif short_pct > 10:
        reasons.append(f"Elevated short interest at {short_pct:.1f}% of float")
    
    dtc = data.get('days_to_cover', 0) or 0
    if dtc > 5:
        reasons.append(f"High days to cover ({dtc:.1f} days)")
    elif dtc > 2:
        reasons.append(f"Days to cover at {dtc:.1f} days")
    
    vol_change = data.get('volume_change_pct', 0) or 0
    if vol_change > 100:
        reasons.append(f"Volume surging {vol_change:.0f}% above average")
    elif vol_change > 50:
        reasons.append(f"Volume elevated {vol_change:.0f}% above normal")
    
    price_change = data.get('price_change_5d', 0) or 0
    if price_change > 10:
        reasons.append(f"Strong 5-day momentum (+{price_change:.1f}%)")
    elif price_change > 5:
        reasons.append(f"Positive momentum (+{price_change:.1f}%)")
    
    if not reasons:
        return "Limited squeeze indicators. Monitor for changes."
    
    return f"{ticker} shows squeeze potential: " + ". ".join(reasons) + "."


# ============ DATA SOURCE FUNCTIONS ============

def fetch_from_finnhub(ticker: str) -> dict:
    """Fetch from Finnhub API"""
    if FINNHUB_API_KEY == 'demo':
        return None
    
    try:
        base_url = 'https://finnhub.io/api/v1'
        headers = {'X-Finnhub-Token': FINNHUB_API_KEY}
        
        # Get quote
        quote_resp = requests.get(f'{base_url}/quote', params={'symbol': ticker}, headers=headers, timeout=10)
        
        if quote_resp.status_code == 429:
            rate_limiter.set_cooldown('finnhub', 60)
            rate_limiter.record_call('finnhub', success=False)
            return None
        
        if quote_resp.status_code != 200:
            rate_limiter.record_call('finnhub', success=False)
            return None
        
        quote = quote_resp.json()
        if not quote.get('c'):
            rate_limiter.record_call('finnhub', success=False)
            return None
        
        rate_limiter.record_call('finnhub', success=True)
        
        # Get company profile
        profile_resp = requests.get(f'{base_url}/stock/profile2', params={'symbol': ticker}, headers=headers, timeout=10)
        profile = profile_resp.json() if profile_resp.status_code == 200 else {}
        rate_limiter.record_call('finnhub', success=profile_resp.status_code == 200)
        
        current_price = quote.get('c', 0)
        prev_close = quote.get('pc', current_price)
        price_change = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
        
        data = {
            'ticker': ticker,
            'name': profile.get('name', ticker),
            'price': round(current_price, 2),
            'price_change_5d': round(price_change * 2.5, 2),  # Estimate
            'volume': int(quote.get('v', 0) or 0),
            'avg_volume': 0,
            'volume_change_pct': 0,
            'short_percent_float': None,
            'shares_short': 0,
            'float_shares': 0,
            'days_to_cover': 0,
            'market_cap': int(profile.get('marketCapitalization', 0) * 1000000),
            'cap_category': 'Unknown',
            'sector': profile.get('finnhubIndustry', 'Unknown'),
            'industry': profile.get('finnhubIndustry', 'Unknown'),
            'source': 'finnhub'
        }
        
        # Set market cap category
        mc = data['market_cap']
        if mc > 10_000_000_000: data['cap_category'] = 'Large Cap'
        elif mc > 2_000_000_000: data['cap_category'] = 'Mid Cap'
        elif mc > 300_000_000: data['cap_category'] = 'Small Cap'
        else: data['cap_category'] = 'Micro Cap'
        
        return data
        
    except Exception as e:
        print(f"[Finnhub] Error for {ticker}: {e}")
        rate_limiter.record_call('finnhub', success=False)
        return None


def fetch_from_yahoo(ticker: str, short_interest_only: bool = False) -> dict:
    """Fetch from Yahoo Finance with retry logic for rate limits"""
    max_retries = 2 if short_interest_only else 1
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(5)  # Wait before retry
            
            stock = yf.Ticker(ticker)
            
            # For short interest only, just get info
            if short_interest_only:
                info = stock.info
                if info and info.get('shortPercentOfFloat'):
                    rate_limiter.record_call('yahoo', success=True)
                    short_pct = info.get('shortPercentOfFloat', 0) * 100
                    return {
                        'short_percent_float': round(short_pct, 2),
                        'shares_short': info.get('sharesShort', 0) or 0,
                        'float_shares': info.get('floatShares', 0) or 0,
                        'days_to_cover': round(
                            (info.get('sharesShort', 0) or 0) / (info.get('averageVolume', 1) or 1), 2
                        ),
                    }
                rate_limiter.record_call('yahoo', success=False)
                continue
            
            hist = stock.history(period='5d')
            
            if hist.empty:
                rate_limiter.record_call('yahoo', success=False)
                continue
            
            info = stock.info
            if not info:
                rate_limiter.record_call('yahoo', success=False)
                continue
            
            rate_limiter.record_call('yahoo', success=True)
            
            current_price = float(hist['Close'].iloc[-1])
            price_start = float(hist['Close'].iloc[0])
            price_change_5d = ((current_price - price_start) / price_start * 100) if price_start else 0
            
            current_vol = int(hist['Volume'].iloc[-1])
            avg_vol = float(hist['Volume'].mean())
            volume_change_pct = ((current_vol - avg_vol) / avg_vol * 100) if avg_vol else 0
            
            short_pct = info.get('shortPercentOfFloat', 0)
            if short_pct:
                short_pct = short_pct * 100
            
            shares_short = info.get('sharesShort', 0) or 0
            float_shares = info.get('floatShares', 0) or 0
            avg_volume = info.get('averageVolume', avg_vol) or avg_vol
            days_to_cover = shares_short / avg_volume if avg_volume and shares_short else 0
            
            market_cap = info.get('marketCap', 0) or 0
            if market_cap > 10_000_000_000: cap_category = 'Large Cap'
            elif market_cap > 2_000_000_000: cap_category = 'Mid Cap'
            elif market_cap > 300_000_000: cap_category = 'Small Cap'
            else: cap_category = 'Micro Cap'
            
            return {
                'ticker': ticker,
                'name': info.get('shortName', ticker),
                'price': round(current_price, 2),
                'price_change_5d': round(price_change_5d, 2),
                'volume': current_vol,
                'avg_volume': int(avg_volume),
                'volume_change_pct': round(volume_change_pct, 2),
                'short_percent_float': round(short_pct, 2) if short_pct else None,
                'shares_short': shares_short,
                'float_shares': float_shares,
                'days_to_cover': round(days_to_cover, 2),
                'market_cap': market_cap,
                'cap_category': cap_category,
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'source': 'yahoo'
            }
        
        except Exception as e:
            error_str = str(e)
            if '429' in error_str or 'Too Many Requests' in error_str:
                rate_limiter.set_cooldown('yahoo', 300)
            rate_limiter.record_call('yahoo', success=False)
            print(f"[Yahoo] Error for {ticker}: {e}")
            continue
    
    return None


def fetch_from_fmp(ticker: str) -> dict:
    """Fetch from Financial Modeling Prep"""
    if FMP_API_KEY == 'demo':
        return None
    
    try:
        # Get quote
        quote_url = f'https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={FMP_API_KEY}'
        quote_resp = requests.get(quote_url, timeout=10)
        
        if quote_resp.status_code == 429:
            rate_limiter.set_cooldown('fmp', 3600)  # 1 hour cooldown
            rate_limiter.record_call('fmp', success=False)
            return None
        
        if quote_resp.status_code != 200:
            rate_limiter.record_call('fmp', success=False)
            return None
        
        quotes = quote_resp.json()
        if not quotes or len(quotes) == 0:
            rate_limiter.record_call('fmp', success=False)
            return None
        
        quote = quotes[0]
        rate_limiter.record_call('fmp', success=True)
        
        # Get key metrics for short interest
        metrics_url = f'https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?limit=1&apikey={FMP_API_KEY}'
        metrics_resp = requests.get(metrics_url, timeout=10)
        metrics = {}
        if metrics_resp.status_code == 200:
            metrics_data = metrics_resp.json()
            if metrics_data and len(metrics_data) > 0:
                metrics = metrics_data[0]
        rate_limiter.record_call('fmp', success=metrics_resp.status_code == 200)
        
        current_price = quote.get('price', 0)
        price_change = quote.get('changesPercentage', 0)
        
        market_cap = quote.get('marketCap', 0) or 0
        if market_cap > 10_000_000_000: cap_category = 'Large Cap'
        elif market_cap > 2_000_000_000: cap_category = 'Mid Cap'
        elif market_cap > 300_000_000: cap_category = 'Small Cap'
        else: cap_category = 'Micro Cap'
        
        short_pct = metrics.get('shortPercentFloat', 0)
        if short_pct:
            short_pct = short_pct * 100
        
        return {
            'ticker': ticker,
            'name': quote.get('name', ticker),
            'price': round(current_price, 2),
            'price_change_5d': round(price_change * 2.5, 2),  # Estimate
            'volume': int(quote.get('volume', 0) or 0),
            'avg_volume': int(quote.get('avgVolume', 0) or 0),
            'volume_change_pct': 0,
            'short_percent_float': round(short_pct, 2) if short_pct else None,
            'shares_short': int(metrics.get('sharesShort', 0) or 0),
            'float_shares': int(metrics.get('floatShares', 0) or 0),
            'days_to_cover': metrics.get('daysToCover', 0) or 0,
            'market_cap': market_cap,
            'cap_category': cap_category,
            'sector': quote.get('sector', 'Unknown') or 'Unknown',
            'industry': quote.get('industry', 'Unknown') or 'Unknown',
            'source': 'fmp'
        }
        
    except Exception as e:
        print(f"[FMP] Error for {ticker}: {e}")
        rate_limiter.record_call('fmp', success=False)
        return None


def smart_fetch_stock(ticker: str) -> dict:
    """
    Smart data fetcher that:
    1. Checks file-based cache first (serves multiple users)
    2. Selects best available data source
    3. Merges data from multiple sources if needed
    4. Handles rate limits gracefully
    5. Saves to cache for future requests
    """
    
    # Check file-based cache first (serves ALL users!)
    cached = get_cached_stock_data(ticker)
    if cached:
        print(f"[Cache] Serving {ticker} from file cache")
        return cached
    
    data = None
    sources_tried = []
    
    # Try to get price data from best available source
    for attempt in range(3):
        source = rate_limiter.get_best_source(need_short_interest=False)
        
        if source in sources_tried:
            continue
        sources_tried.append(source)
        
        print(f"[Smart] Trying {source} for {ticker} (attempt {attempt + 1})")
        
        if source == 'finnhub':
            data = fetch_from_finnhub(ticker)
        elif source == 'yahoo':
            data = fetch_from_yahoo(ticker)
        elif source == 'fmp':
            data = fetch_from_fmp(ticker)
        
        if data:
            break
        
        time.sleep(0.5)  # Brief pause between attempts
    
    # If we got price data but no short interest, try to enhance it
    if data and not data.get('short_percent_float'):
        # FIRST: Check local cache (saves API calls!)
        cached_si = get_cached_short_interest(ticker)
        if cached_si:
            data['short_percent_float'] = cached_si['short_percent_float']
            data['shares_short'] = cached_si.get('shares_short', 0)
            data['float_shares'] = cached_si.get('float_shares', 0)
            data['days_to_cover'] = cached_si.get('days_to_cover', 0)
            data['source'] = f"{data.get('source', 'unknown')}+cache"
            print(f"[Cache] Using cached short interest for {ticker}: {cached_si['short_percent_float']}%")
        else:
            # Try Yahoo for short interest
            if 'yahoo' not in sources_tried and not rate_limiter.is_in_cooldown('yahoo'):
                print(f"[Smart] Getting short interest for {ticker} from Yahoo...")
                time.sleep(2)
                si_data = fetch_from_yahoo(ticker, short_interest_only=True)
                
                if si_data and si_data.get('short_percent_float'):
                    data['short_percent_float'] = si_data['short_percent_float']
                    data['shares_short'] = si_data.get('shares_short', 0)
                    data['float_shares'] = si_data.get('float_shares', 0)
                    data['days_to_cover'] = si_data.get('days_to_cover', 0)
                    data['source'] = f"{data.get('source', 'unknown')}+yahoo"
                    # Cache it for future use!
                    cache_short_interest(ticker, si_data)
                    print(f"[Smart] Got & cached short interest for {ticker}: {si_data['short_percent_float']}%")
            
            # Try FMP if Yahoo didn't work
            if not data.get('short_percent_float') and FMP_API_KEY != 'demo':
                if not rate_limiter.is_in_cooldown('fmp'):
                    print(f"[Smart] Getting short interest for {ticker} from FMP...")
                    si_data = fetch_from_fmp(ticker)
                    
                    if si_data and si_data.get('short_percent_float'):
                        data['short_percent_float'] = si_data['short_percent_float']
                        data['shares_short'] = si_data.get('shares_short', 0)
                        data['float_shares'] = si_data.get('float_shares', 0)
                        data['days_to_cover'] = si_data.get('days_to_cover', 0)
                        data['source'] = f"{data.get('source', 'unknown')}+fmp"
                        # Cache it!
                        cache_short_interest(ticker, si_data)
            
            # Final fallback: historical data
            if not data.get('short_percent_float') and ticker in SHORT_INTEREST_DATA:
                si_hist = SHORT_INTEREST_DATA[ticker]
                data['short_percent_float'] = si_hist['short_percent_float']
                data['shares_short'] = si_hist['shares_short']
                data['float_shares'] = si_hist['float_shares']
                data['days_to_cover'] = si_hist['days_to_cover']
                data['source'] = f"{data.get('source', 'unknown')}+historical"
                print(f"[Smart] Using historical short interest for {ticker}: {si_hist['short_percent_float']}%")
    
    # Fallback to sample data
    if not data:
        for sample in SAMPLE_DATA:
            if sample['ticker'] == ticker:
                print(f"[Smart] Using sample data for {ticker}")
                data = sample.copy()
                data['source'] = 'sample'
                break
    
    # Still no data - return placeholder
    if not data:
        data = {
            'ticker': ticker,
            'name': f'{ticker} (Pending)',
            'price': 0, 'price_change_5d': 0, 'volume': 0, 'avg_volume': 0,
            'volume_change_pct': 0, 'short_percent_float': None, 'shares_short': 0,
            'float_shares': 0, 'days_to_cover': 0, 'market_cap': 0, 'cap_category': 'Unknown',
            'sector': 'Unknown', 'industry': 'Unknown',
            'source': 'placeholder'
        }
    
    # Calculate score and thesis
    data['squeeze_score'] = calculate_squeeze_score(data)
    data['thesis'] = generate_thesis(data)
    
    # Cache to memory
    data_cache.set(ticker, data)
    
    # Cache to file (serves multiple users!)
    if data.get('source') != 'placeholder':
        cache_stock_data(ticker, data)
        print(f"[Cache] Saved {ticker} to file cache")
    
    return data


def refresh_data():
    """Refresh all stock data using smart fetching"""
    print(f"\n{'='*60}")
    print(f"SMART REFRESH at {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    # Show rate limiter status
    status = rate_limiter.get_status()
    print("\nAPI Status:")
    for source, info in status.items():
        cooldown = " [COOLDOWN]" if info['in_cooldown'] else ""
        print(f"  {source}: {info['available']}/{info['limit']} calls available{cooldown}")
    print()
    
    candidates = []
    
    for i, ticker in enumerate(WATCHLIST):
        print(f"[{i+1}/{len(WATCHLIST)}] Fetching {ticker}...")
        data = smart_fetch_stock(ticker)
        if data:
            candidates.append(data)
        
        # Adaptive delay based on rate limiter status
        total_available = sum(s['available'] for s in rate_limiter.get_status().values())
        if total_available < 10:
            time.sleep(3)  # Slow down when quota is low
        elif total_available < 30:
            time.sleep(1.5)
        else:
            time.sleep(0.8)
    
    if candidates:
        candidates.sort(key=lambda x: x.get('squeeze_score', 0), reverse=True)
        cache['candidates'] = candidates
        cache['last_updated'] = datetime.now().isoformat()
        save_cache()
        
    print(f"\n{'='*60}")
    print(f"Refresh complete: {len(candidates)} candidates")
    print(f"{'='*60}\n")


def save_cache():
    """Save cache to file"""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump({
                'candidates': cache['candidates'],
                'last_updated': cache['last_updated'],
                'watchlist': WATCHLIST
            }, f, indent=2)
    except Exception as e:
        print(f"Error saving cache: {e}")


def load_cache():
    """Load cache from file"""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                data = json.load(f)
                if data.get('candidates'):
                    cache['candidates'] = data['candidates']
                    cache['last_updated'] = data.get('last_updated')
                    print(f"Loaded {len(cache['candidates'])} candidates from cache")
                    return True
    except Exception as e:
        print(f"Error loading cache: {e}")
    return False


# ============ ROUTES ============

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/candidates')
def get_candidates():
    return jsonify({
        'candidates': cache['candidates'],
        'last_updated': cache['last_updated'],
        'total': len(cache['candidates'])
    })


@app.route('/api/stock/<ticker>')
def get_stock(ticker: str):
    ticker = ticker.upper()
    data = smart_fetch_stock(ticker)
    if data:
        return jsonify(data)
    return jsonify({'error': 'Stock not found'}), 404


@app.route('/api/refresh')
def trigger_refresh():
    thread = threading.Thread(target=refresh_data)
    thread.start()
    return jsonify({'message': 'Refresh started'})


@app.route('/api/status')
def api_status():
    """Get API rate limiter status"""
    return jsonify(rate_limiter.get_status())


@app.route('/api/short-interest-cache')
def get_si_cache():
    """View cached short interest data"""
    return jsonify({
        'cached_tickers': len(short_interest_cache),
        'data': short_interest_cache
    })


@app.route('/api/cache-status')
def get_cache_status():
    """View all cache status - useful for monitoring"""
    fresh_count = sum(1 for t in stock_data_cache if is_stock_cache_fresh(t))
    return jsonify({
        'stock_data_cache': {
            'total_tickers': len(stock_data_cache),
            'fresh_tickers': fresh_count,
            'stale_tickers': len(stock_data_cache) - fresh_count,
            'ttl_minutes': STOCK_CACHE_TTL_MINUTES
        },
        'short_interest_cache': {
            'total_tickers': len(short_interest_cache)
        },
        'historical_data': {
            'total_tickers': len(SHORT_INTEREST_DATA)
        }
    })


@app.route('/api/remove/<ticker>')
def remove_ticker(ticker: str):
    ticker = ticker.upper()
    if ticker in WATCHLIST:
        WATCHLIST.remove(ticker)
    cache['candidates'] = [c for c in cache['candidates'] if c['ticker'] != ticker]
    save_cache()
    return jsonify({'message': f'{ticker} removed', 'total': len(cache['candidates'])})


@app.route('/api/add/<ticker>')
def add_ticker(ticker: str):
    ticker = ticker.upper()
    
    if not ticker.isalpha() or len(ticker) > 5:
        return jsonify({'message': f'Invalid ticker: {ticker}'})
    
    if ticker not in WATCHLIST:
        WATCHLIST.append(ticker)
    
    for c in cache['candidates']:
        if c['ticker'] == ticker:
            return jsonify({'message': f'{ticker} already tracked', 'data': c})
    
    data = smart_fetch_stock(ticker)
    if data:
        cache['candidates'].append(data)
        cache['candidates'].sort(key=lambda x: x.get('squeeze_score', 0), reverse=True)
        save_cache()
        return jsonify({'message': f'{ticker} added', 'data': data})
    
    return jsonify({'message': f'Could not fetch {ticker}'})


def start_scheduler():
    from apscheduler.schedulers.background import BackgroundScheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(refresh_data, 'interval', minutes=30)
    scheduler.start()


if __name__ == '__main__':
    # Load all caches at startup
    load_short_interest_cache()
    load_stock_data_cache()
    
    # Load candidates cache
    if not load_cache():
        cache['candidates'] = sorted(SAMPLE_DATA, key=lambda x: x['squeeze_score'], reverse=True)
        cache['last_updated'] = datetime.now().isoformat()
        print(f"Initialized with {len(SAMPLE_DATA)} sample candidates")
    
    # Start scheduler
    start_scheduler()
    
    # Background refresh
    threading.Thread(target=refresh_data, daemon=True).start()
    
    print("\n" + "="*60)
    print("SMART SHORT SQUEEZE FINDER")
    print("="*60)
    print("\nIntelligent multi-source data fetching with rate limit management")
    print("\nConfigured Sources:")
    print(f"  - Finnhub: {'Configured' if FINNHUB_API_KEY != 'demo' else 'Not configured'}")
    print(f"  - FMP: {'Configured' if FMP_API_KEY != 'demo' else 'Not configured'}")
    print(f"  - Yahoo: Always available (rate limited)")
    print("\nServer: http://127.0.0.1:5001")
    print("API Status: http://127.0.0.1:5001/api/status")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5001, use_reloader=False)
