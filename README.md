# 🚀 Squozl - Short Squeeze Stock Scanner

Real-time short squeeze detection platform that finds stocks with high squeeze potential.

![Squozl Screenshot](https://img.shields.io/badge/Status-Live-brightgreen)

## Features

- **Real-time Scanner** - Monitors stocks for squeeze potential
- **Smart Data Fetching** - Multi-source API with intelligent rate limiting
- **Squeeze Score** - 0-100 score based on short interest, volume, momentum
- **AI Thesis** - Auto-generated investment thesis for each stock
- **Caching** - File-based cache serves multiple users efficiently
- **Ad-Ready** - Monetization placements built-in

## Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript
- **Data Sources**: Finnhub, FMP, Yahoo Finance
- **Caching**: File-based JSON cache

## Quick Start

```bash
# Clone the repo
git clone https://github.com/pravin12patre/Squozl.git
cd Squozl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Open http://127.0.0.1:5001

## Configuration

Add your API keys in `app.py`:

```python
FINNHUB_API_KEY = 'your_finnhub_key'
FMP_API_KEY = 'your_fmp_key'
```

Get free API keys:
- Finnhub: https://finnhub.io (60 calls/min)
- FMP: https://financialmodelingprep.com (250 calls/day)

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/` | Main dashboard |
| `/api/candidates` | Get all squeeze candidates |
| `/api/stock/<ticker>` | Get single stock data |
| `/api/add/<ticker>` | Add ticker to watchlist |
| `/api/remove/<ticker>` | Remove ticker |
| `/api/refresh` | Force data refresh |
| `/api/status` | Rate limiter status |
| `/api/cache-status` | Cache health |

## Disclaimer

This is for educational purposes only. Not financial advice. Short squeezes are highly volatile and risky.

## License

MIT
