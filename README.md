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

Set your API keys as environment variables:

```bash
# Option 1: Export in terminal
export FINNHUB_API_KEY=your_finnhub_key
export FMP_API_KEY=your_fmp_key

# Option 2: Create .env file (copy from env.example)
cp env.example .env
# Edit .env with your keys
```

Get free API keys:
- **Finnhub**: https://finnhub.io (60 calls/min)
- **FMP**: https://financialmodelingprep.com (250 calls/day)

The app will work without API keys using cached/sample data, but real-time data requires valid keys.

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
