# Financial AI Agent

A comprehensive financial analysis tool powered by Google Gemini AI, LangChain, and real-time market data APIs. This application provides fundamental analysis, technical indicators, and market insights through a web interface.

## Features

- Real-time stock quotes and market data via Polygon API
- Technical analysis with RSI, MACD, Stochastic Oscillator, and VWAP indicators
- Fundamental analysis with financial ratios and metrics
- Latest financial news and company information
- Clean web interface for easy interaction
- RESTful API for programmatic access

## Prerequisites

- Python 3.8 or higher
- Google Gemini API key
- Polygon.io API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd financial-ai-agent
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp env.example .env
```

Edit the `.env` file and add your API keys:
```
GOOGLE_API_KEY=your_google_gemini_api_key
POLYGON_API_KEY=your_polygon_api_key
```

## API Keys Setup

### Google Gemini API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file

### Polygon API Key
1. Sign up at [Polygon.io](https://polygon.io/)
2. Get your API key from the dashboard
3. Copy the key to your `.env` file

## Usage

### Starting the Application

Navigate to the langchain directory and run:
```bash
cd langchain
python agent.py
```

The application will start on `http://localhost:5000`

### Web Interface

Open your browser and navigate to `http://localhost:5000` to access the web interface. You can:

- Ask questions about specific stocks
- Request fundamental analysis
- Get technical indicators
- Retrieve latest market news
- Analyze financial metrics

### Example Queries

- "Should I buy TSLA stock?"
- "Analyze NVDA fundamentals"
- "What is the latest news for AAPL?"
- "Show me technical indicators for GOOGL"
- "Get financial ratios for MSFT"

### API Endpoint

The application exposes a REST API at `/api/financial-agent`:

```bash
curl -X POST http://localhost:5000/api/financial-agent \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze AAPL stock"}'
```

## Project Structure

```
financial-ai-agent/
├── langchain/
│   ├── agent.py          # Main Flask application and LangGraph agent
│   └── index.html        # Web interface
├── .env.example          # Environment variables template
├── .gitignore           # Git ignore file
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Technical Details

### Components

- **LangChain Agent**: Orchestrates tool usage and LLM interactions
- **Google Gemini 2.5 Flash**: Language model for analysis and responses
- **Polygon API**: Real-time market data and financial information
- **yfinance**: Historical stock data and financial metrics
- **TA-Lib**: Technical analysis indicators
- **Flask**: Web server and API

### Tools Available

1. **Stock Price Data**: Historical prices with technical indicators
2. **Financial Metrics**: P/E ratio, debt-to-equity, profit margins
3. **Polygon Last Quote**: Real-time stock quotes
4. **Polygon News**: Latest financial news
5. **Polygon Financials**: Company financial statements
6. **Polygon Aggregates**: Historical price aggregates

## Development

### Running in Development Mode

The Flask application runs in debug mode by default. For production deployment, modify the configuration in `agent.py`.

### Adding New Features

The application uses LangGraph for agent orchestration. To add new capabilities:

1. Create new tools following the `@tool` decorator pattern
2. Add tools to the `tools` list
3. Update the system prompt if needed

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure both Google and Polygon API keys are set in `.env`
2. **Import Errors**: Verify all dependencies are installed via `pip install -r requirements.txt`
3. **Port Conflicts**: Change the port in `agent.py` if 5000 is already in use

### API Rate Limits

- Polygon free tier has rate limits
- Google Gemini has request quotas
- Consider implementing caching for production use

## License

This project is provided as-is for educational and research purposes.
