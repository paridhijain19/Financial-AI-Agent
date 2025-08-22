"""Financial Agent with Gemini 2.5 Flash + Polygon + LangGraph + Flask Backend"""

import os
from dotenv import load_dotenv
load_dotenv()

# --- LLM & Tool Setup ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities.polygon import PolygonAPIWrapper
from langchain_community.tools import (
    PolygonLastQuote,
    PolygonTickerNews,
    PolygonFinancials,
    PolygonAggregates
)

# --- Fundamental Analysis Imports ---
from typing import Union, Dict, Set, List, TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
import yfinance as yf
import datetime as dt
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volume import volume_weighted_average_price
import traceback
import pandas as pd

# --- Flask Setup ---
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Set up API keys from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
polygon_api_key = os.getenv("POLYGON_API_KEY")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

if not polygon_api_key:
    raise ValueError("POLYGON_API_KEY environment variable is required")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key,
    temperature=0.1
)

polygon = PolygonAPIWrapper(polygon_api_key=polygon_api_key)

# --- Fundamental Analysis Tools ---
FUNDAMENTAL_ANALYST_PROMPT = """
You are a fundamental analyst specializing in evaluating company (whose symbol is {company}) performance based on stock prices, technical indicators, and financial metrics. Your task is to provide a comprehensive summary of the fundamental analysis for a given stock.

You have access to the following tools:
1. **get_stock_prices**: Retrieves the latest stock price, historical price data and technical Indicators like RSI, MACD, Drawdown and VWAP.
2. **get_financial_metrics**: Retrieves key financial metrics, such as revenue, earnings per share (EPS), price-to-earnings ratio (P/E), and debt-to-equity ratio.

### Your Task:
1. **Input Stock Symbol**: Use the provided stock symbol to query the tools and gather the relevant information.
2. **Analyze Data**: Evaluate the results from the tools and identify potential resistance, key trends, strengths, or concerns.
3. **Provide Summary**: Write a concise, well-structured summary that highlights:
    - Recent stock price movements, trends and potential resistance.
    - Key insights from technical indicators (e.g., whether the stock is overbought or oversold).
    - Financial health and performance based on financial metrics.

### Constraints:
- Use only the data provided by the tools.
- Avoid speculative language; focus on observable data and trends.
- If any tool fails to provide data, clearly state that in your summary.

### Output Format:
Respond in the following format:
"stock": "<Stock Symbol>",
"price_analysis": "<Detailed analysis of stock price trends>",
"technical_analysis": "<Detailed time series Analysis from ALL technical indicators>",
"financial_analysis": "<Detailed analysis from financial metrics>",
"final Summary": "<Full Conclusion based on the above analyses>"
"Asked Question Answer": "<Answer based on the details and analysis above>"

Ensure that your response is objective, concise, and actionable.
"""

@tool
def get_stock_prices(ticker: str) -> Union[Dict, str]:
    """Fetches historical stock price data and technical indicator for a given ticker."""
    try:
        data = yf.download(
            ticker,
            start=dt.datetime.now() - dt.timedelta(weeks=24*3),
            end=dt.datetime.now(),
            interval='1d'
        )
        df= data.copy()
        if len(df.columns[0]) > 1:
            df.columns = [i[0] for i in df.columns]
        data.reset_index(inplace=True)
        data.Date = data.Date.astype(str)
        
        indicators = {}

        # Momentum Indicators
        rsi_series = RSIIndicator(df['Close'], window=14).rsi().iloc[-12:]
        indicators["RSI"] = {date.strftime('%Y-%m-%d'): int(value) for date, value in rsi_series.dropna().to_dict().items()}
        sto_series = StochasticOscillator(
            df['High'], df['Low'], df['Close'], window=14).stoch().iloc[-12:]
        indicators["Stochastic_Oscillator"] = {date.strftime('%Y-%m-%d'): int(value) for date, value in sto_series.dropna().to_dict().items()}

        macd = MACD(df['Close'])
        macd_series = macd.macd().iloc[-12:]
        indicators["MACD"] = {date.strftime('%Y-%m-%d'): int(value) for date, value in macd_series.to_dict().items()}
        macd_signal_series = macd.macd_signal().iloc[-12:]
        indicators["MACD_Signal"] = {date.strftime('%Y-%m-%d'): int(value) for date, value in macd_signal_series.to_dict().items()}
        
        vwap_series = volume_weighted_average_price(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            volume=df['Volume'],
        ).iloc[-12:]
        indicators["vwap"] = {date.strftime('%Y-%m-%d'): int(value) for date, value in vwap_series.to_dict().items()}
        
        return {'stock_price': data.to_dict(orient='records'), 'indicators': indicators}
    except Exception as e:
        return f"Error fetching price data: {str(e)}"
    
@tool
def get_financial_metrics(ticker: str) -> Union[Dict, str]:
    """Fetches key financial ratios for a given ticker."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'pe_ratio': info.get('forwardPE'),
            'price_to_book': info.get('priceToBook'),
            'debt_to_equity': info.get('debtToEquity'),
            'profit_margins': info.get('profitMargins')
        }
    except Exception as e:
        return f"Error fetching ratios: {str(e)}"

# Combined tools (Polygon + Fundamental Analysis)
polygon_tools = [
    PolygonLastQuote(api_wrapper=polygon),
    PolygonTickerNews(api_wrapper=polygon),
    PolygonFinancials(api_wrapper=polygon),
    PolygonAggregates(api_wrapper=polygon),
]

fundamental_tools = [get_stock_prices, get_financial_metrics]
tools = polygon_tools + fundamental_tools

# --- Prompt & Agent Setup ---
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent
from langchain_core.runnables import RunnablePassthrough
from langchain_core.agents import AgentFinish

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an advanced financial assistant with both real-time market data and fundamental analysis capabilities. "
     "You have access to:\n"
     "1. **Polygon API tools** for real-time quotes, news, financials, and aggregates\n"
     "2. **Fundamental analysis tools** for technical indicators, historical prices, and financial metrics\n\n"
     "**Instructions:**\n"
     "- For questions about fundamental analysis, technical indicators, or 'should I buy/sell' decisions, use get_stock_prices and get_financial_metrics tools\n"
     "- For real-time quotes, latest news, or market data, use Polygon tools\n"
     "- **Always use tools** - do not answer from your own knowledge\n"
     "- If asked for fundamental analysis, provide the structured format with price_analysis, technical_analysis, financial_analysis, and final summary\n"
     "- Be objective and data-driven in your responses\n"
     "- Format your response in a clean, readable way with proper sections and spacing"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent_runnable = create_tool_calling_agent(llm, tools, prompt)

agent = RunnablePassthrough.assign(
    agent_outcome=agent_runnable
)

# --- Tool Execution Handler ---
def execute_tools(data):
    agent_actions = data.pop('agent_outcome')
    
    # Ensure agent_actions is a list
    if not isinstance(agent_actions, list):
        agent_actions = [agent_actions]
    
    for agent_action in agent_actions:
        tool_to_use = {t.name: t for t in tools}[agent_action.tool]
        observation = tool_to_use.invoke(agent_action.tool_input)
        data['intermediate_steps'].append((agent_action, observation))
    
    return data

# --- LangGraph Setup ---
from langgraph.graph import END, StateGraph
from typing import TypedDict, Annotated

class AgentState(TypedDict):
    input: str
    agent_outcome: object
    intermediate_steps: list

def should_continue(data):
    if isinstance(data['agent_outcome'], AgentFinish):
        return "exit"
    else:
        return "continue"

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
workflow.add_node("tools", execute_tools)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {
    "continue": "tools",
    "exit": END
})
workflow.add_edge("tools", "agent")
chain = workflow.compile()

# --- Agent Wrapper ---
def financial_agent(input_text):
    try:
        result = chain.invoke({"input": input_text, "intermediate_steps": []})
        agent_output = result.get("agent_outcome")
        
        # Check if it's an AgentFinish object with return_values
        if hasattr(agent_output, "return_values") and "output" in agent_output.return_values:
            return agent_output.return_values["output"]
        # Check if it's an AgentFinish object with log
        elif hasattr(agent_output, "log"):
            return agent_output.log
        # Fallback to string representation
        else:
            return str(agent_output)
            
    except Exception as e:
        return f"Error processing request: {str(e)}"

# --- Flask Routes ---
@app.route('/')
def index():
    # Read the HTML file content (you'll need to save the HTML as a separate file)
    with open('index.html', 'r') as f:
        html_content = f.read()
    return render_template_string(html_content)

@app.route('/api/financial-agent', methods=['POST'])
def api_financial_agent():
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Process the query
        response = financial_agent(query)
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    print("Starting Financial Agent Server...")
    print("Access the app at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)