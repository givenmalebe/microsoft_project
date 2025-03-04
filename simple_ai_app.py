import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_experimental.agents import create_pandas_dataframe_agent
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
from urllib.parse import quote
import random
import time
import altair as alt
import plotly.express as px
from io import BytesIO
import base64
import zipfile
import openai
import tabulate
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from io import StringIO

# Load environment variables
load_dotenv()

# Initialize OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL_NAME = "gpt-4o-mini"  # Using GPT-4 Turbo for better performance

# Define simplified key statistics to search for
KEY_STATISTICS = [
    'population',
    'total population', 
    'infected population',
    'disease incidence',
    'disease prevalence',
    'demographic distribution',
    'age distribution',
    'sex distribution',
    'gender distribution',
    'geographic distribution',
    'urban population',
    'rural population',
    'mortality rate',
    'life expectancy',
    'healthcare facilities',
    'physician density'
]

# Data sources for statistics research
DATA_SOURCES = {
    'google': 'Google Search',
    'duckduckgo': 'DuckDuckGo'
}

# Replace previous RESEARCH_KEYWORDS with simplified structure
STATISTICS_CATEGORIES = {
    'population_stats': ['population', 'total population', 'population density'],
    'disease_stats': ['infected population', 'disease incidence', 'disease prevalence', 'mortality rate'],
    'demographic_stats': ['demographic distribution', 'age distribution', 'sex distribution', 'gender distribution'],
    'geographic_stats': ['geographic distribution', 'urban population', 'rural population'],
    'healthcare_stats': ['healthcare facilities', 'physician density', 'life expectancy']
}

# Helper functions for the dashboard
def custom_period(date):
    month = date.month
    year = date.year
    if month in [2, 3, 4]:
        return pd.Period(year=year, quarter=1, freq='Q')
    elif month in [5, 6, 7]:
        return pd.Period(year=year, quarter=2, freq='Q')
    elif month in [8, 9, 10]:
        return pd.Period(year=year, quarter=3, freq='Q')
    else:
        return pd.Period(year=year if month != 1 else year-1, quarter=4, freq='Q')

def aggregate_health_data(df, freq):
    if freq == 'Daily':
        # For daily view, just prepare the columns
        df_agg = df.copy()
        return pd.DataFrame({
            'AVG_VALUE': df_agg.groupby('timestamp')['numerical_value'].mean(),
            'INDICATORS': df_agg.groupby('timestamp')['category'].count(),
            'SOURCES': df_agg.groupby('timestamp')['source'].nunique()
        })
    elif freq == 'Q':
        df = df.copy()
        df['CUSTOM_Q'] = df['timestamp'].apply(custom_period)
        df_agg = df.groupby('CUSTOM_Q').agg({
            'numerical_value': 'mean',
            'category': 'count',
            'source': 'nunique'
        }).rename(columns={
            'numerical_value': 'AVG_VALUE',
            'category': 'INDICATORS',
            'source': 'SOURCES'
        })
        return df_agg
    else:
        return df.resample(freq, on='timestamp').agg({
            'numerical_value': 'mean',
            'category': 'count',
            'source': 'nunique'
        }).rename(columns={
            'numerical_value': 'AVG_VALUE',
            'category': 'INDICATORS',
            'source': 'SOURCES'
        })

def format_with_commas(number):
    """Format a number with commas as thousand separators."""
    try:
        return f"{number:,}"
    except:
        return number

def get_analysis_tables():
    """
    Get tables from the analysis results stored in session state.
    """
    if hasattr(st.session_state, 'current_tables') and st.session_state.current_tables:
        return st.session_state.current_tables
    return []

def create_metric_chart(df, column, color, chart_type, height=150, time_frame='Daily'):
    chart_data = df[[column]].copy()
    if time_frame == 'Quarterly':
        chart_data.index = chart_data.index.strftime('%Y Q%q')
    if time_frame == 'Quarterly':
        chart_data.index = chart_data.index.astype(str)
    
    # Create appropriate chart based on type
    if chart_type == 'Area':
        chart = alt.Chart(chart_data.reset_index()).mark_area(
            line={'color': color},
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color=color, offset=0),
                       alt.GradientStop(color='white', offset=1)],
                x1=1,
                x2=1,
                y1=1,
                y2=0
            )
        ).encode(
            alt.X('index:T', title=None),
            alt.Y(f'{column}:Q', title=None),
            tooltip=['index', column]
        ).properties(height=height)
    else:
        chart = alt.Chart(chart_data.reset_index()).mark_bar(color=color).encode(
            alt.X('index:T', title=None),
            alt.Y(f'{column}:Q', title=None),
            tooltip=['index', column]
        ).properties(height=height)

    return chart

def calculate_delta(df, column):
    if len(df) < 2:
        return 0, 0
    current_value = df[column].iloc[-1]
    previous_value = df[column].iloc[-2]
    delta = current_value - previous_value
    delta_percent = (delta / previous_value) * 100 if previous_value != 0 else 0
    return delta, delta_percent

def display_metric(col, title, total_value, df, column, color, time_frame):
    """Display a metric card with a chart"""
    try:
        with col:
            st.metric(title, format_with_commas(total_value))
            chart = create_metric_chart(df, column, color, 'Area', time_frame=time_frame)
            st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        with col:
            st.metric(title, "N/A")
            st.warning(f"Could not display chart: {e}")

def format_table_professionally(data, headers=None, format="fancy_grid", add_analysis=False):
    """
    Format a table in a professional manner for analysis and presentation.
    
    Args:
        data (list or DataFrame): The table data to format
        headers (list, optional): Column headers for the table
        format (str, optional): Table format style (see tabulate styles)
        add_analysis (bool, optional): Whether to add summary statistics
        
    Returns:
        str: Professionally formatted table as string
    """
    if isinstance(data, pd.DataFrame):
        # For DataFrames, extract headers and values
        df = data
        headers = headers or df.columns.tolist()
        table_data = df.values.tolist()
    else:
        # For list data
        table_data = data
        
    # Create the formatted table
    formatted_table = tabulate.tabulate(
        table_data, 
        headers=headers, 
        tablefmt=format,
        numalign="right",
        stralign="left"
    )
    
    # If requested, add summary statistics for numerical columns
    if add_analysis and isinstance(data, pd.DataFrame):
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            analysis = "\n\nTable Summary Statistics:\n"
            summary_data = []
            
            # Calculate key statistics
            for col in numeric_cols:
                try:
                    stats = {
                        "Column": col,
                        "Mean": df[col].mean(),
                        "Median": df[col].median(),
                        "Min": df[col].min(),
                        "Max": df[col].max(),
                        "Std Dev": df[col].std()
                    }
                    summary_data.append(stats)
                except:
                    continue
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                analysis += tabulate.tabulate(
                    summary_df,
                    headers="keys",
                    tablefmt="grid",
                    numalign="right",
                    floatfmt=".2f"
                )
                formatted_table += analysis
    
    return formatted_table

def display_formatted_table(df, title=None, with_stats=False, height=None, use_expander_for_stats=True):
    """
    Display a professionally formatted table with custom styling and optional summary statistics
    
    Parameters:
    - df: pandas DataFrame to display
    - title: Optional title for the table
    - with_stats: Whether to include summary statistics
    - height: Optional height constraint for the table (in pixels)
    - use_expander_for_stats: Whether to use an expander for statistics (set to False when called inside another expander)
    """
    # Apply custom styling
    table_css = """
    <style>
        .dataframe {
            width: 100%;
            border-collapse: collapse;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .dataframe th {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            text-align: left;
            padding: 10px;
            border: 1px solid #ddd;
        }
        .dataframe td {
            padding: 8px;
            border: 1px solid #ddd;
            text-align: left;
        }
        .dataframe tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .dataframe tr:hover {
            background-color: #ddd;
        }
    </style>
    """
    
    # Display title if provided
    if title:
        st.markdown(f"#### {title}")
    
    # Convert DataFrame to HTML with styling
    table_html = table_css + df.to_html(classes='dataframe', escape=False, index=False)
    
    # Display the table with optional height constraint
    if height:
        st.markdown(f'<div style="height: {height}px; overflow-y: auto;">{table_html}</div>', unsafe_allow_html=True)
    else:
        st.markdown(table_html, unsafe_allow_html=True)
    
    # Add summary statistics if requested
    if with_stats:
        if use_expander_for_stats:
            with st.expander("View Summary Statistics"):
                numeric_cols = df.select_dtypes(include=['number'])
                if not numeric_cols.empty:
                    st.table(numeric_cols.describe().T)
                else:
                    st.info("No numerical columns available for statistics.")
        else:
            # Display statistics without an expander
            st.markdown("##### Summary Statistics")
            numeric_cols = df.select_dtypes(include=['number'])
            if not numeric_cols.empty:
                st.table(numeric_cols.describe().T)
            else:
                st.info("No numerical columns available for statistics.")

# Initialize Streamlit
st.set_page_config(page_title="AI Doctor - Chronic Disease Analyzer", page_icon="👨‍⚕️", layout="wide")
st.title("AI Doctor - Chronic Disease Analyzer 👨‍⚕️")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_file' not in st.session_state:
    st.session_state.current_file = os.path.join('data', 'chronic_diseases_data.csv')
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'page' not in st.session_state:
    st.session_state.page = None

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Load existing data if available
if os.path.exists(st.session_state.current_file):
    try:
        st.session_state.data = pd.read_csv(st.session_state.current_file)
    except Exception as e:
        st.warning(f"Could not load existing data: {str(e)}")
        st.session_state.data = pd.DataFrame()

# Predefined analysis prompts
ANALYSIS_PROMPTS = [
    "Provide a comprehensive analysis of the top 3 chronic diseases affecting {country}'s population, including prevalence rates, major risk factors, and healthcare system response.",
    "Analyze the economic burden of chronic diseases on {country}'s healthcare system and national economy, including direct medical costs and productivity losses.",
    "Generate a detailed comparison of chronic disease rates among different demographic groups in {country}.",
    "Identify the trends in chronic disease prevalence over the past 20 years in {country} and project likely scenarios for the next decade.",
    "Analyze the effectiveness of existing prevention and management programs for major chronic diseases in {country}.",
    "Examine the relationship between urbanization, lifestyle changes, and rising rates of chronic diseases in {country}.",
    "Provide an analysis of how climate change and environmental factors are influencing chronic diseases in {country}.",
    "Generate a comparative analysis of {country}'s chronic disease burden versus similar countries.",
    "Analyze the impact of {country}'s healthcare policies on chronic disease outcomes.",
    "Examine how cultural factors and dietary practices influence chronic disease patterns in {country}."
]

# Agent 1: Healthcare Research Agent
class HealthcareResearchAgent:
    def __init__(self, api_key):
        self.llm = ChatOpenAI(temperature=0, model=MODEL_NAME)
        self.statistics = KEY_STATISTICS
        self.data_sources = DATA_SOURCES
        self.categories = STATISTICS_CATEGORIES
        # Add these tools
        self.tools = [
            Tool(
                name="web_scraper",
                func=self.scrape_website,
                description="Scrape website content using Selenium and BeautifulSoup"
            ),
            Tool(
                name="search_engine_api",
                func=self.search_api,
                description="Access search engine APIs for structured results"
            ),
            Tool(
                name="data_validator",
                func=self.validate_source,
                description="Validate data source credibility"
            )
        ]

    def scrape_website(self, url):
        """Scrape website content using Selenium"""
        driver = webdriver.Chrome(options=Options())
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        driver.quit()
        return soup.get_text()

    def validate_source(self, url):
        """Validate data source credibility"""
        trusted_domains = ['who.int', 'cdc.gov', 'nih.gov']
        return any(d in url for d in trusted_domains)

    def search_api(self, query, engine='google', num_results=5):
        """Access search engine APIs for structured results"""
        # This is a wrapper around search_web to maintain compatibility
        return self.search_web(query, engine, num_results)

    def get_random_user_agent(self):
        """Get a random user agent to avoid being blocked"""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
        ]
        return random.choice(user_agents)

    def search_web(self, query, engine='google', num_results=5):
        """Search the web using actual search engines."""
        try:
            st.info(f"Searching {engine} for: {query}")
            
            search_results = []
            
            if engine.lower() == 'google':
                # Search Google
                search_url = f"https://www.google.com/search?q={quote(query)}&num={num_results}"
                headers = {'User-Agent': self.get_random_user_agent()}
                
                response = requests.get(search_url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Parse Google search results
                for result in soup.select('div.g'):
                    try:
                        title_element = result.select_one('h3')
                        if not title_element:
                            continue
                            
                        title = title_element.get_text()
                        link_element = result.select_one('a')
                        link = link_element['href'] if link_element else ""
                        
                        # Clean up the URL
                        if link.startswith('/url?q='):
                            link = link.split('/url?q=')[1].split('&sa=')[0]
                            
                        # Find the snippet
                        snippet_element = result.select_one('div.VwiC3b')
                        snippet = snippet_element.get_text() if snippet_element else ""
                        
                        if title and link:
                            search_results.append({
                                'title': title,
                                'url': link,
                                'snippet': snippet,
                                'source': 'Google'
                            })
                    except Exception as e:
                        continue
                
            elif engine.lower() == 'duckduckgo':
                # Search DuckDuckGo
                search_url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
                headers = {'User-Agent': self.get_random_user_agent()}
                
                response = requests.get(search_url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Parse DuckDuckGo search results
                for result in soup.select('div.result'):
                    try:
                        title_element = result.select_one('a.result__a')
                        if not title_element:
                            continue
                            
                        title = title_element.get_text()
                        link = title_element['href']
                        
                        # Clean up the URL
                        if link.startswith('/'):
                            link_parts = re.search(r'uddg=([^&]+)', link)
                            if link_parts:
                                link = requests.utils.unquote(link_parts.group(1))
                            
                        # Find the snippet
                        snippet_element = result.select_one('a.result__snippet')
                        snippet = snippet_element.get_text() if snippet_element else ""
                        
                        if title and link:
                            search_results.append({
                                'title': title,
                                'url': link,
                                'snippet': snippet,
                                'source': 'DuckDuckGo'
                            })
                    except Exception as e:
                        continue
            
            return search_results
            
        except Exception as e:
            st.error(f"Error in web search: {str(e)}")
            return []

    def extract_numerical_data(self, text):
        """Extract numerical data from text."""
        numbers = []
        # Look for numbers with optional commas and decimal points
        for match in re.finditer(r'(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*%)?', text):
            number_str = match.group(0)
            # Remove commas for processing
            number_str = number_str.replace(',', '')
            # Check if it's a percentage
            is_percent = '%' in number_str
            if is_percent:
                number_str = number_str.replace('%', '')
            
            try:
                number = float(number_str)
                numbers.append({
                    'value': number,
                    'is_percent': is_percent,
                    'context': text[max(0, match.start() - 30):min(len(text), match.end() + 30)]
                })
            except:
                continue
                
        return numbers
        
    def analyze_text_relevance(self, text, statistic):
        """Analyze if text is relevant to the statistic and extract scores."""
        if not text:
            return False, 0.0, []
            
        # Simple relevance check - does the text contain the statistic keyword or synonyms?
        if statistic.lower() in text.lower():
            score = 1.0
            relevance = True
        elif any(syn in text.lower() for syn in self.get_synonyms(statistic)):
            score = 0.7
            relevance = True
        else:
            score = 0.0
            relevance = False
            
        # Extract numbers if relevant
        numbers = self.extract_numerical_data(text) if relevance else []
        
        return relevance, score, numbers
        
    def get_synonyms(self, term):
        """Get synonyms for a search term."""
        term_lower = term.lower()
        synonyms = {
            'population': ['populace', 'inhabitants', 'residents', 'people', 'citizens'],
            'mortality': ['death rate', 'fatality', 'deaths', 'deceased'],
            'prevalence': ['occurrence', 'frequency', 'commonness', 'rate'],
            'incidence': ['rate', 'occurrence', 'frequency', 'cases'],
            'healthcare': ['health care', 'medical care', 'health services']
        }
        
        # Return empty list if no synonyms found
        for key, values in synonyms.items():
            if key in term_lower:
                return values
                
        return []

    def extract_data_from_search_results(self, results, statistic, country):
        """Extract relevant data from search results."""
        data_points = []
        
        if not results:
            return data_points
            
        for result in results:
            try:
                # Check if the result is relevant to our statistic
                is_relevant_title, title_score, title_numbers = self.analyze_text_relevance(result['title'], statistic)
                is_relevant_snippet, snippet_score, snippet_numbers = self.analyze_text_relevance(result['snippet'], statistic)
                
                # If either title or snippet is relevant
                if is_relevant_title or is_relevant_snippet:
                    relevance_score = max(title_score, snippet_score)
                    combined_text = f"{result['title']} - {result['snippet']}"
                    
                    # Combine numbers from title and snippet
                    numbers = title_numbers + snippet_numbers if title_numbers or snippet_numbers else []
                    
                    # Double check that the country is mentioned
                    if country.lower() in combined_text.lower():
                        relevance_score += 0.5  # Bonus for mentioning the country
                    
                    data_point = {
                        'text': combined_text,
                        'numbers': numbers,
                        'source': result.get('source', 'Web Search'),
                        'url': result.get('url', ''),
                        'relevance_score': relevance_score,
                        'has_temporal_data': bool(re.search(r'\b20\d{2}\b', combined_text)),
                        'has_comparison': any(word in combined_text.lower() 
                                            for word in ['increase', 'decrease', 'higher', 'lower', 'compared'])
                    }
                    
                    data_points.append(data_point)
                    
                    # Try to fetch and analyze the linked page for more detailed information
                    try:
                        if result.get('url'):
                            headers = {'User-Agent': self.get_random_user_agent()}
                            page_response = requests.get(result['url'], headers=headers, timeout=8)
                            page_soup = BeautifulSoup(page_response.text, 'html.parser')
                            
                            # Look for specific statistics in the page content
                            for element in page_soup.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'span', 'li']):
                                element_text = element.get_text().strip()
                                
                                if statistic.lower() in element_text.lower() and country.lower() in element_text.lower():
                                    element_numbers = self.extract_numerical_data(element_text)
                                    
                                    if element_numbers:
                                        data_point = {
                                            'text': element_text,
                                            'numbers': element_numbers,
                                            'source': f"Web Page: {result['title']}",
                                            'url': result['url'],
                                            'relevance_score': 2.0,  # Higher score for direct page content
                                            'has_temporal_data': bool(re.search(r'\b20\d{2}\b', element_text)),
                                            'has_comparison': any(word in element_text.lower() 
                                                                for word in ['increase', 'decrease', 'higher', 'lower', 'compared'])
                                        }
                                        data_points.append(data_point)
                            
                            # Also look for tables in the page
                            tables = page_soup.find_all('table')
                            for table in tables:
                                # Extract table headers
                                headers = []
                                for th in table.find_all('th'):
                                    headers.append(th.text.strip())
                                
                                if not headers and table.find('tr'):
                                    # Try to use first row as headers if no th elements
                                    first_row = table.find('tr')
                                    headers = [td.text.strip() for td in first_row.find_all('td')]
                                
                                if not headers:
                                    continue
                                    
                                # Check if table is relevant to our statistic
                                header_text = ' '.join(headers).lower()
                                if statistic.lower() in header_text or country.lower() in header_text:
                                    # Extract rows
                                    rows = []
                                    for tr in table.find_all('tr')[1:] if headers else table.find_all('tr'):  # Skip header row if we have headers
                                        row = []
                                        for td in tr.find_all(['td', 'th']):
                                            row.append(td.text.strip())
                                        if row:
                                            rows.append(row)
                                    
                                    if rows:
                                        data_point = {
                                            'text': f"Table data for {statistic} from {result['title']}",
                                            'numbers': [cell for row in rows for cell in row if self.extract_numerical_data(cell)],
                                            'source': f"Web Table: {result['title']}",
                                            'url': result['url'],
                                            'relevance_score': 2.0,
                                            'has_temporal_data': any(bool(re.search(r'\b20\d{2}\b', ' '.join(row))) for row in rows),
                                            'has_comparison': False,
                                            'is_table': True,
                                            'table_headers': headers,
                                            'table_rows': rows
                                        }
                                        data_points.append(data_point)
                    
                    except Exception as e:
                        st.warning(f"Error fetching page content from {result.get('url', 'unknown URL')}: {str(e)}")
                        continue
            except Exception as e:
                st.warning(f"Error processing search result: {str(e)}")
                continue
        
        return data_points
        
    def search_country_data(self, country):
        """Search for key statistics for a specific country."""
        try:
            country_data = {
                'country': country,
                'timestamp': datetime.now().isoformat(),
                'data': {category: {} for category in self.categories.keys()}
            }
            
            country_code = country.lower().replace(' ', '-')
            
            # Track found statistics to avoid duplicates
            found_statistics = set()
            
            # First, run web searches for each major statistic
            st.info(f"Running web searches for {country} health statistics...")
            
            # Define search engines to use
            engines = ['google', 'duckduckgo']
            current_engine = random.choice(engines)  # Start with a random engine
            
            # Perform searches for each statistic
            for statistic in self.statistics:
                # Skip if we've already found data for this statistic
                if statistic in found_statistics:
                    continue
                
                # Get category for this statistic
                category = self.stat_to_category.get(statistic, 'population_stats')
                
                # Alternate between search engines for better results and to avoid rate limiting
                current_engine = 'google' if current_engine == 'duckduckgo' else 'duckduckgo'
                
                # Create search query
                search_query = f"{country} {statistic} statistics data"
                
                # Get search results
                results = self.search_web(search_query, engine=current_engine)
                
                # Extract data from search results
                data_points = self.extract_data_from_search_results(results, statistic, country)
                
                # Add data points to country data
                if data_points:
                    if statistic not in country_data['data'][category]:
                        country_data['data'][category][statistic] = []
                    
                    country_data['data'][category][statistic].extend(data_points)
                    found_statistics.add(statistic)
                
                # Add a small delay to avoid rate limiting
                time.sleep(random.uniform(0.5, 1.5))
            
            # Now proceed with targeted website searches from the data sources
            st.info(f"Searching for population and key statistics for {country}...")
            
            # Use World Population Review and UN Population as primary sources for population data
            population_sources = [s for s in self.data_sources if 'Population' in s['name']]
            for source in population_sources:
                try:
                    base_url = source['url_template'].format(country=country_code)
                    
                    try:
                        headers = {
                            'User-Agent': self.get_random_user_agent(),
                            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
                        }
                        response = requests.get(base_url, headers=headers, timeout=10)
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Look for population data in prominent elements
                        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'div', 'span']):
                            text = element.get_text().strip()
                            
                            # Check for population statistics
                            for statistic in ['population', 'population density', 'total population']:
                                if statistic in text.lower():
                                    numbers = self.extract_numerical_data(text)
                                    if numbers:
                                        category = 'population_stats'
                                        if statistic not in country_data['data'][category]:
                                            country_data['data'][category][statistic] = []
                                        
                                        data_point = {
                                            'text': text,
                                            'numbers': numbers,
                                            'source': source['name'],
                                            'url': base_url,
                                            'relevance_score': 2.0, # High relevance for population
                                            'has_temporal_data': bool(re.search(r'\b20\d{2}\b', text)),
                                            'has_comparison': False
                                        }
                                        
                                        country_data['data'][category][statistic].append(data_point)
                                        found_statistics.add(statistic)
                    except requests.RequestException as e:
                        st.warning(f"Error accessing {base_url}: {str(e)}")
                        continue
                        
                except Exception as e:
                    st.warning(f"Error processing source {source['name']}: {str(e)}")
                    continue
            
            # Search for all other statistics across remaining data sources
            st.info(f"Searching for disease, demographic, and geographic statistics for {country}...")
            
            for source in self.data_sources:
                try:
                    base_url = source['url_template'].format(country=country_code)
                    urls_to_search = [base_url] + [f"{base_url}/{path}" for path in source['data_paths']]
                    
                    for url in urls_to_search:
                        try:
                            headers = {
                                'User-Agent': self.get_random_user_agent(),
                                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
                            }
                            response = requests.get(url, headers=headers, timeout=10)
                            soup = BeautifulSoup(response.text, 'html.parser')
                            
                            # Look for data tables
                            tables = soup.find_all('table')
                            for table in tables:
                                # Extract table data
                                headers = []
                                for th in table.find_all('th'):
                                    headers.append(th.text.strip())
                                
                                if not headers:
                                    continue
                                
                                rows = []
                                for tr in table.find_all('tr'):
                                    row = []
                                    for td in tr.find_all(['td', 'th']):
                                        row.append(td.text.strip())
                                    if row:
                                        rows.append(row)
                                
                                if rows:
                                    # Determine which category the table belongs to
                                    best_category = None
                                    best_statistic = None
                                    
                                    # Look for keywords in headers
                                    header_text = ' '.join(headers).lower()
                                    for stat in self.statistics:
                                        if stat in header_text:
                                            best_statistic = stat
                                            best_category = self.stat_to_category.get(stat, 'population_stats')
                                            break
                                    
                                    if best_category and best_statistic:
                                        data_point = {
                                            'text': f"Table data for {best_statistic}",
                                            'numbers': [row[1] if len(row) > 1 else '' for row in rows if len(row) > 1],
                                            'source': source['name'],
                                            'url': url,
                                            'relevance_score': 1.8,
                                            'has_temporal_data': any(bool(re.search(r'\b20\d{2}\b', ' '.join(row))) for row in rows),
                                            'has_comparison': False,
                                            'is_table': True,
                                            'table_headers': headers,
                                            'table_rows': rows
                                        }
                                        
                                        if best_statistic not in country_data['data'][best_category]:
                                            country_data['data'][best_category][best_statistic] = []
                                        country_data['data'][best_category][best_statistic].append(data_point)
                            
                            # Search for statistics in text elements
                            for element in soup.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'li']):
                                text = element.get_text().strip()
                                
                                # Check for each statistic
                                for statistic in self.statistics:
                                    if statistic in found_statistics:
                                        continue
                                        
                                    is_relevant, relevance_score, numbers = self.analyze_text_relevance(text, statistic)
                                    
                                    if is_relevant and relevance_score > 1.0:
                                        # Get category for this statistic
                                        category = self.stat_to_category.get(statistic, 'population_stats')
                                        
                                        # Structure the data
                                        data_point = {
                                            'text': text,
                                            'numbers': numbers,
                                            'source': source['name'],
                                            'url': url,
                                            'relevance_score': relevance_score,
                                            'has_temporal_data': bool(re.search(r'\b20\d{2}\b', text)),
                                            'has_comparison': any(word in text.lower() for word in ['increase', 'decrease', 'higher', 'lower', 'compared'])
                                        }
                                        
                                        if statistic not in country_data['data'][category]:
                                            country_data['data'][category][statistic] = []
                                        country_data['data'][category][statistic].append(data_point)
                                        found_statistics.add(statistic)
                            
                        except requests.RequestException as e:
                            st.warning(f"Error accessing {url}: {str(e)}")
                            continue
                            
                except Exception as e:
                    st.warning(f"Error processing source {source['name']}: {str(e)}")
                    continue
            
            # Convert to structured format for DataFrame
            structured_data = []
            
            for category, statistics_data in country_data['data'].items():
                for statistic, data_points in statistics_data.items():
                    for data_point in data_points:
                        entry = {
                            'country': country,
                            'category': category,
                            'indicator': statistic,
                            'description': data_point['text'],
                            'numerical_values': data_point['numbers'],
                            'primary_value': data_point['numbers'][0] if data_point['numbers'] else None,
                            'source': data_point['source'],
                            'url': data_point['url'],
                            'relevance_score': data_point['relevance_score'],
                            'has_temporal_data': data_point['has_temporal_data'],
                            'has_comparison': data_point['has_comparison'],
                            'timestamp': country_data['timestamp'],
                            'is_data_resource': False,
                            'resource_type': '',
                            'is_table': data_point.get('is_table', False)
                        }
                        
                        # If it's a table, add the table data
                        if data_point.get('is_table', False):
                            entry['table_headers'] = data_point.get('table_headers', [])
                            entry['table_rows'] = data_point.get('table_rows', [])
                        
                        structured_data.append(entry)
            
            # Create summary of findings
            summary = self.create_country_summary(structured_data)
            
            return pd.DataFrame(structured_data), summary
            
        except Exception as e:
            return f"Error during research: {str(e)}"

    def create_country_summary(self, structured_data):
        """Create a summary of findings for the country."""
        summary = {
            'total_indicators_found': len(structured_data),
            'categories_coverage': {},
            'key_findings': [],
            'data_quality': {
                'temporal_data_percentage': 0,
                'comparison_data_percentage': 0,
                'numerical_data_percentage': 0
            }
        }
        
        # Calculate statistics
        df = pd.DataFrame(structured_data)
        if not df.empty:
            summary['categories_coverage'] = df.groupby('category').size().to_dict()
            summary['data_quality'] = {
                'temporal_data_percentage': (df['has_temporal_data'].sum() / len(df)) * 100,
                'comparison_data_percentage': (df['has_comparison'].sum() / len(df)) * 100,
                'numerical_data_percentage': (df['numerical_values'].apply(bool).sum() / len(df)) * 100
            }
            
            # Extract key findings (high relevance scores)
            high_relevance_data = df[df['relevance_score'] > 1.5].sort_values('relevance_score', ascending=False)
            summary['key_findings'] = high_relevance_data[['category', 'indicator', 'description']].to_dict('records')[:5]
        
        return summary

# Agent 2: Data Structuring Agent
class DataStructuringAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        self.tools = [
            Tool(
                name="schema_validator",
                func=self.validate_schema,
                description="Validate data against predefined schema"
            ),
            Tool(
                name="normalize_values",
                func=self.normalize_data,
                description="Normalize medical values to standard units"
            )
        ]

    def validate_schema(self, data):
        """Validate data structure against schema"""
        required_fields = ['country', 'indicator', 'value', 'source']
        return all(field in data for field in required_fields)

    def normalize_data(self, data):
        """Normalize medical values to standard units"""
        # Implement normalization logic
        return data

    def structure_data(self, raw_data):
        """Structure the collected data into a consistent format."""
        structured_data = []
        
        # Get base information
        country = raw_data.get('country', '')
        timestamp = raw_data.get('timestamp', '')
        
        # Helper function to extract values from data
        def extract_values(data):
            if isinstance(data, dict):
                if 'data' in data:
                    return data['data']
                return [data]
            elif isinstance(data, list):
                return data
            return []
        
        # Process demographics data
        demographics = extract_values(raw_data.get('demographics', {}))
        if isinstance(demographics, list):
            for demo in demographics:
                if isinstance(demo, dict):
                    entry = {
                        'country': country,
                        'timestamp': timestamp,
                        'category': 'demographics',
                        'indicator': demo.get('indicator', ''),
                        'value': demo.get('value', ''),
                        'unit': demo.get('unit', ''),
                        'year': demo.get('year', ''),
                        'source': demo.get('source', ''),
                        'region': demo.get('region', '')
                    }
                    structured_data.append(entry)
        
        # Process chronic diseases data
        diseases = extract_values(raw_data.get('chronic_diseases', {}))
        if isinstance(diseases, list):
            for disease in diseases:
                if isinstance(disease, dict):
                    entry = {
                        'country': country,
                        'timestamp': timestamp,
                        'category': 'chronic_disease',
                        'indicator': disease.get('disease', ''),
                        'value': disease.get('prevalence', ''),
                        'unit': disease.get('unit', '%'),
                        'year': disease.get('year', ''),
                        'source': disease.get('source', ''),
                        'region': disease.get('region', '')
                    }
                    structured_data.append(entry)
        
        # Process mortality rates
        mortality = extract_values(raw_data.get('mortality_rates', {}))
        if isinstance(mortality, list):
            for rate in mortality:
                if isinstance(rate, dict):
                    entry = {
                        'country': country,
                        'timestamp': timestamp,
                        'category': 'mortality',
                        'indicator': rate.get('cause', ''),
                        'value': rate.get('rate', ''),
                        'unit': rate.get('unit', 'per 100,000'),
                        'year': rate.get('year', ''),
                        'source': rate.get('source', ''),
                        'region': rate.get('region', '')
                    }
                    structured_data.append(entry)
        
        # Process healthcare metrics
        healthcare = extract_values(raw_data.get('healthcare_metrics', {}))
        if isinstance(healthcare, list):
            for metric in healthcare:
                if isinstance(metric, dict):
                    entry = {
                        'country': country,
                        'timestamp': timestamp,
                        'category': 'healthcare',
                        'indicator': metric.get('metric', ''),
                        'value': metric.get('value', ''),
                        'unit': metric.get('unit', ''),
                        'year': metric.get('year', ''),
                        'source': metric.get('source', ''),
                        'region': metric.get('region', '')
                    }
                    structured_data.append(entry)
        
        # Process regions if available
        regions = raw_data.get('regions', [])
        if isinstance(regions, list):
            for region in regions:
                if isinstance(region, str):
                    entry = {
                        'country': country,
                        'timestamp': timestamp,
                        'category': 'region',
                        'indicator': 'administrative_region',
                        'value': region,
                        'unit': '',
                        'year': '',
                        'source': '',
                        'region': region
                    }
                    structured_data.append(entry)
                elif isinstance(region, dict):
                    entry = {
                        'country': country,
                        'timestamp': timestamp,
                        'category': 'region',
                        'indicator': region.get('type', 'administrative_region'),
                        'value': region.get('name', ''),
                        'unit': '',
                        'year': '',
                        'source': region.get('source', ''),
                        'region': region.get('name', '')
                    }
                    structured_data.append(entry)
        
        return structured_data

# Agent 3: Medical Data Cleaning Agent
class MedicalDataCleaningAgent:
    def __init__(self, api_key):
        self.llm = ChatOpenAI(temperature=0, model=MODEL_NAME)
    
    def clean_medical_data(self, df):
        """Clean and standardize medical data."""
        try:
            df_cleaned = df.copy()
            
            # Convert list columns to strings to make them hashable
            for col in ['diagnoses', 'symptoms', 'treatments', 'medications']:
                if col in df_cleaned.columns:
                    # Convert lists to string representation for drop_duplicates
                    df_cleaned[f'{col}_str'] = df_cleaned[col].apply(
                        lambda x: str(x) if isinstance(x, list) else x)
            
            # Remove duplicates based on non-list columns and the string versions of list columns
            cols_for_dedup = [col for col in df_cleaned.columns if not col in ['diagnoses', 'symptoms', 'treatments', 'medications']]
            cols_for_dedup += [f'{col}_str' for col in ['diagnoses', 'symptoms', 'treatments', 'medications'] if f'{col}_str' in df_cleaned.columns]
            df_cleaned = df_cleaned.drop_duplicates(subset=cols_for_dedup)
            
            # Remove the temporary string columns
            for col in ['diagnoses', 'symptoms', 'treatments', 'medications']:
                if f'{col}_str' in df_cleaned.columns:
                    df_cleaned = df_cleaned.drop(columns=[f'{col}_str'])
            
            # Standardize country names
            df_cleaned['country'] = df_cleaned['country'].str.title()
            
            # Clean and standardize indicators
            df_cleaned['indicator'] = df_cleaned['indicator'].str.strip()
            
            # Convert timestamps
            df_cleaned['timestamp'] = pd.to_datetime(df_cleaned['timestamp'])
            
            return df_cleaned
        except Exception as e:
            print(f"Error cleaning medical data: {e}")
            return df

# Agent 4: Medical Analysis Agent
class MedicalAnalysisAgent:
    def __init__(self, api_key):
        self.llm = ChatOpenAI(temperature=0, model=MODEL_NAME)
        self.tools = [
            Tool(
                name="calculate_relative_risk",
                func=self.calculate_rr,
                description="Calculate relative risk for disease factors"
            ),
            Tool(
                name="run_statistical_test",
                func=self.stat_test,
                description="Perform statistical tests (t-test, chi-square)"
            )
        ]

    def calculate_rr(self, exposed_group, control_group):
        """Calculate relative risk"""
        rr = ((exposed_group['cases']/exposed_group['total']) / 
              (control_group['cases']/control_group['total']))
        return round(rr, 2)

    def stat_test(self, data):
        """Perform statistical tests (t-test, chi-square)"""
        # Implement statistical testing logic
        return "Statistical test results"

    def analyze_table_data(self, df, analysis_type="descriptive"):
        """
        Perform professional analysis on tabular data.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            analysis_type (str): Type of analysis to perform
                - "descriptive": Basic summary statistics
                - "trend": Trend analysis for time series data
                - "correlation": Correlation analysis for numerical columns
                - "pattern": Pattern detection in categorical data
        
        Returns:
            dict: Analysis results with formatted tables and insights
        """
        results = {
            "summary": "",
            "tables": [],
            "visualizations": [],
            "insights": []
        }
        
        # Basic data cleaning and type conversion
        # Convert string numbers to float where possible
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Replace commas, handle percentage signs
                    df[col] = df[col].str.replace(',', '').str.replace('%', '')
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
        
        # 1. Basic descriptive statistics
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Generate summary statistics for numeric columns
        if numeric_cols:
            desc_stats = df[numeric_cols].describe().T
            # Add additional metrics
            if len(df) > 1:
                desc_stats['skew'] = df[numeric_cols].skew()
                desc_stats['kurtosis'] = df[numeric_cols].kurtosis()
            
            results["tables"].append({
                "title": "Numeric Column Statistics",
                "data": desc_stats,
                "format": "table"
            })
            
            # Add key insights about numeric columns
            for col in numeric_cols:
                insights = []
                try:
                    # Check for outliers (values > 3 std devs from mean)
                    mean = df[col].mean()
                    std = df[col].std()
                    if not pd.isna(std) and std > 0:
                        outliers = df[abs(df[col] - mean) > 3*std]
                        if len(outliers) > 0:
                            insights.append(f"Column '{col}' has {len(outliers)} potential outliers (values more than 3 standard deviations from the mean).")
                
                    # Check for skewness
                    skew = df[col].skew()
                    if not pd.isna(skew):
                        if abs(skew) > 1:
                            skew_direction = "right" if skew > 0 else "left"
                            insights.append(f"Column '{col}' is highly skewed to the {skew_direction} (skewness = {skew:.2f}).")
                
                    # Range and distribution insights
                    if not pd.isna(mean):
                        insights.append(f"The average value for '{col}' is {mean:.2f}, with values ranging from {df[col].min():.2f} to {df[col].max():.2f}.")
                
                except Exception as e:
                    pass
                
                if insights:
                    results["insights"].extend(insights)
        
        # 2. Frequency analysis for categorical columns
        if categorical_cols and len(df) > 0:
            for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
                try:
                    freq_table = df[col].value_counts().reset_index()
                    freq_table.columns = [col, 'Count']
                    freq_table['Percentage'] = (freq_table['Count'] / len(df) * 100).round(2)
                    
                    # Only include if we have meaningful data
                    if len(freq_table) > 0 and len(freq_table) <= 20:  # Limit to categories that make sense to display
                        results["tables"].append({
                            "title": f"Frequency Distribution: {col}",
                            "data": freq_table,
                            "format": "table"
                        })
                        
                        # Add insights about the distribution
                        top_category = freq_table.iloc[0]
                        results["insights"].append(
                            f"The most common value for '{col}' is '{top_category[col]}' ({top_category['Percentage']}% of data)."
                        )
                except:
                    pass
        
        # 3. Time series analysis if date columns exist
        date_cols = []
        for col in df.columns:
            try:
                if df[col].dtype == 'object':
                    # Try to convert to datetime
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                
                if pd.api.types.is_datetime64_dtype(df[col]):
                    date_cols.append(col)
            except:
                pass
        
        if date_cols and numeric_cols and analysis_type in ["trend", "descriptive"]:
            # Try to identify time series patterns
            for date_col in date_cols:
                for num_col in numeric_cols[:2]:  # Limit to first 2 numeric columns
                    try:
                        # Sort by date and analyze trends
                        trend_df = df[[date_col, num_col]].dropna().sort_values(date_col)
                        if len(trend_df) >= 3:  # Need at least 3 points for trend
                            # Calculate simple trend metrics
                            first_val = trend_df[num_col].iloc[0]
                            last_val = trend_df[num_col].iloc[-1]
                            pct_change = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0
                            
                            # Identify trend direction
                            trend = "increasing" if pct_change > 5 else ("decreasing" if pct_change < -5 else "stable")
                            
                            results["insights"].append(
                                f"'{num_col}' shows a {trend} trend over time ({pct_change:.1f}% change from {trend_df[date_col].min().strftime('%Y-%m-%d')} to {trend_df[date_col].max().strftime('%Y-%m-%d')})."
                            )
                    except:
                        pass
        
        # 4. Correlation analysis for numeric columns
        if len(numeric_cols) >= 2 and len(df) >= 5 and analysis_type in ["correlation", "descriptive"]:
            try:
                corr_matrix = df[numeric_cols].corr().round(2)
                
                # Extract strongest correlations
                correlations = []
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        col1, col2 = numeric_cols[i], numeric_cols[j]
                        corr_val = corr_matrix.loc[col1, col2]
                        if not pd.isna(corr_val) and abs(corr_val) >= 0.5:  # Only strong correlations
                            correlations.append((col1, col2, corr_val))
                
                # Sort by absolute correlation strength
                correlations.sort(key=lambda x: abs(x[2]), reverse=True)
                
                # Add correlation insights
                for col1, col2, corr_val in correlations[:3]:  # Top 3 correlations
                    relationship = "strong positive" if corr_val >= 0.7 else \
                                   "moderate positive" if corr_val >= 0.5 else \
                                   "strong negative" if corr_val <= -0.7 else \
                                   "moderate negative"
                    
                    results["insights"].append(
                        f"There is a {relationship} correlation ({corr_val:.2f}) between '{col1}' and '{col2}'."
                    )
                    
                if correlations:
                    # Create a dataframe of top correlations for display
                    corr_df = pd.DataFrame(correlations, columns=['Variable 1', 'Variable 2', 'Correlation'])
                    results["tables"].append({
                        "title": "Strongest Variable Correlations",
                        "data": corr_df,
                        "format": "table"
                    })
            except:
                pass
        
        # Generate overall summary
        summary_points = []
        
        if len(df) > 0:
            summary_points.append(f"The dataset contains {len(df)} rows and {len(df.columns)} columns.")
            
            if numeric_cols:
                summary_points.append(f"There are {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns.")
            
            if results["insights"]:
                summary_points.append(f"Key insights include: {results['insights'][0]}")
        
        results["summary"] = " ".join(summary_points)
        
        return results

    def analyze_medical_data(self, df, country, prompt):
        """Analyze medical data based on specific prompts."""
        try:
            # Add a tool to analyze tables professionally
            def analyze_df_tables(query=None):
                """Analyze tables in the dataframe and provide professional insights"""
                try:
                    # Filter to only get table data for this country
                    country_df = df[df['country'] == country] if 'country' in df.columns else df
                    
                    # Look for tabular data
                    table_data = country_df[country_df.get('is_table', False) == True]
                    
                    if table_data.empty:
                        return "No tabular data found for analysis."
                    
                    all_analyses = []
                    
                    # Analyze each table
                    for idx, row in table_data.iterrows():
                        if not (row.get('table_headers') and row.get('table_rows')):
                            continue
                            
                        try:
                            # Create a DataFrame from the table data
                            headers = row['table_headers']
                            rows = row['table_rows']
                            
                            if not headers or not rows:
                                continue
                                
                            # Convert rows to proper format
                            table_df = pd.DataFrame(rows, columns=headers)
                            
                            # Different analysis based on the query or content
                            analysis_type = "descriptive"  # default
                            if query:
                                if any(term in query.lower() for term in ["trend", "time", "change", "growth"]):
                                    analysis_type = "trend"
                                elif any(term in query.lower() for term in ["correlation", "relationship", "connect"]):
                                    analysis_type = "correlation"
                                elif any(term in query.lower() for term in ["pattern", "distribution", "category"]):
                                    analysis_type = "pattern"
                            
                            # Perform analysis
                            analysis = self.analyze_table_data(table_df, analysis_type)
                            
                            # Format the analysis as markdown
                            analysis_text = f"## Analysis of {row.get('indicator', 'Table')}\n\n"
                            
                            if analysis["summary"]:
                                analysis_text += f"**Summary:** {analysis['summary']}\n\n"
                            
                            if analysis["insights"]:
                                analysis_text += "### Key Insights\n"
                                for insight in analysis["insights"]:
                                    analysis_text += f"- {insight}\n"
                                analysis_text += "\n"
                            
                            for table_info in analysis["tables"]:
                                if isinstance(table_info["data"], pd.DataFrame):
                                    analysis_text += f"### {table_info['title']}\n"
                                    analysis_text += f"```\n{format_table_professionally(table_info['data'])}\n```\n\n"
                            
                            all_analyses.append(analysis_text)
                        except Exception as e:
                            continue
                    
                    if all_analyses:
                        return "\n".join(all_analyses)
                    else:
                        return "Tables were found but could not be properly analyzed."
                        
                except Exception as e:
                    return f"Error analyzing tables: {str(e)}"
            
            # Create pandas agent with tools
            pandas_kwargs = {
                "prefix": f"""You are a professional healthcare data analyst specializing in chronic disease data for {country}.
                Your task is to analyze the data and provide clear, professional insights.
                When presenting results, use well-structured tables, clear statistics, and professional formatting.
                
                For any tables you present:
                1. Include descriptive headers
                2. Format numbers consistently (e.g., 2 decimal places for percentages)
                3. Sort data in a meaningful order (e.g., by value or alphabetically)
                4. Include summary statistics when relevant
                5. Explain the significance of the data shown
                
                ALWAYS provide a brief interpretation after presenting data tables or charts.
                Highlight key trends, patterns, outliers, or noteworthy findings in the data.
                """,
                "agent_type": AgentType.OPENAI_FUNCTIONS,
                "verbose": True
            }
            
            agent = create_pandas_dataframe_agent(
                self.llm,
                df,
                **pandas_kwargs
            )
            
            # Format prompt with country name and add instructions for professional table formatting
            enhanced_prompt = f"""
            {prompt.format(country=country)}
            
            Important: When presenting tabular data, format it professionally as a data analyst would.
            Follow these principles:
            
            1. Structure tables with clear headers and aligned columns
            2. Include relevant summary statistics (mean, median, min, max) for numerical data
            3. Format numbers consistently with appropriate decimal places
            4. Add a brief interpretation below each table explaining key insights
            5. Highlight notable trends, outliers, or patterns in the data
            6. Use proper labeling for all data points
            
            For time series data, organize chronologically and identify trends.
            For categorical data, sort by frequency or relevance unless another order is more informative.
            
            If there are tables in the data that need analysis, use the analyze_tables tool.
            """
            
            # Add specialized tool for table analysis
            tools = [
                Tool(
                    name="analyze_tables",
                    func=analyze_df_tables,
                    description="Analyze tables in the data and provide professional insights"
                )
            ]
            
            # Use LangChain agent to run the analysis
            agent_with_tools = initialize_agent(
                tools,
                self.llm,
                agent=AgentType.OPENAI_FUNCTIONS,
                verbose=True
            )
            
            # First, check if there are tables that need analysis
            tables_exist = (df.get('is_table', pd.Series([False] * len(df))) == True).any()
            
            if tables_exist:
                # Start with table analysis
                table_analysis = analyze_df_tables()
                
                # Then run the main analysis incorporating the table insights
                full_prompt = f"""
                {enhanced_prompt}
                
                Here is the analysis of tables in the data:
                
                {table_analysis}
                
                Use this information in your analysis and refer to these insights where relevant.
                """
                
                result = agent.run(full_prompt)
            else:
                # Just run the regular analysis
                result = agent.run(enhanced_prompt)
            
            # Post-process to make tables look better
            if "```" in result and ("|" in result or "," in result):
                # Extract table data between markdown code blocks
                import re
                table_blocks = re.findall(r"```(?:csv|markdown)?\s*([\s\S]*?)```", result)
                
                for table_block in table_blocks:
                    # Try to parse as CSV or markdown table
                    try:
                        if "|" in table_block:
                            # Looks like a markdown table
                            lines = [line.strip() for line in table_block.strip().split("\n")]
                            headers = [h.strip() for h in lines[0].strip("|").split("|")]
                            
                            # Skip the separator line
                            data = []
                            for line in lines[2:]:
                                if line.strip():
                                    data.append([cell.strip() for cell in line.strip("|").split("|")])
                            
                            # Format table professionally
                            formatted_table = format_table_professionally(data, headers)
                            
                            # Replace the original table with our professionally formatted one
                            result = result.replace(f"```{table_block}```", f"```\n{formatted_table}\n```")
                        
                        elif "," in table_block:
                            # Looks like CSV data
                            from io import StringIO
                            df_table = pd.read_csv(StringIO(table_block))
                            formatted_table = format_table_professionally(df_table, add_analysis=True)
                            result = result.replace(f"```{table_block}```", f"```\n{formatted_table}\n```")
                    except:
                        # If parsing fails, leave as is
                        continue
            
            return result
        except Exception as e:
            return f"Error during medical analysis: {str(e)}"

# Agent 5: Healthcare Chat Agent
class HealthcareChatAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        self.llm = ChatOpenAI(temperature=0.2, model="gpt-4o-mini")
        self.conversation_history = []
        
        # Initialize tools with try-except for better error handling
        try:
            self.tools = [
                Tool(
                    name="format_data_as_table",
                    func=self.format_data_as_table,
                    description="Format data as a professional table with analysis"
                ),
                Tool(
                    name="analyze_medical_data",
                    func=self.analyze_medical_data,
                    description="Analyze medical data and provide insights"
                ),
                Tool(
                    name="create_visualization",
                    func=self.create_visualization,
                    description="Create data visualizations and charts"
                )
            ]
        except Exception as e:
            st.warning(f"Error initializing tools: {str(e)}")
            self.tools = []

    def format_response(self, response):
        """Format the response to be more presentable."""
        if isinstance(response, dict):
            if 'output' in response:
                return response['output'].strip()
            return '\n'.join(f"{k}: {v}" for k, v in response.items()).strip()
        return str(response).strip()

    def medical_chat(self, user_input, df, country):
        """Handle chat interactions with improved error handling."""
        try:
            # Create the agent with error handling
            agent = initialize_agent(
                self.tools,
                self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
                verbose=True
            )

            # Add conversation context
            context = f"""You are analyzing healthcare data for {country}. 
            Provide direct, professional responses focused on data analysis and insights.
            User query: {user_input}"""

            # Get response from agent
            response = agent.invoke({"input": context})
            formatted_response = self.format_response(response)
            
            return formatted_response

        except Exception as e:
            return f"Error processing request: {str(e)}"

    def format_data_as_table(self, query, df):
        """Format data as a professional table with analysis."""
        try:
            # Filter and prepare data based on query
            relevant_columns = [col for col in df.columns if col.lower() in query.lower()]
            if not relevant_columns:
                relevant_columns = df.columns

            # Create a formatted table
            table_data = df[relevant_columns].head(10)  # Limit to 10 rows for readability
            formatted_table = format_table_professionally(
                table_data,
                headers=relevant_columns,
                add_analysis=True
            )

            return f"""
            📊 **Data Analysis Results**
            
            {formatted_table}
            """

        except Exception as e:
            return f"""
            ⚠️ **Error in Data Formatting**
            
            Unable to format the data table: {str(e)}
            """

    def analyze_medical_data(self, query, df, country):
        """Analyze medical data and provide insights."""
        try:
            analysis_agent = MedicalAnalysisAgent(self.api_key)
            result = analysis_agent.analyze_medical_data(df, country, query)
            return f"""
            🔍 **Medical Data Analysis**
            
            {result}
            """
        except Exception as e:
            return f"""
            ⚠️ **Analysis Error**
            
            Unable to analyze the data: {str(e)}
            """

    def create_visualization(self, query, df):
        """Create data visualizations and charts."""
        try:
            # Determine chart type based on query
            chart_type = 'line'  # default
            if 'bar' in query.lower():
                chart_type = 'bar'
            elif 'scatter' in query.lower():
                chart_type = 'scatter'
            elif 'pie' in query.lower():
                chart_type = 'pie'

            # Create the visualization using the existing metric chart function
            fig = create_metric_chart(
                df,
                column='value' if 'value' in df.columns else df.columns[0],
                color='#29b5e8',
                chart_type=chart_type
            )

            return f"""
            📈 **Visualization**
            
            {fig}
            
            *Chart type: {chart_type}*
            """

        except Exception as e:
            return f"""
            ⚠️ **Visualization Error**
            
            Unable to create the visualization: {str(e)}
            """

@st.cache_resource
def init_agents():
    """Initialize AI agents with proper configuration."""
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        st.error("OpenAI API key not found. Please check your environment variables.")
        return None
    
    agents = {}
        
    try:
        # Initialize agents one by one with separate error handling
        try:
            agents['research'] = HealthcareResearchAgent(api_key)
        except Exception as e:
            st.warning(f"Error initializing research agent: {str(e)}")
            
        try:
            agents['structure'] = DataStructuringAgent(api_key)
        except Exception as e:
            st.warning(f"Error initializing structure agent: {str(e)}")
            
        try:
            agents['clean'] = MedicalDataCleaningAgent(api_key)
        except Exception as e:
            st.warning(f"Error initializing cleaning agent: {str(e)}")
            
        try:
            agents['analyze'] = MedicalAnalysisAgent(api_key)
        except Exception as e:
            st.warning(f"Error initializing analysis agent: {str(e)}")
            
        try:
            agents['chat'] = HealthcareChatAgent(api_key)
        except AttributeError as chat_error:
            if "'HealthcareChatAgent' object has no attribute 'tools'" in str(chat_error):
                st.warning("Chat agent needs to be reinitialized. Please refresh the page.")
            else:
                st.warning(f"Error initializing chat agent: {str(chat_error)}")
        except Exception as e:
            st.warning(f"Error initializing chat agent: {str(e)}")
            
        return agents
        
    except Exception as e:
        st.error(f"Unexpected error initializing agents: {str(e)}")
        return {}

agents = init_agents()

# Sidebar
with st.sidebar:
    st.title("AI Doctor Dashboard")
    
    # Country Selection
    country = st.text_input("Enter Country Name:", placeholder="e.g., United States")
    
    if country:
        # Main Navigation
        st.markdown("---")
        st.header("📊 Navigation")
        # If page is set in session state, use that as the default
        if st.session_state.page:
            default_idx = ["Analysis", "Visualization", "Chat with AI Doctor"].index(st.session_state.page)
            page = st.radio(
                "Select Function",
                ["Analysis", "Visualization", "Chat with AI Doctor"],
                index=default_idx
            )
            # Reset the page session state after using it
            st.session_state.page = None
        else:
            page = st.radio(
                "Select Function",
                ["Analysis", "Visualization", "Chat with AI Doctor"]
            )
        
        # Analysis Settings
        if page == "Analysis":
            st.markdown("---")
            st.header("📈 Analysis Settings")
            if not st.session_state.data.empty:
                selected_prompt = st.selectbox("Analysis Type:", ANALYSIS_PROMPTS)
                analyze_data = st.button("Analyze", use_container_width=True)
        
        # Visualization Settings
        elif page == "Visualization":
            st.markdown("---")
            st.header("📊 Visualization Settings")
            if not st.session_state.data.empty:
                max_date = st.session_state.data['timestamp'].max()
                min_date = st.session_state.data['timestamp'].min()
                
                if isinstance(max_date, pd.Timestamp):
                    max_date = max_date.date()
                if isinstance(min_date, pd.Timestamp):
                    min_date = min_date.date()
                    
                st.markdown("### Time Range")
                start_date = st.date_input(
                    "Start date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date
                )
                end_date = st.date_input(
                    "End date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date
                )
                
                st.markdown("### Display Options")
                time_frame = st.selectbox(
                    "Time Frame",
                    ("Daily", "Weekly", "Monthly", "Quarterly")
                )
                chart_selection = st.selectbox(
                    "Chart Type",
                    ("Bar", "Area")
                )
        
        # Chat Settings
        elif page == "Chat with AI Doctor":
            st.markdown("---")
            st.header("💬 Chat Settings")
            if not st.session_state.data.empty:
                st.markdown("Ask questions about the chronic disease data for " + country)
        
        # Download Options
        if not st.session_state.data.empty:
            st.markdown("---")
            st.header("📥 Download Options")
            st.download_button(
                "Download Full Dataset",
                data=st.session_state.data.to_csv(index=False),
                file_name=f"{country.lower()}_chronic_diseases.csv",
                mime="text/csv",
                use_container_width=True
            )

# Main Page Content
if country:
    if page == "Analysis":
        st.header(f"Chronic Disease Analysis for {country}")
        if not st.session_state.data.empty:
            if analyze_data:
                with st.spinner("Analyzing data..."):
                    result = agents['analyze'].analyze_medical_data(
                        st.session_state.data,
                        country,
                        selected_prompt
                    )
                    st.markdown("### Analysis Results")
                    st.write(result)
                    
                    # Store analysis results and extract tables
                    st.session_state.analysis_results[selected_prompt] = result
                    st.session_state.current_tables = []
                    
                    # Extract tables from the analysis result
                    if "```" in result:
                        table_blocks = re.findall(r"```(?:csv|markdown)?\s*([\s\S]*?)```", result)
                        for idx, table_block in enumerate(table_blocks):
                            try:
                                # Try to parse as DataFrame
                                if "|" in table_block:
                                    # Markdown table
                                    lines = [line.strip() for line in table_block.strip().split("\n")]
                                    headers = [h.strip() for h in lines[0].strip("|").split("|")]
                                    data = []
                                    for line in lines[2:]:  # Skip separator line
                                        if line.strip():
                                            data.append([cell.strip() for cell in line.strip("|").split("|")])
                                    table_df = pd.DataFrame(data, columns=headers)
                                    
                                    # Convert numeric columns to appropriate types
                                    for col in table_df.columns:
                                        try:
                                            table_df[col] = pd.to_numeric(table_df[col], errors='ignore')
                                        except:
                                            pass
                                            
                                elif "," in table_block:
                                    # CSV data
                                    from io import StringIO
                                    table_df = pd.read_csv(StringIO(table_block))
                                else:
                                    continue
                                
                                # Try to infer a better title from the content before the table
                                table_title = f"Table {idx + 1}"
                                table_context = result.split("```")[idx*2]  # Get text before this table
                                
                                # Look for potential title in the last 3 lines before the table
                                context_lines = table_context.strip().split('\n')[-3:]
                                for line in reversed(context_lines):
                                    line = line.strip()
                                    # Look for title-like text (not too long, not too short)
                                    if 10 <= len(line) <= 100 and not line.startswith('|') and not line.startswith('-'):
                                        table_title = line.strip(':#*')
                                        break
                                
                                # Store table in session state
                                st.session_state.current_tables.append({
                                    'title': table_title,
                                    'data': table_df,
                                    'source': selected_prompt
                                })
                                
                                # Display table instead of visualization
                                st.markdown(f"#### {table_title}")
                                st.dataframe(table_df, use_container_width=True)
                                
                                # The visualization will be shown in the Visualization tab
                            except Exception as e:
                                st.warning(f"Could not process table {idx + 1}: {str(e)}")
                                continue
                                
                    # Add a notification to direct users to the Visualization tab
                    if st.session_state.current_tables:
                        st.success(f"✅ {len(st.session_state.current_tables)} tables extracted and shown above. Switch to the Visualization tab to see these tables as interactive charts.")
                        
                        # Show button to go to Visualization tab
                        if st.button("View as Charts in Visualization Tab"):
                            st.session_state.page = "Visualization"
                            st.rerun()
        else:
            st.warning("Please load or upload data first.")
    
    elif page == "Visualization":
        st.header(f"Chronic Disease Dashboard for {country}")
        if not st.session_state.data.empty:
            # Add descriptive text about the visualization tab
            st.info("This tab shows graphical visualizations of the same data tables found in the Analysis tab. Each analysis result is presented as an interactive chart that best represents the data.")
            
            # Get tables from analysis
            analysis_tables = [] if not hasattr(st.session_state, 'current_tables') else st.session_state.current_tables
            
            if analysis_tables:
                st.subheader("Analysis Visualizations")
                
                with st.expander("Visualization Settings", expanded=False):
                    use_ai_charts = st.checkbox("Use AI to automatically select best chart type", value=True)
                    st.info("When enabled, AI will analyze each table and recommend the best visualization. Disable to manually select chart types.")
                
                # Display each table with visualizations
                for i, table_info in enumerate(analysis_tables):
                    st.markdown(f"### {table_info['title']}")
                    
                    # Visualization section
                    if use_ai_charts:
                        with st.spinner(f"AI is creating the best visualization for {table_info['title']}..."):
                            # Use pandas agent to create intelligent visualization
                            fig = create_pandas_agent_visualization(table_info['data'], table_info['title'])
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                # Fall back to default visualization if AI can't create one
                                st.warning("AI visualization failed. Using default chart type...")
                                fig = create_table_visualization(table_info['data'], table_info['title'])
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("Could not generate any visualization for this data.")
                    else:
                        # Manual chart selection
                        col1, col2 = st.columns([3, 1])
                        with col2:
                            chart_type = st.selectbox(
                                f"Chart Type",
                                ["bar", "line", "scatter", "pie", "heatmap"],
                                key=f"chart_type_{i}"
                            )
                        
                        # Create and display visualization
                        fig = create_table_visualization(table_info['data'], table_info['title'], chart_type)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Could not create visualization with the selected options.")
            else:
                st.info("No analysis tables found. Please go to the Analysis tab and run an analysis first.")
                
                # Show button to go to Analysis tab
                if st.button("Go to Analysis Tab"):
                    st.session_state.page = "Analysis"
                    st.rerun()
            
    elif page == "Chat with AI Doctor":
        st.header(f"Chat with AI Doctor about {country}")
        if not st.session_state.data.empty:
            # Initialize chat history if not exists
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # Chat input
            user_input = st.text_input("Ask a question about the health data:", key="chat_input")
            
            if user_input:
                with st.spinner("Processing..."):
                    response = agents['chat'].medical_chat(user_input, st.session_state.data, country)
                    st.session_state.chat_history.append({"user": user_input, "assistant": response})
            
            # Display chat history in reverse order (newest first)
            if st.session_state.chat_history:
                st.markdown("### Previous Conversations")
                for message in reversed(st.session_state.chat_history):
                    with st.container():
                        st.markdown(f"**Q:** {message['user']}")
                        st.markdown(f"**A:** {message['assistant']}")
                        st.markdown("---")
        else:
            st.warning("Please load or upload data first.")
else:
    st.info("👈 Please enter a country name in the sidebar to begin.") 

# Function to create automatic visualizations for tables
def create_table_visualization(df, title=None, chart_type=None):
    """
    Create appropriate visualization for a table using pandas agent intelligence.
    
    Args:
        df (pd.DataFrame): DataFrame to visualize
        title (str, optional): Title for the visualization
        chart_type (str, optional): Override chart type (bar, line, scatter, pie, etc.)
        
    Returns:
        plotly figure or None if visualization couldn't be created
    """
    try:
        # Detect numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        
        if len(df) == 0 or not numeric_cols:
            return None
            
        # If chart_type is not specified, determine best chart type based on data
        if not chart_type:
            # Simple heuristic for chart type selection:
            # - If many data points with one numeric column -> line chart
            # - If few categories with one numeric column -> bar chart
            # - If two numeric columns -> scatter plot
            # - If percentages/proportions that sum to ~100% -> pie chart
            
            if len(df) > 10 and len(numeric_cols) == 1:
                chart_type = 'line'
            elif len(df) <= 10 and len(numeric_cols) == 1:
                chart_type = 'bar'
            elif len(numeric_cols) >= 2:
                chart_type = 'scatter'
            elif len(df) <= 8 and len(numeric_cols) == 1:
                # Check if values might represent percentages that add to ~100
                values = df[numeric_cols[0]].dropna()
                if 80 <= values.sum() <= 120 and 0 <= values.min() and values.max() <= 100:
                    chart_type = 'pie'
                else:
                    chart_type = 'bar'
            else:
                chart_type = 'bar'  # Default
        
        # Generate plot based on chart type
        if chart_type == 'bar':
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                # Use first categorical and first numeric column
                fig = px.bar(
                    df, 
                    x=categorical_cols[0], 
                    y=numeric_cols[0],
                    title=title or f"Bar Chart of {numeric_cols[0]} by {categorical_cols[0]}",
                    color=categorical_cols[0] if len(df[categorical_cols[0]].unique()) <= 10 else None,
                    height=400
                )
            else:
                # Just use index and first numeric column
                fig = px.bar(
                    df, 
                    y=numeric_cols[0],
                    title=title or f"Bar Chart of {numeric_cols[0]}",
                    height=400
                )
                
        elif chart_type == 'line':
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                # Try to find a date column first
                date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() 
                             or 'year' in col.lower() or df[col].dtype == 'datetime64[ns]']
                
                if date_cols:
                    x_col = date_cols[0]
                else:
                    x_col = categorical_cols[0]
                    
                # Check if multiple lines should be shown (if there's a good category column)
                category_col = None
                good_category_cols = [col for col in categorical_cols 
                                      if col != x_col and 2 <= df[col].nunique() <= 10]
                
                if good_category_cols:
                    category_col = good_category_cols[0]
                    
                fig = px.line(
                    df, 
                    x=x_col, 
                    y=numeric_cols[0],
                    color=category_col,
                    title=title or f"Trend of {numeric_cols[0]} over {x_col}",
                    height=400,
                    markers=True
                )
            else:
                # Just use index and first numeric column
                fig = px.line(
                    df, 
                    y=numeric_cols[0],
                    title=title or f"Trend of {numeric_cols[0]}",
                    height=400,
                    markers=True
                )
        
        elif chart_type == 'scatter':
            if len(numeric_cols) >= 2:
                # Check if there's a good column for color coding (categorical with few unique values)
                color_col = None
                if categorical_cols:
                    good_color_cols = [col for col in categorical_cols if 2 <= df[col].nunique() <= 10]
                    if good_color_cols:
                        color_col = good_color_cols[0]
                
                fig = px.scatter(
                    df, 
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    color=color_col,
                    title=title or f"Relationship between {numeric_cols[0]} and {numeric_cols[1]}",
                    height=400
                )
            else:
                # Default to bar chart if we don't have two numeric columns
                fig = px.bar(
                    df, 
                    y=numeric_cols[0],
                    title=title or f"Bar Chart of {numeric_cols[0]}",
                    height=400
                )
                
        elif chart_type == 'pie':
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                # Use first categorical for names and first numeric for values
                fig = px.pie(
                    df, 
                    names=categorical_cols[0],
                    values=numeric_cols[0],
                    title=title or f"Distribution of {numeric_cols[0]} by {categorical_cols[0]}",
                    height=400
                )
            else:
                # Just use index for names and first numeric column for values
                fig = px.pie(
                    df, 
                    names=df.index,
                    values=numeric_cols[0],
                    title=title or f"Distribution of {numeric_cols[0]}",
                    height=400
                )
        
        else:
            # Default to bar chart
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                fig = px.bar(
                    df, 
                    x=categorical_cols[0], 
                    y=numeric_cols[0],
                    title=title or f"Bar Chart of {numeric_cols[0]} by {categorical_cols[0]}",
                    height=400
                )
            else:
                fig = px.bar(
                    df, 
                    y=numeric_cols[0],
                    title=title or f"Bar Chart of {numeric_cols[0]}",
                    height=400
                )
        
        # Apply common layout improvements
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif"),
            title={
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )
        
        return fig
    
    except Exception as e:
        st.warning(f"Could not create visualization: {str(e)}")
        return None

# Function to use pandas agent to recommend and create visualizations
def create_pandas_agent_visualization(df, title=None):
    """
    Use a pandas agent to intelligently create a visualization for the dataframe.
    
    Args:
        df (pd.DataFrame): DataFrame to visualize
        title (str, optional): Title for the visualization
        
    Returns:
        plotly figure or None if visualization couldn't be created
    """
    try:
        # Skip if DataFrame is empty or too small
        if len(df) < 2 or df.empty:
            return None
        
        # Create a pandas agent to analyze and create the best visualization
        agent = create_pandas_dataframe_agent(
            ChatOpenAI(
                temperature=0, 
                model=MODEL_NAME,
                api_key=OPENAI_API_KEY
            ),
            df,
            verbose=False,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            prefix="""You are a data visualization expert. Your task is to recommend the best 
            visualization type for this dataset and explain why it's appropriate. 
            Consider the data types, number of records, and potential insights."""
        )
        
        # Ask the agent to recommend a visualization
        recommendation = agent.run(f"""
        Analyze this dataframe and recommend the best visualization type.
        Consider:
        1. The types of columns (numerical, categorical, datetime)
        2. The number of records ({len(df)})
        3. The potential patterns or insights to highlight
        
        Return your answer in this format:
        CHART_TYPE: (one of bar, line, scatter, pie, heatmap)
        X_COLUMN: (column name for x-axis or None)
        Y_COLUMN: (column name for y-axis)
        COLOR_BY: (column to use for color differentiation or None)
        TITLE: (suggested title for the chart)
        REASON: (brief explanation of why this visualization is best)
        """)
        
        # Parse the agent's recommendation
        lines = recommendation.strip().split('\n')
        viz_specs = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                viz_specs[key.strip()] = value.strip()
        
        # Extract visualization specifications
        chart_type = viz_specs.get('CHART_TYPE', 'bar').lower()
        x_column = viz_specs.get('X_COLUMN', None)
        if x_column == 'None' or x_column == 'null':
            x_column = None
        
        y_column = viz_specs.get('Y_COLUMN', None)
        if y_column == 'None' or y_column == 'null':
            y_column = df.select_dtypes(include=['number']).columns[0] if not df.select_dtypes(include=['number']).empty else df.columns[0]
        
        color_by = viz_specs.get('COLOR_BY', None)
        if color_by == 'None' or color_by == 'null':
            color_by = None
            
        chart_title = viz_specs.get('TITLE', title or 'Data Visualization')
        
        # Create the visualization based on the recommendation
        if chart_type == 'bar':
            fig = px.bar(
                df, 
                x=x_column, 
                y=y_column,
                color=color_by,
                title=chart_title,
                height=400
            )
        elif chart_type == 'line':
            fig = px.line(
                df, 
                x=x_column, 
                y=y_column,
                color=color_by,
                title=chart_title,
                height=400,
                markers=True
            )
        elif chart_type == 'scatter':
            fig = px.scatter(
                df, 
                x=x_column, 
                y=y_column,
                color=color_by,
                title=chart_title,
                height=400
            )
        elif chart_type == 'pie':
            fig = px.pie(
                df, 
                names=x_column if x_column else df.index,
                values=y_column,
                title=chart_title,
                height=400
            )
        elif chart_type == 'heatmap':
            # For heatmap, we need to pivot the data if it's not already in the right format
            if x_column and y_column and len(df[x_column].unique()) <= 30 and len(df[y_column].unique()) <= 30:
                # If we have a third column for values
                value_col = [col for col in df.select_dtypes(include=['number']).columns if col != x_column and col != y_column]
                if value_col:
                    pivot_df = df.pivot_table(index=y_column, columns=x_column, values=value_col[0], aggfunc='mean')
                    fig = px.imshow(
                        pivot_df,
                        title=chart_title,
                        height=400
                    )
                else:
                    # We'll create a count heatmap
                    pivot_df = pd.crosstab(df[y_column], df[x_column])
                    fig = px.imshow(
                        pivot_df,
                        title=chart_title,
                        height=400
                    )
            else:
                # Fall back to correlation heatmap if the data isn't suitable for a regular heatmap
                numeric_df = df.select_dtypes(include=['number'])
                if len(numeric_df.columns) >= 2:
                    corr_matrix = numeric_df.corr()
                    fig = px.imshow(
                        corr_matrix,
                        title=f"Correlation Heatmap: {chart_title}",
                        height=400
                    )
                else:
                    # Fall back to bar chart
                    fig = px.bar(
                        df, 
                        x=x_column, 
                        y=y_column,
                        color=color_by,
                        title=chart_title,
                        height=400
                    )
        else:
            # Default to bar chart
            fig = px.bar(
                df, 
                x=x_column, 
                y=y_column,
                color=color_by,
                title=chart_title,
                height=400
            )
        
        # Apply common layout improvements
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif"),
            title={
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )
        
        return fig
    
    except Exception as e:
        st.warning(f"Could not create intelligent visualization: {str(e)}")
        # Fall back to basic visualization
        return create_table_visualization(df, title)