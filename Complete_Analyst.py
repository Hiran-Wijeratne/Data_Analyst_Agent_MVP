#!/usr/bin/env python3
"""
Data Analyst Agent - A conversational data analysis tool with LLM integration
Run with: streamlit run data_analyst_agent.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import json
import io
import os
import re
import sys
import logging
import traceback
from typing import Dict, Any, List, Optional, Tuple
import time
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')

# Platform-specific imports
import platform
if platform.system() != 'Windows':
    import signal
else:
    signal = None

# LLM Integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    st.error("OpenAI package not installed. Install with: pip install openai")

# ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    from sklearn.preprocessing import LabelEncoder
    from scipy import stats
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("Scikit-learn not available. ML features will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    """Context manager for timing out code execution - Windows compatible"""
    if signal and hasattr(signal, 'SIGALRM'):
        # Unix/Linux systems
        def signal_handler(signum, frame):
            raise TimeoutException("Code execution timed out")
        
        old_handler = signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows or systems without SIGALRM - just yield without timeout
        yield

class SafeCodeExecutor:
    """Safe execution environment for generated code"""
    
    ALLOWED_IMPORTS = {
        'pandas', 'numpy', 'matplotlib.pyplot', 'plotly.express', 'plotly.graph_objects',
        'sklearn', 'scipy.stats', 'math', 'statistics', 'datetime', 'json', 'io'
    }
    
    FORBIDDEN_PATTERNS = [
        r'import\s+(?!pandas|numpy|matplotlib|plotly|sklearn|scipy|math|statistics|datetime|json|io|sns|seaborn)',
        r'__import__',
        r'eval\s*\(',
        r'exec\s*\(',
        r'compile\s*\(',
        r'open\s*\(',
        r'file\s*\(',
        r'input\s*\(',
        r'raw_input\s*\(',
        r'subprocess',
        r'os\.',
        r'sys\.',
        r'requests',
        r'urllib',
        r'socket',
        r'network',
    ]
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.results = {}
        
    def is_code_safe(self, code: str) -> Tuple[bool, str]:
        """Check if code is safe to execute"""
        print(f"SAFETY CHECK: Starting safety validation")
        
        # Check for malformed import patterns first
        if '**import**' in code:
            print("SAFETY CHECK: Found **import** pattern - BLOCKING")
            return False, "Malformed import pattern detected and blocked"
        
        # Don't block __import__ since we need it for print() to work
        forbidden_patterns_filtered = [p for p in self.FORBIDDEN_PATTERNS if p != r'__import__']
        
        for pattern in forbidden_patterns_filtered:
            if re.search(pattern, code, re.IGNORECASE):
                print(f"SAFETY CHECK: Found forbidden pattern: {pattern}")
                return False, f"Forbidden pattern detected: {pattern}"
        
        print("SAFETY CHECK: All patterns passed - code is safe")
        return True, ""
    
    def execute(self, code: str, timeout_seconds: int = 10) -> Dict[str, Any]:
        """Safely execute code with timeout and restrictions"""
        print("EXECUTION DEBUG: Starting code execution")
        print(f"EXECUTION DEBUG: Input code: {repr(code)}")
        
        # Check for malformed patterns early
        if '**import**' in code:
            print("EXECUTION DEBUG: Found **import** pattern in input code")
            return {"success": False, "error": f"Code contains '**import**' pattern. Raw code:\n{repr(code)}"}
            
        is_safe, error_msg = self.is_code_safe(code)
        if not is_safe:
            print(f"EXECUTION DEBUG: Code safety check failed: {error_msg}")
            return {"success": False, "error": f"Unsafe code: {error_msg}"}
        
        # Prepare safe execution environment with ALL necessary builtins
        safe_globals = {
            '__builtins__': {
                # Core functions
                'len': len, 'range': range, 'enumerate': enumerate,
                'zip': zip, 'map': map, 'filter': filter,
                'sum': sum, 'min': min, 'max': max, 'abs': abs,
                'round': round, 'sorted': sorted, 'list': list,
                'dict': dict, 'tuple': tuple, 'set': set,
                'str': str, 'int': int, 'float': float, 'bool': bool,
                'print': print, 'type': type, 'isinstance': isinstance,
                # Essential for print() and other operations
                '__import__': __import__,
                'repr': repr, 'format': format, 'getattr': getattr,
                'hasattr': hasattr, 'setattr': setattr,
                # Exception handling
                'Exception': Exception, 'ValueError': ValueError,
                'TypeError': TypeError, 'KeyError': KeyError,
                # None and other constants
                'None': None, 'True': True, 'False': False,
            },
            'pd': pd,
            'np': np,
            'plt': plt,
            'px': px,
            'go': go,
            'df': self.df.copy(),
            'json': json,
            'stats': stats if 'stats' in globals() else None,
        }
        
        if SKLEARN_AVAILABLE:
            safe_globals.update({
                'RandomForestClassifier': RandomForestClassifier,
                'RandomForestRegressor': RandomForestRegressor,
                'train_test_split': train_test_split,
                'accuracy_score': accuracy_score,
                'mean_squared_error': mean_squared_error,
                'r2_score': r2_score,
                'LabelEncoder': LabelEncoder,
            })
        
        # Pre-process code to handle common patterns
        processed_code = self._preprocess_code(code)
        
        try:
            print("EXECUTION DEBUG: Starting actual code execution")
            # Capture output
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            with timeout(timeout_seconds):
                exec(processed_code, safe_globals)
            
            output = captured_output.getvalue()
            sys.stdout = old_stdout
            
            print("EXECUTION DEBUG: Code executed successfully")
            
            # Extract variables that might be results
            result_vars = {}
            for key, value in safe_globals.items():
                if key not in ['__builtins__', 'pd', 'np', 'plt', 'px', 'go', 'df', 'json', 'stats'] and not key.startswith('_'):
                    if not callable(value) and not isinstance(value, type):
                        try:
                            json.dumps(value, default=str)
                            result_vars[key] = value
                        except:
                            result_vars[key] = str(value)
            
            return {
                "success": True,
                "output": output,
                "variables": result_vars,
                "plots": self._extract_plots()
            }
            
        except TimeoutException:
            sys.stdout = old_stdout
            print("EXECUTION DEBUG: Code execution timed out")
            return {"success": False, "error": "Code execution timed out"}
        except Exception as e:
            sys.stdout = old_stdout
            error_msg = str(e)
            print(f"EXECUTION DEBUG: Exception during execution: {error_msg}")
            print(f"EXECUTION DEBUG: Exception type: {type(e).__name__}")
            print(f"EXECUTION DEBUG: Full traceback:")
            print(traceback.format_exc())
            
            return {"success": False, "error": f"Execution error: {error_msg}", "traceback": traceback.format_exc()}
    
    def _preprocess_code(self, code: str) -> str:
        """Preprocess code to handle common patterns and imports"""
        print("PREPROCESSING: Starting code preprocessing")
        
        if not code or not code.strip():
            return code
            
        lines = code.split('\n')
        processed_lines = []
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                processed_lines.append(line)
                continue
            
            # Handle import statements - convert to comments
            if stripped.startswith('import ') or stripped.startswith('from '):
                indent = len(line) - len(line.lstrip())
                new_line = ' ' * indent + f"# {stripped} (already available)"
                print(f"PREPROCESSING: Line {line_num}: Converting import to comment")
                processed_lines.append(new_line)
                continue
            
            # Handle plt.show() calls
            if 'plt.show()' in line:
                indent = len(line) - len(line.lstrip())
                new_line = ' ' * indent + "# plt.show() - plot captured automatically"
                processed_lines.append(new_line)
                continue
                
            # Keep other lines as-is
            processed_lines.append(line)
        
        result = '\n'.join(processed_lines)
        print("PREPROCESSING: Preprocessing complete")
        
        return result
    
    def _extract_plots(self) -> List[Dict]:
        """Extract matplotlib plots"""
        plots = []
        figs = plt.get_fignums()
        for fig_num in figs:
            fig = plt.figure(fig_num)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            plots.append({
                'type': 'matplotlib',
                'data': buf.getvalue(),
                'figure_num': fig_num
            })
        return plots

class LLMAnalyst:
    """Julius - The AI Data Analyst"""
    
    def __init__(self, api_key: str):
        if OPENAI_AVAILABLE and api_key:
            openai.api_key = api_key
            self.client = openai.OpenAI(api_key=api_key)
            self.available = True
        else:
            self.available = False
        
        self.conversation_history = []
    
    def analyze_dataset(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        """Main analysis function that generates plan, code, and insights"""
        if not self.available:
            return self._fallback_analysis(df, question)
        
        dataset_info = self._get_dataset_info(df)
        
        # Enhanced prompt that explicitly forbids imports
        prompt = f"""You are Julius, an expert data analyst assistant with a friendly, conversational personality.

Dataset Information:
{dataset_info}

Previous conversation context:
{self._format_conversation_history()}

User Question: {question}

Please provide a response in the following JSON format:
{{
    "plan": "Brief 1-3 line description of what you'll do to answer this question",
    "code": "Python code for data analysis. CRITICAL: Do NOT include ANY import statements - all libraries are pre-loaded. Use 'df' as the dataset variable. Available: pandas (pd), numpy (np), matplotlib.pyplot (plt), plotly.express (px), plotly.graph_objects (go), sklearn, scipy.stats",
    "explanation": "Detailed explanation of what the code will show and what insights can be drawn",
    "interpretation": "Interpretation of what the results mean and actionable insights",
    "suggestions": ["suggestion1", "suggestion2", "suggestion3"]
}}

CRITICAL CODE REQUIREMENTS:
1. NO IMPORT STATEMENTS: Never include import statements - all libraries are already available
2. NO plt.show(): Don't include plt.show() - plots are captured automatically  
3. Use 'df' for the dataset, 'pd' for pandas, 'np' for numpy, 'plt' for matplotlib
4. Include print() statements to show key results

Example correct code:
data_types = df.dtypes.to_dict()
missing_values = df.isnull().sum().to_dict()
print("Data Types:", data_types)
print("Missing Values:", missing_values)
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are Julius, a data analyst. Always respond with valid JSON. Never include import statements in code."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Debug logging
            print("=" * 80)
            print("DEBUG: RAW GPT-4 API RESPONSE")
            print("=" * 80)
            print("RESPONSE:", repr(content))
            print("=" * 80)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_content = json_match.group()
                
                try:
                    result = json.loads(json_content)
                    
                    # Log the extracted code
                    if 'code' in result:
                        print("DEBUG: EXTRACTED CODE:")
                        print(repr(result['code']))
                        
                        # Check for problematic patterns
                        if '**import**' in result['code']:
                            print("FOUND **import** PATTERN IN CODE!")
                        if 'import' in result['code'] and not result['code'].startswith('#'):
                            print("Found import statements in code - this should not happen!")
                    
                    self.conversation_history.append({"question": question, "response": result})
                    return result
                    
                except json.JSONDecodeError as e:
                    print(f"JSON DECODE ERROR: {e}")
                    return self._fallback_analysis(df, question)
            else:
                print("NO JSON FOUND IN RESPONSE!")
                return self._fallback_analysis(df, question)
                
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return self._fallback_analysis(df, question)
    
    def _get_dataset_info(self, df: pd.DataFrame) -> str:
        """Generate dataset summary for context"""
        return f"""
Shape: {df.shape}
Columns: {list(df.columns)}
Data types: {dict(df.dtypes)}
Missing values: {dict(df.isnull().sum())}
Sample data:
{df.head(3).to_string()}
"""
    
    def _format_conversation_history(self) -> str:
        """Format conversation history for context"""
        if not self.conversation_history:
            return "No previous conversation."
        
        history = []
        for i, conv in enumerate(self.conversation_history[-3:]):
            history.append(f"Q{i+1}: {conv['question']}")
            if 'plan' in conv['response']:
                history.append(f"A{i+1}: {conv['response']['plan']}")
        
        return "\n".join(history)
    
    def _fallback_analysis(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        """Fallback analysis when LLM is not available"""
        question_lower = question.lower()
        suggestions = self._generate_suggestions(df)
        
        if any(word in question_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return {
                "plan": "Hello! I'm Julius, your data analyst assistant. Let me show you what we can discover in your dataset!",
                "code": f"""
print("Welcome! I'm your AI data analyst.")
print(f"I can see you have a dataset with {{df.shape[0]}} rows and {{df.shape[1]}} columns.")
print("\\nHere's a quick overview:")
print(df.info())
print("\\nLet's start exploring!")
for i, suggestion in enumerate(suggestions[:4], 1):
    print(f"{{i}}. {{suggestion}}")
""",
                "explanation": "Welcoming the user and providing an overview of available analyses",
                "suggestions": suggestions
            }
        elif 'summary' in question_lower or 'describe' in question_lower:
            return {
                "plan": "I'll generate a comprehensive statistical summary of your dataset, including data types, missing values, and key statistics.",
                "code": """
print("DATASET SUMMARY REPORT")
print("=" * 50)
print(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\\nColumn Information:")
print(df.dtypes)
print("\\nStatistical Summary:")
print(df.describe())
print("\\nMissing Values:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("No missing values found!")
print("\\nData Quality:")
print(f"Duplicate rows: {df.duplicated().sum()}")
""",
                "explanation": "Comprehensive dataset overview with statistics and data quality metrics",
                "suggestions": suggestions
            }
        elif 'correlation' in question_lower:
            return {
                "plan": "I'll calculate correlations between numeric variables and create a heatmap to visualize relationships.",
                "code": """
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 1:
    print("CORRELATION ANALYSIS")
    print("=" * 40)
    corr_matrix = df[numeric_cols].corr()
    print("Correlation Matrix:")
    print(corr_matrix)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    plt.title('Correlation Heatmap', fontsize=16, pad=20)
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45, ha='right')
    plt.yticks(range(len(numeric_cols)), numeric_cols)
    
    # Add correlation values as text
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                    ha='center', va='center', 
                    color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
    
    plt.tight_layout()
    
    # Find strongest correlations
    print("\\nStrongest Correlations:")
    corr_pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.3:
                corr_pairs.append((numeric_cols[i], numeric_cols[j], corr_val))
    
    if corr_pairs:
        for col1, col2, corr_val in sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            strength = "Strong" if abs(corr_val) > 0.7 else "Moderate"
            direction = "positive" if corr_val > 0 else "negative"
            print(f"  • {col1} ↔ {col2}: {corr_val:.3f} ({strength} {direction})")
    else:
        print("  No strong correlations found (all < 0.3)")
else:
    print("Need at least 2 numeric columns for correlation analysis")
""",
                "explanation": "Visualizing relationships between numeric variables with correlation heatmap",
                "suggestions": suggestions
            }
        else:
            return {
                "plan": "Let me provide you with a comprehensive overview of your dataset and analysis options.",
                "code": """
print("DATASET ANALYSIS OVERVIEW")
print("=" * 50)
print(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\\nColumns and Types:")
for col, dtype in df.dtypes.items():
    non_null = df[col].count()
    null_pct = (df[col].isnull().sum() / len(df)) * 100
    unique = df[col].nunique()
    print(f"  • {col}: {dtype} ({non_null} non-null, {unique} unique, {null_pct:.1f}% missing)")

print("\\nSample Data:")
print(df.head(3))

print("\\nWhat I can help you analyze:")
analysis_suggestions = [
    "Show me summary statistics and data quality metrics",
    "Create distribution plots for numeric variables", 
    "Analyze correlations between numeric variables",
    "Show categorical variable distributions",
    "Compare variables across different categories",
    "Build a predictive model",
    "Detect outliers in the data",
    "Perform statistical tests"
]
for idx, suggestion in enumerate(analysis_suggestions, 1):
    print(f"  {idx}. {suggestion}")
""",
                "explanation": "Comprehensive dataset overview with analysis suggestions",
                "suggestions": suggestions
            }
    
    def _generate_suggestions(self, df: pd.DataFrame) -> List[str]:
        """Generate analysis suggestions based on dataset characteristics"""
        suggestions = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        suggestions.append("Show me summary statistics and data quality metrics")
        
        if len(numeric_cols) >= 1:
            suggestions.append(f"Create distribution plots for numeric variables")
            suggestions.append(f"Analyze outliers in {numeric_cols[0]}")
        
        if len(categorical_cols) >= 1:
            suggestions.append(f"Show me the distribution of {categorical_cols[0]}")
        
        if len(numeric_cols) >= 2:
            suggestions.append("Analyze correlations between numeric variables")
            suggestions.append(f"Create scatter plot of {numeric_cols[0]} vs {numeric_cols[1]}")
        
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            suggestions.append(f"Compare {numeric_cols[0]} across {categorical_cols[0]} categories")
        
        if len(numeric_cols) >= 3:
            suggestions.append(f"Build a predictive model for {numeric_cols[0]}")
        
        return suggestions[:8]

def create_sample_dataset() -> pd.DataFrame:
    """Create a sample dataset for demonstration"""
    np.random.seed(42)
    n = 1000
    
    data = {
        'customer_id': range(1, n + 1),
        'age': np.random.randint(18, 80, n),
        'income': np.random.normal(50000, 15000, n),
        'spending_score': np.random.randint(1, 100, n),
        'gender': np.random.choice(['Male', 'Female'], n),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], n),
        'purchase_amount': np.random.exponential(100, n),
        'satisfaction': np.random.randint(1, 6, n),
        'is_premium': np.random.choice([0, 1], n, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    df['income'] = np.clip(df['income'], 20000, 150000)
    df['purchase_amount'] = np.clip(df['purchase_amount'], 10, 1000)
    
    return df

def perform_quick_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """Quick exploratory data analysis"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    results = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': df[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {},
        'categorical_summary': {col: df[col].value_counts().head().to_dict() for col in categorical_cols}
    }
    
    return results

def perform_ml_baseline(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """Perform baseline ML analysis"""
    if not SKLEARN_AVAILABLE:
        return {"error": "Scikit-learn not available"}
    
    if target_col not in df.columns:
        return {"error": f"Target column '{target_col}' not found"}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    if len(numeric_cols) == 0:
        return {"error": "No numeric features available"}
    
    X = df[numeric_cols].fillna(0)
    y = df[target_col].fillna(0)
    
    is_classification = df[target_col].dtype == 'object' or df[target_col].nunique() < 10
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if is_classification:
            if df[target_col].dtype == 'object':
                le = LabelEncoder()
                y_train = le.fit_transform(y_train.astype(str))
                y_test = le.transform(y_test.astype(str))
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            results = {
                "model_type": "Classification",
                "accuracy": accuracy,
                "features_used": numeric_cols,
                "feature_importance": dict(zip(numeric_cols, model.feature_importances_))
            }
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results = {
                "model_type": "Regression",
                "mse": mse,
                "r2_score": r2,
                "features_used": numeric_cols,
                "feature_importance": dict(zip(numeric_cols, model.feature_importances_))
            }
        
        return results
    
    except Exception as e:
        return {"error": f"ML analysis failed: {str(e)}"}

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Data Analyst Agent",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 Data Analyst Agent")
    st.markdown("*Conversational data analysis powered by AI*")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
    if 'analyst' not in st.session_state:
        st.session_state.analyst = None
    
    # Sidebar
    with st.sidebar:
        st.divider()
        
        # File upload
        st.header("📁 Dataset")
        uploaded_file = st.file_uploader(
            "Upload CSV, XLSX, or JSON",
            type=['csv', 'xlsx', 'json'],
            help="Upload your dataset file"
        )
        
        # Sample data option
        if st.button("📋 Load Sample Dataset"):
            st.session_state.current_df = create_sample_dataset()
            st.success("Sample dataset loaded!")
        
        # Process uploaded file
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.current_df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    st.session_state.current_df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    st.session_state.current_df = pd.read_json(uploaded_file)
                
                st.success(f"✅ Loaded {uploaded_file.name}")
                st.info(f"Shape: {st.session_state.current_df.shape}")
                
            except Exception as e:
                st.error(f"Error loading file: {e}")
        
        st.divider()
        
        # Quick actions
        st.header("⚡ Quick Actions")
        if st.session_state.current_df is not None:
            
            if st.button("📊 Dataset Summary"):
                results = perform_quick_eda(st.session_state.current_df)
                st.session_state.chat_history.append({
                    'question': 'Dataset Summary',
                    'plan': 'Generate comprehensive dataset overview',
                    'results': results,
                    'type': 'quick_action'
                })
            
            # ML Baseline
            numeric_cols = st.session_state.current_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                target_col = st.selectbox("🤖 ML Target Column", numeric_cols)
                if st.button("🎯 ML Baseline"):
                    results = perform_ml_baseline(st.session_state.current_df, target_col)
                    st.session_state.chat_history.append({
                        'question': f'ML Baseline Analysis (target: {target_col})',
                        'plan': f'Train RandomForest model to predict {target_col}',
                        'results': results,
                        'type': 'ml_analysis'
                    })
    
    # Main content area
    if st.session_state.current_df is None:
        st.info("👆 Please upload a dataset or load the sample dataset from the sidebar to get started.")
        st.markdown("""
        ### Features:
        - 🤖 **AI-Powered Analysis**: Ask questions in natural language
        - 📊 **Interactive Visualizations**: Automatic chart generation
        - 🔍 **Safe Code Execution**: Sandboxed environment
        - ⚡ **Quick Actions**: One-click EDA, ML, and statistical tests
        """)
        return
    
    # Dataset preview
    with st.expander("📋 Dataset Preview", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", st.session_state.current_df.shape[0])
        with col2:
            st.metric("Columns", st.session_state.current_df.shape[1])
        with col3:
            st.metric("Missing Values", st.session_state.current_df.isnull().sum().sum())
        
        st.dataframe(st.session_state.current_df.head(), width='stretch')
        
        # Column info
        col_info = pd.DataFrame({
            'Column': st.session_state.current_df.columns,
            'Type': st.session_state.current_df.dtypes.astype(str),
            'Non-Null': st.session_state.current_df.count(),
            'Null': st.session_state.current_df.isnull().sum(),
            'Unique': st.session_state.current_df.nunique()
        })
        st.dataframe(col_info, width='stretch')
    
    # Chat interface
    st.header("💬 Chat with your Data Analyst")
    
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**Q{i+1}:** {chat['question']}")
            
            if chat.get('type') == 'analysis':
                if chat.get('plan'):
                    st.markdown(f"**Julius's Plan:** {chat['plan']}")
                
                if chat.get('explanation'):
                    st.markdown("**What This Analysis Will Show:**")
                    st.info(chat['explanation'])
                
                if chat.get('code'):
                    with st.expander("💻 Generated Code", expanded=False):
                        st.code(chat['code'], language='python')
                        
                        if st.button(f"▶️ Run Code", key=f"run_{i}"):
                            executor = SafeCodeExecutor(st.session_state.current_df)
                            result = executor.execute(chat['code'])
                            chat['execution_result'] = result
                            st.rerun()
                
                if 'execution_result' in chat:
                    result = chat['execution_result']
                    if result['success']:
                        st.markdown("**Analysis Results:**")
                        
                        if result['output']:
                            st.code(result['output'])
                        
                        if result.get('plots'):
                            for plot in result['plots']:
                                if plot['type'] == 'matplotlib':
                                    st.image(plot['data'])
                        
                        if chat.get('interpretation'):
                            st.markdown("**💡 What These Results Mean:**")
                            st.success(chat['interpretation'])
                        
                        if result.get('variables'):
                            with st.expander("🔍 Additional Details"):
                                st.json(result['variables'])
                    else:
                        st.error(f"Execution failed: {result['error']}")
                        
                        # Provide direct analysis for data types and missing values
                        if 'data type' in chat['question'].lower() or 'missing' in chat['question'].lower():
                            st.markdown("**📊 Direct Analysis:**")
                            
                            dtypes_df = pd.DataFrame({
                                'Column': st.session_state.current_df.columns,
                                'Data Type': st.session_state.current_df.dtypes.astype(str)
                            })
                            st.dataframe(dtypes_df, width='stretch')
                            
                            missing_data = st.session_state.current_df.isnull().sum()
                            missing_df = pd.DataFrame({
                                'Column': st.session_state.current_df.columns,
                                'Missing Count': missing_data.values,
                                'Missing %': (missing_data.values / len(st.session_state.current_df) * 100).round(1)
                            })
                            st.dataframe(missing_df, width='stretch')
                
                if chat.get('suggestions'):
                    with st.expander("💡 Follow-up suggestions:", expanded=False):
                        suggestions = chat['suggestions'] if isinstance(chat['suggestions'], list) else [chat['suggestions']]
                        for j, suggestion in enumerate(suggestions[:4]):
                            if st.button(f"🔍 {suggestion}", key=f"suggestion_{i}_{j}"):
                                if not st.session_state.analyst:
                                    st.session_state.analyst = LLMAnalyst("")
                                
                                with st.spinner("Julius is analyzing..."):
                                    response = st.session_state.analyst.analyze_dataset(
                                        st.session_state.current_df, 
                                        suggestion
                                    )
                                
                                new_chat_entry = {
                                    'question': suggestion,
                                    'plan': response.get('plan', ''),
                                    'code': response.get('code', ''),
                                    'explanation': response.get('explanation', ''),
                                    'interpretation': response.get('interpretation', ''),
                                    'suggestions': response.get('suggestions', []),
                                    'type': 'analysis'
                                }
                                
                                st.session_state.chat_history.append(new_chat_entry)
                                
                                if response.get('code'):
                                    executor = SafeCodeExecutor(st.session_state.current_df)
                                    result = executor.execute(response['code'])
                                    new_chat_entry['execution_result'] = result
                                
                                st.rerun()
            
            elif chat.get('type') == 'quick_action':
                results = chat.get('results', {})
                if 'shape' in results:
                    st.write(f"**Dataset Shape:** {results['shape']}")
                if 'missing_values' in results:
                    st.write("**Missing Values:**")
                    st.json(results['missing_values'])
                if 'numeric_summary' in results:
                    st.write("**Numeric Summary:**")
                    st.dataframe(pd.DataFrame(results['numeric_summary']))
            
            elif chat.get('type') == 'ml_analysis':
                results = chat.get('results', {})
                if 'error' in results:
                    st.error(results['error'])
                else:
                    st.write(f"**Model Type:** {results.get('model_type', 'Unknown')}")
                    if 'accuracy' in results:
                        st.metric("Accuracy", f"{results['accuracy']:.3f}")
                    if 'r2_score' in results:
                        st.metric("R² Score", f"{results['r2_score']:.3f}")
                    if 'mse' in results:
                        st.metric("MSE", f"{results['mse']:.3f}")
                    
                    if 'feature_importance' in results:
                        st.write("**Feature Importance:**")
                        importance_df = pd.DataFrame(
                            list(results['feature_importance'].items()),
                            columns=['Feature', 'Importance']
                        ).sort_values('Importance', ascending=False)
                        st.dataframe(importance_df)
            
            st.divider()
    
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        user_question = st.text_input(
            "Ask Julius about your data:",
            placeholder="e.g., 'Show me the distribution of age' or 'What are the correlations?'",
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            analyze_clicked = st.form_submit_button("🚀 Analyze")
        with col2:
            clear_clicked = st.form_submit_button("🗑️ Clear Chat")
    
    # Handle form submissions
    if clear_clicked:
        st.session_state.chat_history = []
        st.rerun()
    
    if analyze_clicked and user_question:
        if not st.session_state.analyst:
            st.session_state.analyst = LLMAnalyst("")
        
        with st.spinner("Julius is analyzing your data..."):
            response = st.session_state.analyst.analyze_dataset(
                st.session_state.current_df, 
                user_question
            )
        
        chat_entry = {
            'question': user_question,
            'plan': response.get('plan', ''),
            'code': response.get('code', ''),
            'explanation': response.get('explanation', ''),
            'interpretation': response.get('interpretation', ''),
            'suggestions': response.get('suggestions', []),
            'type': 'analysis'
        }
        
        st.session_state.chat_history.append(chat_entry)
        
        if response.get('code'):
            print("STEP 1: About to execute code from GPT-4 response")
            print(f"STEP 1: Code: {repr(response['code'])}")
            
            executor = SafeCodeExecutor(st.session_state.current_df)
            result = executor.execute(response['code'])
            chat_entry['execution_result'] = result
            
            print("STEP 2: Code execution completed")
            print(f"STEP 2: Success: {result.get('success', False)}")
            if not result.get('success'):
                print(f"STEP 2: Error: {result.get('error', 'No error message')}")
        
        st.rerun()
    
    # Dynamic sample questions
    if not st.session_state.chat_history:
        st.markdown("### 💡 Sample Questions:")
        
        if st.session_state.current_df is not None:
            numeric_cols = st.session_state.current_df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = st.session_state.current_df.select_dtypes(include=['object']).columns.tolist()
            
            sample_questions = [
                "What are the data types and missing values in this dataset?",
                "Show me a summary of this dataset"
            ]
            
            if len(numeric_cols) >= 1:
                sample_questions.append(f"Show me the distribution of {numeric_cols[0]}")
            
            if len(numeric_cols) >= 2:
                sample_questions.append(f"What's the correlation between {numeric_cols[0]} and {numeric_cols[1]}?")
            
            if len(categorical_cols) >= 1:
                sample_questions.append(f"Show me the distribution of {categorical_cols[0]}")
            
            if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                sample_questions.append(f"Compare {numeric_cols[0]} across {categorical_cols[0]} categories")
            
            cols = st.columns(2)
            for i, question in enumerate(sample_questions[:6]):
                with cols[i % 2]:
                    if st.button(question, key=f"sample_{i}"):
                        if not st.session_state.analyst:
                            st.session_state.analyst = LLMAnalyst("")
                        
                        with st.spinner("Julius is analyzing..."):
                            response = st.session_state.analyst.analyze_dataset(
                                st.session_state.current_df, 
                                question
                            )
                        
                        chat_entry = {
                            'question': question,
                            'plan': response.get('plan', ''),
                            'code': response.get('code', ''),
                            'explanation': response.get('explanation', ''),
                            'interpretation': response.get('interpretation', ''),
                            'suggestions': response.get('suggestions', []),
                            'type': 'analysis'
                        }
                        
                        st.session_state.chat_history.append(chat_entry)
                        
                        if response.get('code'):
                            executor = SafeCodeExecutor(st.session_state.current_df)
                            result = executor.execute(response['code'])
                            chat_entry['execution_result'] = result
                        
                        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("**Data Analyst Agent** - Built with Streamlit • Powered by OpenAI GPT-4")

if __name__ == "__main__":
    main()