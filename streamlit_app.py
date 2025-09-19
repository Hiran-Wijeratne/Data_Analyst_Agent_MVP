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

import concurrent.futures




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
class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    """
    Context manager for timing out code execution.
    Works cross-platform (Windows, Linux, macOS) and inside Streamlit.
    """
    # Executor for running code in a separate thread
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    
    # Wrapper that will run the code inside the 'with' block
    future = executor.submit(lambda: None)  # placeholder
    
    try:
        yield future  # The user can call future.result(timeout=seconds) to run code
    finally:
        executor.shutdown(wait=False)







class SafeCodeExecutor:
    """Safe execution environment for generated code"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.results = {}
        
    def is_code_safe(self, code: str) -> Tuple[bool, str]:
        """Check if code is safe to execute"""
        if '**import**' in code:
            return False, "Malformed import pattern detected"
        
        forbidden_patterns = [
            r'eval\s*\(', r'exec\s*\(', r'compile\s*\(',
            r'open\s*\(', r'file\s*\(', r'subprocess',
            r'os\.', r'sys\.', r'requests', r'urllib'
        ]
        
        for pattern in forbidden_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Forbidden pattern detected: {pattern}"
        
        return True, ""
    
    def execute(self, code: str, timeout_seconds: int = 10) -> Dict[str, Any]:
        """Safely execute code with timeout and restrictions"""
        print(f"EXECUTION DEBUG: Raw input code:")
        print(f"'{code}'")
        print(f"Code repr: {repr(code)}")
        
        if '**import**' in code:
            return {"success": False, "error": f"Code contains '**import**' pattern"}
        
        # Check for empty or invalid code
        if not code or not code.strip():
            return {"success": False, "error": "Empty code provided"}
            
        is_safe, error_msg = self.is_code_safe(code)
        if not is_safe:
            return {"success": False, "error": f"Unsafe code: {error_msg}"}
        
        # Prepare safe execution environment
        safe_globals = {
            '__builtins__': {
                'len': len, 'range': range, 'enumerate': enumerate,
                'zip': zip, 'map': map, 'filter': filter,
                'sum': sum, 'min': min, 'max': max, 'abs': abs,
                'round': round, 'sorted': sorted, 'list': list,
                'dict': dict, 'tuple': tuple, 'set': set,
                'str': str, 'int': int, 'float': float, 'bool': bool,
                'print': print, 'type': type, 'isinstance': isinstance,
                '__import__': __import__, 'repr': repr, 'format': format,
                'getattr': getattr, 'hasattr': hasattr, 'setattr': setattr,
                'Exception': Exception, 'ValueError': ValueError,
                'TypeError': TypeError, 'KeyError': KeyError,
                'None': None, 'True': True, 'False': False,
            },
            'pd': pd, 'np': np, 'plt': plt, 'px': px, 'go': go,
            'df': self.df.copy(), 'json': json,
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
        
        # Add statistical functions from scipy.stats if available
        if SKLEARN_AVAILABLE and 'stats' in globals():
            safe_globals.update({
                'ttest_ind': stats.ttest_ind,
                'ttest_1samp': stats.ttest_1samp,
                'chi2_contingency': stats.chi2_contingency,
                'pearsonr': stats.pearsonr,
                'spearmanr': stats.spearmanr,
                'f_oneway': stats.f_oneway,
            })
        
        # Ensure stats module is available for t-tests
        if 'stats' in globals() and stats is not None:
            safe_globals['stats'] = stats
        
        processed_code = self._preprocess_code(code)
        
        print(f"EXECUTION DEBUG: Processed code:")
        print(f"'{processed_code}'")
        print(f"Processed repr: {repr(processed_code)}")
        
        # Check if processed code is valid Python syntax
        try:
            compile(processed_code, '<string>', 'exec')
        except SyntaxError as e:
            return {
                "success": False, 
                "error": f"Syntax error in generated code: {str(e)}\nLine {e.lineno}: {e.text}",
                "original_code": code,
                "processed_code": processed_code
            }
        
        try:
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            with timeout(timeout_seconds):
                exec(processed_code, safe_globals)
            
            output = captured_output.getvalue()
            sys.stdout = old_stdout
            
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
            return {"success": False, "error": "Code execution timed out"}
        except Exception as e:
            sys.stdout = old_stdout
            return {"success": False, "error": f"Execution error: {str(e)}", "traceback": traceback.format_exc()}
    
    def _preprocess_code(self, code: str) -> str:
        """Preprocess code to handle common patterns"""
        if not code or not code.strip():
            return code
        
        # Remove markdown code block formatting if present
        if code.strip().startswith('```'):
            print("PREPROCESSING: Removing markdown code blocks")
            lines = code.split('\n')
            # Remove first line if it's ```python or ```
            if lines[0].strip().startswith('```'):
                lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            code = '\n'.join(lines)
            print(f"PREPROCESSING: Code after markdown removal: {repr(code)}")
            
        lines = code.split('\n')
        processed_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            if not stripped or stripped.startswith('#'):
                processed_lines.append(line)
                continue
            
            if stripped.startswith('import ') or stripped.startswith('from '):
                indent = len(line) - len(line.lstrip())
                new_line = ' ' * indent + f"# {stripped} (already available)"
                processed_lines.append(new_line)
                continue
            
            if 'plt.show()' in line:
                indent = len(line) - len(line.lstrip())
                new_line = ' ' * indent + "# plt.show() - plot captured automatically"
                processed_lines.append(new_line)
                continue
            
            # Fix Period plotting issues
            if 'plt.plot(' in line and '.index,' in line:
                line = line.replace('.index,', '.index.astype(str),')
            elif 'plt.plot(' in line and 'monthly_' in line and '.index' in line:
                line = line.replace('.index', '.index.astype(str)')
                
            processed_lines.append(line)
        
        result = '\n'.join(processed_lines)
        print(f"PREPROCESSING: Final processed code: {repr(result)}")
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
            # Don't store the API key directly to avoid exposure
            self.client = openai.OpenAI(api_key=api_key)
            self.available = True
            # Test the connection without storing the key
            try:
                # Quick test to validate the key works
                self.client.models.list()
            except Exception as e:
                logger.error("API key validation failed")
                self.available = False
        else:
            self.available = False
        
        self.conversation_history = []
    
    def analyze_dataset(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        """Main analysis function that generates plan, code, and insights"""
        if not self.available:
            return self._fallback_analysis(df, question)
        
        dataset_info = self._get_dataset_info(df)
        
        prompt = f"""You are Julius, an expert data analyst who specializes in creating insightful visualizations.

Dataset Information:
{dataset_info}

Previous conversation context:
{self._format_conversation_history()}

User Question: {question}

Please provide a response in the following JSON format:
{{
    "plan": "Brief description of what visualization/analysis you'll create",
    "code": "Raw Python code WITHOUT markdown formatting. Do NOT wrap in ```python or ``` blocks. Just plain Python code.",
    "explanation": "Detailed explanation of what the visualization shows and statistical insights",
    "interpretation": "Interpretation of patterns, trends, and actionable insights from the visualization",
    "suggestions": ["suggestion1", "suggestion2", "suggestion3"]
}}

CRITICAL CODE FORMAT REQUIREMENTS:
- NO MARKDOWN CODE BLOCKS: Do NOT use ```python or ``` - just raw Python code
- NO IMPORT STATEMENTS: All libraries are already loaded
- Raw Python code only in the "code" field

AVAILABLE LIBRARIES ONLY (DO NOT USE ANY OTHERS):
- pandas (as pd) - for data manipulation
- numpy (as np) - for numerical operations  
- matplotlib.pyplot (as plt) - for plotting
- plotly.express (as px) - for interactive plots
- plotly.graph_objects (as go) - for advanced plotly plots
- scipy.stats (as stats) - for statistical functions like ttest_ind, pearsonr

FORBIDDEN LIBRARIES (DO NOT USE):
- seaborn, sns - NOT AVAILABLE
- itertools - NOT AVAILABLE  
- combinations - NOT AVAILABLE
- missingno - NOT AVAILABLE
- Any other libraries not listed above

VISUALIZATION REQUIREMENTS:
1. ALWAYS CREATE PLOTS: Generate matplotlib or plotly visualizations
2. USE ONLY AVAILABLE LIBRARIES: Only use pandas, numpy, matplotlib.pyplot, plotly, scipy.stats
3. STATISTICAL ANALYSIS: Include correlations, t-tests when appropriate
4. MULTIPLE PLOTS: Create subplots when analyzing multiple variables

CODE REQUIREMENTS:
- NO IMPORT STATEMENTS: All libraries are already loaded
- NO plt.show(): Plots are captured automatically
- For correlation heatmaps: Use plt.imshow() with matplotlib, NOT seaborn
- For multiple plots: Use plt.subplot() or plt.subplots()
- Period objects: Use .astype(str) for plotting dates

CORRECT CORRELATION HEATMAP EXAMPLE:
```python
# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()

# Create heatmap using matplotlib (NOT seaborn)
plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
plt.title('Correlation Heatmap')
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)

# Add correlation values as text
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        plt.text(j, i, f'{{corr_matrix.iloc[i, j]:.2f}}', 
                ha='center', va='center')
```

CORRECT SCATTER PLOTS EXAMPLE:
```python
# Create scatter plots for numeric column pairs
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) >= 2:
    plt.figure(figsize=(15, 5))
    plot_num = 1
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            if plot_num <= 3:  # Limit to 3 plots
                plt.subplot(1, 3, plot_num)
                plt.scatter(df[numeric_cols[i]], df[numeric_cols[j]], alpha=0.6)
                plt.xlabel(numeric_cols[i])
                plt.ylabel(numeric_cols[j])
                plt.title(f'{{numeric_cols[i]}} vs {{numeric_cols[j]}}')
                
                # Add correlation
                corr_val = df[numeric_cols[i]].corr(df[numeric_cols[j]])
                plt.text(0.05, 0.95, f'r = {{corr_val:.3f}}', 
                        transform=plt.gca().transAxes)
                plot_num += 1
```

CORRECT T-TEST EXAMPLE:
```python
# T-test analysis with visualizations
group_col = 'Status'  # example categorical column
value_col = 'Resolution_Time (hrs)'  # example numeric column

# Get groups
groups = df[group_col].unique()[:2]  # Take first 2 groups
group1_data = df[df[group_col] == groups[0]][value_col].dropna()
group2_data = df[df[group_col] == groups[1]][value_col].dropna()

# Perform t-test using stats.ttest_ind
t_stat, p_value = stats.ttest_ind(group1_data, group2_data)

# Create visualization
plt.figure(figsize=(12, 8))

# Box plot comparison
plt.subplot(2, 2, 1)
plt.boxplot([group1_data, group2_data], labels=[groups[0], groups[1]])
plt.title(f'{{value_col}} by {{group_col}}')
plt.ylabel(value_col)

# Histogram comparison  
plt.subplot(2, 2, 2)
plt.hist(group1_data, alpha=0.7, label=f'{{groups[0]}} (n={{len(group1_data)}})', bins=20)
plt.hist(group2_data, alpha=0.7, label=f'{{groups[1]}} (n={{len(group2_data)}})', bins=20)
plt.title('Distribution Comparison')
plt.legend()

plt.tight_layout()

# Print results
print(f"T-test Results:")
print(f"Group 1 ({{groups[0]}}): Mean={{group1_data.mean():.3f}}, N={{len(group1_data)}}")
print(f"Group 2 ({{groups[1]}}): Mean={{group2_data.mean():.3f}}, N={{len(group2_data)}}")
print(f"T-statistic: {{t_stat:.4f}}")
print(f"P-value: {{p_value:.4f}}")
print(f"Significant: {{'Yes' if p_value < 0.05 else 'No'}} (α=0.05)")
```
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are Julius, a data analyst. Always respond with valid JSON. Prioritize visualizations over summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_content = json_match.group()
                try:
                    result = json.loads(json_content)
                    self.conversation_history.append({"question": question, "response": result})
                    return result
                except json.JSONDecodeError:
                    return self._fallback_analysis(df, question)
            else:
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
        
        if 'summary' in question_lower or 'describe' in question_lower:
            return {
                "plan": "I'll create comprehensive visualizations showing data distributions, missing values patterns, and statistical summaries.",
                "code": """
# Create comprehensive data visualization dashboard
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Dataset Summary Dashboard', fontsize=16, y=0.98)

# Get column types
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Plot 1: Missing values visualization
ax1 = axes[0, 0]
missing_data = df.isnull().sum()
missing_cols = missing_data[missing_data > 0]
if len(missing_cols) > 0:
    ax1.bar(range(len(missing_cols)), missing_cols.values, color='coral')
    ax1.set_title('Missing Values by Column')
    ax1.set_xticks(range(len(missing_cols)))
    ax1.set_xticklabels(missing_cols.index, rotation=45, ha='right')
else:
    ax1.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_title('Missing Values Status')

# Plot 2: Data types distribution
ax2 = axes[0, 1]
dtype_counts = df.dtypes.value_counts()
ax2.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', startangle=90)
ax2.set_title('Data Types Distribution')

# Plot 3: Numeric distribution (first numeric column)
ax3 = axes[0, 2]
if len(numeric_cols) > 0:
    first_numeric = numeric_cols[0]
    ax3.hist(df[first_numeric].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_title(f'Distribution of {first_numeric}')
    ax3.grid(True, alpha=0.3)
else:
    ax3.text(0.5, 0.5, 'No Numeric Columns', ha='center', va='center', transform=ax3.transAxes)

# Plot 4: Categorical distribution (first categorical column)
ax4 = axes[1, 0]
if len(categorical_cols) > 0:
    first_categorical = categorical_cols[0]
    value_counts = df[first_categorical].value_counts().head(10)
    ax4.bar(range(len(value_counts)), value_counts.values, color='lightcoral')
    ax4.set_title(f'Top Categories in {first_categorical}')
    ax4.set_xticks(range(len(value_counts)))
    ax4.set_xticklabels(value_counts.index, rotation=45, ha='right')
else:
    ax4.text(0.5, 0.5, 'No Categorical Columns', ha='center', va='center', transform=ax4.transAxes)

# Plot 5: Correlation heatmap
ax5 = axes[1, 1]
if len(numeric_cols) >= 2:
    corr_matrix = df[numeric_cols[:5]].corr()
    im = ax5.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax5.set_title('Correlation Heatmap')
    ax5.set_xticks(range(len(corr_matrix.columns)))
    ax5.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax5.set_yticks(range(len(corr_matrix.columns)))
    ax5.set_yticklabels(corr_matrix.columns)
else:
    ax5.text(0.5, 0.5, 'Need 2+ Numeric\\nColumns', ha='center', va='center', transform=ax5.transAxes)

# Plot 6: Dataset metrics
ax6 = axes[1, 2]
info_data = ['Rows', 'Columns', 'Numeric', 'Text', 'Missing']
info_values = [df.shape[0], df.shape[1], len(numeric_cols), len(categorical_cols), df.isnull().sum().sum()]
bars = ax6.bar(info_data, info_values, color=['skyblue', 'lightgreen', 'gold', 'lightcoral', 'orange'])
ax6.set_title('Dataset Metrics')
for bar, value in zip(bars, info_values):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(info_values)*0.01, 
             str(value), ha='center', va='bottom')

plt.tight_layout()

# Print key statistics
print("📊 DATASET SUMMARY STATISTICS")
print("=" * 50)
print(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Numeric columns: {len(numeric_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")
print(f"Total missing values: {df.isnull().sum().sum()}")
print(f"Data completeness: {((df.size - df.isnull().sum().sum()) / df.size * 100):.1f}%")
""",
                "explanation": "Creating a comprehensive visualization dashboard with missing values, distributions, correlations, and statistical summaries",
                "suggestions": suggestions
            }
        elif 'monthly' in question_lower and ('complaint' in question_lower or 'report' in question_lower):
            return {
                "plan": "I'll analyze monthly complaint patterns by converting dates and creating a time series plot.",
                "code": """
# Convert date column to datetime
df['Date_Reported'] = pd.to_datetime(df['Date_Reported'])

# Extract month-year and count complaints
df['Month_Year'] = df['Date_Reported'].dt.to_period('M')
monthly_complaints = df.groupby('Month_Year').size()

print("MONTHLY COMPLAINTS ANALYSIS")
print("=" * 40)

# Create comprehensive time series visualization
plt.figure(figsize=(15, 10))

# Plot 1: Line plot
plt.subplot(2, 2, 1)
months_str = monthly_complaints.index.astype(str)
plt.plot(range(len(monthly_complaints)), monthly_complaints.values, marker='o', linewidth=2, markersize=6)
plt.title('Monthly Distribution of Complaints', fontsize=14)
plt.xlabel('Time Period')
plt.ylabel('Number of Complaints')
plt.xticks(range(len(monthly_complaints)), months_str, rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# Plot 2: Bar chart
plt.subplot(2, 2, 2)
plt.bar(range(len(monthly_complaints)), monthly_complaints.values, color='skyblue', alpha=0.7)
plt.title('Monthly Complaints (Bar Chart)')
plt.xlabel('Time Period')
plt.ylabel('Number of Complaints')
plt.xticks(range(len(monthly_complaints)), months_str, rotation=45, ha='right')

# Plot 3: Statistics box
plt.subplot(2, 2, 3)
plt.axis('off')
stats_text = f'''
Monthly Statistics:
Average: {monthly_complaints.mean():.1f}
Median: {monthly_complaints.median():.1f}
Highest: {monthly_complaints.max()} ({monthly_complaints.idxmax()})
Lowest: {monthly_complaints.min()} ({monthly_complaints.idxmin()})
Total: {monthly_complaints.sum()}
Std Dev: {monthly_complaints.std():.1f}
'''
plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center', fontfamily='monospace')

# Plot 4: Moving average
plt.subplot(2, 2, 4)
if len(monthly_complaints) >= 3:
    moving_avg = monthly_complaints.rolling(window=3, center=True).mean()
    plt.plot(range(len(monthly_complaints)), monthly_complaints.values, 'o-', label='Monthly', alpha=0.7)
    plt.plot(range(len(moving_avg)), moving_avg.values, 's-', label='3-Month Avg', linewidth=2)
    plt.title('Trend Analysis')
    plt.xlabel('Time Period')
    plt.ylabel('Number of Complaints')
    plt.legend()
    plt.xticks(range(len(monthly_complaints)), months_str, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)

plt.tight_layout()

print(f"Monthly complaint counts:")
for month, count in monthly_complaints.items():
    print(f"  {month}: {count} complaints")
""",
                "explanation": "Creating a comprehensive time series analysis of monthly complaints with proper Period object handling",
                "suggestions": suggestions
            }
        else:
            return {
                "plan": "Let me create visualizations to give you a comprehensive overview of your dataset.",
                "code": """
print("CREATING COMPREHENSIVE DATA VISUALIZATIONS")
print("=" * 50)

# Get column types
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Create main visualization
plt.figure(figsize=(16, 12))

# Plot 1: First numeric column distribution
if len(numeric_cols) > 0:
    plt.subplot(2, 3, 1)
    plt.hist(df[numeric_cols[0]].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {numeric_cols[0]}')
    plt.xlabel(numeric_cols[0])
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

# Plot 2: First categorical column
if len(categorical_cols) > 0:
    plt.subplot(2, 3, 2)
    value_counts = df[categorical_cols[0]].value_counts().head(10)
    plt.bar(range(len(value_counts)), value_counts.values, color='lightcoral')
    plt.title(f'Distribution of {categorical_cols[0]}')
    plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')

# Plot 3: Correlation heatmap
if len(numeric_cols) >= 2:
    plt.subplot(2, 3, 3)
    corr_matrix = df[numeric_cols[:5]].corr()
    plt.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.title('Correlation Heatmap')
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)

# Plot 4: Missing values
plt.subplot(2, 3, 4)
missing_data = df.isnull().sum()
missing_cols = missing_data[missing_data > 0]
if len(missing_cols) > 0:
    plt.bar(range(len(missing_cols)), missing_cols.values, color='orange')
    plt.title('Missing Values by Column')
    plt.xticks(range(len(missing_cols)), missing_cols.index, rotation=45, ha='right')
else:
    plt.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Missing Values Status')

# Plot 5: Box plot (first numeric)
if len(numeric_cols) > 0:
    plt.subplot(2, 3, 5)
    plt.boxplot(df[numeric_cols[0]].dropna())
    plt.title(f'Box Plot of {numeric_cols[0]}')
    plt.ylabel(numeric_cols[0])

# Plot 6: Scatter plot (if 2+ numeric columns)
if len(numeric_cols) >= 2:
    plt.subplot(2, 3, 6)
    plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.6)
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])
    plt.title(f'{numeric_cols[0]} vs {numeric_cols[1]}')
    
    # Add correlation
    corr_val = df[numeric_cols[0]].corr(df[numeric_cols[1]])
    plt.text(0.05, 0.95, f'r = {corr_val:.3f}', transform=plt.gca().transAxes,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()

print(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Numeric columns: {len(numeric_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")
""",
                "explanation": "Creating multiple visualizations to explore different aspects of your dataset",
                "suggestions": suggestions
            }
    
    def _generate_suggestions(self, df: pd.DataFrame) -> List[str]:
        """Generate analysis suggestions based on dataset characteristics"""
        suggestions = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        suggestions.append("Create a comprehensive dashboard with multiple visualizations")
        
        if len(numeric_cols) >= 1:
            suggestions.append(f"Create distribution plots and box plots for {numeric_cols[0]}")
        
        if len(numeric_cols) >= 2:
            suggestions.append("Show me correlation heatmap and scatter plots")
        
        if len(categorical_cols) >= 1:
            suggestions.append(f"Create bar charts and pie charts for {categorical_cols[0]}")
        
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            suggestions.append(f"Compare {numeric_cols[0]} across {categorical_cols[0]} categories with box plots")
        
        return suggestions[:6]

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

def perform_ttest(df: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Any]:
    """Perform t-test analysis with visualization"""
    if group_col not in df.columns or value_col not in df.columns:
        return {"error": "Specified columns not found"}
    
    groups = df[group_col].unique()
    if len(groups) != 2:
        return {"error": "T-test requires exactly 2 groups"}
    
    try:
        group1_data = df[df[group_col] == groups[0]][value_col].dropna()
        group2_data = df[df[group_col] == groups[1]][value_col].dropna()
        
        t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Box plot comparison
        plt.subplot(2, 2, 1)
        data_to_plot = [group1_data, group2_data]
        plt.boxplot(data_to_plot, labels=[groups[0], groups[1]])
        plt.title(f'{value_col} by {group_col}')
        plt.ylabel(value_col)
        
        # Subplot 2: Histogram comparison
        plt.subplot(2, 2, 2)
        plt.hist(group1_data, alpha=0.7, label=f'{groups[0]} (n={len(group1_data)})', bins=20)
        plt.hist(group2_data, alpha=0.7, label=f'{groups[1]} (n={len(group2_data)})', bins=20)
        plt.title('Distribution Comparison')
        plt.xlabel(value_col)
        plt.ylabel('Frequency')
        plt.legend()
        
        # Subplot 3: Statistical summary
        plt.subplot(2, 2, (3, 4))
        plt.axis('off')
        
        effect_size = abs(group1_data.mean() - group2_data.mean()) / np.sqrt(((len(group1_data)-1)*group1_data.var() + (len(group2_data)-1)*group2_data.var()) / (len(group1_data) + len(group2_data) - 2))
        
        results_text = f"""
T-Test Results
{'='*20}
Group 1 ({groups[0]}):
  Mean: {group1_data.mean():.3f}
  Std:  {group1_data.std():.3f}
  N:    {len(group1_data)}

Group 2 ({groups[1]}):
  Mean: {group2_data.mean():.3f}
  Std:  {group2_data.std():.3f}
  N:    {len(group2_data)}

Test Statistics:
  T-statistic: {t_stat:.4f}
  P-value:     {p_value:.4f}
  Significant: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)
  Effect Size: {effect_size:.3f}
        """
        
        plt.text(0.1, 0.5, results_text, fontsize=12, verticalalignment='center', 
                fontfamily='monospace', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        
        results = {
            "group1": groups[0],
            "group1_mean": group1_data.mean(),
            "group1_std": group1_data.std(),
            "group1_count": len(group1_data),
            "group2": groups[1],
            "group2_mean": group2_data.mean(),
            "group2_std": group2_data.std(),
            "group2_count": len(group2_data),
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "effect_size": effect_size
        }
        
        return results
    
    except Exception as e:
        return {"error": f"T-test failed: {str(e)}"}

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
        st.header("⚙️ Configuration")
        
        st.markdown("""
        **🔒 API Key Security:**
        - Your API key is kept secure in the session
        - Not logged or stored permanently
        - Use environment variable OPENAI_API_KEY for added security
        """)
        
        # API Key input with better security
        api_key = st.secrets.get("OPENAI_API_KEY", "")
        
        # Check for environment variable if no key entered
        if not api_key:
            env_key = os.getenv("OPENAI_API_KEY", "")
            if env_key:
                api_key = env_key
                st.info("✅ Using API key from environment variable")         
        
        if api_key:
            if not st.session_state.analyst or not getattr(st.session_state.analyst, 'available', False):
                st.session_state.analyst = LLMAnalyst(api_key)
                if st.session_state.analyst.available:
                    st.success("✅ LLM Connected")
                else:
                    st.error("❌ LLM Connection Failed - Check your API key")
        elif not api_key:
            st.warning("⚠️ No API key - using fallback mode")
            st.markdown("**To use AI features:**")
            st.markdown("1. Get OpenAI API key from https://platform.openai.com")
            st.markdown("2. Enter it above, OR")
            st.markdown("3. Set environment variable: `OPENAI_API_KEY=your-key`")
        
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
            
            # T-test interface
            numeric_cols = st.session_state.current_df.select_dtypes(include=[np.number]).columns
            categorical_cols = st.session_state.current_df.select_dtypes(include=['object']).columns
            
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                st.subheader("🧪 Statistical T-Test")
                col1, col2 = st.columns(2)
                with col1:
                    group_col = st.selectbox("📊 Group Column", categorical_cols)
                with col2:
                    value_col = st.selectbox("📈 Value Column", numeric_cols)
                
                if st.button("🧪 Perform T-Test"):
                    results = perform_ttest(st.session_state.current_df, group_col, value_col)
                    st.session_state.chat_history.append({
                        'question': f'T-test: {group_col} vs {value_col}',
                        'plan': f'Compare means of {value_col} between {group_col} groups',
                        'results': results,
                        'type': 't_test'
                    })
                    st.rerun()
            
            # ML Baseline
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
        - 📊 **Interactive Visualizations**: Automatic chart generation with smart chart type selection
        - 🔍 **Statistical Testing**: T-tests, correlations, ANOVA, and more
        - ⚡ **Advanced Visualizations**: Correlation matrices, distribution analysis, time series
        - 🎨 **Plot Analysis**: AI-powered interpretation of your visualizations
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
    st.header("💬 Chat with Julius")
    
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
                        if result['error']:
                            st.error(f"Execution failed: {result['error']}")
                            
                            # Show debug information if syntax error
                            if 'Syntax error' in result['error']:
                                with st.expander("Debug Information", expanded=True):
                                    st.write("**Original Code:**")
                                    st.code(result.get('original_code', 'Not available'))
                                    st.write("**Processed Code:**")
                                    st.code(result.get('processed_code', 'Not available'))
                            
                            # Provide direct analysis as fallback
                            if 'correlation' in chat['question'].lower() or 'scatter' in chat['question'].lower():
                                st.markdown("**📊 Fallback Correlation Analysis:**")
                                
                                numeric_cols = st.session_state.current_df.select_dtypes(include=[np.number]).columns
                                if len(numeric_cols) >= 2:
                                    # Create correlation matrix
                                    corr_matrix = st.session_state.current_df[numeric_cols].corr()
                                    
                                    # Display correlation heatmap using plotly
                                    fig = px.imshow(corr_matrix, 
                                                   text_auto=True, 
                                                   aspect="auto",
                                                   color_continuous_scale='RdBu_r',
                                                   title="Correlation Heatmap")
                                    st.plotly_chart(fig)
                                    
                                    # Create scatter plots
                                    if len(numeric_cols) >= 2:
                                        col1, col2 = numeric_cols[0], numeric_cols[1]
                                        fig2 = px.scatter(st.session_state.current_df, 
                                                         x=col1, y=col2,
                                                         title=f"{col1} vs {col2}")
                                        st.plotly_chart(fig2)
                                        
                                        # Show correlation value
                                        corr_val = st.session_state.current_df[col1].corr(st.session_state.current_df[col2])
                                        st.metric("Correlation Coefficient", f"{corr_val:.3f}")
                                else:
                                    st.info("Need at least 2 numeric columns for correlation analysis")
                            
                            elif 'data type' in chat['question'].lower() or 'missing' in chat['question'].lower():
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
            
            elif chat.get('type') == 't_test':
                results = chat.get('results', {})
                if 'error' in results:
                    st.error(results['error'])
                else:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**{results['group1']}**")
                        st.metric("Mean", f"{results['group1_mean']:.3f}")
                        st.metric("Std", f"{results['group1_std']:.3f}")
                        st.metric("Count", results['group1_count'])
                    
                    with col2:
                        st.write(f"**{results['group2']}**")
                        st.metric("Mean", f"{results['group2_mean']:.3f}")
                        st.metric("Std", f"{results['group2_std']:.3f}")
                        st.metric("Count", results['group2_count'])
                    
                    st.write("**Statistical Test Results:**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("T-Statistic", f"{results['t_statistic']:.4f}")
                    with col2:
                        st.metric("P-Value", f"{results['p_value']:.4f}")
                    with col3:
                        significance = "Yes" if results['significant'] else "No"
                        st.metric("Significant (α=0.05)", significance)
                    with col4:
                        st.metric("Effect Size", f"{results.get('effect_size', 0):.3f}")
            
            st.divider()
    
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        user_question = st.text_input(
            "Ask Julius about your data:",
            placeholder="e.g., 'Create a correlation heatmap', 'Compare groups with t-test', 'Show me time series trends'",
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
            executor = SafeCodeExecutor(st.session_state.current_df)
            result = executor.execute(response['code'])
            chat_entry['execution_result'] = result
        
        st.rerun()
    
    # Dynamic sample questions - VISUALIZATION FOCUSED
    if not st.session_state.chat_history:
        st.markdown("### 💡 Sample Questions (Visualization Focused):")
        
        if st.session_state.current_df is not None:
            numeric_cols = st.session_state.current_df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = st.session_state.current_df.select_dtypes(include=['object']).columns.tolist()
            
            sample_questions = [
                "Create a comprehensive dashboard with multiple visualizations",
                "Show me correlation heatmap and scatter plots"
            ]
            
            if len(numeric_cols) >= 1:
                sample_questions.extend([
                    f"Create distribution plots and box plots for {numeric_cols[0]}",
                    f"Show me outlier analysis with visualizations for {numeric_cols[0]}"
                ])
            
            if len(numeric_cols) >= 2:
                sample_questions.append(f"Create scatter plot matrix for all numeric variables")
            
            if len(categorical_cols) >= 1:
                sample_questions.extend([
                    f"Create bar charts and pie charts for {categorical_cols[0]}",
                    f"Show me cross-tabulation visualization for categorical variables"
                ])
            
            if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                sample_questions.extend([
                    f"Create box plots comparing {numeric_cols[0]} across {categorical_cols[0]}",
                    f"Perform t-test analysis with visualizations for {categorical_cols[0]} and {numeric_cols[0]}"
                ])
            
            # Check for date columns
            date_cols = [col for col in st.session_state.current_df.columns 
                        if any(keyword in col.lower() for keyword in ['date', 'time', 'month', 'year'])]
            if date_cols:
                sample_questions.append(f"Create time series analysis and trend plots")
            
            cols = st.columns(2)
            for i, question in enumerate(sample_questions[:8]):
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
    st.markdown("**Data Analyst Agent** - Built with Streamlit • Powered by OpenAI GPT-4 • Advanced Statistical Analysis & Visualizations")

if __name__ == "__main__":
    main()