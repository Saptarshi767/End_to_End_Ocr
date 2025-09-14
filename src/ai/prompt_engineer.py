"""
Prompt engineering component for data analysis tasks.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .nlp_processor import ParsedIntent, QuestionType, AggregationType
from ..core.models import DataSchema, ColumnInfo, DataType


class PromptTemplate(Enum):
    """Available prompt templates."""
    QUERY_GENERATION = "query_generation"
    DATA_ANALYSIS = "data_analysis"
    EXPLANATION = "explanation"
    SUGGESTION = "suggestion"
    ERROR_DIAGNOSIS = "error_diagnosis"


@dataclass
class PromptContext:
    """Context for prompt generation."""
    schema: Optional[DataSchema] = None
    intent: Optional[ParsedIntent] = None
    sample_data: Optional[Dict[str, List[Any]]] = None
    previous_queries: Optional[List[str]] = None
    error_message: Optional[str] = None


class PromptEngineer:
    """Generate optimized prompts for different data analysis tasks."""
    
    def __init__(self):
        self.templates = {
            PromptTemplate.QUERY_GENERATION: self._query_generation_template,
            PromptTemplate.DATA_ANALYSIS: self._data_analysis_template,
            PromptTemplate.EXPLANATION: self._explanation_template,
            PromptTemplate.SUGGESTION: self._suggestion_template,
            PromptTemplate.ERROR_DIAGNOSIS: self._error_diagnosis_template
        }
    
    def generate_prompt(self, template: PromptTemplate, context: PromptContext) -> str:
        """
        Generate prompt for specific template and context.
        
        Args:
            template: Prompt template to use
            context: Context information
            
        Returns:
            Generated prompt string
        """
        if template not in self.templates:
            raise ValueError(f"Unknown template: {template}")
        
        return self.templates[template](context)
    
    def _query_generation_template(self, context: PromptContext) -> str:
        """Generate prompt for SQL query generation."""
        if not context.schema:
            raise ValueError("Schema required for query generation")
        
        # Build schema description
        schema_desc = self._build_schema_description(context.schema)
        
        # Build question context
        question = context.intent.original_question if context.intent else "Analyze the data"
        
        # Determine query complexity
        complexity_hints = ""
        if context.intent:
            if context.intent.question_type == QuestionType.AGGREGATION:
                complexity_hints = "\n- Use appropriate aggregation functions (SUM, COUNT, AVG, etc.)"
            elif context.intent.question_type == QuestionType.FILTERING:
                complexity_hints = "\n- Include WHERE clauses for filtering"
            elif context.intent.question_type == QuestionType.COMPARISON:
                complexity_hints = "\n- Use GROUP BY for comparisons across categories"
            elif context.intent.question_type == QuestionType.TREND:
                complexity_hints = "\n- Order by date/time columns for trend analysis"
            elif context.intent.question_type == QuestionType.RANKING:
                complexity_hints = "\n- Use ORDER BY and LIMIT for ranking queries"
        
        # Add sample data context if available
        sample_context = ""
        if context.sample_data:
            sample_context = f"\n\nSample data preview:\n{self._format_sample_data(context.sample_data)}"
        
        prompt = f"""You are an expert SQL query generator. Generate a precise SQL query to answer the user's question.

Database Schema:
{schema_desc}
{sample_context}

User Question: "{question}"

Requirements:
- Generate only valid SQL syntax
- Use table name 'data_table' 
- Return only the SQL query without explanation
- Ensure the query directly answers the question{complexity_hints}
- Use appropriate column names from the schema
- Handle potential NULL values appropriately

SQL Query:"""
        
        return prompt.strip()
    
    def _data_analysis_template(self, context: PromptContext) -> str:
        """Generate prompt for data analysis insights."""
        schema_desc = ""
        if context.schema:
            schema_desc = f"Data Schema:\n{self._build_schema_description(context.schema)}\n\n"
        
        sample_context = ""
        if context.sample_data:
            sample_context = f"Sample Data:\n{self._format_sample_data(context.sample_data)}\n\n"
        
        question = context.intent.original_question if context.intent else "Analyze this data"
        
        prompt = f"""You are a data analyst. Provide insights and analysis based on the user's question.

{schema_desc}{sample_context}User Question: "{question}"

Please provide:
1. Key insights from the data
2. Notable patterns or trends
3. Recommendations based on the analysis
4. Potential follow-up questions for deeper analysis

Keep your response concise and actionable."""
        
        return prompt.strip()
    
    def _explanation_template(self, context: PromptContext) -> str:
        """Generate prompt for explaining query results or analysis."""
        question = context.intent.original_question if context.intent else "the analysis"
        
        prompt = f"""You are a data analyst explaining results to a business user.

User asked: "{question}"

Please explain:
1. What the results show in plain language
2. Why these results are significant
3. What actions or decisions this data supports
4. Any limitations or caveats to consider

Use clear, non-technical language that a business stakeholder would understand."""
        
        return prompt.strip()
    
    def _suggestion_template(self, context: PromptContext) -> str:
        """Generate prompt for suggesting follow-up questions or analyses."""
        schema_desc = ""
        if context.schema:
            schema_desc = f"Available data includes:\n{self._build_simple_schema_description(context.schema)}\n\n"
        
        current_question = context.intent.original_question if context.intent else "data analysis"
        
        prompt = f"""You are a data analyst suggesting follow-up analyses.

{schema_desc}The user just asked: "{current_question}"

Based on the available data and their current question, suggest 3-5 relevant follow-up questions that would provide additional business insights. 

Format each suggestion as:
- Question: [specific question]
- Why: [brief explanation of business value]

Focus on actionable insights that would help with business decisions."""
        
        return prompt.strip()
    
    def _error_diagnosis_template(self, context: PromptContext) -> str:
        """Generate prompt for diagnosing query or analysis errors."""
        error_msg = context.error_message or "Unknown error occurred"
        
        schema_desc = ""
        if context.schema:
            schema_desc = f"Schema:\n{self._build_schema_description(context.schema)}\n\n"
        
        question = context.intent.original_question if context.intent else "data query"
        
        prompt = f"""You are a database expert helping diagnose query errors.

{schema_desc}User Question: "{question}"
Error Message: {error_msg}

Please provide:
1. What likely caused this error
2. How to fix the issue
3. A corrected approach if applicable
4. Tips to avoid similar errors

Be specific and actionable in your response."""
        
        return prompt.strip()
    
    def _build_schema_description(self, schema: DataSchema) -> str:
        """Build detailed schema description."""
        lines = []
        lines.append("Table: data_table")
        lines.append("Columns:")
        
        for col in schema.columns:
            col_desc = f"  - {col.name} ({col.data_type.value})"
            
            if col.sample_values:
                sample_str = ", ".join(str(v) for v in col.sample_values[:3])
                col_desc += f" - Examples: {sample_str}"
            
            if not col.nullable:
                col_desc += " [NOT NULL]"
            
            lines.append(col_desc)
        
        lines.append(f"\nTotal rows: {schema.row_count}")
        
        return "\n".join(lines)
    
    def _build_simple_schema_description(self, schema: DataSchema) -> str:
        """Build simple schema description for suggestions."""
        categories = {
            'Numeric': [],
            'Text': [],
            'Date': [],
            'Boolean': []
        }
        
        for col in schema.columns:
            if col.data_type in [DataType.NUMBER, DataType.CURRENCY]:
                categories['Numeric'].append(col.name)
            elif col.data_type == DataType.TEXT:
                categories['Text'].append(col.name)
            elif col.data_type == DataType.DATE:
                categories['Date'].append(col.name)
            elif col.data_type == DataType.BOOLEAN:
                categories['Boolean'].append(col.name)
        
        lines = []
        for category, columns in categories.items():
            if columns:
                lines.append(f"- {category}: {', '.join(columns)}")
        
        return "\n".join(lines)
    
    def _format_sample_data(self, sample_data: Dict[str, List[Any]]) -> str:
        """Format sample data for prompt context."""
        if not sample_data:
            return "No sample data available"
        
        lines = []
        
        # Get column names
        columns = list(sample_data.keys())
        if not columns:
            return "No sample data available"
        
        # Header
        lines.append(" | ".join(columns))
        lines.append("-" * (len(" | ".join(columns))))
        
        # Data rows (limit to first 3 rows)
        max_rows = min(3, len(next(iter(sample_data.values()))))
        
        for i in range(max_rows):
            row = []
            for col in columns:
                if i < len(sample_data[col]):
                    value = str(sample_data[col][i])
                    # Truncate long values
                    if len(value) > 20:
                        value = value[:17] + "..."
                    row.append(value)
                else:
                    row.append("NULL")
            lines.append(" | ".join(row))
        
        return "\n".join(lines)
    
    def optimize_prompt_for_model(self, prompt: str, model_name: str) -> str:
        """
        Optimize prompt for specific model characteristics.
        
        Args:
            prompt: Original prompt
            model_name: Target model name
            
        Returns:
            Optimized prompt
        """
        # Model-specific optimizations
        if 'gpt-3.5' in model_name.lower():
            # GPT-3.5 works better with clear structure and explicit instructions
            return self._optimize_for_gpt35(prompt)
        elif 'gpt-4' in model_name.lower():
            # GPT-4 can handle more complex reasoning
            return self._optimize_for_gpt4(prompt)
        elif 'claude' in model_name.lower():
            # Claude prefers conversational style
            return self._optimize_for_claude(prompt)
        else:
            # Default optimization
            return prompt
    
    def _optimize_for_gpt35(self, prompt: str) -> str:
        """Optimize prompt for GPT-3.5."""
        # Add explicit formatting instructions
        if "SQL Query:" in prompt:
            prompt += "\n\nFormat: Return only the SQL query, no additional text."
        
        return prompt
    
    def _optimize_for_gpt4(self, prompt: str) -> str:
        """Optimize prompt for GPT-4."""
        # GPT-4 can handle more nuanced instructions
        return prompt
    
    def _optimize_for_claude(self, prompt: str) -> str:
        """Optimize prompt for Claude."""
        # Claude prefers polite, conversational tone
        if prompt.startswith("You are"):
            return "Please help as " + prompt[7:].lower()
        elif "Generate a SQL query" in prompt:
            return prompt + "\n\nPlease provide a clear, well-formatted response."
        
        return prompt
    
    def create_few_shot_examples(self, template: PromptTemplate, 
                                context: PromptContext) -> List[Dict[str, str]]:
        """
        Create few-shot examples for better prompt performance.
        
        Args:
            template: Prompt template
            context: Context information
            
        Returns:
            List of example input/output pairs
        """
        if template == PromptTemplate.QUERY_GENERATION:
            return self._query_generation_examples(context)
        elif template == PromptTemplate.DATA_ANALYSIS:
            return self._data_analysis_examples(context)
        else:
            return []
    
    def _query_generation_examples(self, context: PromptContext) -> List[Dict[str, str]]:
        """Generate few-shot examples for query generation."""
        examples = [
            {
                "input": "What is the total sales amount?",
                "output": "SELECT SUM(sales_amount) as total_sales FROM data_table"
            },
            {
                "input": "Show me customers with sales greater than 1000",
                "output": "SELECT * FROM data_table WHERE sales_amount > 1000"
            },
            {
                "input": "What are the top 5 products by sales?",
                "output": "SELECT product_name, SUM(sales_amount) as total_sales FROM data_table GROUP BY product_name ORDER BY total_sales DESC LIMIT 5"
            }
        ]
        
        # Filter examples based on available schema
        if context.schema:
            schema_columns = {col.name for col in context.schema.columns}
            filtered_examples = []
            
            for example in examples:
                # Check if example uses columns that exist in schema
                output = example["output"]
                if any(col in output for col in schema_columns):
                    filtered_examples.append(example)
            
            return filtered_examples
        
        return examples
    
    def _data_analysis_examples(self, context: PromptContext) -> List[Dict[str, str]]:
        """Generate few-shot examples for data analysis."""
        return [
            {
                "input": "Analyze sales performance",
                "output": "Based on the sales data, I can see that total revenue is $X with an average order value of $Y. The top-performing category is Z, representing 40% of total sales. I recommend focusing marketing efforts on this category while investigating why other categories are underperforming."
            }
        ]