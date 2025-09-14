"""
Query generation engine for converting natural language to SQL/Pandas queries.
"""

import re
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

from .nlp_processor import ParsedIntent, QuestionType, AggregationType
from ..core.models import DataSchema, Query, ValidationResult, DataType


class QueryType(Enum):
    """Types of queries that can be generated."""
    SELECT = "select"
    AGGREGATE = "aggregate"
    FILTER = "filter"
    GROUP_BY = "group_by"
    ORDER_BY = "order_by"
    JOIN = "join"


@dataclass
class QueryPlan:
    """Execution plan for a query."""
    query_type: QueryType
    operations: List[Dict[str, Any]]
    estimated_complexity: str
    requires_joins: bool = False
    optimization_hints: List[str] = None


class QueryGenerator:
    """Generate SQL and Pandas queries from natural language intents."""
    
    def __init__(self):
        self.operator_mapping = {
            'equals': '=',
            'greater_than': '>',
            'less_than': '<',
            'greater_equal': '>=',
            'less_equal': '<=',
            'not_equal': '!=',
            'contains': 'LIKE',
            'starts_with': 'LIKE',
            'ends_with': 'LIKE',
            'is': '=',
            'greater than': '>',
            'less than': '<',
            'at least': '>=',
            'at most': '<=',
            'minimum': '>=',
            'maximum': '<='
        }
        
        self.aggregation_mapping = {
            AggregationType.COUNT: 'COUNT',
            AggregationType.SUM: 'SUM',
            AggregationType.AVERAGE: 'AVG',
            AggregationType.MIN: 'MIN',
            AggregationType.MAX: 'MAX',
            AggregationType.MEDIAN: 'MEDIAN',
            AggregationType.STD: 'STDDEV'
        }
    
    def generate_sql_query(self, intent: ParsedIntent, schema: DataSchema, 
                          table_name: str = "data_table") -> Query:
        """
        Generate SQL query from parsed intent.
        
        Args:
            intent: Parsed natural language intent
            schema: Data schema information
            table_name: Name of the table to query
            
        Returns:
            Query object with SQL and parameters
        """
        if intent.question_type == QuestionType.AGGREGATION:
            return self._generate_aggregation_query(intent, schema, table_name)
        elif intent.question_type == QuestionType.FILTERING:
            return self._generate_filter_query(intent, schema, table_name)
        elif intent.question_type == QuestionType.COMPARISON:
            return self._generate_comparison_query(intent, schema, table_name)
        elif intent.question_type == QuestionType.RANKING:
            return self._generate_ranking_query(intent, schema, table_name)
        elif intent.question_type == QuestionType.TREND:
            return self._generate_trend_query(intent, schema, table_name)
        elif intent.question_type == QuestionType.DISTRIBUTION:
            return self._generate_distribution_query(intent, schema, table_name)
        else:
            return self._generate_basic_select_query(intent, schema, table_name)
    
    def generate_pandas_query(self, intent: ParsedIntent, schema: DataSchema) -> str:
        """
        Generate Pandas query string from parsed intent.
        
        Args:
            intent: Parsed natural language intent
            schema: Data schema information
            
        Returns:
            Pandas query string
        """
        if intent.question_type == QuestionType.AGGREGATION:
            return self._generate_pandas_aggregation(intent, schema)
        elif intent.question_type == QuestionType.FILTERING:
            return self._generate_pandas_filter(intent, schema)
        elif intent.question_type == QuestionType.COMPARISON:
            return self._generate_pandas_comparison(intent, schema)
        elif intent.question_type == QuestionType.RANKING:
            return self._generate_pandas_ranking(intent, schema)
        elif intent.question_type == QuestionType.TREND:
            return self._generate_pandas_trend(intent, schema)
        elif intent.question_type == QuestionType.DISTRIBUTION:
            return self._generate_pandas_distribution(intent, schema)
        else:
            return self._generate_pandas_basic_select(intent, schema)
    
    def _generate_aggregation_query(self, intent: ParsedIntent, schema: DataSchema, 
                                  table_name: str) -> Query:
        """Generate SQL for aggregation queries."""
        columns = intent.columns if intent.columns else self._get_numeric_columns(schema)
        
        if not columns:
            # Default to count all
            sql = f"SELECT COUNT(*) as total_count FROM {table_name}"
            return Query(sql=sql, columns=['total_count'])
        
        # Build aggregation clauses
        agg_clauses = []
        result_columns = []
        
        for column in columns:
            if intent.aggregation:
                agg_func = self.aggregation_mapping.get(intent.aggregation, 'COUNT')
                if intent.aggregation == AggregationType.COUNT:
                    agg_clause = f"{agg_func}(*)"
                    result_col = f"count_all"
                else:
                    agg_clause = f"{agg_func}({column})"
                    result_col = f"{agg_func.lower()}_{column}"
            else:
                # Default to SUM for numeric columns, COUNT for others
                col_info = next((c for c in schema.columns if c.name == column), None)
                if col_info and col_info.data_type in [DataType.NUMBER, DataType.CURRENCY]:
                    agg_clause = f"SUM({column})"
                    result_col = f"sum_{column}"
                else:
                    agg_clause = f"COUNT({column})"
                    result_col = f"count_{column}"
            
            agg_clauses.append(f"{agg_clause} as {result_col}")
            result_columns.append(result_col)
        
        sql = f"SELECT {', '.join(agg_clauses)} FROM {table_name}"
        
        # Add filters if present
        if intent.filters:
            where_clause = self._build_where_clause(intent.filters)
            sql += f" WHERE {where_clause}"
        
        return Query(sql=sql, columns=result_columns)
    
    def _generate_filter_query(self, intent: ParsedIntent, schema: DataSchema, 
                             table_name: str) -> Query:
        """Generate SQL for filtering queries."""
        columns = intent.columns if intent.columns else ['*']
        
        sql = f"SELECT {', '.join(columns)} FROM {table_name}"
        
        if intent.filters:
            where_clause = self._build_where_clause(intent.filters)
            sql += f" WHERE {where_clause}"
        
        return Query(sql=sql, columns=columns)
    
    def _generate_comparison_query(self, intent: ParsedIntent, schema: DataSchema, 
                                 table_name: str) -> Query:
        """Generate SQL for comparison queries."""
        columns = intent.columns if intent.columns else self._get_numeric_columns(schema)
        
        if len(columns) < 2:
            # Add a grouping column if we only have one column
            categorical_cols = self._get_categorical_columns(schema)
            if categorical_cols:
                columns.append(categorical_cols[0])
        
        if len(columns) >= 2:
            # Group by categorical column, aggregate numeric column
            group_col = columns[-1]  # Assume last column is categorical
            agg_cols = columns[:-1]  # All others are for aggregation
            
            agg_clauses = []
            result_columns = [group_col]
            
            for col in agg_cols:
                agg_clauses.append(f"AVG({col}) as avg_{col}")
                result_columns.append(f"avg_{col}")
            
            sql = f"SELECT {group_col}, {', '.join(agg_clauses)} FROM {table_name}"
            sql += f" GROUP BY {group_col}"
            sql += f" ORDER BY {group_col}"
            
            return Query(sql=sql, columns=result_columns)
        else:
            # Fallback to basic select
            return self._generate_basic_select_query(intent, schema, table_name)
    
    def _generate_ranking_query(self, intent: ParsedIntent, schema: DataSchema, 
                              table_name: str) -> Query:
        """Generate SQL for ranking queries."""
        columns = intent.columns if intent.columns else self._get_numeric_columns(schema)
        
        # Extract limit from numbers in the intent
        limit = None
        if intent.entities and 'numbers' in intent.entities:
            numbers = intent.entities['numbers']
            if numbers:
                limit = int(numbers[0])
        
        if not limit:
            limit = 10  # Default limit
        
        if columns:
            order_col = columns[0]
            # Determine if we want top (DESC) or bottom (ASC)
            order_direction = "DESC"  # Default to top
            if any(word in intent.original_question.lower() for word in ['bottom', 'worst', 'lowest']):
                order_direction = "ASC"
            
            sql = f"SELECT * FROM {table_name} ORDER BY {order_col} {order_direction} LIMIT {limit}"
            return Query(sql=sql, columns=['*'], limit=limit)
        else:
            return self._generate_basic_select_query(intent, schema, table_name)
    
    def _generate_trend_query(self, intent: ParsedIntent, schema: DataSchema, 
                            table_name: str) -> Query:
        """Generate SQL for trend queries."""
        # Look for date columns
        date_columns = self._get_date_columns(schema)
        numeric_columns = intent.columns if intent.columns else self._get_numeric_columns(schema)
        
        if date_columns and numeric_columns:
            date_col = date_columns[0]
            value_col = numeric_columns[0]
            
            sql = f"""
            SELECT {date_col}, AVG({value_col}) as avg_{value_col}
            FROM {table_name}
            GROUP BY {date_col}
            ORDER BY {date_col}
            """
            
            return Query(sql=sql.strip(), columns=[date_col, f"avg_{value_col}"])
        else:
            return self._generate_basic_select_query(intent, schema, table_name)
    
    def _generate_distribution_query(self, intent: ParsedIntent, schema: DataSchema, 
                                   table_name: str) -> Query:
        """Generate SQL for distribution queries."""
        columns = intent.columns if intent.columns else self._get_categorical_columns(schema)
        
        if columns:
            group_col = columns[0]
            sql = f"""
            SELECT {group_col}, COUNT(*) as count,
                   COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {table_name}) as percentage
            FROM {table_name}
            GROUP BY {group_col}
            ORDER BY count DESC
            """
            
            return Query(sql=sql.strip(), columns=[group_col, 'count', 'percentage'])
        else:
            return self._generate_basic_select_query(intent, schema, table_name)
    
    def _generate_basic_select_query(self, intent: ParsedIntent, schema: DataSchema, 
                                   table_name: str) -> Query:
        """Generate basic SELECT query."""
        columns = intent.columns if intent.columns else ['*']
        sql = f"SELECT {', '.join(columns)} FROM {table_name}"
        
        if intent.filters:
            where_clause = self._build_where_clause(intent.filters)
            sql += f" WHERE {where_clause}"
        
        return Query(sql=sql, columns=columns)
    
    def _build_where_clause(self, filters: List[Dict[str, Any]]) -> str:
        """Build WHERE clause from filter conditions."""
        conditions = []
        
        for filter_condition in filters:
            column = filter_condition['column']
            operator = filter_condition['operator']
            value = filter_condition['value']
            
            # Map operator
            sql_operator = self.operator_mapping.get(operator, operator)
            
            # Handle LIKE operators
            if sql_operator == 'LIKE':
                if operator == 'contains':
                    value = f"'%{value}%'"
                elif operator == 'starts_with':
                    value = f"'{value}%'"
                elif operator == 'ends_with':
                    value = f"'%{value}'"
                else:
                    value = f"'{value}'"
            else:
                # Quote string values
                if isinstance(value, str) and not value.replace('.', '').isdigit():
                    value = f"'{value}'"
            
            conditions.append(f"{column} {sql_operator} {value}")
        
        return ' AND '.join(conditions)
    
    def _generate_pandas_aggregation(self, intent: ParsedIntent, schema: DataSchema) -> str:
        """Generate Pandas aggregation query."""
        columns = intent.columns if intent.columns else self._get_numeric_columns(schema)
        
        if not columns:
            return "df.shape[0]  # Count all rows"
        
        if intent.aggregation == AggregationType.COUNT:
            return f"df[{columns}].count()"
        elif intent.aggregation == AggregationType.SUM:
            return f"df[{columns}].sum()"
        elif intent.aggregation == AggregationType.AVERAGE:
            return f"df[{columns}].mean()"
        elif intent.aggregation == AggregationType.MIN:
            return f"df[{columns}].min()"
        elif intent.aggregation == AggregationType.MAX:
            return f"df[{columns}].max()"
        elif intent.aggregation == AggregationType.MEDIAN:
            return f"df[{columns}].median()"
        else:
            return f"df[{columns}].describe()"
    
    def _generate_pandas_filter(self, intent: ParsedIntent, schema: DataSchema) -> str:
        """Generate Pandas filter query."""
        if not intent.filters:
            columns = intent.columns if intent.columns else None
            if columns:
                return f"df[{columns}]"
            else:
                return "df"
        
        conditions = []
        for filter_condition in intent.filters:
            column = filter_condition['column']
            operator = filter_condition['operator']
            value = filter_condition['value']
            
            if operator in ['equals', 'is', '=']:
                conditions.append(f"(df['{column}'] == '{value}')")
            elif operator in ['greater_than', '>']:
                conditions.append(f"(df['{column}'] > {value})")
            elif operator in ['less_than', '<']:
                conditions.append(f"(df['{column}'] < {value})")
            elif operator in ['greater_equal', '>=']:
                conditions.append(f"(df['{column}'] >= {value})")
            elif operator in ['less_equal', '<=']:
                conditions.append(f"(df['{column}'] <= {value})")
            elif operator in ['not_equal', '!=']:
                conditions.append(f"(df['{column}'] != '{value}')")
            elif operator == 'contains':
                conditions.append(f"df['{column}'].str.contains('{value}')")
        
        filter_expr = ' & '.join(conditions)
        columns = intent.columns if intent.columns else None
        
        if columns:
            return f"df.loc[{filter_expr}, {columns}]"
        else:
            return f"df[{filter_expr}]"
    
    def _generate_pandas_comparison(self, intent: ParsedIntent, schema: DataSchema) -> str:
        """Generate Pandas comparison query."""
        columns = intent.columns if intent.columns else self._get_numeric_columns(schema)
        
        if len(columns) >= 2:
            return f"df.groupby('{columns[-1]}')[{columns[:-1]}].mean()"
        else:
            return f"df[{columns}].describe()"
    
    def _generate_pandas_ranking(self, intent: ParsedIntent, schema: DataSchema) -> str:
        """Generate Pandas ranking query."""
        columns = intent.columns if intent.columns else self._get_numeric_columns(schema)
        
        # Extract limit
        limit = 10
        if intent.entities and 'numbers' in intent.entities:
            numbers = intent.entities['numbers']
            if numbers:
                limit = int(numbers[0])
        
        if columns:
            order_col = columns[0]
            ascending = any(word in intent.original_question.lower() 
                          for word in ['bottom', 'worst', 'lowest'])
            
            return f"df.nlargest({limit}, '{order_col}')" if not ascending else f"df.nsmallest({limit}, '{order_col}')"
        else:
            return f"df.head({limit})"
    
    def _generate_pandas_trend(self, intent: ParsedIntent, schema: DataSchema) -> str:
        """Generate Pandas trend query."""
        date_columns = self._get_date_columns(schema)
        numeric_columns = intent.columns if intent.columns else self._get_numeric_columns(schema)
        
        if date_columns and numeric_columns:
            date_col = date_columns[0]
            value_col = numeric_columns[0]
            return f"df.groupby('{date_col}')['{value_col}'].mean().sort_index()"
        else:
            return "df.describe()"
    
    def _generate_pandas_distribution(self, intent: ParsedIntent, schema: DataSchema) -> str:
        """Generate Pandas distribution query."""
        columns = intent.columns if intent.columns else self._get_categorical_columns(schema)
        
        if columns:
            col = columns[0]
            return f"df['{col}'].value_counts(normalize=True)"
        else:
            return "df.describe()"
    
    def _generate_pandas_basic_select(self, intent: ParsedIntent, schema: DataSchema) -> str:
        """Generate basic Pandas select."""
        columns = intent.columns if intent.columns else None
        
        if columns:
            return f"df[{columns}]"
        else:
            return "df"
    
    def _get_numeric_columns(self, schema: DataSchema) -> List[str]:
        """Get numeric columns from schema."""
        return [col.name for col in schema.columns 
                if col.data_type in [DataType.NUMBER, DataType.CURRENCY]]
    
    def _get_categorical_columns(self, schema: DataSchema) -> List[str]:
        """Get categorical columns from schema."""
        return [col.name for col in schema.columns 
                if col.data_type in [DataType.TEXT, DataType.BOOLEAN]]
    
    def _get_date_columns(self, schema: DataSchema) -> List[str]:
        """Get date columns from schema."""
        return [col.name for col in schema.columns 
                if col.data_type == DataType.DATE]
    
    def validate_query(self, query: Query, schema: DataSchema) -> ValidationResult:
        """
        Validate generated query against schema.
        
        Args:
            query: Generated query
            schema: Data schema
            
        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []
        
        # Check if columns exist in schema
        schema_columns = {col.name for col in schema.columns}
        
        for column in query.columns:
            if column != '*' and column not in schema_columns:
                # Check if it's an alias (contains 'as' or '_')
                if ' as ' not in column.lower() and not any(
                    col in column for col in schema_columns
                ):
                    errors.append(f"Column '{column}' not found in schema")
        
        # Basic SQL syntax validation
        sql_lower = query.sql.lower()
        
        # Check for required keywords
        if not any(keyword in sql_lower for keyword in ['select', 'from']):
            errors.append("Query must contain SELECT and FROM clauses")
        
        # Check for potential SQL injection
        dangerous_keywords = ['drop', 'delete', 'truncate', 'alter', 'create']
        if any(keyword in sql_lower for keyword in dangerous_keywords):
            errors.append("Query contains potentially dangerous SQL keywords")
        
        # Performance warnings
        if 'select *' in sql_lower and 'limit' not in sql_lower:
            warnings.append("Consider limiting results when selecting all columns")
        
        if 'group by' in sql_lower and 'order by' not in sql_lower:
            warnings.append("Consider adding ORDER BY for consistent results")
        
        is_valid = len(errors) == 0
        confidence = 1.0 if is_valid else max(0.0, 1.0 - len(errors) * 0.2)
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            confidence=confidence
        )
    
    def optimize_query(self, query: Query, schema: DataSchema) -> Query:
        """
        Optimize query for better performance.
        
        Args:
            query: Original query
            schema: Data schema
            
        Returns:
            Optimized query
        """
        optimized_sql = query.sql
        
        # Add LIMIT if not present and selecting all columns
        if 'select *' in optimized_sql.lower() and 'limit' not in optimized_sql.lower():
            optimized_sql += " LIMIT 1000"
        
        # Add indexes hints for large tables (if we had that information)
        # This is a placeholder for more sophisticated optimization
        
        return Query(
            sql=optimized_sql,
            parameters=query.parameters,
            columns=query.columns,
            limit=query.limit
        )
    
    def create_query_plan(self, intent: ParsedIntent, schema: DataSchema) -> QueryPlan:
        """
        Create execution plan for the query.
        
        Args:
            intent: Parsed intent
            schema: Data schema
            
        Returns:
            QueryPlan with execution strategy
        """
        operations = []
        complexity = 'simple'
        requires_joins = False
        
        # Analyze operations needed
        if intent.question_type == QuestionType.AGGREGATION:
            operations.append({'type': 'aggregate', 'function': intent.aggregation})
            if intent.filters:
                operations.append({'type': 'filter', 'conditions': len(intent.filters)})
        
        elif intent.question_type == QuestionType.FILTERING:
            operations.append({'type': 'filter', 'conditions': len(intent.filters or [])})
        
        elif intent.question_type == QuestionType.COMPARISON:
            operations.append({'type': 'group_by'})
            operations.append({'type': 'aggregate'})
            complexity = 'medium'
        
        elif intent.question_type == QuestionType.RANKING:
            operations.append({'type': 'order_by'})
            operations.append({'type': 'limit'})
        
        elif intent.question_type == QuestionType.TREND:
            operations.append({'type': 'group_by'})
            operations.append({'type': 'aggregate'})
            operations.append({'type': 'order_by'})
            complexity = 'medium'
        
        # Determine complexity
        if len(operations) > 3 or len(intent.columns) > 3:
            complexity = 'complex'
        elif len(operations) > 1 or (intent.filters and len(intent.filters) > 1):
            complexity = 'medium'
        
        optimization_hints = []
        if complexity == 'complex':
            optimization_hints.append("Consider adding indexes on filter columns")
        if intent.filters and len(intent.filters) > 2:
            optimization_hints.append("Multiple filters may benefit from compound indexes")
        
        return QueryPlan(
            query_type=QueryType.SELECT,  # Default type
            operations=operations,
            estimated_complexity=complexity,
            requires_joins=requires_joins,
            optimization_hints=optimization_hints
        )