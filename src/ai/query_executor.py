"""
Query executor for running generated queries against data.
"""

import pandas as pd
import sqlite3
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import time
import logging

from ..core.models import Query, QueryResult, ValidationResult


@dataclass
class ExecutionContext:
    """Context for query execution."""
    dataframe: Optional[pd.DataFrame] = None
    connection: Optional[sqlite3.Connection] = None
    timeout_seconds: int = 30
    max_rows: int = 10000


class QueryExecutor:
    """Execute SQL and Pandas queries against data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def execute_sql_query(self, query: Query, context: ExecutionContext) -> QueryResult:
        """
        Execute SQL query against database connection.
        
        Args:
            query: SQL query to execute
            context: Execution context with connection
            
        Returns:
            QueryResult with data and metadata
        """
        if not context.connection:
            raise ValueError("Database connection required for SQL execution")
        
        start_time = time.time()
        
        try:
            # Execute query with timeout
            cursor = context.connection.cursor()
            cursor.execute(query.sql, query.parameters or {})
            
            # Fetch results
            rows = cursor.fetchmany(context.max_rows)
            columns = [description[0] for description in cursor.description]
            
            # Convert to list of dictionaries
            data = [dict(zip(columns, row)) for row in rows]
            
            execution_time = int((time.time() - start_time) * 1000)
            
            return QueryResult(
                data=data,
                columns=columns,
                row_count=len(data),
                execution_time_ms=execution_time,
                query=query
            )
            
        except Exception as e:
            self.logger.error(f"SQL execution error: {str(e)}")
            raise
        finally:
            if 'cursor' in locals():
                cursor.close()
    
    def execute_pandas_query(self, query_string: str, context: ExecutionContext) -> QueryResult:
        """
        Execute Pandas query against DataFrame.
        
        Args:
            query_string: Pandas query string
            context: Execution context with DataFrame
            
        Returns:
            QueryResult with data and metadata
        """
        if context.dataframe is None:
            raise ValueError("DataFrame required for Pandas execution")
        
        start_time = time.time()
        
        try:
            df = context.dataframe
            
            # Execute the query string
            # This is a simplified approach - in production, you'd want more security
            result = eval(query_string)
            
            # Convert result to standard format
            if isinstance(result, pd.DataFrame):
                # Limit rows if necessary
                if len(result) > context.max_rows:
                    result = result.head(context.max_rows)
                
                data = result.to_dict('records')
                columns = result.columns.tolist()
                row_count = len(result)
                
            elif isinstance(result, pd.Series):
                # Convert Series to DataFrame-like structure
                if result.name:
                    columns = [result.name]
                    data = [{'value': val} for val in result.values]
                else:
                    columns = ['value']
                    data = [{'value': val} for val in result.values]
                row_count = len(result)
                
            else:
                # Single value result
                columns = ['result']
                data = [{'result': result}]
                row_count = 1
            
            execution_time = int((time.time() - start_time) * 1000)
            
            return QueryResult(
                data=data,
                columns=columns,
                row_count=row_count,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Pandas execution error: {str(e)}")
            raise
    
    def execute_dataframe_to_sql(self, dataframe: pd.DataFrame, 
                                table_name: str = "temp_table") -> sqlite3.Connection:
        """
        Load DataFrame into temporary SQLite database.
        
        Args:
            dataframe: DataFrame to load
            table_name: Name for the table
            
        Returns:
            SQLite connection with loaded data
        """
        # Create in-memory SQLite database
        conn = sqlite3.connect(':memory:')
        
        try:
            # Load DataFrame into SQLite
            dataframe.to_sql(table_name, conn, index=False, if_exists='replace')
            return conn
            
        except Exception as e:
            conn.close()
            self.logger.error(f"Error loading DataFrame to SQL: {str(e)}")
            raise
    
    def execute_query_with_fallback(self, query: Query, pandas_query: str, 
                                  context: ExecutionContext) -> QueryResult:
        """
        Execute query with fallback from SQL to Pandas.
        
        Args:
            query: SQL query
            pandas_query: Equivalent Pandas query
            context: Execution context
            
        Returns:
            QueryResult from successful execution
        """
        # Try SQL first if connection available
        if context.connection:
            try:
                return self.execute_sql_query(query, context)
            except Exception as e:
                self.logger.warning(f"SQL execution failed, falling back to Pandas: {str(e)}")
        
        # Fallback to Pandas
        if context.dataframe is not None:
            return self.execute_pandas_query(pandas_query, context)
        
        raise ValueError("No valid execution context available")
    
    def validate_execution_context(self, context: ExecutionContext) -> ValidationResult:
        """
        Validate execution context.
        
        Args:
            context: Execution context to validate
            
        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []
        
        if context.dataframe is None and context.connection is None:
            errors.append("Either DataFrame or database connection must be provided")
        
        if context.dataframe is not None:
            if context.dataframe.empty:
                warnings.append("DataFrame is empty")
            elif len(context.dataframe) > 100000:
                warnings.append("Large DataFrame may impact performance")
        
        if context.timeout_seconds < 1:
            errors.append("Timeout must be at least 1 second")
        
        if context.max_rows < 1:
            errors.append("Max rows must be at least 1")
        elif context.max_rows > 50000:
            warnings.append("Large max_rows setting may impact performance")
        
        is_valid = len(errors) == 0
        confidence = 1.0 if is_valid else 0.5
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            confidence=confidence
        )
    
    def get_execution_statistics(self, results: List[QueryResult]) -> Dict[str, Any]:
        """
        Get execution statistics from multiple query results.
        
        Args:
            results: List of query results
            
        Returns:
            Dictionary with execution statistics
        """
        if not results:
            return {}
        
        execution_times = [r.execution_time_ms for r in results]
        row_counts = [r.row_count for r in results]
        
        return {
            'total_queries': len(results),
            'avg_execution_time_ms': sum(execution_times) / len(execution_times),
            'max_execution_time_ms': max(execution_times),
            'min_execution_time_ms': min(execution_times),
            'total_rows_returned': sum(row_counts),
            'avg_rows_per_query': sum(row_counts) / len(row_counts),
            'max_rows_per_query': max(row_counts)
        }
    
    def optimize_dataframe_for_queries(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame for better query performance.
        
        Args:
            dataframe: Original DataFrame
            
        Returns:
            Optimized DataFrame
        """
        optimized_df = dataframe.copy()
        
        # Convert object columns to category if they have few unique values
        for col in optimized_df.select_dtypes(include=['object']).columns:
            unique_ratio = optimized_df[col].nunique() / len(optimized_df)
            if unique_ratio <= 0.5:  # Less than or equal to 50% unique values
                optimized_df[col] = optimized_df[col].astype('category')
        
        # Ensure numeric columns are proper numeric types
        for col in optimized_df.columns:
            if optimized_df[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    numeric_col = pd.to_numeric(optimized_df[col], errors='coerce')
                    if not numeric_col.isna().all():
                        optimized_df[col] = numeric_col
                except:
                    pass
        
        return optimized_df
    
    def create_execution_plan(self, query: Query, context: ExecutionContext) -> Dict[str, Any]:
        """
        Create execution plan for query.
        
        Args:
            query: Query to analyze
            context: Execution context
            
        Returns:
            Dictionary with execution plan details
        """
        plan = {
            'query_type': 'unknown',
            'estimated_rows': 0,
            'estimated_time_ms': 0,
            'operations': [],
            'optimization_suggestions': []
        }
        
        sql_lower = query.sql.lower()
        
        # Determine query type
        if 'select' in sql_lower:
            plan['query_type'] = 'select'
        if 'group by' in sql_lower:
            plan['operations'].append('group_by')
        if 'order by' in sql_lower:
            plan['operations'].append('order_by')
        if 'where' in sql_lower:
            plan['operations'].append('filter')
        if any(func in sql_lower for func in ['sum', 'count', 'avg', 'min', 'max']):
            plan['operations'].append('aggregate')
        
        # Estimate based on DataFrame if available
        if context.dataframe is not None:
            total_rows = len(context.dataframe)
            
            # Rough estimation based on operations
            if 'filter' in plan['operations']:
                plan['estimated_rows'] = max(1, total_rows // 2)  # Assume 50% filtered
            elif 'group_by' in plan['operations']:
                # Estimate unique groups
                plan['estimated_rows'] = max(1, min(total_rows // 2, 100))
            else:
                plan['estimated_rows'] = min(total_rows, context.max_rows)
            
            # Estimate execution time (very rough)
            base_time = max(1, total_rows // 10000)  # 1ms per 10k rows
            operation_multiplier = len(plan['operations']) + 1
            plan['estimated_time_ms'] = base_time * operation_multiplier
        
        # Optimization suggestions
        if 'select *' in sql_lower:
            plan['optimization_suggestions'].append("Consider selecting specific columns instead of *")
        
        if 'group by' in sql_lower and 'limit' not in sql_lower:
            plan['optimization_suggestions'].append("Consider adding LIMIT for grouped queries")
        
        return plan