"""
Enhanced Table Validation and Manual Editing Module
Provides comprehensive table validation and manual editing capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime


def render_validation_page():
    """Render enhanced table validation page with manual editing capabilities"""
    if not st.session_state.selected_document:
        st.info("👆 Please upload and process a document first, or select one from the sidebar")
        return
    
    doc_name = st.session_state.selected_document
    
    if doc_name not in st.session_state.processed_documents:
        st.error(f"Document '{doc_name}' not found in processed documents")
        return
    
    doc_data = st.session_state.processed_documents[doc_name]
    df = doc_data['dataframe'].copy()
    metadata = doc_data.get('metadata', {})
    
    st.title("🔍 Table Validation & Manual Editing")
    st.markdown(f"Reviewing and editing tables from: **{doc_name}**")
    
    # Initialize editing state
    if f'editing_df_{doc_name}' not in st.session_state:
        st.session_state[f'editing_df_{doc_name}'] = df.copy()
    
    current_df = st.session_state[f'editing_df_{doc_name}']
    
    # Real confidence indicators
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        confidence = metadata.get('confidence_avg', 0)
        st.metric("Confidence Score", f"{confidence:.1%}")
    with col2:
        tables_found = len(metadata.get('table_regions', []))
        st.metric("Table Regions", str(tables_found))
    with col3:
        data_quality = "Good" if confidence > 0.7 else "Fair" if confidence > 0.5 else "Poor"
        st.metric("Data Quality", data_quality)
    with col4:
        st.metric("Current Size", f"{len(current_df)}×{len(current_df.columns)}")
    
    # Validation Issues Detection
    st.markdown("---")
    st.subheader("🔍 Validation Issues")
    
    validation_issues = detect_validation_issues(current_df)
    
    if validation_issues:
        for issue_type, issues in validation_issues.items():
            if issues:
                with st.expander(f"⚠️ {issue_type} ({len(issues)} found)", expanded=True):
                    for issue in issues:
                        st.warning(issue)
    else:
        st.success("✅ No validation issues detected!")
    
    # Manual Editing Tools
    st.markdown("---")
    st.subheader("🛠️ Manual Editing Tools")
    
    # Create tabs for different editing modes
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Cell Editor", "🗑️ Delete Rows/Columns", "➕ Add Rows/Columns", "🔧 Bulk Operations"])
    
    with tab1:
        current_df = render_cell_editor(current_df, doc_name)
    
    with tab2:
        current_df = render_delete_operations(current_df, doc_name)
    
    with tab3:
        current_df = render_add_operations(current_df, doc_name)
    
    with tab4:
        current_df = render_bulk_operations(current_df, doc_name)
    
    # Update session state
    st.session_state[f'editing_df_{doc_name}'] = current_df
    
    # Data Editor with enhanced features
    st.markdown("---")
    st.subheader("📊 Interactive Table Editor")
    
    if current_df.empty:
        st.warning("No table data available")
        return
    
    # Enhanced data editor with column configuration
    column_config = {}
    for col in current_df.columns:
        if current_df[col].dtype in ['int64', 'float64']:
            column_config[col] = st.column_config.NumberColumn(
                col,
                help=f"Edit {col} values",
                format="%.2f" if current_df[col].dtype == 'float64' else "%d"
            )
        else:
            column_config[col] = st.column_config.TextColumn(
                col,
                help=f"Edit {col} values",
                max_chars=200
            )
    
    # Display the editable dataframe
    edited_df = st.data_editor(
        current_df,
        width='stretch',
        num_rows="dynamic",
        column_config=column_config,
        key=f"main_editor_{doc_name}",
        hide_index=False
    )
    
    # Update if changes detected
    if not current_df.equals(edited_df):
        st.session_state[f'editing_df_{doc_name}'] = edited_df
        st.info("📝 Changes detected in table editor!")
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("💾 Save Changes", type="primary"):
            # Save changes to main storage
            st.session_state.processed_documents[doc_name]['dataframe'] = edited_df.copy()
            st.session_state.extracted_tables[doc_name] = edited_df.copy()
            st.session_state.processed_documents[doc_name]['last_saved'] = datetime.now()
            st.success("✅ Changes saved successfully!")
    
    with col2:
        if st.button("🔄 Reset to Original"):
            st.session_state[f'editing_df_{doc_name}'] = df.copy()
            st.info("🔄 Table reset to original state")
            st.rerun()
    
    with col3:
        if st.button("🤖 Auto-Clean Data"):
            cleaned_df = auto_clean_data(edited_df)
            st.session_state[f'editing_df_{doc_name}'] = cleaned_df
            st.success("✅ Data auto-cleaned!")
            st.rerun()
    
    with col4:
        if st.button("✅ Validate Data"):
            validation_results = validate_table_data(edited_df)
            display_validation_results(validation_results)
    
    with col5:
        # Export options
        csv = edited_df.to_csv(index=False)
        st.download_button(
            "📥 Export CSV",
            csv,
            f"{doc_name}_edited.csv",
            "text/csv"
        )
    
    # Data Statistics and Preview
    render_data_statistics(edited_df)


def detect_validation_issues(df):
    """Detect common validation issues in the dataframe"""
    issues = {
        "Empty Cells": [],
        "Duplicate Rows": [],
        "Data Type Issues": [],
        "Formatting Issues": []
    }
    
    # Check for empty cells
    for col in df.columns:
        empty_count = df[col].isna().sum() + (df[col] == '').sum()
        if empty_count > 0:
            issues["Empty Cells"].append(f"Column '{col}' has {empty_count} empty cells")
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues["Duplicate Rows"].append(f"Found {duplicates} duplicate rows")
    
    # Check for data type inconsistencies
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if column should be numeric
            numeric_count = 0
            for val in df[col].dropna():
                try:
                    float(str(val).replace(',', '').replace('$', '').replace('%', ''))
                    numeric_count += 1
                except:
                    pass
            
            if numeric_count > len(df[col].dropna()) * 0.8:  # 80% numeric
                issues["Data Type Issues"].append(f"Column '{col}' appears to contain numeric data but is stored as text")
    
    # Check for formatting issues
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check for inconsistent spacing
            has_leading_space = df[col].astype(str).str.startswith(' ').any()
            has_trailing_space = df[col].astype(str).str.endswith(' ').any()
            
            if has_leading_space or has_trailing_space:
                issues["Formatting Issues"].append(f"Column '{col}' has inconsistent spacing")
    
    return {k: v for k, v in issues.items() if v}


def render_cell_editor(df, doc_name):
    """Render individual cell editing interface"""
    st.markdown("**Edit individual cells by selecting row and column:**")
    
    if df.empty:
        st.info("No data to edit")
        return df
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        selected_row = st.selectbox("Select Row", range(len(df)), format_func=lambda x: f"Row {x+1}")
    
    with col2:
        selected_col = st.selectbox("Select Column", df.columns)
    
    with col3:
        current_value = df.iloc[selected_row, df.columns.get_loc(selected_col)]
        new_value = st.text_input("New Value", value=str(current_value), key=f"cell_edit_{doc_name}")
        
        if st.button("Update Cell", key=f"update_cell_{doc_name}"):
            df.iloc[selected_row, df.columns.get_loc(selected_col)] = new_value
            st.success(f"✅ Updated cell at Row {selected_row+1}, Column '{selected_col}'")
            st.rerun()
    
    return df


def render_delete_operations(df, doc_name):
    """Render row and column deletion interface"""
    st.markdown("**Delete rows or columns:**")
    
    if df.empty:
        st.info("No data to delete")
        return df
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Delete Rows:**")
        if len(df) > 0:
            rows_to_delete = st.multiselect(
                "Select rows to delete",
                range(len(df)),
                format_func=lambda x: f"Row {x+1}: {' | '.join(str(df.iloc[x, i])[:20] for i in range(min(3, len(df.columns))))}"
            )
            
            if rows_to_delete and st.button("🗑️ Delete Selected Rows", key=f"delete_rows_{doc_name}"):
                df = df.drop(df.index[rows_to_delete]).reset_index(drop=True)
                st.success(f"✅ Deleted {len(rows_to_delete)} rows")
                st.rerun()
    
    with col2:
        st.markdown("**Delete Columns:**")
        if len(df.columns) > 1:  # Keep at least one column
            cols_to_delete = st.multiselect("Select columns to delete", df.columns)
            
            if cols_to_delete and st.button("🗑️ Delete Selected Columns", key=f"delete_cols_{doc_name}"):
                df = df.drop(columns=cols_to_delete)
                st.success(f"✅ Deleted {len(cols_to_delete)} columns")
                st.rerun()
        else:
            st.info("Cannot delete - need at least one column")
    
    return df


def render_add_operations(df, doc_name):
    """Render row and column addition interface"""
    st.markdown("**Add new rows or columns:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Add New Row:**")
        if st.button("➕ Add Empty Row", key=f"add_row_{doc_name}"):
            new_row = pd.Series([''] * len(df.columns), index=df.columns)
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
            st.success("✅ Added new empty row")
            st.rerun()
        
        if len(df) > 0 and st.button("📋 Duplicate Last Row", key=f"dup_row_{doc_name}"):
            last_row = df.iloc[-1].copy()
            df = pd.concat([df, last_row.to_frame().T], ignore_index=True)
            st.success("✅ Duplicated last row")
            st.rerun()
    
    with col2:
        st.markdown("**Add New Column:**")
        new_col_name = st.text_input("Column Name", key=f"new_col_name_{doc_name}")
        default_value = st.text_input("Default Value", key=f"new_col_value_{doc_name}")
        
        if new_col_name and st.button("➕ Add Column", key=f"add_col_{doc_name}"):
            if new_col_name not in df.columns:
                df[new_col_name] = default_value
                st.success(f"✅ Added column '{new_col_name}'")
                st.rerun()
            else:
                st.error("Column name already exists!")
    
    return df


def render_bulk_operations(df, doc_name):
    """Render bulk editing operations (with Rename Column and Replace Top Row options)"""
    st.markdown("**Bulk operations for multiple cells:**")
    
    if df.empty:
        st.info("No data for bulk operations")
        return df
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Find & Replace:**")
        find_text = st.text_input("Find", key=f"find_{doc_name}")
        replace_text = st.text_input("Replace with", key=f"replace_{doc_name}")
        selected_columns = st.multiselect("In columns (leave empty for all)", df.columns)
        
        if find_text and st.button("🔄 Replace All", key=f"replace_all_{doc_name}"):
            cols_to_process = selected_columns if selected_columns else df.columns
            count = 0
            for col in cols_to_process:
                # operate on any dtype but convert to str for contains/replace
                mask = df[col].astype(str).str.contains(find_text, na=False)
                if mask.any():
                    # replace only on masked rows
                    df.loc[mask, col] = df.loc[mask, col].astype(str).str.replace(find_text, replace_text, regex=False)
                    count += int(mask.sum())
            st.success(f"✅ Replaced {count} occurrences")
            st.rerun()
    
    with col2:
        st.markdown("**Column Operations:**")
        target_column = st.selectbox("Select Column", df.columns, key=f"bulk_col_{doc_name}")
        
        operation = st.selectbox("Operation", [
            "Trim whitespace",
            "Convert to uppercase",
            "Convert to lowercase",
            "Remove special characters",
            "Convert to numbers"
        ], key=f"bulk_op_{doc_name}")
        
        if st.button("🔧 Apply Operation", key=f"apply_op_{doc_name}"):
            if operation == "Trim whitespace":
                df[target_column] = df[target_column].astype(str).str.strip()
            elif operation == "Convert to uppercase":
                df[target_column] = df[target_column].astype(str).str.upper()
            elif operation == "Convert to lowercase":
                df[target_column] = df[target_column].astype(str).str.lower()
            elif operation == "Remove special characters":
                df[target_column] = df[target_column].astype(str).str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
            elif operation == "Convert to numbers":
                try:
                    df[target_column] = pd.to_numeric(df[target_column].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
                except Exception:
                    st.error("Could not convert to numbers")
            
            st.success(f"✅ Applied '{operation}' to column '{target_column}'")
            st.rerun()
    
    # ----------------------------
    # Rename column section (new)
    # ----------------------------
    st.markdown("---")
    st.subheader("🔁 Rename Column")
    rename_col1, rename_col2, rename_col3 = st.columns([2, 2, 1])
    
    with rename_col1:
        col_to_rename = st.selectbox("Select column to rename", df.columns, key=f"rename_select_{doc_name}")
    with rename_col2:
        new_col_name = st.text_input("New column name", key=f"rename_input_{doc_name}")
    with rename_col3:
        if st.button("✏️ Rename", key=f"rename_btn_{doc_name}"):
            # Validation checks
            if not new_col_name or new_col_name.strip() == "":
                st.error("Column name cannot be empty")
            elif new_col_name in df.columns:
                st.error(f"A column named '{new_col_name}' already exists")
            else:
                df = df.rename(columns={col_to_rename: new_col_name})
                st.success(f"✅ Renamed column '{col_to_rename}' → '{new_col_name}'")
                st.rerun()
    
    # ----------------------------
    # Replace top row with column names (new)
    # ----------------------------
    st.markdown("---")
    st.subheader("🔃 Replace Top Row with Column Names")
    rc1, rc2, rc3 = st.columns([2, 2, 1])
    
    with rc1:
        row_index_for_header = st.number_input(
            "Row index to use as header (0-based)",
            min_value=0,
            max_value=max(0, len(df)-1),
            value=0,
            step=1,
            key=f"header_row_idx_{doc_name}"
        )
    with rc2:
        trim_headers = st.checkbox("Trim whitespace from header values", value=True, key=f"trim_headers_{doc_name}")
    with rc3:
        if st.button("🔁 Replace headers", key=f"replace_headers_{doc_name}"):
            # Safety checks
            if len(df) == 0:
                st.error("Cannot replace headers on an empty table")
            elif row_index_for_header < 0 or row_index_for_header >= len(df):
                st.error("Selected row index is out of range")
            else:
                # read the chosen row as strings
                new_headers = df.iloc[row_index_for_header].astype(str).tolist()
                if trim_headers:
                    new_headers = [h.strip() for h in new_headers]
                # ensure uniqueness by appending index to duplicates
                seen = {}
                final_headers = []
                for i, h in enumerate(new_headers):
                    if h == "" or h.lower() == "nan":
                        h = f"col_{i+1}"
                    base = h
                    if h in seen:
                        seen[h] += 1
                        h = f"{base}_{seen[base]}"
                    else:
                        seen[h] = 0
                    final_headers.append(h)
                # apply and drop the header row (optional: keep it)
                df.columns = final_headers
                # drop the row used as header to avoid duplicate header row in data
                df = df.drop(df.index[row_index_for_header]).reset_index(drop=True)
                st.success("✅ Replaced table headers from selected row and removed that row from data")
                st.rerun()
    
    return df



def auto_clean_data(df):
    """Automatically clean common data issues"""
    cleaned_df = df.copy()
    
    # Remove completely empty rows and columns
    cleaned_df = cleaned_df.dropna(how='all').dropna(axis=1, how='all')
    
    # Strip whitespace from text columns
    for col in cleaned_df.select_dtypes(include=['object']).columns:
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    # Try to convert numeric-looking columns
    for col in cleaned_df.select_dtypes(include=['object']).columns:
        # Check if column looks numeric
        sample_values = cleaned_df[col].dropna().astype(str)
        if len(sample_values) > 0:
            numeric_count = 0
            for val in sample_values:
                try:
                    float(val.replace(',', '').replace('$', '').replace('%', ''))
                    numeric_count += 1
                except:
                    pass
            
            if numeric_count > len(sample_values) * 0.8:  # 80% numeric
                try:
                    cleaned_df[col] = pd.to_numeric(
                        cleaned_df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True),
                        errors='coerce'
                    )
                except:
                    pass
    
    return cleaned_df


def validate_table_data(df):
    """Comprehensive table validation"""
    results = {
        'passed': [],
        'warnings': [],
        'errors': []
    }
    
    # Check basic structure
    if len(df) == 0:
        results['errors'].append("Table is empty")
        return results
    
    if len(df.columns) == 0:
        results['errors'].append("Table has no columns")
        return results
    
    results['passed'].append(f"Table structure: {len(df)} rows × {len(df.columns)} columns")
    
    # Check for missing data
    missing_data = df.isnull().sum()
    total_missing = missing_data.sum()
    
    if total_missing == 0:
        results['passed'].append("No missing values found")
    else:
        missing_pct = (total_missing / (len(df) * len(df.columns))) * 100
        if missing_pct > 20:
            results['warnings'].append(f"High missing data: {missing_pct:.1f}% of cells are empty")
        else:
            results['warnings'].append(f"Some missing data: {total_missing} empty cells ({missing_pct:.1f}%)")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates == 0:
        results['passed'].append("No duplicate rows found")
    else:
        results['warnings'].append(f"Found {duplicates} duplicate rows")
    
    # Check data consistency
    for col in df.columns:
        unique_count = df[col].nunique()
        if unique_count == 1:
            results['warnings'].append(f"Column '{col}' has only one unique value")
        elif unique_count == len(df):
            results['passed'].append(f"Column '{col}' has all unique values")
    
    return results


def display_validation_results(results):
    """Display validation results in a user-friendly format"""
    if results['errors']:
        st.error("❌ Validation Errors:")
        for error in results['errors']:
            st.error(f"• {error}")
    
    if results['warnings']:
        st.warning("⚠️ Validation Warnings:")
        for warning in results['warnings']:
            st.warning(f"• {warning}")
    
    if results['passed']:
        st.success("✅ Validation Passed:")
        for passed in results['passed']:
            st.success(f"• {passed}")


def render_data_statistics(df):
    """Render comprehensive data statistics"""
    if df.empty:
        return
    
    st.markdown("---")
    st.subheader("📈 Data Statistics & Preview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Basic Information:**")
        st.write(f"• **Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.write(f"• **Memory usage:** {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        st.write(f"• **Missing values:** {df.isnull().sum().sum()}")
        st.write(f"• **Duplicate rows:** {df.duplicated().sum()}")
    
    with col2:
        st.markdown("**Column Information:**")
        for col in df.columns:
            dtype = df[col].dtype
            unique_count = df[col].nunique()
            st.write(f"• **{col}:** {dtype} ({unique_count} unique)")
    
    # Data preview with pagination
    st.markdown("**Data Preview:**")
    
    # Pagination controls
    rows_per_page = st.selectbox("Rows per page", [5, 10, 20, 50], index=1)
    total_pages = (len(df) - 1) // rows_per_page + 1
    
    if total_pages > 1:
        page = st.selectbox("Page", range(1, total_pages + 1))
        start_idx = (page - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, len(df))
        display_df = df.iloc[start_idx:end_idx]
        st.write(f"Showing rows {start_idx + 1}-{end_idx} of {len(df)}")
    else:
        display_df = df.head(rows_per_page)
    
    st.dataframe(display_df, width='stretch')