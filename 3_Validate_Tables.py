import streamlit as st
import pandas as pd
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Validate and Transform Tables",
    page_icon="✅",
    layout="wide"
)

st.title("✅ Validate and Transform Tables")
st.write("Here you can review the extracted table data, clean it up, and make corrections before exporting.")

# --- Helper Functions ---
def initialize_session_state():
    """Initialize a sample DataFrame in session state if not present."""
    if 'edited_df' not in st.session_state:
        # In a real app, this data would come from a previous OCR step
        data = {
            'Column 1': ['Header A', 'Data 1A', 'Data 2A', 'Data 3A'],
            'Column 2': ['Header B', 'Data 1B', 'Data 2B', 'Data 3B'],
            'Column 3': ['Header C', 'Data 1C', 'Data 2C', 'Data 3C'],
        }
        df = pd.DataFrame(data)
        st.session_state.edited_df = df
        st.session_state.original_df = df.copy()

def reset_data():
    """Reset the DataFrame to its original state."""
    st.session_state.edited_df = st.session_state.original_df.copy()
    st.success("Data has been reset to its original state.")
    st.rerun()

# --- Main App ---
initialize_session_state()

# --- Sidebar for Actions ---
with st.sidebar:
    st.header("Transformation Tools")
    st.write("Use these tools to clean your data.")

    if st.button("🔄 Reset to Original"):
        reset_data()

    # 1. Promote Headers
    st.subheader("Promote Headers")
    if st.button("Use First Row as Headers"):
        df = st.session_state.edited_df.copy()
        if not df.empty and len(df) > 0:
            new_header = df.iloc[0]
            df = df[1:].reset_index(drop=True)
            df.columns = new_header
            st.session_state.edited_df = df
            st.success("First row has been promoted to headers.")
            st.rerun()
        else:
            st.warning("Table is empty or has no rows to promote.")

    # 2. Delete Rows
    st.subheader("Delete Rows")
    df = st.session_state.edited_df
    if not df.empty:
        rows_to_delete = st.multiselect(
            "Select row(s) to delete (by index):",
            options=df.index.tolist(),
            key="delete_rows_multiselect"
        )
        if st.button("Delete Selected Rows") and rows_to_delete:
            df = df.drop(rows_to_delete).reset_index(drop=True)
            st.session_state.edited_df = df
            st.success(f"Deleted rows: {rows_to_delete}")
            st.rerun()

    # 3. Delete Columns
    st.subheader("Delete Columns")
    if not df.empty:
        cols_to_delete = st.multiselect(
            "Select column(s) to delete:",
            options=df.columns.tolist(),
            key="delete_cols_multiselect"
        )
        if st.button("Delete Selected Columns") and cols_to_delete:
            df = df.drop(columns=cols_to_delete)
            st.session_state.edited_df = df
            st.success(f"Deleted columns: {cols_to_delete}")
            st.rerun()

    # 4. Rename Columns
    st.subheader("Rename Column")
    if not df.empty:
        col_to_rename = st.selectbox(
            "Select column to rename:",
            options=df.columns.tolist(),
            index=None,
            placeholder="Choose a column..."
        )
        if col_to_rename:
            new_col_name = st.text_input(
                f"New name for '{col_to_rename}':",
                value=col_to_rename,
                key=f"rename_{col_to_rename}"
            )
            if st.button(f"Rename '{col_to_rename}'"):
                if new_col_name and new_col_name != col_to_rename:
                    st.session_state.edited_df.rename(
                        columns={col_to_rename: new_col_name},
                        inplace=True
                    )
                    st.success(f"Renamed '{col_to_rename}' to '{new_col_name}'.")
                    st.rerun()
                elif not new_col_name:
                    st.warning("New column name cannot be empty.")
                else:
                    st.info("No changes made.")


# --- Display Table ---
st.header("Current Table Data")
st.info("You can directly edit cell values in the table below. Use the transformation tools in the sidebar for structural changes.")

# Use st.data_editor to allow for interactive editing
edited_df = st.data_editor(
    st.session_state.edited_d,
    num_rows="dynamic", # Allows adding/deleting rows from the editor UI
    use_container_width=True,
    key="data_editor"
)

# The output of st.data_editor is the modified dataframe.
# We update the session state if it has changed.
if not edited_df.equals(st.session_state.edited_df):
    st.session_state.edited_df = edited_df
    st.rerun()

st.markdown("---")
st.header("Export Data")
st.write("Once you are satisfied with the data, you can export it.")

col1, col2 = st.columns(2)

with col1:
    st.download_button(
        label="📥 Download as CSV",
        data=st.session_state.edited_df.to_csv(index=False).encode('utf-8'),
        file_name='validated_table.csv',
        mime='text/csv',
    )

with col2:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        st.session_state.edited_df.to_excel(writer, index=False, sheet_name='Sheet1')
    excel_data = output.getvalue()
    st.download_button(
        label="📥 Download as Excel",
        data=excel_data,
        file_name='validated_table.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


