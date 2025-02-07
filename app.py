import streamlit as st
import pandas as pd
from utils.menu_processor import MenuProcessor
from utils.substitutions import get_substitution_rules, add_substitution_rule
from utils.database import init_db, get_db, SubstitutionRule
from typing import Generator
import io

# Initialize database
init_db()

def get_db_session() -> Generator:
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()

def main():
    st.set_page_config(
        page_title="Menu Allergen Converter",
        page_icon="🍽️",
        layout="wide"
    )

    st.title("🍽️ School Menu Allergen Converter")
    st.write("Transform your school menu into allergen-free versions while maintaining the original format.")

    # Get database session
    db = next(get_db_session())

    # Sidebar for configuration
    st.sidebar.title("Settings")

    # Allergen selection
    allergens = st.sidebar.multiselect(
        "Select allergens to exclude:",
        ["Gluten", "Dairy", "Nuts", "Eggs", "Soy"],
        default=["Gluten", "Dairy"]
    )

    # Add custom substitution rules
    st.sidebar.subheader("Add Custom Substitution")
    with st.sidebar.form("new_rule"):
        allergen = st.selectbox("Allergen", ["Gluten", "Dairy", "Nuts", "Eggs", "Soy"])
        original = st.text_input("Original ingredient")
        replacement = st.text_input("Replacement ingredient")

        if st.form_submit_button("Add Rule"):
            if original and replacement:
                add_substitution_rule(allergen, original, replacement, db)
                st.success("Rule added successfully!")
            else:
                st.error("Please fill in both original and replacement ingredients.")

    # View existing custom rules
    st.sidebar.subheader("Custom Rules")
    custom_rules = db.query(SubstitutionRule).all()
    if custom_rules:
        for rule in custom_rules:
            st.sidebar.text(f"{rule.allergen}: {rule.original} → {rule.replacement}")
    else:
        st.sidebar.text("No custom rules added yet.")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload your menu file (CSV format)",
        type=["csv", "txt"]
    )

    if uploaded_file:
        try:
            # Read the file content
            content = uploaded_file.getvalue().decode('utf-8')

            # Initialize processor with raw content
            processor = MenuProcessor(content)

            # Get custom substitution rules with database access
            custom_rules = get_substitution_rules(allergens, db)

            # Process menu with both custom rules and allergens for AI processing
            modified_df, changes = processor.convert_menu(custom_rules, allergens)

            # Display original and modified menus side by side
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Menu")
                st.dataframe(processor.original_df, use_container_width=True)

            with col2:
                st.subheader("Allergen-Free Menu")
                st.dataframe(modified_df, use_container_width=True)

            # Display changes
            if changes:
                st.subheader("Changes Made")
                for change in changes:
                    st.info(change)

            # Export options
            st.subheader("Export Modified Menu")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Export as CSV"):
                    csv = modified_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="modified_menu.csv",
                        mime="text/csv"
                    )

            with col2:
                if st.button("Export as Excel"):
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        modified_df.to_excel(writer, index=False)

                    buffer.seek(0)
                    st.download_button(
                        label="Download Excel",
                        data=buffer,
                        file_name="modified_menu.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please ensure your file follows the expected format.")

    # Help section
    with st.expander("Help & Instructions"):
        st.markdown("""
        ### How to use this tool:
        1. Select the allergens you want to exclude using the sidebar
        2. Add custom substitution rules if needed
        3. Upload your menu file (CSV format)
        4. Review the changes in the side-by-side view
        5. Export the modified menu in your preferred format

        ### Menu File Format:
        Your menu file should be in CSV format with:
        - One row per day
        - Different weeks in separate columns
        - Each cell containing:
          - B: (Breakfast)
          - L: (Lunch)
          - S: (Snack)

        ### Supported Features:
        - Multiple allergen exclusions
        - Custom substitution rules
        - Automatic substitution suggestions
        - Change highlighting
        - Export to CSV or Excel
        """)

if __name__ == "__main__":
    main()