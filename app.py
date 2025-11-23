import streamlit as st
import pandas as pd
from utils.menu_processor import MenuProcessor
from utils.substitutions import get_substitution_rules, add_substitution_rule
from utils.database import init_db, get_db, SubstitutionRule
from utils.confetti import show_confetti
from utils.excel_exporter import export_to_excel
from typing import Generator
import hashlib

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

    # Initialize session state
    if 'processed_results' not in st.session_state:
        st.session_state.processed_results = None
    if 'current_file_hash' not in st.session_state:
        st.session_state.current_file_hash = None
    if 'current_allergens' not in st.session_state:
        st.session_state.current_allergens = None

    # Get database session
    db = next(get_db_session())

    # Sidebar for configuration
    st.sidebar.title("Settings")

    # Allergen selection
    allergens = st.sidebar.multiselect(
        "Select allergens to exclude:",
        ["Gluten", "Dairy", "Nuts", "Egg Products", "Soy", "Fish"],
        default=["Gluten", "Dairy"]
    )

    # Add custom substitution rules
    st.sidebar.subheader("Add Custom Substitution")
    with st.sidebar.form("new_rule"):
        allergen = st.selectbox("Allergen", ["Gluten", "Dairy", "Nuts", "Egg Products", "Soy", "Fish"])
        original = st.text_input("Original ingredient")
        replacement = st.text_input("Replacement ingredient")

        if st.form_submit_button("Add Rule"):
            if original and replacement:
                add_substitution_rule(allergen, original, replacement, db)
                st.success("Rule added successfully!")
                show_confetti()
            else:
                st.error("Please fill in both original and replacement ingredients.")

    # View existing custom rules with delete option - organized by allergen
    st.sidebar.subheader("Custom Rules")
    custom_rules = db.query(SubstitutionRule).all()
    if custom_rules:
        # Group rules by allergen
        allergen_groups = {}
        for rule in custom_rules:
            if rule.allergen not in allergen_groups:
                allergen_groups[rule.allergen] = []
            allergen_groups[rule.allergen].append(rule)
        
        # Display rules grouped by allergen
        for allergen in sorted(allergen_groups.keys()):
            st.sidebar.markdown(f"**{allergen}**")
            for rule in allergen_groups[allergen]:
                col1, col2 = st.sidebar.columns([4, 1])
                with col1:
                    st.text(f"{rule.original} → {rule.replacement}")
                with col2:
                    if st.button("🗑️", key=f"delete_{rule.id}"):
                        from utils.substitutions import delete_substitution_rule
                        if delete_substitution_rule(rule.id, db):
                            st.success("Rule deleted!")
                            st.rerun()
            st.sidebar.divider()
    else:
        st.sidebar.text("No custom rules added yet.")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload your menu file (Excel format)",
        type=["xlsm", "xlsx", "xls"]
    )

    if uploaded_file:
        try:
            # Read the file content
            content_bytes = uploaded_file.getvalue()

            # Calculate hash of file content to detect changes
            file_hash = hashlib.md5(content_bytes).hexdigest()
            
            # Check if this is a new file or settings changed - clear results if so
            allergens_tuple = tuple(sorted(allergens))
            if (file_hash != st.session_state.current_file_hash or
                allergens_tuple != st.session_state.current_allergens):
                st.session_state.processed_results = None
                st.session_state.current_file_hash = file_hash
                st.session_state.current_allergens = allergens_tuple

            # Initialize processor with raw content
            processor = MenuProcessor(content_bytes)

            # Show original menu preview
            st.subheader("Original Menu Preview")
            st.dataframe(processor.original_df, use_container_width=True)

            # Add Run button
            st.markdown("---")
            run_button = st.button("🚀 Run Conversion", type="primary", use_container_width=True)

            # Only process when Run button is clicked
            if run_button:
                with st.spinner("Processing your menu..."):
                    # Get custom substitution rules with database access
                    custom_rules = get_substitution_rules(allergens, db)

                    # Process menu with both custom rules and allergens for AI processing
                    modified_df, changes, summary = processor.convert_menu(custom_rules, allergens)

                    # Store results in session state including the processor for Excel export
                    st.session_state.processed_results = {
                        'modified_df': modified_df,
                        'changes': changes,
                        'original_df': processor.original_df,
                        'processor': processor,
                        'summary': summary
                    }
                    
                    # Show confetti for successful processing
                    if changes:
                        show_confetti()

            # Display results if they exist in session state
            if st.session_state.processed_results:
                results = st.session_state.processed_results
                
                # Display original and modified menus side by side
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Original Menu")
                    st.dataframe(results['original_df'], use_container_width=True)

                with col2:
                    st.subheader("Allergen-Free Menu")
                    st.dataframe(results['modified_df'], use_container_width=True)

                # Display changes
                summary = results.get('summary', {})
                replaced_meals = summary.get('replaced', [])
                unreplaced_meals = summary.get('unreplaced', [])

                if replaced_meals:
                    st.subheader("Replaced meals")
                    for item in replaced_meals:
                        st.success(
                            f"Week {item['week']} {item['day']} ({item['meal_type']}): "
                            f"{item['original']} → {item['replacement']}"
                        )

                if unreplaced_meals:
                    st.subheader("Unreplaced meals (no allergen conflicts)")
                    for item in unreplaced_meals:
                        st.info(
                            f"Week {item['week']} {item['day']} ({item['meal_type']}): {item['text']}"
                        )

                if results['changes']:
                    st.subheader("Raw change log")
                    for change in results['changes']:
                        st.caption(change)

                # Export options
                st.subheader("Export Modified Menu")
                
                # Generate Excel file with red highlighting for substitutions
                excel_buffer = export_to_excel(results['modified_df'], results['processor'])
                
                st.download_button(
                    label="📥 Download Excel (with highlighted substitutions)",
                    data=excel_buffer,
                    file_name="modified_menu.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    on_click=show_confetti,
                    use_container_width=True
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
        3. Upload your menu file (Excel format)
        4. Click the "Run Conversion" button to process
        5. Review the changes in the side-by-side view
        6. Download the Excel file with highlighted substitutions

        ### Menu File Format:
        Upload an Excel file that matches the provided `Menu_Allergy_Sub.xlsm` template:
        - Columns for Week 1-4 along the top row
        - Monday-Friday listed down the left side
        - Each meal cell contains three labeled lines:
          - `B:` for Breakfast
          - `L:` for Lunch
          - `S:` for Snack

        ### Supported Features:
        - Multiple allergen exclusions
        - Custom substitution rules
        - AI-powered substitution suggestions
        - Excel export with red highlighting for substituted ingredients
        - Original Excel-style layout preserved in output
        """)


if __name__ == "__main__":
    main()