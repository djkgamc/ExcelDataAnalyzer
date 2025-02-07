import streamlit as st
import pandas as pd
from utils.menu_processor import MenuProcessor
from utils.substitutions import get_substitution_rules

def main():
    st.set_page_config(
        page_title="Menu Allergen Converter",
        page_icon="🍽️",
        layout="wide"
    )

    st.title("🍽️ School Menu Allergen Converter")
    st.write("Transform your school menu into allergen-free versions while maintaining the original format.")

    # Sidebar for configuration
    st.sidebar.title("Settings")
    
    # Allergen selection
    allergens = st.sidebar.multiselect(
        "Select allergens to exclude:",
        ["Gluten", "Dairy", "Nuts", "Eggs", "Soy"],
        default=["Gluten"]
    )

    # File upload
    uploaded_file = st.file_uploader(
        "Upload your menu file (CSV or Excel)",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file:
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Initialize processor
            processor = MenuProcessor(df)
            
            # Get substitution rules
            rules = get_substitution_rules(allergens)
            
            # Process menu
            modified_df, changes = processor.convert_menu(rules)

            # Display original and modified menus side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Menu")
                st.dataframe(df, use_container_width=True)
            
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
                    output = modified_df.to_excel(index=False)
                    st.download_button(
                        label="Download Excel",
                        data=output,
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
        2. Upload your menu file (CSV or Excel format)
        3. Review the changes in the side-by-side view
        4. Export the modified menu in your preferred format
        
        ### Supported Features:
        - Multiple allergen exclusions
        - Automatic substitution suggestions
        - Change highlighting
        - Export to CSV or Excel
        """)

if __name__ == "__main__":
    main()
