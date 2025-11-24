import streamlit as st
import pandas as pd
from utils.menu_processor import MenuProcessor
from utils.substitutions import get_substitution_rules, add_substitution_rule
from utils.database import init_db, get_db, SubstitutionRule
from utils.confetti import show_confetti
from utils.excel_exporter import export_to_excel
from typing import Generator
import hashlib
import streamlit.components.v1 as components

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

            # Add Run button
            st.markdown("---")
            run_button = st.button("🚀 Run Conversion", type="primary", width=300)

            # Only process when Run button is clicked
            if run_button:
                # Create placeholder for reasoning text display
                st.markdown("**AI Reasoning:**")
                reasoning_display = st.empty()
                reasoning_display.text("Waiting for AI to start reasoning...")
                
                # Accumulate reasoning text for display
                accumulated_reasoning = [""]
                saw_json_marker = [False]
                
                def update_reasoning(chunk: str):
                    """Callback to update reasoning display with new chunks in real-time"""
                    if chunk:
                        # If we've already hit the JSON marker, ignore further chunks for the reasoning box
                        if saw_json_marker[0]:
                            return
                        accumulated_reasoning[0] += chunk
                        # Stop updating the reasoning box once the JSON marker appears
                        if "===JSON===" in accumulated_reasoning[0]:
                            saw_json_marker[0] = True
                            accumulated_reasoning[0] = accumulated_reasoning[0].split("===JSON===")[0].rstrip()
                        # Update the placeholder with st.text() for real-time streaming
                        reasoning_display.text(accumulated_reasoning[0] if accumulated_reasoning[0] else "...")
                
                with st.spinner("Processing your menu..."):
                    # Get custom substitution rules with database access
                    custom_rules = get_substitution_rules(allergens, db)

                    # Process menu with both custom rules and allergens for AI processing
                    # Pass progress callback for streaming reasoning
                    modified_df, changes, summary = processor.convert_menu(
                        custom_rules, allergens, progress_callback=update_reasoning
                    )
                
                # Final update - convert to text area for better readability after completion
                if accumulated_reasoning[0]:
                    reasoning_display.text_area(
                        "Reasoning",
                        value=accumulated_reasoning[0],
                        height=200,
                        disabled=True,
                        label_visibility="collapsed",
                        key="reasoning_display_final"
                    )

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
                    # Play a short sound and show a notification in the browser
                    try:
                        components.html("""
<script>
(function(){
  try {
    const ctx = new (window.AudioContext||window.webkitAudioContext)();
    const o = ctx.createOscillator();
    const g = ctx.createGain();
    o.type='sine'; o.frequency.value=880;
    o.connect(g); g.connect(ctx.destination);
    g.gain.setValueAtTime(0.0001, ctx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.2, ctx.currentTime+0.01);
    o.start();
    g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime+0.25);
    o.stop(ctx.currentTime+0.3);
  } catch(e){}
  try {
    if ("Notification" in window) {
      if (Notification.permission === "granted") {
        new Notification("Allergen substitutions ready");
      } else if (Notification.permission !== "denied") {
        Notification.requestPermission().then(p=>{
          if (p==="granted"){ new Notification("Allergen substitutions ready"); }
        });
      }
    }
  } catch(e){}
})();
</script>
""", height=0)
                    except Exception:
                        pass

            # Display results if they exist in session state
            if st.session_state.processed_results:
                results = st.session_state.processed_results
                
                # Display substitutions list
                summary = results.get('summary', {})
                replaced_meals = summary.get('replaced', [])
                unreplaced_meals = summary.get('unreplaced', [])

                if replaced_meals:
                    st.subheader("Substitutions Made")
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
                    width=320
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