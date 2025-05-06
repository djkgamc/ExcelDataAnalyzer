import streamlit as st

def show_confetti():
    """Show a confetti animation using JavaScript."""
    # Canvas confetti JS library CDN link
    confetti_js = "https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"
    
    # JavaScript for triggering confetti - using double braces to escape JavaScript curly braces in f-string
    confetti_code = f"""
    <script src="{confetti_js}"></script>
    <script>
        // Trigger confetti
        var count = 200;
        var defaults = {{
            origin: {{ y: 0.7 }},
            zIndex: 10000
        }};

        function fire(particleRatio, opts) {{
            confetti(Object.assign({{}}, defaults, opts, {{
                particleCount: Math.floor(count * particleRatio)
            }}));
        }}

        fire(0.25, {{
            spread: 26,
            startVelocity: 55,
        }});
        fire(0.2, {{
            spread: 60,
        }});
        fire(0.35, {{
            spread: 100,
            decay: 0.91,
            scalar: 0.8
        }});
        fire(0.1, {{
            spread: 120,
            startVelocity: 25,
            decay: 0.92,
            scalar: 1.2
        }});
        fire(0.1, {{
            spread: 120,
            startVelocity: 45,
        }});
    </script>
    """
    
    # Display using html component
    st.components.v1.html(confetti_code, height=0)