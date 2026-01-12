"""
Main entry point for the Transformer Translation Application
"""
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    from src.web.app import app
    
    print("Starting Transformer Translation Web App...")
    print("Visit: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
