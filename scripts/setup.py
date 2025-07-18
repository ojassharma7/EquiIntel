#!/usr/bin/env python3
"""
EquIntel Setup Script
Initializes the environment and downloads necessary components
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import platform


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)


def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "raw_pdfs",
        "parsed_data", 
        "embeddings",
        "metadata",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✅ Created {directory}/")


def setup_environment():
    """Set up environment configuration"""
    print("⚙️ Setting up environment...")
    
    env_file = Path(".env")
    example_env = Path("config.example.env")
    
    if not env_file.exists() and example_env.exists():
        shutil.copy(example_env, env_file)
        print("✅ Created .env file from template")
        print("📝 Please edit .env file with your configuration")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("⚠️ No .env template found, creating basic .env file")
        with open(env_file, 'w') as f:
            f.write("# EquIntel Configuration\n")
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
            f.write("LLM_TYPE=openai\n")


def download_embedding_model():
    """Download the embedding model"""
    print("🧠 Downloading embedding model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # This will download the model on first use
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model ready")
        
    except Exception as e:
        print(f"⚠️ Embedding model will be downloaded on first use: {e}")


def test_installation():
    """Test the installation"""
    print("🧪 Testing installation...")
    
    try:
        # Test imports
        import lancedb
        import sentence_transformers
        import streamlit
        import langchain
        import pymupdf
        import pdfplumber
        
        print("✅ All core dependencies imported successfully")
        
        # Test vector store
        from app.vector_store import VectorStore
        vector_store = VectorStore()
        print("✅ Vector store initialized successfully")
        
        # Test document parser
        from app.document_parser import DocumentParser
        parser = DocumentParser()
        print("✅ Document parser initialized successfully")
        
        print("✅ Installation test passed!")
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False
    
    return True


def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*50)
    print("🚀 EquIntel Setup Complete!")
    print("="*50)
    
    print("\n📋 Next Steps:")
    print("1. Edit .env file with your OpenAI API key (if using OpenAI)")
    print("2. Add PDF documents to the raw_pdfs/ directory")
    print("3. Run the pipeline: python scripts/pipeline.py --action pipeline")
    print("4. Or start the web UI: streamlit run ui/app.py")
    
    print("\n🔧 Quick Start Commands:")
    print("  # Download sample data")
    print("  python scripts/download_sample_data.py")
    print("")
    print("  # Process documents")
    print("  python scripts/pipeline.py --action pipeline")
    print("")
    print("  # Run a query")
    print("  python scripts/pipeline.py --action query --query 'What are the main risks mentioned?'")
    print("")
    print("  # Start web interface")
    print("  streamlit run ui/app.py")
    
    print("\n📚 Documentation:")
    print("  - README.md: Project overview and usage")
    print("  - scripts/pipeline.py --help: Command line options")
    
    print("\n🎯 Example Queries:")
    print("  - 'What are the main risks mentioned in NVIDIA documents?'")
    print("  - 'Compare revenue growth between different companies'")
    print("  - 'What are the key AI initiatives mentioned in earnings calls?'")


def main():
    """Main setup function"""
    print("📊 EquIntel Setup")
    print("="*30)
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    install_dependencies()
    
    # Create directories
    create_directories()
    
    # Setup environment
    setup_environment()
    
    # Download embedding model
    download_embedding_model()
    
    # Test installation
    if test_installation():
        print_next_steps()
    else:
        print("❌ Setup completed with errors. Please check the output above.")


if __name__ == "__main__":
    main() 