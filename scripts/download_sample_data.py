#!/usr/bin/env python3
"""
Download Sample Data for EquIntel
Downloads sample financial documents for testing the system
"""

import os
import requests
from pathlib import Path
import zipfile
import tempfile


def download_file(url: str, filename: str, directory: str = "raw_pdfs"):
    """Download a file from URL"""
    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    
    file_path = directory / filename
    
    if file_path.exists():
        print(f"✅ {filename} already exists, skipping...")
        return str(file_path)
    
    print(f"📥 Downloading {filename}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Downloaded {filename}")
        return str(file_path)
    
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        return None


def download_sample_documents():
    """Download sample financial documents"""
    print("🚀 Downloading sample financial documents...")
    
    # Sample documents (these are example URLs - replace with actual ones)
    sample_docs = [
        {
            "name": "sample_10k.pdf",
            "url": "https://example.com/sample_10k.pdf",
            "description": "Sample 10-K filing"
        },
        {
            "name": "sample_earnings_call.pdf", 
            "url": "https://example.com/sample_earnings_call.pdf",
            "description": "Sample earnings call transcript"
        }
    ]
    
    downloaded_files = []
    
    for doc in sample_docs:
        print(f"\n📄 {doc['description']}")
        file_path = download_file(doc['url'], doc['name'])
        if file_path:
            downloaded_files.append(file_path)
    
    print(f"\n✅ Downloaded {len(downloaded_files)} sample documents")
    return downloaded_files


def create_sample_text_files():
    """Create sample text files for testing"""
    print("📝 Creating sample text files...")
    
    raw_pdfs_dir = Path("raw_pdfs")
    raw_pdfs_dir.mkdir(exist_ok=True)
    
    # Sample 10-K content
    sample_10k = """
    NVIDIA CORPORATION
    10-K ANNUAL REPORT
    FISCAL YEAR ENDED JANUARY 29, 2023
    
    ITEM 1. BUSINESS
    
    NVIDIA Corporation ("NVIDIA," "we," "us," "our," or the "Company") is a world leader in visual computing. We pioneered a supercharged form of computing loved by the most demanding computer users in the world — scientists, designers, artists, and gamers. It was the first to bring programmable shading to consumers, and it is the only company with substantial expertise in both the visual and artificial intelligence, or AI, sides of computer graphics.
    
    Our two platforms address four large and growing markets — Gaming, Data Center, Professional Visualization, and Automotive. We serve these markets with our GPU, Tegra processor, and related software, which are based on our proprietary technologies.
    
    RISK FACTORS
    
    Our business is subject to numerous risks and uncertainties, including those described in this section, that could adversely affect our business, financial condition, results of operations, and cash flows. These risks include, but are not limited to, the following:
    
    • We depend on third parties to manufacture, assemble, package, and test our products, and if these third parties are unable to do so on a timely basis or at all, our business could be harmed.
    • Our business is highly dependent on the success of our GPU products, and if we are unable to successfully develop and introduce new products, our business could be harmed.
    • We face intense competition in our markets, and if we are unable to compete effectively, our business could be harmed.
    """
    
    # Sample earnings call content
    sample_earnings_call = """
    NVIDIA CORPORATION
    Q4 2023 EARNINGS CALL TRANSCRIPT
    FEBRUARY 22, 2023
    
    PARTICIPANTS
    Colette Kress - Executive Vice President and Chief Financial Officer
    Jensen Huang - Founder, President and Chief Executive Officer
    
    OPERATOR: Good day, and welcome to the NVIDIA Fourth Quarter Fiscal Year 2023 Earnings Conference Call. All participants will be in listen-only mode. After today's presentation, there will be an opportunity to ask questions.
    
    JENSEN HUANG: Thank you, operator. Good afternoon, everyone. Thank you for joining us today. I'm Jensen Huang, NVIDIA's founder and CEO.
    
    We had a strong finish to fiscal 2023, with record revenue of $6.05 billion, up 2% sequentially and down 21% year-over-year. Gaming revenue was $1.83 billion, down 46% year-over-year. Data Center revenue was $3.62 billion, up 11% year-over-year.
    
    The AI revolution is driving exponential growth in computing requirements, and NVIDIA is at the center of it. Our Data Center platform is powering the AI revolution across every industry.
    
    We're seeing strong demand for our AI infrastructure, particularly for large language models and generative AI applications. Our H100 GPU is in high demand, and we're working to increase supply to meet customer needs.
    
    Looking ahead, we expect continued strong growth in our Data Center business as AI adoption accelerates across industries.
    """
    
    # Write sample files
    with open(raw_pdfs_dir / "NVIDIA_10K_2023.txt", 'w') as f:
        f.write(sample_10k)
    
    with open(raw_pdfs_dir / "NVIDIA_Earnings_Q4_2023.txt", 'w') as f:
        f.write(sample_earnings_call)
    
    print("✅ Created sample text files")
    return [
        str(raw_pdfs_dir / "NVIDIA_10K_2023.txt"),
        str(raw_pdfs_dir / "NVIDIA_Earnings_Q4_2023.txt")
    ]


def main():
    """Main function"""
    print("📊 EquIntel Sample Data Downloader")
    print("=" * 40)
    
    # Create sample text files (since we can't download actual PDFs)
    sample_files = create_sample_text_files()
    
    print(f"\n📁 Sample files created in raw_pdfs/ directory:")
    for file_path in sample_files:
        print(f"  - {Path(file_path).name}")
    
    print("\n🚀 Next steps:")
    print("1. Run the pipeline: python scripts/pipeline.py --action pipeline")
    print("2. Or start the UI: streamlit run ui/app.py")
    print("3. Or run a query: python scripts/pipeline.py --action query --query 'What are the main risks mentioned in NVIDIA documents?'")


if __name__ == "__main__":
    main() 