"""
Integration Guide: Real Estate Dataset → RAG System
Dataset: USA Real Estate Dataset from Kaggle
URL: https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset
"""

import pandas as pd
import os
from pathlib import Path

# ========================================
# STEP 1: Download the dataset
# ========================================

"""
1. Go to: https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset
2. Click "Download" (you'll need a Kaggle account - free)
3. Extract realtor-data.zip
4. You'll get: realtor-data.csv (~280MB, 2.2M rows)

Or use Kaggle API:
pip install kaggle
kaggle datasets download -d ahmedshahriarsakib/usa-real-estate-dataset
unzip usa-real-estate-dataset.zip
"""

# ========================================
# STEP 2: Load and explore the data
# ========================================

def explore_dataset(csv_path: str = "realtor-data.csv"):
    """Quick exploration of the dataset."""
    
    df = pd.read_csv(csv_path)
    
    print("="*60)
    print("DATASET OVERVIEW")
    print("="*60)
    print(f"Total properties: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nSample row:")
    print(df.iloc[0].to_dict())
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    return df


# ========================================
# STEP 3: Convert to documents for RAG
# ========================================

def create_property_document(row: pd.Series) -> str:
    """
    Convert a DataFrame row into a text document for RAG.
    
    This is the key function - it creates natural language text
    that the LLM can understand and retrieve from.
    """
    
    # Extract key fields (handle missing values)
    address = f"{row.get('street', 'N/A')}, {row.get('city', 'N/A')}, {row.get('state', 'N/A')} {row.get('zip_code', 'N/A')}"
    price = f"${row.get('price', 0):,.0f}" if pd.notna(row.get('price')) else "Price not available"
    bedrooms = int(row.get('bed', 0)) if pd.notna(row.get('bed')) else 0
    bathrooms = row.get('bath', 0) if pd.notna(row.get('bath')) else 0
    sqft = f"{int(row.get('house_size', 0)):,}" if pd.notna(row.get('house_size')) else "N/A"
    acre_lot = row.get('acre_lot', 0) if pd.notna(row.get('acre_lot')) else 0
    
    # Create natural language document
    doc = f"""
Property Listing: {address}

Price: {price}
Bedrooms: {bedrooms} | Bathrooms: {bathrooms} | Square Feet: {sqft}
Lot Size: {acre_lot:.2f} acres

Property Type: {row.get('status', 'N/A')}
    """.strip()
    
    return doc


def generate_documents_from_dataset(
    csv_path: str = "realtor-data.csv",
    output_dir: str = "./documents",
    sample_size: int = 100,  # Start small for testing
    filter_city: str = None  # Optional: filter by city
):
    """
    Generate text documents from the CSV dataset.
    
    Args:
        csv_path: Path to realtor-data.csv
        output_dir: Where to save generated documents
        sample_size: How many properties to include
        filter_city: Optional city filter (e.g., "Charlottesville")
    """
    
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter by city if specified
    if filter_city:
        df = df[df['city'].str.contains(filter_city, case=False, na=False)]
        print(f"Filtered to {len(df)} properties in {filter_city}")
    
    # Sample if dataset is large
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} properties")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate documents
    print(f"\nGenerating documents...")
    for idx, row in df.iterrows():
        doc_text = create_property_document(row)
        
        # Create filename from address (sanitize for filesystem)
        city = str(row.get('city', 'unknown')).replace(' ', '_')
        state = str(row.get('state', 'XX'))
        filename = f"property_{city}_{state}_{idx}.txt"
        
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(doc_text)
    
    print(f"✅ Created {len(df)} documents in {output_dir}/")
    print(f"\nSample document:")
    print("-" * 60)
    print(create_property_document(df.iloc[0]))
    print("-" * 60)
    
    return len(df)


# ========================================
# STEP 4: Create a market report document
# ========================================

def create_market_report(
    csv_path: str = "realtor-data.csv",
    output_dir: str = "./documents",
    state: str = "Virginia"
):
    """
    Create a market summary document with statistics.
    This gives your RAG system aggregate data to answer
    questions like "What's the median price in Virginia?"
    """
    
    df = pd.read_csv(csv_path)
    
    # Filter by state
    df_state = df[df['state'] == state].copy()
    
    if len(df_state) == 0:
        print(f"No data found for {state}")
        return
    
    # Calculate statistics
    median_price = df_state['price'].median()
    mean_price = df_state['price'].mean()
    median_sqft = df_state['house_size'].median()
    median_bed = df_state['bed'].median()
    median_bath = df_state['bath'].median()
    
    # Group by city for top markets
    city_stats = df_state.groupby('city').agg({
        'price': 'median',
        'house_size': 'median'
    }).sort_values('price', ascending=False).head(10)
    
    # Create report
    report = f"""
{state} Real Estate Market Report

Overview:
The {state} real estate market analysis is based on {len(df_state):,} property listings.

Statewide Statistics:
- Median Home Price: ${median_price:,.0f}
- Average Home Price: ${mean_price:,.0f}
- Median Square Footage: {median_sqft:,.0f}
- Median Bedrooms: {median_bed:.0f}
- Median Bathrooms: {median_bath:.1f}

Top 10 Markets by Median Price:
"""
    
    for city, stats in city_stats.iterrows():
        report += f"- {city}: ${stats['price']:,.0f} median, {stats['house_size']:,.0f} sq ft\n"
    
    report += f"""

Market Conditions:
Properties in this dataset range from ${df_state['price'].min():,.0f} to ${df_state['price'].max():,.0f}.
The typical home has {median_bed:.0f} bedrooms and {median_bath:.1f} bathrooms.
    """
    
    # Save report
    filepath = os.path.join(output_dir, f"market_report_{state.replace(' ', '_')}.txt")
    with open(filepath, 'w') as f:
        f.write(report.strip())
    
    print(f"✅ Created market report: {filepath}")
    print("\nReport preview:")
    print("-" * 60)
    print(report[:500])
    print("-" * 60)


# ========================================
# STEP 5: Put it all together
# ========================================

def main():
    """
    Full pipeline: CSV → Documents → Ready for RAG
    """
    
    csv_path = "realtor-data.csv"
    output_dir = "./documents"
    
    # Check if dataset exists
    if not os.path.exists(csv_path):
        print(f"❌ Dataset not found at {csv_path}")
        print("\nTo download:")
        print("1. Visit: https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset")
        print("2. Click Download")
        print("3. Extract realtor-data.csv to this directory")
        print("\nOr use Kaggle API:")
        print("  pip install kaggle")
        print("  kaggle datasets download -d ahmedshahriarsakib/usa-real-estate-dataset")
        print("  unzip usa-real-estate-dataset.zip")
        return
    
    print("="*60)
    print("REAL ESTATE DATASET → RAG INTEGRATION")
    print("="*60)
    
    # Explore dataset first
    df = explore_dataset(csv_path)
    
    print("\n" + "="*60)
    print("GENERATING DOCUMENTS")
    print("="*60)
    
    # Option 1: Sample 100 random properties
    print("\nOption 1: Random sample of 100 properties")
    num_docs = generate_documents_from_dataset(
        csv_path=csv_path,
        output_dir=output_dir,
        sample_size=100
    )
    
    # Option 2: All properties from a specific city
    # Uncomment to use:
    # print("\nOption 2: All properties in Charlottesville")
    # num_docs = generate_documents_from_dataset(
    #    csv_path=csv_path,
    #    output_dir=output_dir,
    #    filter_city="Charlottesville",
    #    sample_size=1000
    # )
    
    # Create market report
    print("\n" + "="*60)
    print("CREATING MARKET REPORT")
    print("="*60)
    create_market_report(
        csv_path=csv_path,
        output_dir=output_dir,
        state="Virginia"
    )
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print(f"✅ Created {num_docs} property documents")
    print(f"✅ Created market report")
    print(f"\nNow run your RAG system:")
    print("  1. Delete old index: rm -rf chroma_db")
    print("  2. Run RAG system: python rag_system.py")
    print("  3. Start API: uvicorn api:app --reload")
    print("\nTest queries:")
    print('  - "What properties are available in Virginia?"')
    print('  - "What is the median price in Virginia?"')
    print('  - "Show me properties under $500,000"')


if __name__ == "__main__":
    main()


# ========================================
# BONUS: Advanced filtering
# ========================================

def create_filtered_dataset(
    csv_path: str = "realtor-data.csv",
    output_dir: str = "./documents",
    min_price: int = 200000,
    max_price: int = 1000000,
    states: list = ["Virginia", "North Carolina", "Maryland"],
    min_bedrooms: int = 3
):
    """
    Create a focused dataset with specific filters.
    
    Good for: Creating a realistic, curated dataset for your demo.
    """
    
    df = pd.read_csv(csv_path)
    
    # Apply filters
    df_filtered = df[
        (df['price'] >= min_price) &
        (df['price'] <= max_price) &
        (df['state'].isin(states)) &
        (df['bed'] >= min_bedrooms) &
        (df['price'].notna()) &
        (df['house_size'].notna())
    ].copy()
    
    print(f"Filtered from {len(df):,} to {len(df_filtered):,} properties")
    print(f"Price range: ${min_price:,} - ${max_price:,}")
    print(f"States: {', '.join(states)}")
    print(f"Min bedrooms: {min_bedrooms}")
    
    # Sample if still too large
    if len(df_filtered) > 500:
        df_filtered = df_filtered.sample(n=500, random_state=42)
        print(f"Sampled down to 500 properties")
    
    # Generate documents
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, row in df_filtered.iterrows():
        doc_text = create_property_document(row)
        city = str(row['city']).replace(' ', '_')
        state = str(row['state'])
        filename = f"property_{city}_{state}_{idx}.txt"
        
        with open(os.path.join(output_dir, filename), 'w') as f:
            f.write(doc_text)
    
    print(f"✅ Created {len(df_filtered)} filtered documents")
    
    return len(df_filtered)
