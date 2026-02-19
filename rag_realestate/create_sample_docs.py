# rag_realestate/create_sample_docs.py

import os
os.makedirs("documents", exist_ok=True)

docs = {
    "property_listing.txt": """
Property Listing: 123 Oak Street, Charlottesville VA 22901
Price: $485,000
Bedrooms: 4 | Bathrooms: 2.5 | Square Feet: 2,340
Description: Beautiful colonial-style home in Frys Spring neighborhood.
Features hardwood floors, updated kitchen with granite countertops.
HOA Fees: None
Property Tax: $4,200/year
School District: Charlottesville City Schools
Recent Updates: New roof 2022, HVAC system 2021, water heater 2023
Zoning: R-1 Residential | Lot Size: 0.35 acres
    """,

    "market_report.txt": """
Charlottesville Real Estate Market Report Q4 2024
Median home prices increased 8.3% year-over-year reaching $412,000.
Days on Market: Average 23 days down from 31 days in 2023.
Inventory: 2.1 months supply indicating seller market conditions.
Top Neighborhoods:
- Belmont: Median $380,000 up 12%
- Frys Spring: Median $425,000 up 9%
- Downtown: Median $520,000 up 6%
Rental Market: Average 2BR rent $1,850/month. Vacancy rate 3.2%.
Forecast 2025: Analysts predict 5-7% appreciation.
    """,

    "investment_guide.txt": """
Real Estate Investment Key Metrics

Cap Rate: Net Operating Income divided by Property Value.
Good cap rate is 5-10% depending on market.
Example: $24,000 NOI divided by $400,000 property equals 6% cap rate.

Cash-on-Cash Return: Annual Cash Flow divided by Total Cash Invested.
Target 8-12% for most investors.

The 1% Rule: Monthly rent should be at least 1% of purchase price.
Example: $300,000 property should rent for $3,000 per month.

Common Expenses:
- Property management: 8-12% of gross rent
- Vacancy allowance: 5-10%
- Maintenance: 1% of property value annually
- Insurance: $1,200-2,400 per year
    """
}

for filename, content in docs.items():
    with open(f"documents/{filename}", "w") as f:
        f.write(content)
    print(f"Created {filename}")

print("All documents created!")
