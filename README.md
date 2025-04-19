# Indego Bike-Share Growth Analysis

This script analyzes Indego Bike-Share data to generate a two-slide executive presentation highlighting growth trends, anomalies, and recommendations.

## Prerequisites

- Python 3.9 or higher
- Required Python packages (install using `pip install -r requirements.txt`):
  - pandas
  - numpy
  - matplotlib
  - python-pptx
  - requests

## Data Sources

The script can automatically download and process Indego Bike-Share data from their website. Data is available from 2015 Q2 to 2024 Q4.

Alternatively, you can manually place the following CSV files in the `data` directory:

1. `trips_summary.csv` - Quarterly trip aggregates
2. `segments_summary.csv` - Quarterly breakdown by passholder and bike type
3. `station_util.csv` - Station-level utilization rates

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the analysis:
   ```bash
   python indego_analysis.py
   ```

   The script will:
   - Download and process raw data from the Indego website (if needed)
   - Generate summary files
   - Create visualizations
   - Generate the presentation

3. Find the output presentation in the `output` directory:
   - `Perpay_Indego_Presentation.pptx`

## Output

The script generates:
- A PowerPoint presentation with two slides
- Growth trends and anomaly detection charts
- Key metrics and recommendations

## File Structure

```
.
├── data/
│   ├── indego-trips-*.csv (raw data files)
│   ├── trips_summary.csv
│   ├── segments_summary.csv
│   └── station_util.csv
├── output/
│   ├── trends.png
│   ├── anomalies.png
│   └── Perpay_Indego_Presentation.pptx
├── indego_analysis.py
├── requirements.txt
└── README.md
``` 