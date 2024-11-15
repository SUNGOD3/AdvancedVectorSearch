# AdvancedVectorSearch

## Dataset Setup

This project uses the [BIGANN dataset](http://corpus-texmex.irisa.fr/). To download & run the dataset:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the datasets
   ```bash
   # /benchmarks
   python BIGANN_downloader.py
   ```
3. Run the BIGANN datasets
   ```bash
   # /benchmarks
   python BIGANN_run_benchmarks.py
   ```

Note. This project also designs its own datasets (random vectors), if you want to use it:
   ```bash
   # /benchmarks
   python run_benchmarks.py
   ```