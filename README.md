# 📈 Corporate Analysis Report Generator

This is a web application built with Flask (Python) that allows users to generate detailed analysis reports for companies listed on the Vietnamese stock market. The application fetches data from local CSV files and the `vnstock` library, then aggregates, visualizes, and exports a professional report as a PDF file.

## 🌟 Key Features

* **Simple Web Interface:** Users only need to enter a stock ticker (e.g., `DHG`, `FPT`) and a report date.
* **Data Aggregation:** Automatically fetches and processes data from multiple sources (historical prices, company information, financial statements).
* **Comprehensive Analysis:** The report includes:
    * Market Overview (VNINDEX).
    * Company Profile and Business Strategy.
    * Price and Volume Chart Analysis.
    * Key Financial Ratios (Profitability, Liquidity, Valuation, etc.).
    * Income Statement & Balance Sheet extracts.
    * Comparison with industry peers.
* **Data Visualization:** Uses Plotly and Matplotlib to create clear and interactive charts.
* **PDF Export:** Automatically generates a professional PDF report, ready for viewing or sharing, using WeasyPrint.

## 📄 Sample Output Report

The application generates a PDF file similar to the `DHG_2024-12-31 00_00_00 (20).pdf` file included in this repository.

*(It is highly recommended to take a screenshot of the first page of your PDF report and add it here for a better visual preview)*

`[Screenshot of the DHG.pdf report cover page]`

## 🛠️ Technology Stack

* **Backend:** Python, Flask
* **Data Retrieval:** `vnstock`
* **Data Processing:** `pandas`, `numpy`
* **Visualization:** `plotly`, `matplotlib`
* **PDF Generation:** `weasyprint`
* **Frontend:** `HTML`, `Jinja2` (for the report template)

## 🚀 Setup and Installation

### 1. Prerequisites

* Python 3.8+
* Git

### 2. Installation Guide

1.  **Clone this repository:**
    ```bash
    git clone [YOUR-GITHUB-REPOSITORY-URL]
    cd [your-project-directory-name]
    ```

2.  **(Recommended) Create and activate a virtual environment:**
    * On macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    * On Windows:
        ```bash
        python -m venv venv
        venv\Scripts\activate
        ```

3.  **Install the required libraries:**
    (You can create a `requirements.txt` file, paste the lines below into it, and then run `pip install -r requirements.txt`)
    ```text
    Flask
    pandas
    vnstock
    matplotlib
    plotly
    weasyprint
    numpy
    prettytable
    ```
    Or, install them directly:
    ```bash
    pip install Flask pandas vnstock matplotlib plotly weasyprint numpy prettytable
    ```

4.  **Organize Project Structure (Very Important):**
    Flask requires a specific folder structure to find templates and static files.
    * Create a folder named `templates` and move `index.html` and `report.html` into it.
    * Create a folder named `data` and move your three `.csv` files into it.

    Your final project structure should look like this:
    ```
    your-project-folder/
    ├── app.py
    ├── README.md
    |
    ├── data/
    │   ├── thongtin.xlsx - Sheet1.csv
    │   ├── TM.csv
    │   └── Vietnam_Price_sheet2.csv
    │
    └── templates/
        ├── index.html
        └── report.html
    ```

### 3. Running the Application

1.  From the project's root directory (the one containing `app.py`), run the command:
    ```bash
    python app.py
    ```

2.  Open your web browser and navigate to:
    [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 💡 How to Use

1.  On the home page (`index.html`), enter a **Company Ticker** (e.g., `DHG`).
2.  Select a **Report Date**.
3.  Click the "**Tạo Báo Cáo**" (Generate Report) button.
4.  Wait a few moments for the server to process the data. Your browser will automatically download the complete PDF report.

## ⚖️ Disclaimer

This report is generated automatically for academic purposes, based on publicly available information, proprietary data, and other sources (Vnstock data) believed to be reliable, but which have not been independently verified. The author makes no representation or warranty as to the accuracy, correctness, or completeness of the information in this report. This report is for informational purposes only and is not an offer or solicitation to buy or sell any securities mentioned herein. Past performance, if any, is not indicative of future results. Investors must make their own investment decisions based on independent opinions tailored to their specific financial situation or investment objectives.
