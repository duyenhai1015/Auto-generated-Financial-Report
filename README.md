# 📈 Corporate Analysis Report Generator

This is a web application built with Flask (Python) that allows users to generate detailed analysis reports for companies listed on the Vietnamese stock market. The application fetches data from local CSV files and the `vnstock` library, then aggregates, visualizes, and exports a professional report as a PDF file.

## 🌟 Key Features

* **Simple Web Interface:** Users only need to enter a stock ticker (e.g., `DHG`, `FPT`) and a report date.
* **Data Aggregation:** Automatically fetches and processes data from multiple sources (historical prices, company information, financial statements).
* **Comprehensive Analysis:** The report includes:
    * Market Overview (VNINDEX).
      <img width="849" height="516" alt="image" src="https://github.com/user-attachments/assets/1dcaff85-9a8d-405e-b73c-7a6106c0a226" />
      <img width="736" height="845" alt="image" src="https://github.com/user-attachments/assets/6e3d5892-5904-4e82-91dd-839b38b80fae" />
      <img width="631" height="544" alt="image" src="https://github.com/user-attachments/assets/e77a4080-d326-4012-885a-d94ce3fc1e1e" />
      <img width="672" height="843" alt="image" src="https://github.com/user-attachments/assets/2cf764df-f526-49c7-8efd-c0476d8dc700" />
      <img width="669" height="520" alt="image" src="https://github.com/user-attachments/assets/8b17fd0a-18f0-478e-87ec-4406409765cb" />
      <img width="639" height="809" alt="image" src="https://github.com/user-attachments/assets/e7f07e1f-acc8-49e4-9613-6d64001c70ed" />
      <img width="623" height="518" alt="image" src="https://github.com/user-attachments/assets/6b78bf4e-0abd-4b8c-8bf3-dcd3ef3372af" />
      <img width="675" height="677" alt="image" src="https://github.com/user-attachments/assets/c47eb22a-2c01-435c-bcdc-56a414343ec3" />





    * Company Profile and Business Strategy.
      <img width="698" height="552" alt="image" src="https://github.com/user-attachments/assets/85cbbe35-a324-4553-b554-d55f0e0279ce" />
      <img width="664" height="814" alt="image" src="https://github.com/user-attachments/assets/4bf6678f-2b50-40bb-bdcf-41d11fb6445a" />
      <img width="663" height="760" alt="image" src="https://github.com/user-attachments/assets/918cb3de-ec3c-4f0a-a794-c3fba79191b2" />
    * Price and Volume Chart Analysis.
      <img width="668" height="704" alt="image" src="https://github.com/user-attachments/assets/b0ac6484-b982-4541-b4dc-bb1c2f315c0b" />
      <img width="660" height="778" alt="image" src="https://github.com/user-attachments/assets/6b49c621-84ef-4718-a25e-543f86aec806" />

  
    * Key Financial Ratios (Profitability, Liquidity, Valuation, etc.).
      <img width="660" height="843" alt="image" src="https://github.com/user-attachments/assets/a8aeb797-194c-4bbe-b66b-bcc4ce448ab4" />
      <img width="661" height="534" alt="image" src="https://github.com/user-attachments/assets/83173117-8265-45ed-8995-bb2fb34a1fe6" />
      <img width="653" height="446" alt="image" src="https://github.com/user-attachments/assets/9206259f-57dd-4411-b4ae-3fbd414c5c42" />
      <img width="660" height="525" alt="image" src="https://github.com/user-attachments/assets/c0c20b52-a331-4842-a146-ab9ae673d6a3" />


    * Income Statement & Balance Sheet extracts.
      <img width="660" height="750" alt="image" src="https://github.com/user-attachments/assets/c345f50f-591b-401a-bcbc-9ca1b6c84a26" />
      <img width="668" height="596" alt="image" src="https://github.com/user-attachments/assets/df92585a-e38b-41d7-a8e8-518d9802b007" />



    * Comparison with industry peers.
      <img width="676" height="709" alt="image" src="https://github.com/user-attachments/assets/75baebe7-f330-4127-a3d6-670f3c4d1749" />
      <img width="665" height="481" alt="image" src="https://github.com/user-attachments/assets/5e1dd8c5-52d4-475d-822d-59130fa3118a" />
      ![Uploading image.png…]()



* **Data Visualization:** Uses Plotly and Matplotlib to create clear and interactive charts.
* **PDF Export:** Automatically generates a professional PDF report, ready for viewing or sharing, using WeasyPrint.

## 📄 Sample Output Report

The application generates a PDF file similar to the `DHG_2024-12-31 00_00_00 (20).pdf` file included in this repository.

<img width="935" height="852" alt="image" src="https://github.com/user-attachments/assets/39b1c3f9-9096-4453-9fd0-98706e655f87" />


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
