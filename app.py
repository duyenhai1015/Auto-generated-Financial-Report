import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from flask import Flask, render_template, request
import json
import os
import matplotlib
matplotlib.use('Agg')  # Chọn backend không yêu cầu GUI
import matplotlib.pyplot as plt
import io
import base64
from vnstock import Vnstock
from vnstock import Listing, Quote, Company, Finance, Trading, Screener 
from vnstock.explorer.vci import Company
import numpy as np
from prettytable import PrettyTable
import plotly.graph_objects as go
from weasyprint import HTML
from flask import send_file
from plotly.subplots import make_subplots
import plotly.io as pio
from io import BytesIO
from datetime import timedelta
import traceback
from datetime import datetime


FILE_PATH1 = "data/Vietnam_Price_sheet2.csv"

# Hàm đọc và xử lý dữ liệu
def load_data_TA():
    df = pd.read_csv(FILE_PATH1, dtype=str, low_memory=False, encoding="utf-8")
    df_long = df.melt(id_vars=["Name", "Code"], var_name="Date", value_name="Close_Price")
    invalid_dates = ["RIC", "Start Date", "Exchange", "Sector", "Activity"]
    df_long = df_long[~df_long["Date"].isin(invalid_dates)]
    df_long["Date"] = pd.to_datetime(df_long["Date"], format="%Y-%m-%d", errors="coerce")
    df_long["Close_Price"] = pd.to_numeric(df_long["Close_Price"], errors="coerce")
    df_long = df_long.dropna(subset=["Date", "Close_Price"])
    df_long = df_long.sort_values(by=["Code", "Date"])
    return df_long

def load_data_TA():
    df = pd.read_csv(FILE_PATH1, dtype=str, low_memory=False, encoding="utf-8")
    df_long = df.melt(id_vars=["Name", "Code"], var_name="Date", value_name="Close_Price")
    invalid_dates = ["RIC", "Start Date", "Exchange", "Sector", "Activity"]
    df_long = df_long[~df_long["Date"].isin(invalid_dates)]
    df_long["Date"] = df_long["Date"].astype(str).str.split().str[0]
    df_long["Date"] = pd.to_datetime(df_long["Date"], format="%Y-%m-%d", errors="coerce")
    df_long["Close_Price"] = pd.to_numeric(df_long["Close_Price"], errors="coerce")
    df_long = df_long.dropna(subset=["Date", "Close_Price"])
    df_long = df_long.sort_values(by=["Code", "Date"])
    return df_long
# Hàm tính MA
def calculate_moving_averages(df_long, ma_periods):
    for ma in ma_periods:
        df_long[f"MA{ma}"] = df_long.groupby("Code")["Close_Price"].transform(
            lambda x: x.rolling(window=ma, min_periods=1).mean()
        )
    return df_long

# Hàm tính số lượng cổ phiếu trên MA
def count_stocks_above_ma(df_long, ma_periods):
    above_ma_counts = df_long.copy()
    for ma in ma_periods:
        above_ma_counts[f"Above_MA{ma}"] = above_ma_counts["Close_Price"] > above_ma_counts[f"MA{ma}"]
    return above_ma_counts.groupby("Date")[[f"Above_MA{ma}" for ma in ma_periods]].sum()

# Hàm tính số lượng MA đang tăng
def count_increasing_ma(df_long, ma_periods):
    df_ma_increase = df_long.copy()
    for ma in ma_periods:
        df_ma_increase[f"Increase_MA{ma}"] = df_ma_increase[f"MA{ma}"].diff() > 0
    return df_ma_increase.groupby("Date")[[f"Increase_MA{ma}" for ma in ma_periods]].sum()


def plot_trend_MA_chart(selected_date=None):
    df_long = load_data_TA()
    ma_periods = [10, 20, 50, 100, 200]
    df_long = calculate_moving_averages(df_long, ma_periods)
    df_above_ma_final = count_stocks_above_ma(df_long, ma_periods)
    df_ma_increase_final = count_increasing_ma(df_long, ma_periods)
    end_date = pd.to_datetime(selected_date)
    start_date = end_date - pd.DateOffset(months=6)

    df_above_ma_final = df_above_ma_final.loc[start_date:end_date]
    df_ma_increase_final = df_ma_increase_final.loc[start_date:end_date]

# Tạo layout 2x2 với Plotly Subplots
    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        "SLCP có giá nằm trên các MA tương ứng (D)",
        "SLCP có MA tương ứng đang tăng (D)",
        "SLCP trend template ngày (D) và tuần (W)",
        "SLCP trend template và biến thiên (D)"
    ], specs=[[{}, {}], [{"secondary_y": False}, {"secondary_y": True}]])

    # Biểu đồ 1: Số lượng cổ phiếu có giá trên MA
    for ma in ma_periods:
            fig.add_trace(go.Scatter(x=df_above_ma_final.index, y=df_above_ma_final[f"Above_MA{ma}"], mode='lines',
                                     name=f"Above MA{ma}"), row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Số lượng CP", row=1, col=1)
    fig.update_layout(showlegend=True)

        # Biểu đồ 2: Số lượng cổ phiếu có MA tăng
    for ma in ma_periods:
        fig.add_trace(
            go.Scatter(x=df_ma_increase_final.index, y=df_ma_increase_final[f"Increase_MA{ma}"], mode='lines',
                           name=f"Increase MA{ma}"), row=1, col=2)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(title_text="Số lượng CP", row=1, col=2)

        # Biểu đồ 3: Xu hướng ngày và tuần
    df_trend = df_above_ma_final.mean(axis=1).rolling(window=10).mean()
    df_trend_weekly = df_above_ma_final.mean(axis=1).rolling(window=50).mean()
    fig.add_trace(
        go.Scatter(x=df_trend.index, y=df_trend, mode='lines', name="TrendTP_D", line=dict(color='orange')), row=2,
            col=1)
    fig.add_trace(go.Scatter(x=df_trend_weekly.index, y=df_trend_weekly, mode='lines', name="TrendTP_W",
                                 line=dict(color='blue')), row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Trend", row=2, col=1)

        # Biểu đồ 4: Xu hướng và biến thiên với 2 trục tung
    trend_diff = df_trend.diff().fillna(0)
    fig.add_trace(go.Bar(x=trend_diff.index, y=trend_diff, name="Biến thiên",
                             marker_color=['red' if x < 0 else 'green' for x in trend_diff]), row=2, col=2,
                      secondary_y=False)
    fig.add_trace(
            go.Scatter(x=df_trend.index, y=df_trend, mode='lines', name="TrendTP_D", line=dict(color='orange')), row=2,
            col=2, secondary_y=True)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_yaxes(title_text="Biến thiên", row=2, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Trend", row=2, col=2, secondary_y=True)

    chart1 = fig.to_image(format="png", width=2200, height=1500, scale=2)

    # Chuyển đổi hình ảnh thành base64
    chart1=base64.b64encode(chart1).decode("utf-8")
       
    return chart1

#Giao dịch theo ngành và nhà đầu tư
def load_data_GD(file_path):
    """Đọc dữ liệu từ CSV và chuyển đổi cột 'ngày' thành datetime.date"""
    df = pd.read_csv(file_path)
    df['ngày'] = pd.to_datetime(df['ngày'], format='%Y-%m-%d').dt.date
    return df

def investor_flow(df, selected_date):
    """Lọc dữ liệu trong 30 ngày trước và bao gồm ngày đã chọn"""
    selected_date = pd.to_datetime(selected_date).date()
    start_date = selected_date - timedelta(days=30)
    df_filtered = df[(df['ngày'] >= start_date) & (df['ngày'] <= selected_date)].copy()
    return df_filtered

def create_combined_plot(df_filtered, matching_columns):
    """Tạo một figure chứa cả hai biểu đồ"""
    # Tính tổng giao dịch mua và bán ròng
    matching_columns = list(matching_columns)
    total_buy = df_filtered[df_filtered[matching_columns].sum(axis=1) > 0][matching_columns].sum().sum() / 1e9
    total_sell = df_filtered[df_filtered[matching_columns].sum(axis=1) < 0][matching_columns].sum().sum() / 1e9

    # Xử lý dữ liệu theo ngành
    df_filtered['Giá trị ròng'] = df_filtered[matching_columns].sum(axis=1) / 1e9
    df_sorted = df_filtered.groupby('ngành')['Giá trị ròng'].sum().reset_index()
    df_sorted['ngành'] = df_sorted['ngành'].str.replace(' L2', '', regex=True)
    df_sorted = df_sorted.sort_values(by='Giá trị ròng', ascending=True)
    df_sorted['Mua ròng'] = df_sorted['Giá trị ròng'].apply(lambda x: x if x > 0 else 0)
    df_sorted['Bán ròng'] = df_sorted['Giá trị ròng'].apply(lambda x: x if x < 0 else 0)

    # Tạo subplot với 1 hàng 2 cột
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=("Tổng giao dịch", "Giao dịch theo ngành"), 
        column_widths=[0.25, 0.75],  # Tăng width cột 1, giảm width cột 2
        horizontal_spacing=0.2  # Tăng khoảng cách giữa 2 biểu đồ
    )

    # Biểu đồ 1: Tổng giao dịch mua bán ròng
    fig.add_trace(go.Bar(
        x=[''], y=[total_buy], marker_color='#2196F3', name="Mua ròng", width=[0.2],  # ✅ Màu xanh dương + giảm width
        text=f"{total_buy:,.1f}bn", textposition="inside", insidetextanchor="middle",
        hovertext=f"Mua ròng: {total_buy:.1f}T", hoverinfo="text"
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=[''], y=[total_sell], marker_color='#F44336', name="Bán ròng", width=[0.2],  # ✅ Màu đỏ + giảm width
        text=f"{total_sell:,.1f}bn", textposition="inside", insidetextanchor="middle",
        hovertext=f"Bán ròng: {total_sell:.1f}T", hoverinfo="text"
    ), row=1, col=1)

    # Biểu đồ 2: Giao dịch theo ngành
    fig.add_trace(go.Bar(
        y=df_sorted['ngành'], x=df_sorted['Mua ròng'], orientation='h', marker_color='#2196F3', name="Mua ròng",
        text=[f"{val:,.1f}bn" if val > 0 else "" for val in df_sorted['Mua ròng']], textposition="outside",
        hoverinfo="text"
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        y=df_sorted['ngành'], x=df_sorted['Bán ròng'], orientation='h', marker_color='#F44336', name="Bán ròng",
        text=[f"{val:,.1f}bn" if val < 0 else "" for val in df_sorted['Bán ròng']], textposition="outside",
        hoverinfo="text"
    ), row=1, col=2)

    # Cấu hình layout
    fig.update_layout(
        barmode='relative', showlegend=True,
        xaxis1=dict(title="Tổng giá trị giao dịch (Tỷ VNĐ)"),
        xaxis2=dict(title="Giao dịch theo ngành (Tỷ VNĐ)"),
        margin=dict(l=100, r=100, t=50, b=50)
    )

    chart = fig.to_image(format="png", width=1400, height=700, scale=2)

    # Chuyển đổi hình ảnh thành base64
    chart=base64.b64encode(chart).decode("utf-8")
       
    return chart

# Lấy dữ liệu VNINDEX từ API
def get_vnindex_data(start_date='2019-01-02', end_date='2025-03-19'):
    from vnstock import Vnstock
    stock = Vnstock().stock(symbol='ACB', source='VCI')
    df_index = stock.quote.history(symbol='VNINDEX', start='2019-01-02', end='2025-03-19', interval='1D')
    df_index['Date'] = pd.to_datetime(df_index['time'])
    return df_index

df_index = get_vnindex_data()

# Hiển thị tổng quan VNINDEX và vẽ biểu đồ
def vnindex_overview(selected_date):
    selected_date = pd.to_datetime(selected_date)
    
    index_today = df_index[df_index['Date'] == selected_date]
    index_yesterday = df_index[df_index['Date'] == (selected_date - pd.Timedelta(days=1))]
    high_price = index_today['high'].values[0]
    low_price = index_today['low'].values[0]
    volume = index_today['volume'].values[0]
    
    if not index_today.empty and not index_yesterday.empty:
        vnindex_change = index_today['close'].values[0] - index_yesterday['close'].values[0]
        vnindex_percent_change = (vnindex_change / index_yesterday['close'].values[0]) * 100
        # Nếu VNINDEX tăng
        if vnindex_change > 0:
           vnindex_summary = f"VNINDEX tại {selected_date.date()} tăng <span style='color:green'>{vnindex_change:.2f}</span> điểm (<span style='color:green'>{vnindex_percent_change:.2f}</span>%). Cao nhất: {high_price}, Thấp nhất: {low_price}, Khối lượng giao dịch: {volume} <span style='color:green'>↑</span>"
            # Phân tích ý nghĩa khi VNINDEX tăng
           vnindex_summary += f"<br>VNINDEX tăng cho thấy sự lạc quan của thị trường. Nhà đầu tư có thể kỳ vọng vào sự phục hồi kinh tế và niềm tin vào các chính sách vĩ mô tích cực. Các ngành dẫn dắt sự tăng trưởng có thể bao gồm Công nghệ, Tiêu dùng và Năng lượng."
        # Nếu VNINDEX giảm
        else:
           vnindex_summary = f"VNINDEX tại {selected_date.date()} giảm <span style='color:red'>{vnindex_change:.2f}</span> điểm (<span style='color:red'>{vnindex_percent_change:.2f}</span>%). Cao nhất: {high_price}, Thấp nhất: {low_price}, Khối lượng giao dịch: {volume} <span style='color:red'>↓</span>"
           # Phân tích ý nghĩa khi VNINDEX giảm mạnh hơn 5 điểm
           vnindex_summary += f"<br>VNINDEX giảm mạnh có thể phản ánh sự lo ngại từ nhà đầu tư về tình hình kinh tế vĩ mô, bao gồm các yếu tố như lạm phát, tăng lãi suất và bất ổn chính trị. Điều này có thể dẫn đến một làn sóng bán tháo cổ phiếu, gây áp lực lên thị trường. Một yếu tố khác có thể giải thích cho sự giảm điểm là sự điều chỉnh sau một đợt tăng mạnh"


    else:
        vnindex_summary = "Không có dữ liệu VNINDEX cho ngày này."

    # Lấy dữ liệu VNINDEX từ năm 2019 đến ngày đã chọn để vẽ biểu đồ
    start_date = pd.to_datetime('2019-01-01')
    df_index_year = df_index[(df_index['Date'] >= start_date) & (df_index['Date'] <= selected_date)]

    plt.figure(figsize=(10, 5))
    plt.plot(df_index_year['Date'], df_index_year['close'], linestyle='-', color='b')
    plt.xlabel('Ngày')
    plt.ylabel('VNINDEX')
    plt.title('Biểu đồ VNINDEX theo thời gian (đến hiện tại)')
    plt.grid()

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    
    # Chuyển đổi hình ảnh thành base64
    vnindex_chart = base64.b64encode(img.getvalue()).decode("utf-8")  
    # Đóng lại để không giữ tài nguyên
    plt.close()

    return vnindex_summary, vnindex_chart,vnindex_percent_change

EXCEL_PATH = "data/Cleaned_Vietnam_Marketcap.xlsx"

def load_and_process_data():
    # Load the Excel file and process the data
    xls = pd.ExcelFile(EXCEL_PATH, engine="openpyxl")
    df1 = xls.parse("Sheet1").fillna({"Sector": "Unknown Sector"})
    df1["Sector"] = df1["Sector"].replace("-", "Uncategorized")
    # Xóa các dòng có "Unknown Sector" hoặc "Uncategorized"
    df1 = df1[~df1["Sector"].isin(["Unknown Sector", "Uncategorized"])]

    df2 = xls.parse("Sheet2")
    df2.columns = [str(col).replace(" 00:00:00", "") for col in df2.columns]
    df2["Name"] = df2["Name"].str.replace(" - MARKET VALUE", "", regex=False)
    df2["Code"] = df2["Code"].str.replace("(MV)", "", regex=False)

    merged_df = df2.merge(df1[["Name", "Sector"]], on="Name", how="left")
    date_columns = merged_df.columns[2:-1]
    merged_df[date_columns] = merged_df[date_columns].apply(pd.to_numeric, errors='coerce')

    return df1, merged_df, date_columns

# Load data globally
DF1, MERGED_DF, DATE_COLUMNS = load_and_process_data()

def plot_market_cap(selected_date=None):
    SECTOR_MARKETCAP_T = MERGED_DF.groupby("Sector")[DATE_COLUMNS].sum().T
    SECTOR_MARKETCAP_T.index = pd.to_datetime(SECTOR_MARKETCAP_T.index)
    
    # If selected_date is provided, filter the data until that date
    if selected_date:
        selected_date = pd.to_datetime(selected_date)
        SECTOR_MARKETCAP_T = SECTOR_MARKETCAP_T[SECTOR_MARKETCAP_T.index <= selected_date]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    top_sectors = SECTOR_MARKETCAP_T.sum().nlargest(5).index
    for sector in top_sectors:
        ax.plot(SECTOR_MARKETCAP_T.index, SECTOR_MARKETCAP_T[sector], label=sector)
    
    ax.set_xlabel("Thời gian")
    ax.set_ylabel("Tổng vốn hóa thị trường (VNĐ)")
    ax.set_title("Biểu đồ vốn hóa thị trường của top 5 ngành vốn hóa lớn nhất (đến hiện tại)")
    ax.legend(loc="upper left")
    ax.grid(True)

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    
    # Chuyển đổi hình ảnh thành base64
    market_cap_chart = base64.b64encode(img.getvalue()).decode("utf-8")  
    # Đóng lại để không giữ tài nguyên
    plt.close()

    return market_cap_chart

# Tính tổng vốn hóa theo ngành và lấy top 5 ngành
SECTOR_MARKETCAP_T = MERGED_DF.groupby("Sector")[DATE_COLUMNS].sum().T
SECTOR_MARKETCAP_T.index = pd.to_datetime(SECTOR_MARKETCAP_T.index)
top_sectors = SECTOR_MARKETCAP_T.sum().nlargest(5).index

# Đọc dữ liệu giá cổ phiếu
price = pd.read_csv("data/Processed_Vietnam_Price_Long.csv")

# Chuyển cột 'Date' từ int64 sang chuỗi với định dạng ngày (YYYYMMDD)
price['Date'] = price['Date'].astype(str)
price['Date'] = pd.to_datetime(price['Date'], format='%Y%m%d')

def plot_sector_value_trends(end_date, merged_df, df_price, top_sectors):
    # Tính start_date cách 1 năm so với end_date
    start_date = end_date - pd.DateOffset(years=5)
    
    # Lọc các cổ phiếu trong top 5 ngành
    top_stocks = merged_df[merged_df['Sector'].isin(top_sectors)]
    filtered_price_data = df_price[df_price['Code'].isin(top_stocks['Code'])]
    
    # Lọc dữ liệu trong khoảng thời gian từ start_date đến end_date
    filtered_price_data = filtered_price_data[(filtered_price_data['Date'] >= start_date) & (filtered_price_data['Date'] <= end_date)]
    
    # Tính giá trị trung bình hàng ngày cho mỗi ngành (dùng 'Value' thay vì 'Price')
    avg_value_per_day = filtered_price_data.groupby(['Sector', 'Date']).agg({'Value': 'mean'}).reset_index()
    
    # Loại bỏ các ngày không có dữ liệu
    avg_value_per_day = avg_value_per_day.dropna(subset=['Value'])

    # Vẽ đồ thị
    plt.figure(figsize=(12, 6))
    
    # Vẽ giá trị trung bình theo ngành
    for sector in top_sectors:
        sector_data = avg_value_per_day[avg_value_per_day['Sector'] == sector]
        plt.plot(sector_data['Date'], sector_data['Value'], label=sector)

    # Thêm thông tin cho đồ thị
    plt.title(f"Diễn biến giá trị cổ phiếu của 5 ngành có vốn hóa lớn nhất (tính từ {start_date.date()} đến {end_date.date()})")
    plt.xlabel("Ngày")
    plt.ylabel("Giá trị cổ phiếu trung bình")
    plt.legend(title="Ngành")
    plt.xticks(rotation=45)
    plt.grid(True)

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    
    # Chuyển đổi hình ảnh thành base64
    trend_chart = base64.b64encode(img.getvalue()).decode("utf-8")  
    # Đóng lại để không giữ tài nguyên
    plt.close()

    return trend_chart



# Hàm lấy thông tin báo cáo thị trường
def market_overview(selected_date=None, selected_stock=None):
    # Đọc dữ liệu từ file CSV
    df_price = pd.read_csv("data/Processed_Vietnam_Price_Long.csv")
    df_KQKD = pd.read_csv("data/KQKD.csv")

    df_volume = pd.read_csv("data/Processed_Vietnam_Volume_Long.csv")

    # Chuyển cột 'Date' từ int64 sang chuỗi với định dạng ngày (YYYYMMDD)
    df_price['Date'] = df_price['Date'].astype(str)
    df_volume['Date'] = df_volume['Date'].astype(str)
    
    # Chuyển cột 'Date' từ chuỗi sang kiểu datetime
    df_price['Date'] = pd.to_datetime(df_price['Date'], format='%Y%m%d')
    df_volume['Date'] = pd.to_datetime(df_volume['Date'], format='%Y%m%d')

    # Gộp dữ liệu giá và khối lượng giao dịch
    df = pd.merge(df_price, df_volume, on=['Date', 'Code'], suffixes=('_Price', '_Volume'))

    # Lọc theo ngày giao dịch nếu có chọn
    if selected_date:
        df = df[df["Date"] == selected_date]

    # Lọc theo mã CP nếu có chọn
    if selected_stock:
        df = df[df["Code"] == selected_stock]

    # Kiểm tra xem có dữ liệu không
    if df.empty:
        return "Không có dữ liệu cho ngày hoặc mã cổ phiếu đã chọn."

    # Sắp xếp theo mã CK và ngày giao dịch
    df = df.sort_values(by=["Code", "Date"])

    # Tính giá đóng cửa hôm qua
    df["close_yesterday"] = df.groupby("Code")["Value_Price"].shift(1)
    
    # Loại bỏ dòng không có giá hôm qua
    df = df.dropna()

    # Tính mức thay đổi giá và phần trăm thay đổi
    df["change"] = df["Value_Price"] - df["close_yesterday"]
    df["percent_change"] = (df["change"] / df["close_yesterday"]) * 100  

    # Chuyển đổi kiểu dữ liệu đảm bảo "percent_change" là số
    df["percent_change"] = pd.to_numeric(df["percent_change"], errors="coerce")
    
    # Lọc cổ phiếu có KLGD > 10,000
    df = df[df["Value_Volume"] > 10000]

    # Lấy top 10 tăng và giảm giá
    top_gainers = df.nlargest(10, "percent_change")[["Code", "Value_Price", "percent_change", "Value_Volume"]].copy()
    top_losers = df.nsmallest(10, "percent_change")[["Code", "Value_Price", "percent_change", "Value_Volume"]].copy()


    # Định dạng dấu và màu sắc
    def format_change(value):
        return f'<span style="color:green">+{value:.2f}%</span>' if value > 0 else f'<span style="color:red">{value:.2f}%</span>'
    
    top_gainers["percent_change"] = top_gainers["percent_change"].map(format_change)
    top_losers["percent_change"] = top_losers["percent_change"].map(format_change)
    top_gainers["Value_Price"] = top_gainers["Value_Price"].map(lambda x: f"{x:,.0f}")
    top_losers["Value_Price"] = top_losers["Value_Price"].map(lambda x: f"{x:,.0f}")
    top_gainers["Value_Volume"] = top_gainers["Value_Volume"].map(lambda x: f"{x:,.0f}")
    top_losers["Value_Volume"] = top_losers["Value_Volume"].map(lambda x: f"{x:,.0f}")

      # Tạo bảng HTML
    html_table = f"""
    <style>
        .stock-table {{ width: 80%; margin: auto; border-collapse: collapse; font-size: 14px; }}
        .stock-table th, .stock-table td {{ padding: 8px; text-align: center; border: 1px solid #ddd; }}
        .stock-table th {{ background-color: #f4f4f4; font-weight: bold; }}
        .green-title {{ color: green; font-weight: bold; }}
        .red-title {{ color: red; font-weight: bold; }}
        .italic-text {{ font-style: italic; text-align: center; margin-top: 10px; }}
    </style>
    <table class='stock-table'>
        <tr><th colspan='4' class='green-title'>🔺 Top 10 Cổ phiếu Tăng Giá </th></tr>
        <tr><th>MCK</th><th>Giá</th><th>% Thay đổi</th><th>KLGD</th></tr>
        {''.join(f"<tr><td>{row['Code']}</td><td>{row['Value_Price']}</td><td>{row['percent_change']}</td><td>{row['Value_Volume']}</td></tr>" for _, row in top_gainers.iterrows())}
    </table>
    <br>
    <table class='stock-table'>
        <tr><th colspan='4' class='red-title'>🔻 Top 10 Cổ phiếu Giảm Giá </th></tr>
        <tr><th>MCK</th><th>Giá</th><th>% Thay đổi</th><th>KLGD</th></tr>
        {''.join(f"<tr><td>{row['Code']}</td><td>{row['Value_Price']}</td><td>{row['percent_change']}</td><td>{row['Value_Volume']}</td></tr>" for _, row in top_losers.iterrows())}
    </table>
    <p class='italic-text'>Khối lượng giao dịch (KLGD) trên 10,000 đơn vị.</p>

    """
    return html_table



#TỔNG QUAN GIAO DỊCH
    
def load_data_NN(file_path):
    df = pd.read_csv(file_path)
    df['ngày'] = pd.to_datetime(df['ngày'], format='%Y-%m-%d').dt.date
    return df

def plot_investor_flow(df, selected_date):
    # **Chuyển đổi ngày chọn thành datetime**
    selected_date = pd.to_datetime(selected_date).date()
    start_date = selected_date - timedelta(days=30)  # Lấy dữ liệu từ 30 ngày trước

    # **Lọc dữ liệu**
    df_filtered = df[(df['ngày'] >= start_date) & (df['ngày'] <= selected_date)]

    # **Tính tổng theo ngày**
    df_grouped = df_filtered.groupby("ngày").sum()

    # Tạo figure cho Khớp lệnh & Thỏa thuận
    fig_khop = go.Figure()
    fig_thoa_thuan = go.Figure()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    investor_list = ["cá_nhân", "tổ_chức_trong_nước", "tự_doanh", "nước_ngoài"]

    # Xác định giá trị lớn nhất để tự động scale trục Y
    all_values = []

    for investor in investor_list:
        df_col_khop = f"{investor}_khớp_ròng"
        df_col_thoa_thuan = f"{investor}_thỏa_thuận_ròng"

        if df_col_khop in df_grouped.columns:
            all_values.extend(df_grouped[df_col_khop].tolist())

        if df_col_thoa_thuan in df_grouped.columns:
            all_values.extend(df_grouped[df_col_thoa_thuan].tolist())

    # **Tự động điều chỉnh trục Y**
    y_max = max(abs(v / 1e9) for v in all_values if v != 0)
    y_range = [-y_max * 1.1, y_max * 1.1]

    for index, investor in enumerate(investor_list):
        color = colors[index % len(colors)]
        df_col_khop = f"{investor}_khớp_ròng"
        df_col_thoa_thuan = f"{investor}_thỏa_thuận_ròng"

        if df_col_khop in df_grouped.columns:
            fig_khop.add_trace(go.Bar(
                x=df_grouped.index,
                y=df_grouped[df_col_khop] / 1e9,  # Chia cho 1e9 để hiển thị Tỷ VND
                name=investor.replace('_', ' ').title(),
                marker_color=color,
                text=[f"{int(x)} bn" if abs(x) > 0 else "" for x in df_grouped[df_col_khop] / 1e9],  # Làm tròn số
                textposition="inside",  # Số liệu nằm trong cột
                insidetextanchor="middle"  # **Giữ text ở chính giữa**
            ))

        if df_col_thoa_thuan in df_grouped.columns:
            fig_thoa_thuan.add_trace(go.Bar(
                x=df_grouped.index,
                y=df_grouped[df_col_thoa_thuan] / 1e9,  # Chia cho 1e9 để hiển thị Tỷ VND
                name=investor.replace('_', ' ').title(),
                marker_color=color,
                text=[f"{int(x)} bn" if abs(x) > 0 else "" for x in df_grouped[df_col_thoa_thuan] / 1e9],
                # Làm tròn số
                textposition="inside",  # Số liệu nằm trong cột
                insidetextanchor="middle"  # **Giữ text ở chính giữa**
            ))

    fig_khop.update_layout(
        barmode='relative',
        bargap=0.1,  # Cột sát nhau nhưng không dính
        xaxis=dict(type="category", tickangle=-45),  # Xoay ngày thành xéo
        yaxis=dict(visible=False),  # Ẩn trục Y
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5),  # Chú thích nằm ngang
    )

    fig_thoa_thuan.update_layout(
        barmode='relative',
        bargap=0.1,
        xaxis=dict(type="category", tickangle=-45),  # Xoay ngày thành xéo
        yaxis=dict(visible=False),  # Ẩn trục Y
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5),  # Chú thích nằm ngang
    )
    

    img_bytes_TT = fig_thoa_thuan.to_image(format="png")
    img_bytes_KL = fig_khop.to_image(format="png")


    # Chuyển đổi hình ảnh thành base64
    base64_thoa_thuan= base64.b64encode(img_bytes_TT).decode("utf-8")
    base64_khop=base64.b64encode(img_bytes_KL).decode("utf-8")
    
    
    return base64_khop, base64_thoa_thuan

# Hàm lấy thông tin công ty
def get_company_info(company_code):
    df_KQKD = pd.read_csv("data/KQKD.csv")
    # Lọc thông tin công ty theo mã cổ phiếu
    company_info = df_KQKD[df_KQKD['Mã'] == company_code]

    if company_info.empty:
        return None  # Nếu không tìm thấy mã cổ phiếu, trả về None

    # Lấy thông tin tên công ty và ngành
    company_name = company_info['Tên công ty'].iloc[0]
    industry_level1 = company_info['Ngành ICB - cấp 1'].iloc[0]
    industry_level2 = company_info['Ngành ICB - cấp 2'].iloc[0]
    industry_level3 = company_info['Ngành ICB - cấp 3'].iloc[0]

    # Tạo mô tả ngành đầy đủ
    industry_full_description = f"{industry_level1} > {industry_level2} > {industry_level3}"

    return company_name, industry_full_description

#Hàm lấy thông tin Doanh nghiệp 
def get_company_inf(company_code):
    # Đọc dữ liệu từ file Excel
    df_tt = pd.read_excel("data/thongtin.xlsx")
    
    # Lọc thông tin công ty theo mã công ty
    company_detail = df_tt[df_tt['Mã CK'] == company_code]

    # Kiểm tra xem cột 'Thông tin' có tồn tại và lấy thông tin hồ sơ công ty
    if 'Thông tin' in company_detail.columns:
        company_profile = company_detail['Thông tin'].iloc[0]
        
        # Kiểm tra nếu có thông tin hồ sơ công ty
        if pd.notna(company_profile):
            company_inf =  company_profile  # Khởi tạo từ điển thông tin công ty với hồ sơ

            # Thêm các chi tiết bổ sung vào company_inf
            for col in company_detail.columns:
                if col not in ['Mã CK', 'Thông tin']:  # Đảm bảo không lấy các cột không cần thiết
                    value = company_detail[col].iloc[0]
                    if pd.notna(value):
                        # Chuyển tên cột thành dạng chuẩn (lowercase và thay dấu cách bằng dấu gạch dưới)
                        company_inf[col.lower().replace(' ', '_')] = value

            return company_inf  # Trả về thông tin công ty
        else:
            return None  # Nếu không có thông tin hồ sơ, trả về None
    else:
        return None  # Nếu không tìm thấy cột 'Thông tin', trả về None
    
#Hàm lấy thông tin doanh nghiệp từ vnstock
def get_company_overview(company_code):
    company = Vnstock().stock(symbol=company_code, source='TCBS').company
    df_company_overview = company.overview()  # Trả về DataFrame
    stock_data =  df_company_overview.filter(items=[
    'short_name', 'symbol', 'exchange',
    'website', 'outstanding_share', 'no_shareholders',
    'foreign_percent', 'stock_rating'
])

    return stock_data
#vị thế công ty 
def get_company_position(company_code):
    company = Vnstock().stock(symbol=company_code, source='TCBS').company
    df_company_profile = company.profile()  # Trả về DataFrame
    company_position = df_company_profile.filter(items=[
    'business_strategies'
])
    return company_position
#chiến lược kinh doanh
def get_company_business_strategy(company_code):
    company = Vnstock().stock(symbol=company_code, source='TCBS').company
    df_company_business_strategy = company.profile()  # Trả về DataFrame
    company_key_developments =  df_company_business_strategy.filter(items=[
    'key_developments'
])
    return company_key_developments


#****THÔNG TIN CỔ PHIẾU****

EXCEL_PATH = "data/Cleaned_Vietnam_Marketcap.xlsx"

def load_marketcap_data():
    """Tải dữ liệu MarketCap từ file Excel"""
    xls = pd.ExcelFile(EXCEL_PATH, engine="openpyxl")
    df_marketcap = xls.parse("Sheet2")
    
    # Làm sạch dữ liệu
    df_marketcap.columns = [str(col).replace(" 00:00:00", "") for col in df_marketcap.columns]
    df_marketcap["Name"] = df_marketcap["Name"].str.replace(" - MARKET VALUE", "", regex=False)
    df_marketcap["Code"] = df_marketcap["Code"].str.replace("(MV)", "", regex=False)
    
    return df_marketcap


def get_stock_overview(selected_date, selected_stock):
    # Đọc dữ liệu từ CSV
    df_price = pd.read_csv("data/Processed_Vietnam_Price_Long.csv")
    df_volume = pd.read_csv("data/Processed_Vietnam_Volume_Long.csv")

    # Chuyển cột 'Date' sang datetime
    df_price['Date'] = pd.to_datetime(df_price['Date'].astype(str), format='%Y%m%d')
    df_volume['Date'] = pd.to_datetime(df_volume['Date'].astype(str), format='%Y%m%d')

    # Xác định ngày gần nhất
    max_date = pd.to_datetime(selected_date) if selected_date else df_price['Date'].max()
    prev_date = df_price[df_price["Date"] < max_date]["Date"].max()

    # Lọc dữ liệu trong 52 tuần trước ngày được chọn
    df_price_sorted = df_price[df_price['Date'] < max_date].sort_values(by='Date', ascending=False)
    df_price_valid = df_price_sorted.groupby('Code').head(260)
    df_52w = df_price_valid.groupby("Code")["Value"].agg(H52W="max", L52W="min").reset_index()

    # Lấy giá của ngày được chọn và ngày trước đó
    df_latest_price = df_price[df_price["Date"] == max_date][["Code", "Value"]]
    df_prev_price = df_price[df_price["Date"] == prev_date][["Code", "Value"]]
    df_prev_price.columns = ["Code", "Prev_Value"]

    # Merge dữ liệu giá
    df_latest_price = df_latest_price.merge(df_prev_price, on="Code", how="left")

    # Tính %Change
    df_latest_price["Change"] = np.where(
        df_latest_price["Prev_Value"].notna() & (df_latest_price["Prev_Value"] != 0),
        ((df_latest_price["Value"] - df_latest_price["Prev_Value"]) / df_latest_price["Prev_Value"]) * 100,
        np.nan
    )

    # Làm tròn Change
    df_latest_price["Change"] = df_latest_price["Change"].round(2)

    def format_change(value):
        if pd.isna(value):
            return '<span style="color:black;">N/A</span>'
        formatted_value = f"{abs(value):.2f}%"
        if value > 0:
            return f'<span style="color:green;">+{formatted_value}</span>'
        elif value < 0:
            return f'<span style="color:red;">-{formatted_value}</span>'
        else:
            return '<span style="color:black;">0.00%</span>'

    # Áp dụng định dạng
    df_latest_price["Formatted_Change"] = df_latest_price["Change"].apply(format_change)

    # Lấy khối lượng giao dịch hôm nay
    df_volume_today = df_volume[df_volume["Date"] == max_date][["Code", "Value"]]
    df_volume_today.columns = ["Code", "Volume_Today"]

    # Tải dữ liệu MarketCap
    df_marketcap = load_marketcap_data()
    if str(max_date.date()) in df_marketcap.columns:
        df_marketcap = df_marketcap[["Code", str(max_date.date())]]
        df_marketcap.columns = ["Code", "MarketCap"]
    else:
        df_marketcap["MarketCap"] = None

    # Gộp dữ liệu
    df_final = df_latest_price.merge(df_52w, on="Code", how="left")
    df_final = df_final.merge(df_marketcap, on="Code", how="left")
    df_final = df_final.merge(df_volume_today, on="Code", how="left")

    # Lọc theo mã cổ phiếu nếu có chọn
    if selected_stock:
        df_final = df_final[df_final["Code"] == selected_stock]

    return df_final

def analyze_stock(stock_over, vnindex_change):
    # Biến chứa các kết quả phân tích
    analysis = []

    if stock_over is not None and not stock_over.empty:
        current_value = float(stock_over["Value"].iloc[0])
        previous_value = float(stock_over["Prev_Value"].iloc[0])

        # So sánh giá cổ phiếu
        if current_value > previous_value:
            analysis.append(f"Giá cổ phiếu {stock_over['Code'].iloc[0]} hôm nay tăng so với phiên trước, cho thấy lực cầu đang tốt hoặc có thể đang thu hút dòng tiền ngắn hạn. Việc giá vượt qua mức hôm trước có thể là tín hiệu tích cực trong ngắn hạn, nhất là nếu đi kèm với thanh khoản tăng.")
        elif current_value < previous_value:
            analysis.append(f"Giá cổ phiếu {stock_over['Code'].iloc[0]} hôm nay giảm so với phiên trước, phản ánh áp lực bán hoặc tâm lý thị trường tiêu cực trong ngắn hạn. Cần theo dõi thêm các yếu tố hỗ trợ như vùng giá 52 tuần, tin tức doanh nghiệp hoặc xu hướng thị trường chung.")
        else:
            analysis.append(f"Giá cổ phiếu {stock_over['Code'].iloc[0]} không thay đổi so với phiên trước, cho thấy sự lưỡng lự của nhà đầu tư hoặc trạng thái cân bằng cung cầu tạm thời.")
        
        # Xuống dòng giữa các phần phân tích
        analysis.append("<br>")

        # So sánh với mức giá 52 tuần
        if "H52W" in stock_over and "L52W" in stock_over:
            high_52w = float(stock_over["H52W"].iloc[0])
            low_52w = float(stock_over["L52W"].iloc[0])

            if current_value >= high_52w * 0.9:
                analysis.append(f"So với mức cao nhất và thấp nhất trong 52 tuần, cổ phiếu đang gần vùng đỉnh 52 tuần, có thể phản ánh kỳ vọng tích cực của thị trường hoặc định giá đang cao.")
            elif current_value <= low_52w * 1.1:
                analysis.append(f"So với mức cao nhất và thấp nhất trong 52 tuần, cổ phiếu đang gần vùng đáy 52 tuần, điều này có thể phản ánh một cơ hội đầu tư hấp dẫn đối với nhà đầu tư giá trị, hoặc thị trường đang lo ngại về triển vọng của doanh nghiệp.")
            else:
                analysis.append(f"So với mức cao nhất và thấp nhất trong 52 tuần, cổ phiếu đang trong vùng trung tính so với biên độ 52 tuần, phản ánh trạng thái tích lũy hoặc chưa có xu hướng rõ ràng.")
        else:
            analysis.append("Không đủ dữ liệu về giá 52 tuần để đánh giá vị trí hiện tại của cổ phiếu.")
        
        # Xuống dòng giữa các phần phân tích
        analysis.append("<br>")

        # So sánh với VNIndex
        if vnindex_change is not None:
            try:
                 #Kiểm tra và xử lý vnindex_change
                if isinstance(vnindex_change, str):
                   vnindex_change = vnindex_change.replace('%', '').strip()
                vnindex_change_float = float(vnindex_change)

                # Kiểm tra và xử lý Formatted_Change
                formatted_change = stock_over["Change"].iloc[0]

                # Nếu là kiểu số, chuyển thành chuỗi trước khi xử lý
                if isinstance(formatted_change, float):
                    formatted_change = str(formatted_change)

                # Loại bỏ ký tự '%' và chuyển thành số thực
                stock_change = float(formatted_change.replace('%', '').strip())

                if stock_change > vnindex_change_float:
                    analysis.append("So với giá VNINDEX tại ngày này, ta thấy hiệu suất của cổ phiếu tốt hơn so với thị trường chung.")
                elif stock_change < vnindex_change_float:
                    analysis.append("So với giá VNINDEX tại ngày này, ta thấy Cổ phiếu có diễn biến kém hơn chỉ số thị trường.")
                else:
                    analysis.append("So với giá VNINDEX tại ngày này, ta thấy Cổ phiếu diễn biến tương đồng với xu hướng của thị trường chung.")
            except ValueError:
                analysis.append("Dữ liệu VNIndex hoặc Formatted_Change không hợp lệ.")
        
        # Xuống dòng giữa các phần phân tích
        analysis.append("<br>")

        # Tổng kết tâm lý thị trường
        market_sentiment = "tâm lý tích cực" if current_value > previous_value else "tâm lý thận trọng hoặc tiêu cực"
        analysis.append(f"Tổng thể, cổ phiếu {stock_over['Code'].iloc[0]} đang phản ánh {market_sentiment} của nhà đầu tư trong ngắn hạn. Cần kết hợp thêm phân tích kỹ thuật, tình hình doanh nghiệp và xu hướng ngành để đưa ra quyết định đầu tư hợp lý.")
    
    else:
        analysis.append("Không đủ dữ liệu để phân tích diễn biến giá cổ phiếu.")
    
    # Trả về kết quả phân tích dưới dạng chuỗi với dấu <br> để hiển thị trong HTML
    return "<br>".join(analysis)


def format_number_us(value):
    """Định dạng số theo kiểu Mỹ, bỏ phần thập phân nếu là số nguyên"""
    if pd.notna(value):  # Kiểm tra nếu không phải NaN
        if value == int(value):  # Nếu là số nguyên, chỉ hiển thị phần nguyên
            return "{:,}".format(int(value))
        return "{:,.2f}".format(value)  # Nếu là số thực, giữ lại 2 chữ số thập phân
    return "N/A"

#***biểu đồ giá & KLGD & VNINDEX
# Lấy dữ liệu VNINDEX từ API
def get_vnindex_data(start_date, end_date):
    """Lấy dữ liệu VNINDEX từ API"""
    stock = Vnstock().stock(symbol='ACB', source='VCI')  # API cần symbol để truy cập
    df_index = stock.quote.history(symbol='VNINDEX', start=start_date, end=end_date, interval='1D')

    if df_index is not None and not df_index.empty:
        df_index['Date'] = pd.to_datetime(df_index['time'])
        df_index = df_index[['Date', 'close']]  # Giữ lại cột giá đóng cửa
        df_index.rename(columns={'close': 'VNINDEX'}, inplace=True)
    return df_index

# Hàm lấy dữ liệu tổng quan thị trường
def get_market_overview(selected_date, selected_stock):
    """Tạo DataFrame chứa giá cổ phiếu, khối lượng và VNINDEX"""
    
    # Đọc dữ liệu từ file CSV
    df_price = pd.read_csv("data/Processed_Vietnam_Price_Long.csv")
    df_volume = pd.read_csv("data/Processed_Vietnam_Volume_Long.csv")

    # Chuyển cột 'Date' sang kiểu datetime
    for df in [df_price, df_volume]:
        df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m%d')

    # Xác định thời gian 1 năm trước từ ngày chọn
    start_date = selected_date - pd.DateOffset(years=5)

    # Lọc dữ liệu trong khoảng thời gian 1 năm
    df_price = df_price[(df_price['Date'] >= start_date) & (df_price['Date'] <= selected_date)]
    df_volume = df_volume[(df_volume['Date'] >= start_date) & (df_volume['Date'] <= selected_date)]

    # Gộp dữ liệu giá và khối lượng giao dịch
    df_merged = pd.merge(df_price, df_volume, on=['Date', 'Code'], suffixes=('_Price', '_Volume'))

    # Lọc theo mã cổ phiếu
    df_merged = df_merged[df_merged["Code"] == selected_stock]

    # Lấy dữ liệu VNINDEX
    df_index = get_vnindex_data(start_date.strftime('%Y-%m-%d'), selected_date.strftime('%Y-%m-%d'))

    # Gộp với VNINDEX theo Date
    df_final = pd.merge(df_merged, df_index, on="Date", how="left")

    # Tính %Change
    df_final['%Change_Stock'] = df_final['Value_Price'].pct_change(fill_method=None) * 100
    df_final['%Change_VNINDEX'] = df_final['VNINDEX'].pct_change(fill_method=None) * 100

    return df_final

def plot_charts(df, selected_stock):
    """Vẽ hai biểu đồ riêng biệt: Biến động %Change và Giá cổ phiếu - Khối lượng giao dịch"""

    # Loại bỏ các ngày không có dữ liệu
    df = df.dropna(subset=['Date', '%Change_Stock', '%Change_VNINDEX', 'Value_Volume', 'Value_Price'])

    # Biểu đồ 1: %Change so với VNINDEX
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df['Date'], df['%Change_Stock'], linestyle='-', color='g', label=f'{selected_stock}')
    ax1.plot(df['Date'], df['%Change_VNINDEX'], linestyle='-', color='b', label='VNINDEX')

    ax1.set_title(f'Biến động %Change của {selected_stock} so với VNINDEX')
    ax1.set_ylabel('% Thay đổi')
    ax1.axhline(0, color='black', linewidth=0.8)  # Đường gốc 0%
    ax1.legend()
    ax1.grid()

    # Lưu biểu đồ dưới dạng base64
    img1 = io.BytesIO()
    plt.savefig(img1, format='png', bbox_inches='tight')
    img1.seek(0)
    price_and_vnindex_chart = f'<img src="data:image/png;base64,{base64.b64encode(img1.getvalue()).decode()}"/>'
    plt.close(fig1)  # Đóng figure sau khi lưu xong

    # Biểu đồ 2: Biến động giá và khối lượng giao dịch
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    # Chuyển cột Date sang dạng string để tránh trục X bị giãn cách
    df['Date'] = df['Date'].astype(str)

    # ✅ Chỉ dùng index làm trục X để tránh khoảng trống giữa các ngày
    ax2.bar(df.index, df['Value_Volume'], color='navy', alpha=0.8, label='Khối lượng giao dịch', width=1)

    # Vẽ đường giá cổ phiếu trên trục phụ
    ax3 = ax2.twinx()
    ax3.plot(df.index, df['Value_Price'], linestyle='-', color='crimson', label='Giá cổ phiếu')

    ax2.set_ylabel('Cổ phiếu (nghìn CP)', color='navy')
    ax3.set_ylabel('Giá', color='crimson')
    ax2.set_title(f'Biến động giá và khối lượng giao dịch của {selected_stock}')


    ax2.legend(loc='upper left')
    ax3.legend(loc='upper right')
    # ✅ Chỉ hiển thị một số ngày trên trục X
    num_labels = 5  # Số lượng nhãn hiển thị (có thể điều chỉnh)
    xticks_positions = np.linspace(0, len(df) - 1, num_labels, dtype=int)
    ax2.set_xticks(df.iloc[xticks_positions].index)
    ax2.set_xticklabels(df.iloc[xticks_positions]['Date'], rotation=0, ha='center')


    # Lưu biểu đồ dưới dạng base64
    img2 = io.BytesIO()
    plt.savefig(img2, format='png', bbox_inches='tight')
    img2.seek(0)
    price_and_volume_chart = f'<img src="data:image/png;base64,{base64.b64encode(img2.getvalue()).decode()}"/>'
    plt.close(fig2)  # Đóng figure sau khi lưu xong

    return price_and_vnindex_chart, price_and_volume_chart

#Giao dịch đầu tư nước ngoài***

def load_data():
    file_paths = {
        "data/FT1921_cleaned.csv",
        "data/FT2123_cleaned.csv",
        "data/FT2325_cleaned.csv"
    }
    
    dfs = [pd.read_csv(path) for path in file_paths]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["Date", "Net.F_Val"])
    df = df.sort_values(by="Date").reset_index(drop=True)
    return df

def plot_foreign_trading(ticker_selected, selected_date):
    """ Vẽ biểu đồ giao dịch Nhà Đầu Tư Nước Ngoài trong 2 tuần gần nhất, có nhãn số liệu. """
    # Tải dữ liệu
    data = load_data()
    if data is None:
        return "⚠️ Không thể tải dữ liệu!"

    # Chuyển đổi cột Date về kiểu datetime
    data["Date"] = pd.to_datetime(data["Date"])

    # Chuyển đổi ngày được chọn
    selected_date = pd.to_datetime(selected_date, dayfirst=True)

    # Xác định khoảng thời gian 2 tuần trước ngày được chọn
    start_date = selected_date - pd.Timedelta(days=14)
    end_date = selected_date

    # Lọc dữ liệu theo mã cổ phiếu và thời gian
    filtered_data = data[(data["Ticker"] == ticker_selected) & 
                         (data["Date"] >= start_date) & 
                         (data["Date"] <= end_date)].copy()

    # Loại bỏ dữ liệu rỗng và sắp xếp lại
    filtered_data = filtered_data.dropna(subset=["Net.F_Val"]).sort_values(by="Date").reset_index(drop=True)

    # Kiểm tra nếu không có dữ liệu sau khi lọc
    if filtered_data.empty:
        return "⚠️ Không có dữ liệu trong khoảng thời gian đã chọn."

    # Dữ liệu mua ròng và bán ròng
    buy_data = filtered_data[filtered_data["Net.F_Val"] >= 0]
    sell_data = filtered_data[filtered_data["Net.F_Val"] < 0]

    # Vẽ biểu đồ
    fig = go.Figure()

    # Cột mua ròng (có nhãn)
    fig.add_trace(go.Bar(
        x=buy_data["Date"].dt.strftime("%d-%m-%Y"),  # Chuyển ngày về dạng chuỗi
        y=buy_data["Net.F_Val"], 
        name="Mua ròng", 
        marker_color="#007bff", 
        opacity=0.9,
        text=buy_data["Net.F_Val"].apply(lambda x: f"{x:,.0f}"),  # Định dạng số liệu
        textposition='outside',
        textfont=dict(size=14, color="black", weight="bold"),  # Kích thước và màu chữ
    ))

    # Cột bán ròng (có nhãn)
    fig.add_trace(go.Bar(
        x=sell_data["Date"].dt.strftime("%d-%m-%Y"),  # Chuyển ngày về dạng chuỗi
        y=sell_data["Net.F_Val"], 
        name="Bán ròng", 
        marker_color="#dc3545", 
        opacity=0.9,
        text=sell_data["Net.F_Val"].apply(lambda x: f"{x:,.0f}"),  # Định dạng số liệu
        textposition='outside',
        textfont=dict(size=14, color="black", weight="bold"),  # Kích thước và màu chữ
    ))

     # Căn giữa tiêu đề, in đậm
    fig.update_layout(
        title=dict(
            text=f"<b>Giao dịch Nhà Đầu Tư Nước Ngoài - {ticker_selected}</b>",
            x=0.5,  # Căn giữa
            font=dict(size=18, family="Arial", color="black")  # In đậm
        ),
        yaxis_title="Giá trị ròng",
        barmode="relative",
        template="plotly_white"
    )
        # Xuất hình ảnh ra một buffer
    img_bytes = fig.to_image(format="png")

    # Chuyển đổi hình ảnh thành base64
    fig = base64.b64encode(img_bytes).decode("utf-8")
    
    return fig


#***CHỈ SỐ TÀI CHÍNH***


def extract_financial_data(selected_stock):
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv("data/market_average_by_code.csv")

    # Kiểm tra nếu mã cổ phiếu không tồn tại
    if selected_stock not in df["Mã"].unique():
        return "⚠️ Mã cổ phiếu không tồn tại!"

    # Lọc dữ liệu theo mã cổ phiếu
    df_selected = df[df["Mã"] == selected_stock]

    # Danh sách cột cần lấy (loại bỏ 'Date')
    columns = ["Date", "ROA (%)", "ROE (%)", "ROS (%)", "EBIT Margin (%)", 
               "Gross Profit Margin (%)"]
    df_selected = df_selected[columns]

    # Chuyển đổi cột Date thành số nguyên để bỏ .0
    df_selected["Date"] = df_selected["Date"].astype(int)

    # Xoay bảng (Date thành cột ngang, chỉ số thành hàng dọc)
    df_transposed = df_selected.set_index("Date").transpose()

    # Đặt lại tên index thành "Chỉ số tài chính" để bỏ cột 'Date' dư thừa
    df_transposed.index.name = None  # Xóa hoàn toàn tên index để không hiển thị "Date"
    df_transposed.columns.name = "Chỉ số tài chính"  # Đây là phần quan trọng!

    # Làm tròn số chỉ còn 2 chữ số sau dấu thập phân
    df_transposed = df_transposed.applymap(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)

    # Chuyển thành bảng HTML có style đẹp hơn
    financial_table = df_transposed.to_html(classes="table table-striped", border=0)

    return financial_table


##Đòn bẩy tài chính
def extract_le_data(selected_stock):
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv("data/market_average_by_code.csv")

    # Kiểm tra nếu mã cổ phiếu không tồn tại
    if selected_stock not in df["Mã"].unique():
        return "⚠️ Mã cổ phiếu không tồn tại!"

    # Lọc dữ liệu theo mã cổ phiếu
    df_selected = df[df["Mã"] == selected_stock]

    # Danh sách cột cần lấy (loại bỏ 'Date')
    columns = ["Date", "D/A (%)", "D/E (%)", "E/A (%)"]
    
    df_selected = df_selected[columns]

    # Chuyển đổi cột Date thành số nguyên để bỏ .0
    df_selected["Date"] = df_selected["Date"].astype(int)

    # Xoay bảng (Date thành cột ngang, chỉ số thành hàng dọc)
    df_transposed = df_selected.set_index("Date").transpose()

    # Đặt lại tên index thành "Chỉ số tài chính" để bỏ cột 'Date' dư thừa
    df_transposed.index.name = None  # Xóa hoàn toàn tên index để không hiển thị "Date"
    df_transposed.columns.name = "Chỉ số tài chính"  # Đây là phần quan trọng!

    # Làm tròn số chỉ còn 2 chữ số sau dấu thập phân
    df_transposed = df_transposed.applymap(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)

    # Chuyển thành bảng HTML có style đẹp hơn
    le_table = df_transposed.to_html(classes="table table-striped", border=0)

    return le_table

def extract_li_data(selected_stock):
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv("data/market_average_by_code.csv")

    # Kiểm tra nếu mã cổ phiếu không tồn tại
    if selected_stock not in df["Mã"].unique():
        return "⚠️ Mã cổ phiếu không tồn tại!"

    # Lọc dữ liệu theo mã cổ phiếu
    df_selected = df[df["Mã"] == selected_stock]

    # Danh sách cột cần lấy (loại bỏ 'Date')
    columns = ["Date", "Current Ratio", "Quick Ratio"]
    
    df_selected = df_selected[columns]

    # Chuyển đổi cột Date thành số nguyên để bỏ .0
    df_selected["Date"] = df_selected["Date"].astype(int)

    # Xoay bảng (Date thành cột ngang, chỉ số thành hàng dọc)
    df_transposed = df_selected.set_index("Date").transpose()

    # Đặt lại tên index thành "Chỉ số tài chính" để bỏ cột 'Date' dư thừa
    df_transposed.index.name = None  # Xóa hoàn toàn tên index để không hiển thị "Date"
    df_transposed.columns.name = "Chỉ số tài chính"  # Đây là phần quan trọng!

    # Làm tròn số chỉ còn 2 chữ số sau dấu thập phân
    df_transposed = df_transposed.applymap(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)

    # Chuyển thành bảng HTML có style đẹp hơn
    li_table = df_transposed.to_html(classes="table table-striped", border=0)

    return li_table

# Lấy dữ liệu định giá từ API
def get_price_data(selected_stock):
    from vnstock import Vnstock
    stock = Vnstock().stock(symbol=selected_stock, source='VCI')
    dinh_gia=stock.finance.ratio(period='year', lang='vi', dropna=True).head()
    return dinh_gia

def extract_dinh_gia_data(selected_stock):
    df = get_price_data(selected_stock)
    df = pd.DataFrame(df)

    # Làm phẳng MultiIndex columns
    df.columns = ['{} - {}'.format(l1, l2) if l1 != '' else l2 for l1, l2 in df.columns]

    # Chọn các cột cần thiết
    columns = ['Meta - Năm', 'Chỉ tiêu định giá - P/E', 'Chỉ tiêu định giá - P/B',
               'Chỉ tiêu định giá - P/S', 'Chỉ tiêu định giá - P/Cash Flow',
               'Chỉ tiêu định giá - EPS (VND)', 'Chỉ tiêu định giá - BVPS (VND)']

    df_selected = df[columns].copy()

    # Đổi tên cột cho gọn gàng
    df_selected.columns = ["Năm", "P/E", "P/B", "P/S", "P/Cash Flow", "EPS (VND)", "BVPS (VND)"]

    # Chuyển đổi cột Năm sang int
    df_selected["Năm"] = df_selected["Năm"].astype(int)

    # Xoay bảng
    df_transposed = df_selected.set_index("Năm").transpose()
    df_transposed = df_transposed[sorted(df_transposed.columns)]
    df_transposed.index.name = None
    df_transposed.columns.name = "Chỉ số tài chính"

    # Làm tròn số
    df_transposed = df_transposed.applymap(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)

    # HTML
    dinh_gia_table = df_transposed.to_html(classes="table table-striped", border=0)

    return dinh_gia_table

def extract_financial_data_from_api_and_analyze(selected_stock):
    df_activity = extract_financial_data(selected_stock)
    df_le = extract_le_data(selected_stock)
    df_li= extract_li_data(selected_stock)
    df_dinh_gia_data = extract_dinh_gia_data(selected_stock)
    # Chuyển đổi HTML thành DataFrame thực sự để có thể phân tích
    df_activity = pd.read_html(df_activity)[0]
    df_le = pd.read_html(df_le)[0]
    df_li = pd.read_html(df_li)[0]
    df_dinh_gia_data = pd.read_html(df_dinh_gia_data)[0]
    # Loại bỏ khoảng trắng trong tên cột
    df_activity.columns = df_activity.columns.str.strip()
    df_le.columns = df_le.columns.str.strip()
    df_li.columns = df_li.columns.str.strip()
    df_dinh_gia_data.columns = df_dinh_gia_data.columns.str.strip()

    # Chuyển hàng đầu tiên thành cột và loại bỏ hàng đầu tiên
    df_activity = df_activity.set_index('Chỉ số tài chính').transpose()
    df_le = df_le.set_index('Chỉ số tài chính').transpose()
    df_li = df_li.set_index('Chỉ số tài chính').transpose()
    df_dinh_gia_data = df_dinh_gia_data.set_index('Chỉ số tài chính').transpose()

# Phân tích tổng hợp tình hình tài chính
    financial_analysis = ""

    # **Phân tích hiệu quả hoạt động**
    roa = df_activity["ROA (%)"].iloc[-1]
    roe = df_activity["ROE (%)"].iloc[-1]
    ros = df_activity["ROS (%)"].iloc[-1]
    ebit_margin = df_activity["EBIT Margin (%)"].iloc[-1]
    gross_margin = df_activity["Gross Profit Margin (%)"].iloc[-1]

    financial_analysis += f"**Phân tích hiệu quả hoạt động**:\n"
    if roa > 5:
        financial_analysis += f"- ROA ({roa}%) cho thấy công ty có khả năng sinh lời tốt từ tài sản.\n"
    else:
        financial_analysis += f"- ROA ({roa}%) thấp có thể cho thấy công ty chưa tận dụng tài sản hiệu quả.\n"
    
    if roe > 10:
        financial_analysis += f"- ROE ({roe}%) cho thấy công ty có khả năng sinh lời cao từ vốn chủ sở hữu.\n"
    else:
        financial_analysis += f"- ROE ({roe}%) thấp có thể là dấu hiệu của việc sử dụng vốn chưa hiệu quả.\n"

    if ros > 10:
        financial_analysis += f"- ROS ({ros}%) cho thấy công ty có tỷ suất lợi nhuận từ doanh thu tốt.\n"
    else:
        financial_analysis += f"- ROS ({ros}%) thấp có thể là dấu hiệu của chi phí sản xuất hoặc bán hàng cao.\n"

    financial_analysis += f"- EBIT Margin ({ebit_margin}%) và Gross Profit Margin ({gross_margin}%) cho thấy công ty có biên lợi nhuận tốt trong hoạt động sản xuất kinh doanh .\n"

    # **Phân tích đòn bẩy tài chính**
    da = df_le["D/A (%)"].iloc[-1]
    de = df_le["D/E (%)"].iloc[-1]
    ea = df_le["E/A (%)"].iloc[-1]

    financial_analysis += f"\n**Phân tích đòn bẩy tài chính**:\n"
    if da > 50:
        financial_analysis += f"- D/A ({da}%) cao cho thấy công ty đang sử dụng nhiều nợ trong cấu trúc tài chính.\n"
    else:
        financial_analysis += f"- D/A ({da}%) thấp cho thấy công ty ít sử dụng nợ.\n"
    
    if de > 100:
        financial_analysis += f"- D/E ({de}%) cao, có thể là dấu hiệu công ty đang chịu áp lực nợ nần lớn.\n"
    else:
        financial_analysis += f"- D/E ({de}%) thấp, có thể cho thấy công ty đang duy trì nợ ở mức độ an toàn.\n"
    
    if ea > 50:
        financial_analysis += f"- E/A ({ea}%) cao cho thấy công ty chủ yếu sử dụng vốn chủ sở hữu.\n"
    else:
        financial_analysis += f"- E/A ({ea}%) thấp cho thấy công ty sử dụng nợ nhiều hơn so với vốn chủ sở hữu.\n"

    # **Phân tích thanh khoản**
    current_ratio = df_li["Current Ratio"].iloc[-1]
    quick_ratio = df_li["Quick Ratio"].iloc[-1]

    financial_analysis += f"\n**Phân tích thanh khoản**:\n"
    if current_ratio > 1.5:
        financial_analysis += f"- Current Ratio ({current_ratio}) cho thấy công ty có khả năng thanh toán nợ ngắn hạn tốt.\n"
    else:
        financial_analysis += f"- Current Ratio ({current_ratio}) thấp, công ty có thể gặp khó khăn trong thanh toán nợ ngắn hạn.\n"
    
    if quick_ratio > 1:
        financial_analysis += f"- Quick Ratio ({quick_ratio}) cho thấy công ty có khả năng thanh toán các khoản nợ ngắn hạn mà không cần bán hàng tồn kho.\n"
    else:
        financial_analysis += f"- Quick Ratio ({quick_ratio}) thấp có thể chỉ ra rằng công ty gặp khó khăn khi thanh toán nợ mà không sử dụng hàng tồn kho.\n"

    # **Phân tích định giá **
    pe_value = df_dinh_gia_data['P/E'].iloc[-1]  # Lấy P/E năm gần nhất
    
    financial_analysis += f"\n**Phân tích định giá (P/E)**:\n"
    if pe_value > 25:
        financial_analysis += f"- P/E ({pe_value}) cho thấy cổ phiếu có thể đang bị định giá cao.\n"
    elif pe_value < 10:
        financial_analysis += f"- P/E ({pe_value}) cho thấy cổ phiếu có thể đang bị định giá thấp.\n"
    else:
        financial_analysis += f"- P/E ({pe_value}) cho thấy cổ phiếu có mức định giá hợp lý.\n"


    # Trả về kết quả phân tích tổng hợp
    return financial_analysis

#Biểu đồ doanh thu và lợi nhuận 

def bieu_do_doanh_loi(selected_stock):
    # Đọc dữ liệu từ file CSV
    data = pd.read_csv("data/KQKD.csv")

    if selected_stock not in data["Mã"].unique():
        print("⚠️ Mã cổ phiếu không tồn tại!")
        return

    # Lọc dữ liệu theo mã cổ phiếu
    df_selected = data[data["Mã"] == selected_stock]

    years = df_selected['Năm'].values
    if len(years) == 0:
        raise ValueError("⚠️ Không có dữ liệu năm cho mã cổ phiếu này.")
    revenue_data = df_selected['Doanh thu thuần'].values / 1e9
    profit_data = df_selected['Lợi nhuận sau thuế thu nhập doanh nghiệp'].values / 1e9

        # Tạo biểu đồ doanh thu và lợi nhuận
    plt.figure(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(years))
        
    plt.bar(x - width/2, revenue_data, width, label='Doanh thu thuần', color='#3498db')
    plt.bar(x + width/2, profit_data, width, label='Lợi nhuận sau thuế', color='#2ecc71')
        
    plt.xlabel('Năm', fontsize=12)
    plt.ylabel('Tỷ VNĐ', fontsize=12)
    plt.title('Doanh thu và Lợi nhuận qua các năm', fontsize=14, fontweight='bold')
    plt.xticks(x, [str(int(year)) for year in years], fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
        
        # Thêm số liệu lên biểu đồ
    for i, v in enumerate(revenue_data):
        plt.text(i - width/2, v + 0.1, f'{v:.1f}', ha='center', fontsize=9)
        
    for i, v in enumerate(profit_data):
        plt.text(i + width/2, v + 0.1, f'{v:.1f}', ha='center', fontsize=9)

    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    
    # Chuyển đổi hình ảnh thành base64
    bieu_do_doanh_loi = base64.b64encode(img.getvalue()).decode("utf-8")  
    # Đóng lại để không giữ tài nguyên
    plt.close()

    return bieu_do_doanh_loi

#Biểu đồ cơ cấu nợ
def bieu_do_no_von(selected_stock):
    # Đọc dữ liệu từ file CSV
    data = pd.read_csv("data/BCDKT.csv")

    if selected_stock not in data["Mã"].unique():
        print("⚠️ Mã cổ phiếu không tồn tại!")
        return

    # Lọc dữ liệu theo mã cổ phiếu
    df_selected = data[data["Mã"] == selected_stock]

    years = df_selected['Năm'].values
    if len(years) == 0:
        raise ValueError("⚠️ Không có dữ liệu năm cho mã cổ phiếu này.")

    # Lấy dữ liệu tài chính
    assets_data = df_selected['TỔNG CỘNG TÀI SẢN'].values / 1e9
    liabilities_data = df_selected['NỢ PHẢI TRẢ'].values / 1e9
    equity_data = df_selected['VỐN CHỦ SỞ HỮU'].values / 1e9

    x = np.arange(len(years))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, assets_data, width, label='Tổng tài sản', color='#3498db')
    plt.bar(x, liabilities_data, width, label='Nợ phải trả', color='#e74c3c')
    plt.bar(x + width, equity_data, width, label='Vốn chủ sở hữu', color='#2ecc71')

    plt.xlabel('Năm', fontsize=12)
    plt.ylabel('Tỷ VNĐ', fontsize=12)
    plt.title(f'Cơ cấu tài sản, nợ, vốn qua các năm - {selected_stock}', fontsize=14, fontweight='bold')
    plt.xticks(x, [str(int(year)) for year in years], fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    for i, v in enumerate(assets_data):
        if v > 0:
            plt.text(i - width, v, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

    for i, v in enumerate(liabilities_data):
        if v > 0:
            plt.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

    for i, v in enumerate(equity_data):
        if v > 0:
            plt.text(i + width, v, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    
    # Chuyển đổi hình ảnh thành base64
    bieu_do_no_von = base64.b64encode(img.getvalue()).decode("utf-8")  
    # Đóng lại để không giữ tài nguyên
    plt.close()

    return bieu_do_no_von

def bieu_do_so_sanh(selected_stock):
    # Đọc dữ liệu từ file CSV
    data = pd.read_csv("data/market_average_by_code.csv")
    công_ty_data = pd.read_csv("data/KQKD.csv")

    if selected_stock not in data["Mã"].unique():
        print("⚠️ Mã cổ phiếu không tồn tại!")
        return

    # Lọc dữ liệu công ty được chọn
    df_selected = data[data["Mã"] == selected_stock]

    if df_selected.empty:
        raise ValueError("⚠️ Không có dữ liệu cho mã cổ phiếu này trong file ROA/ROE/ROS.")

    # Lấy ngành ICB cấp 1 từ file công ty
    ngành = công_ty_data[công_ty_data["Mã"] == selected_stock]["Ngành ICB - cấp 1"].values
    if len(ngành) == 0:
        raise ValueError("⚠️ Không tìm thấy ngành của mã cổ phiếu.")
    ngành = ngành[0]

    # Lọc các công ty cùng ngành từ KQKD rồi match với bảng ROA
    cùng_ngành_mã = công_ty_data[công_ty_data["Ngành ICB - cấp 1"] == ngành]["Mã"].unique()
    ngành_data = data[data["Mã"].isin(cùng_ngành_mã)]

    # Tính trung bình ngành
    metrics = ['ROA (%)', 'ROE (%)', 'ROS (%)']
    try:
        company_values = [df_selected[m].mean() for m in metrics]
        sector_values = [ngành_data[m].mean() for m in metrics]
    except KeyError as e:
        print(f"⚠️ Cột bị thiếu trong dữ liệu: {e}")
        return

    # Vẽ biểu đồ
    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, company_values, width, label=selected_stock, color='#3498db')
    plt.bar(x + width/2, sector_values, width, label='Trung bình ngành', color='#e74c3c')

    plt.xlabel('Chỉ số (Trung bình 5 năm)', fontsize=12)
    plt.ylabel('Phần trăm (%)', fontsize=12)
    plt.title(f'So sánh {selected_stock} với trung bình ngành ({ngành})', fontsize=14, fontweight='bold')
    plt.xticks(x, metrics, fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    for i, v in enumerate(company_values):
        plt.text(i - width/2, v + 0.3, f'{v:.1f}%', ha='center', fontsize=9)

    for i, v in enumerate(sector_values):
        plt.text(i + width/2, v + 0.3, f'{v:.1f}%', ha='center', fontsize=9)

     # Lưu biểu đồ vào buffer
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    
    # Chuyển đổi hình ảnh thành base64
    bieu_do_so_sanh = base64.b64encode(img.getvalue()).decode("utf-8")
    
    # Đóng lại để không giữ tài nguyên
    plt.close()

    return bieu_do_so_sanh

#Vẽ pie_chart 

def ve_pie_chart_top5(mã_công_ty):
    # Đọc dữ liệu
    df = pd.read_csv("data/KQKD.csv")

    # Làm sạch
    df = df.dropna(subset=['Năm', 'Doanh thu bán hàng và cung cấp dịch vụ'])
    df['Năm'] = df['Năm'].astype(int)

    # Lọc theo công ty
    công_ty_data = df[df['Mã'] == mã_công_ty]
    if công_ty_data.empty:
        print(f"Không tìm thấy dữ liệu cho công ty {mã_công_ty}.")
        return

    # Xác định ngành và năm mới nhất
    ngành = công_ty_data['Ngành ICB - cấp 1'].iloc[0]
    năm_mới_nhất = công_ty_data['Năm'].max()

    # Lọc các công ty cùng ngành và cùng năm
    ngành_data = df[(df['Ngành ICB - cấp 1'] == ngành) & (df['Năm'] == năm_mới_nhất)]

    # Tính tổng doanh thu theo mã
    doanh_thu_theo_cty = ngành_data.groupby('Mã')['Doanh thu bán hàng và cung cấp dịch vụ'].sum().reset_index()

    # Tính tổng doanh thu toàn ngành
    tổng_doanh_thu = doanh_thu_theo_cty['Doanh thu bán hàng và cung cấp dịch vụ'].sum()

    # Thêm cột thị phần %
    doanh_thu_theo_cty['Thị phần (%)'] = doanh_thu_theo_cty['Doanh thu bán hàng và cung cấp dịch vụ'] / tổng_doanh_thu * 100

    # Lấy top 5
    top5 = doanh_thu_theo_cty.sort_values(by='Thị phần (%)', ascending=False).head(5)

    # Gộp phần còn lại
    phần_còn_lại = 100 - top5['Thị phần (%)'].sum()
    top5 = top5._append({'Mã': 'Khác', 'Thị phần (%)': phần_còn_lại}, ignore_index=True)

    # Vẽ Pie Chart
    labels = [f"{row['Mã']} ({row['Thị phần (%)']:.1f}%)" for _, row in top5.iterrows()]
    sizes = top5['Thị phần (%)']
    colors = plt.cm.Paired.colors[:len(labels)]

    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 12}
    )
    ax.axis('equal')
    plt.title(f"Top 5 thị phần doanh thu ngành {ngành} - Năm {năm_mới_nhất}", fontsize=14)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    
    # Chuyển đổi hình ảnh thành base64
    pie_chart = base64.b64encode(img.getvalue()).decode("utf-8")
    
    # Đóng lại để không giữ tài nguyên
    plt.close()
    return pie_chart

#so sánh với các công ty trong cùng ngành 
def so_sanh_chi_so_tai_chinh(mã_công_ty):
    # Đọc dữ liệu từ file "KQKD.csv"
    df = pd.read_csv("data/KQKD.csv")

    # Làm sạch dữ liệu và chỉ lấy dữ liệu năm 2024
    df = df.dropna(subset=['Năm', 'Doanh thu bán hàng và cung cấp dịch vụ'])
    df['Năm'] = df['Năm'].astype(int)
    df_2024 = df[df['Năm'] == 2024]

    # Lọc dữ liệu của công ty theo mã cổ phiếu
    công_ty_data = df_2024[df_2024['Mã'] == mã_công_ty]
    if công_ty_data.empty:
        print(f"Không tìm thấy dữ liệu cho công ty {mã_công_ty}.")
        return

    # Xác định ngành và năm mới nhất (chỉ lấy năm 2024)
    ngành = công_ty_data['Ngành ICB - cấp 1'].iloc[0]
    năm_mới_nhất = 2024

    # Lọc các công ty cùng ngành và cùng năm
    ngành_data = df_2024[(df_2024['Ngành ICB - cấp 1'] == ngành) & (df_2024['Năm'] == năm_mới_nhất)]

    # Tính tổng doanh thu theo mã
    doanh_thu_theo_cty = ngành_data.groupby('Mã')['Doanh thu bán hàng và cung cấp dịch vụ'].sum().reset_index()

    # Tính tổng doanh thu toàn ngành
    tổng_doanh_thu = doanh_thu_theo_cty['Doanh thu bán hàng và cung cấp dịch vụ'].sum()

    # Thêm cột thị phần % cho mỗi công ty
    doanh_thu_theo_cty['Thị phần (%)'] = doanh_thu_theo_cty['Doanh thu bán hàng và cung cấp dịch vụ'] / tổng_doanh_thu * 100

    # Lấy top 5 công ty có thị phần lớn nhất
    top5 = doanh_thu_theo_cty.sort_values(by='Thị phần (%)', ascending=False).head(5)

    # Đọc dữ liệu ROE, ROA, ROS từ file "market_average_by_code.csv"
    df_roa_roe_ros = pd.read_csv("data/market_average_by_code.csv")
    df_roa_roe_ros = df_roa_roe_ros.dropna(subset=['ROE (%)', 'ROA (%)', 'ROS (%)'])

    # Lọc dữ liệu ROE, ROA, ROS của Top 5 công ty và lấy giá trị trung bình (nếu có nhiều dòng)
    top5_data = df_roa_roe_ros[df_roa_roe_ros['Mã'].isin(top5['Mã'])][['Mã', 'ROE (%)', 'ROA (%)', 'ROS (%)']]
    top5_data = top5_data.groupby('Mã').agg({'ROE (%)': 'mean', 'ROA (%)': 'mean', 'ROS (%)': 'mean'}).reset_index()

    # Thêm cột "Năm đầu tiên" vào bảng: nhóm theo "Mã" và lấy giá trị năm nhỏ nhất
    năm_đầu_tiên = df_2024.groupby('Mã')['Năm'].min().loc[top5['Mã']]
    
    # Thêm cột "Năm đầu tiên" vào top5_data (sẽ không sử dụng trong biểu đồ)
    top5_data['Năm đầu tiên'] = top5_data['Mã'].map(năm_đầu_tiên)

    # Vẽ biểu đồ cột so sánh ROE, ROA, ROS của các công ty Top 5, bỏ cột "Năm đầu tiên"
    top5_data.set_index('Mã')[['ROE (%)', 'ROA (%)', 'ROS (%)']].plot(kind='bar', figsize=(10, 6))
    plt.title(f"So sánh ROE, ROA, ROS - Top 5 thị phần ngành {ngành} năm {năm_mới_nhất}", fontsize=14)
    plt.ylabel("Tỷ lệ (%)")
    plt.xticks(rotation=0)
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    
    # Chuyển đổi hình ảnh thành base64
    top_chart = base64.b64encode(img.getvalue()).decode("utf-8")
    
    # Đóng lại để không giữ tài nguyên
    plt.close()
    return top_chart


#***Bảng cân đối kế toán**
def extract_balance_sheet(selected_stock):
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv("data/BCDKT.csv")

    # Kiểm tra nếu mã cổ phiếu không tồn tại
    if selected_stock not in df["Mã"].unique():
        return "⚠️ Mã cổ phiếu không tồn tại!"

    # Lọc dữ liệu theo mã cổ phiếu
    df_selected = df[df["Mã"] == selected_stock]

    # Danh sách cột cần lấy (loại bỏ 'Date')
    columns = ["Năm", "TÀI SẢN NGẮN HẠN","Tiền và tương đương tiền","Đầu tư tài chính ngắn hạn","Các khoản phải thu ngắn hạn","Hàng tồn kho, ròng",
               "Tài sản ngắn hạn khác","TÀI SẢN DÀI HẠN","Phải thu dài hạn","Tài sản cố định","GTCL TSCĐ hữu hình","GTCL Tài sản thuê tài chính",
               "GTCL tài sản cố định vô hình","Giá trị ròng tài sản đầu tư","Tài sản dở dang dài hạn"
               ,"Đầu tư dài hạn","Tài sản dài hạn khác","Lợi thế thương mại","TỔNG CỘNG TÀI SẢN"
               ,"NỢ PHẢI TRẢ","Nợ ngắn hạn","Phải trả người bán ngắn hạn","Người mua trả tiền trước ngắn hạn","Doanh thu chưa thực hiện ngắn hạn"
               ,"Vay và nợ thuê tài chính ngắn hạn","Nợ dài hạn","VỐN CHỦ SỞ HỮU","Vốn và các quỹ","Vốn góp của chủ sở hữu","Thặng dư vốn cổ phần","Vốn khác",
               "Lãi chưa phân phối","LNST chưa phân phối lũy kế đến cuối kỳ trước","LNST chưa phân phối kỳ này","Lợi ích cổ đông không kiểm soát"
               ,"Nguồn kinh phí và quỹ khác","TỔNG CỘNG NGUỒN VỐN"]
    
    df_selected = df_selected[columns]

    # Chuyển đổi cột Date thành số nguyên để bỏ .0
    df_selected["Năm"] = df_selected["Năm"].astype(int)

    # Xoay bảng (Date thành cột ngang, chỉ số thành hàng dọc)
    df_transposed = df_selected.set_index("Năm").transpose()

    # Đặt lại tên index thành "Chỉ số tài chính" để bỏ cột 'Date' dư thừa
    df_transposed.index.name = None  # Xóa hoàn toàn tên index để không hiển thị "Date"
    df_transposed.columns.name = "Cân đối kế toán - Triệu VND"  # Đây là phần quan trọng!

    # Làm tròn số chỉ còn 2 chữ số sau dấu thập phân
    df_transposed = df_transposed.applymap(lambda x: round(x / 1_000_000, 2) if isinstance(x, (int, float)) else x)
    df_transposed = df_transposed.applymap(format_number_us)

    # Chuyển thành bảng HTML có style đẹp hơn
    balance_sheet = df_transposed.to_html(classes="table table-striped", border=0)

    return balance_sheet

##Báo cáo KQKD*
#***Bảng cân đối kế toán**
def extract_income_statement(selected_stock):
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv("data/KQKD.csv")

    # Kiểm tra nếu mã cổ phiếu không tồn tại
    if selected_stock not in df["Mã"].unique():
        return "⚠️ Mã cổ phiếu không tồn tại!"

    # Lọc dữ liệu theo mã cổ phiếu
    df_selected = df[df["Mã"] == selected_stock]

    # Danh sách cột cần lấy (loại bỏ 'Date')
    columns = ["Năm", "Doanh thu bán hàng và cung cấp dịch vụ","Doanh thu thuần","Lợi nhuận gộp về bán hàng và cung cấp dịch vụ","Doanh thu hoạt động tài chính","Chi phí tài chính",
               "Trong đó: Chi phí lãi vay","Lãi/lỗ từ công ty liên doanh","Chi phí bán hàng","Chi phí quản lý doanh  nghiệp","Lợi nhuận thuần từ hoạt động kinh doanh","Lợi nhuận khác",
               "Tổng lợi nhuận kế toán trước thuế","Chi phí thuế thu nhập doanh nghiệp","Lợi nhuận sau thuế thu nhập doanh nghiệp"
               ,"Lợi ích của cổ đông thiểu số","Cổ đông của Công ty mẹ","Lãi cơ bản trên cổ phiếu","Lãi trước thuế"
               ,"Khấu hao TSCĐ","Tổng lợi nhuận kế toán trước thuế","Lợi nhuận sau thuế thu nhập doanh nghiệp"]
    
    df_selected = df_selected[columns]

    # Chuyển đổi cột Date thành số nguyên để bỏ .0
    df_selected["Năm"] = df_selected["Năm"].astype(int)

    # Xoay bảng (Date thành cột ngang, chỉ số thành hàng dọc)
    df_transposed = df_selected.set_index("Năm").transpose()

    # Đặt lại tên index thành "Chỉ số tài chính" để bỏ cột 'Date' dư thừa
    df_transposed.index.name = None  # Xóa hoàn toàn tên index để không hiển thị "Date"
    df_transposed.columns.name = "Kết quả kinh doanh - Triệu VND"  # Đây là phần quan trọng!

    # Làm tròn số chỉ còn 2 chữ số sau dấu thập phân
    df_transposed = df_transposed.applymap(lambda x: round(x / 1_000_000, 2) if isinstance(x, (int, float)) else x)
    df_transposed = df_transposed.applymap(format_number_us)

    # Chuyển thành bảng HTML có style đẹp hơn
    income_statement = df_transposed.to_html(classes="table table-striped", border=0)

    return income_statement

#Biểu đồ doanh thu lợi nhuận 
def vẽ_biểu_đồ_tăng_trưởng(df, mã_công_ty):
    # Lọc dữ liệu của công ty theo mã cổ phiếu
    công_ty_data = df[df['Mã'] == mã_công_ty]

    if công_ty_data.empty:
        print(f"Không tìm thấy dữ liệu cho công ty với mã cổ phiếu {mã_công_ty}.")
        return

    # Lấy Ngành ICB cấp 1 của công ty
    ngành = công_ty_data['Ngành ICB - cấp 1'].iloc[0]

    # Lọc dữ liệu của tất cả công ty trong cùng ngành
    ngành_data = df[df['Ngành ICB - cấp 1'] == ngành]

    # Loại bỏ các dòng có giá trị NaN trong cột 'Năm'
    df = df.dropna(subset=['Năm'])

    # Ép kiểu cột 'Năm' thành kiểu int (số nguyên)
    df['Năm'] = df['Năm'].astype(int)

    # Tính tỷ lệ tăng trưởng doanh thu và lợi nhuận của công ty
    công_ty_data = công_ty_data[['Năm', 'Doanh thu bán hàng và cung cấp dịch vụ', 'Lợi nhuận sau thuế thu nhập doanh nghiệp']]
    công_ty_data['Tăng trưởng Doanh thu'] = công_ty_data['Doanh thu bán hàng và cung cấp dịch vụ'].pct_change() * 100
    công_ty_data['Tăng trưởng Lợi nhuận'] = công_ty_data['Lợi nhuận sau thuế thu nhập doanh nghiệp'].pct_change() * 100

    # Tính tổng doanh thu và lợi nhuận của ngành theo năm
    ngành_data = ngành_data[['Năm', 'Doanh thu bán hàng và cung cấp dịch vụ', 'Lợi nhuận sau thuế thu nhập doanh nghiệp']]
    ngành_tăng_trưởng = ngành_data.groupby('Năm').agg({
        'Doanh thu bán hàng và cung cấp dịch vụ': 'sum',
        'Lợi nhuận sau thuế thu nhập doanh nghiệp': 'sum'
    })

    # Tính tỷ lệ tăng trưởng doanh thu và lợi nhuận của ngành
    ngành_tăng_trưởng['Tăng trưởng Doanh thu'] = ngành_tăng_trưởng['Doanh thu bán hàng và cung cấp dịch vụ'].pct_change() * 100
    ngành_tăng_trưởng['Tăng trưởng Lợi nhuận'] = ngành_tăng_trưởng['Lợi nhuận sau thuế thu nhập doanh nghiệp'].pct_change() * 100

    # Vẽ biểu đồ tăng trưởng doanh thu
    plt.figure(figsize=(12, 6))
    plt.plot(công_ty_data['Năm'], công_ty_data['Tăng trưởng Doanh thu'], label=f'Tăng trưởng Doanh thu {mã_công_ty}', marker='o', color='blue')
    plt.plot(ngành_tăng_trưởng.index, ngành_tăng_trưởng['Tăng trưởng Doanh thu'], label=f'Tăng trưởng Doanh thu Ngành {ngành}', marker='x', color='orange')
    plt.title('So sánh Tăng trưởng Doanh thu giữa Công ty và Ngành')
    plt.xlabel('Năm')
    plt.ylabel('Tỷ lệ tăng trưởng (%)')
    plt.xticks(công_ty_data['Năm'])  # Chỉ hiển thị những năm có trong dữ liệu của công ty

    # Thêm giá trị lên các điểm trong biểu đồ doanh thu
    for i, txt in enumerate(công_ty_data['Tăng trưởng Doanh thu']):
        plt.text(công_ty_data['Năm'].iloc[i], txt, f'{txt:.2f}%', color='blue', ha='center', va='bottom', fontsize=9)

    # Thêm giá trị lên các điểm trong biểu đồ doanh thu của ngành
    for i, txt in enumerate(ngành_tăng_trưởng['Tăng trưởng Doanh thu']):
        plt.text(ngành_tăng_trưởng.index[i], txt, f'{txt:.2f}%', color='orange', ha='center', va='bottom', fontsize=9)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    revenue_chart = f'<img src="data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}"/>'
    plt.close()


    # Vẽ biểu đồ tăng trưởng lợi nhuận
    plt.figure(figsize=(12, 6))
    plt.plot(công_ty_data['Năm'], công_ty_data['Tăng trưởng Lợi nhuận'], label=f'Tăng trưởng Lợi nhuận {mã_công_ty}', marker='o', color='green')
    plt.plot(ngành_tăng_trưởng.index, ngành_tăng_trưởng['Tăng trưởng Lợi nhuận'], label=f'Tăng trưởng Lợi nhuận Ngành {ngành}', marker='x', color='red')
    plt.title('So sánh Tăng trưởng Lợi nhuận giữa Công ty và Ngành')
    plt.xlabel('Năm')
    plt.ylabel('Tỷ lệ tăng trưởng (%)')
    plt.xticks(công_ty_data['Năm'])  # Chỉ hiển thị những năm có trong dữ liệu của công ty

    # Thêm giá trị lên các điểm trong biểu đồ lợi nhuận
    for i, txt in enumerate(công_ty_data['Tăng trưởng Lợi nhuận']):
        plt.text(công_ty_data['Năm'].iloc[i], txt, f'{txt:.2f}%', color='green', ha='center', va='bottom', fontsize=9)

    # Thêm giá trị lên các điểm trong biểu đồ lợi nhuận của ngành
    for i, txt in enumerate(ngành_tăng_trưởng['Tăng trưởng Lợi nhuận']):
        plt.text(ngành_tăng_trưởng.index[i], txt, f'{txt:.2f}%', color='red', ha='center', va='bottom', fontsize=9)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    profit_chart = f'<img src="data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}"/>'
    plt.close()
    return revenue_chart, profit_chart

# Khởi tạo Flask app
app = Flask(__name__)

# Route cho trang chủ
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Lấy dữ liệu từ form
        selected_date = request.form.get("report_date")
        selected_stock = request.form.get("company_code")

        if selected_date:
            selected_date = pd.to_datetime(selected_date)  # Chuyển ngày về dạng datetime

        # Lấy thông tin công ty
        company_name, industry_full_description = get_company_info(selected_stock)
        html_table = market_overview(selected_date=None, selected_stock=None)
        vnindex_summary, vnindex_chart= vnindex_overview(selected_date)
        market_cap_chart= plot_market_cap(selected_date=None)
        company_inf = get_company_inf(selected_stock)
        stock_data = get_company_overview(selected_stock)
        if not stock_data.empty:
             stock_data_dict = stock_data.iloc[0].to_dict()  # Chuyển thành dictionary
        else:
             stock_data_dict = None  # Nếu không có dữ liệu, truyền None
        company_position= get_company_position(selected_stock)
        if not company_position.empty:
            company_position_dict = company_position.iloc[0].to_dict()  # Chuyển thành dictionary
            company_position_dict=company_position_dict.get("business_strategies", "").strip()
        else:
            company_position_dict = None  # Nếu không có dữ liệu, truyền None

        company_business_strategy=get_company_business_strategy(selected_stock)
        if not company_business_strategy.empty:
            company_business_strategy_dict = company_business_strategy.iloc[0].to_dict()  # Chuyển thành dictionary
            company_business_strategy_dict=company_business_strategy_dict.get("key_developments", "").strip()
        else:
            company_business_strategy_dict = None  # Nếu không có dữ liệu, truyền None
        
        stock_overview = get_stock_overview(selected_date, selected_stock)

        if not stock_overview.empty:
             stock_overview_dict = stock_overview.iloc[0].to_dict()  

        # Chuyển đổi số sang định dạng Mỹ
             for key in ["Value","Prev_Value", "H52W", "L52W", "Volume_Today", "MarketCap"]:
                if key in stock_overview_dict:
                    stock_overview_dict[key] = format_number_us(stock_overview_dict[key])
        else:
                stock_overview_dict = None
        

        df_final=get_market_overview(selected_date, selected_stock)
        price_and_vnindex_chart, price_and_volume_chart= plot_charts(df_final, selected_stock)

        
       
    
        # Tạo báo cáo
        report_data = market_overview(selected_date, selected_stock)
        return render_template("report.html", price_and_vnindex_chart=price_and_vnindex_chart, price_and_volume_chart=price_and_volume_chart,
                               stock_price=stock_overview_dict,business_strategy=company_business_strategy_dict, stock_data=stock_data_dict,
                               company_profile = company_inf,company_position=company_position_dict, vnindex_summary = vnindex_summary,
                                market_cap_chart=market_cap_chart, vnindex_chart=vnindex_chart, report_data=report_data, market_overview = html_table,
                                report_date= selected_date, company_code=selected_stock, company_name=company_name, industry_full_description=industry_full_description)

    # Nếu là GET, chỉ hiển thị form
    return render_template("index.html")


# Route cho báo cáo (report)
@app.route("/report", methods=["GET", "POST"])
def report():
    selected_date = request.form.get("report_date")
    selected_stock = request.form.get("company_code")
    selected_date = pd.to_datetime(selected_date)
    

    # Lấy thông tin công ty
    company_name, industry_full_description = get_company_info(selected_stock)
    html_table = market_overview(selected_date=None, selected_stock=None)
    vnindex_summary, vnindex_chart,vnindex_percent_change= vnindex_overview(selected_date)
    market_cap_chart= plot_market_cap(selected_date=None)
    company_inf = get_company_inf(selected_stock)
    stock_data = get_company_overview(selected_stock)
    if not stock_data.empty:
        stock_data_dict = stock_data.iloc[0].to_dict()  # Chuyển thành dictionary
    else:
        stock_data_dict = None  # Nếu không có dữ liệu, truyền None
    company_position= get_company_position(selected_stock)
    if not company_position.empty:
        company_position_dict = company_position.iloc[0].to_dict()  # Chuyển thành dictionary
        company_position_dict=company_position_dict.get("business_strategies", "").strip()
    else:
        company_position_dict = None  # Nếu không có dữ liệu, truyền None

    company_business_strategy=get_company_business_strategy(selected_stock)
    if not company_business_strategy.empty:
        company_business_strategy_dict = company_business_strategy.iloc[0].to_dict()  # Chuyển thành dictionary
        company_business_strategy_dict=company_business_strategy_dict.get("key_developments", "").strip()
    else:
        company_business_strategy_dict = None  # Nếu không có dữ liệu, truyền None


    stock_overview = get_stock_overview(selected_date, selected_stock)
    if not stock_overview.empty:
             stock_overview_dict = stock_overview.iloc[0].to_dict()  

        # Chuyển đổi số sang định dạng Mỹ
             for key in ["Value","Prev_Value", "H52W", "L52W", "Volume_Today", "MarketCap"]:
                if key in stock_overview_dict:
                    stock_overview_dict[key] = format_number_us(stock_overview_dict[key])
    else:
                stock_overview_dict = None


    stock_an = get_stock_overview(selected_date, selected_stock)
    stock_over=analyze_stock(stock_an, vnindex_percent_change)

    df_final=get_market_overview(selected_date, selected_stock)
    price_and_vnindex_chart,  price_and_volume_chart= plot_charts(df_final, selected_stock)

    financial_table= extract_financial_data(selected_stock)
    foreign_trading_chart=plot_foreign_trading(selected_stock, selected_date)
    balance_sheet=extract_balance_sheet(selected_stock)
    income_statement=extract_income_statement(selected_stock)
    trend_MA_chart=plot_trend_MA_chart(selected_date)

    df = load_data_NN("data/output.csv")
    base64_khop, base64_thoa_thuan= plot_investor_flow(df, selected_date)

    file_path = "data/output.csv"  # Đường dẫn đến file CSV
    df_GD = load_data_GD(file_path)
    df_filtered = investor_flow(df_GD, selected_date)
    # Danh sách các cột giao dịch (cần thay đổi cho đúng với dữ liệu thực tế)
    matching_columns = ['cá_nhân_tổng_gt_ròng', 'tổ_chức_trong_nước_tổng_gt_ròng', 'nước_ngoài_tổng_gt_ròng']
    matching_columns = list(matching_columns)
    # Hiển thị biểu đồ kết hợp
    combined_plot=create_combined_plot(df_filtered, matching_columns)

    df_KQKD = pd.read_csv("data/KQKD.csv")
    revenue_chart, profit_chart=vẽ_biểu_đồ_tăng_trưởng(df_KQKD, selected_stock)
    le_indicators=extract_le_data(selected_stock)
    li_indicators=extract_li_data(selected_stock)
    dinh_gia_indicators=extract_dinh_gia_data(selected_stock)
    financial_analysis=extract_financial_data_from_api_and_analyze(selected_stock)
    co_cau_chart=bieu_do_no_von(selected_stock)
    doanh_loi_chart=bieu_do_doanh_loi(selected_stock)
    so_sanh_chart=bieu_do_so_sanh(selected_stock)
    pie_chart=ve_pie_chart_top5(selected_stock)
    top_chart=so_sanh_chi_so_tai_chinh(selected_stock)
    end_date = selected_date # Ngày kết thúc
    value_trends=plot_sector_value_trends(end_date, MERGED_DF, price, top_sectors)


       
    # Tạo báo cáo
    report_data = market_overview(selected_date, selected_stock)

    report_date = selected_date.strftime('%Y-%m-%d')
    return render_template("report.html",stock_over=stock_over,vnindex_change=vnindex_percent_change,value_trends=value_trends,top_chart=top_chart, pie_chart=pie_chart, so_sanh_chart=so_sanh_chart, co_cau_chart=co_cau_chart, doanh_loi_chart=doanh_loi_chart,financial_analysis= financial_analysis,dinh_gia_indicators=dinh_gia_indicators, li_indicators=li_indicators, le_indicators=le_indicators, revenue_chart=revenue_chart, profit_chart=profit_chart, combined_plot=combined_plot, base64_khop=base64_khop, base64_thoa_thuan=base64_thoa_thuan, ma_chart=trend_MA_chart, income_statement= income_statement, balance_sheet=balance_sheet, foreign_trading_chart=foreign_trading_chart, financial_indicators=financial_table,price_and_vnindex_chart= price_and_vnindex_chart, price_and_volume_chart=price_and_volume_chart,
                           stock_price=stock_overview_dict, business_strategy=company_business_strategy_dict, stock_data=stock_data_dict,
                           company_profile = company_inf,company_position=company_position_dict, vnindex_summary = vnindex_summary,
                           market_cap_chart=market_cap_chart, vnindex_chart=vnindex_chart, market_overview = html_table, report_data=report_data,
                             report_date= report_date,company_code=selected_stock, company_name=company_name, industry_full_description=industry_full_description)


@app.route("/download_report", methods=["POST"])
def download_report():
    selected_date = request.form.get("report_date")
    selected_stock = request.form.get("company_code")
    selected_date = pd.to_datetime(selected_date)
    

    # Lấy thông tin công ty
    company_name, industry_full_description = get_company_info(selected_stock)
    html_table = market_overview(selected_date=None, selected_stock=None)
    vnindex_summary, vnindex_chart,vnindex_percent_change= vnindex_overview(selected_date)
    market_cap_chart= plot_market_cap(selected_date=None)
    company_inf = get_company_inf(selected_stock)
    stock_data = get_company_overview(selected_stock)
    if not stock_data.empty:
        stock_data_dict = stock_data.iloc[0].to_dict()  # Chuyển thành dictionary
    else:
        stock_data_dict = None  # Nếu không có dữ liệu, truyền None
    company_position= get_company_position(selected_stock)
    if not company_position.empty:
        company_position_dict = company_position.iloc[0].to_dict()  # Chuyển thành dictionary
        company_position_dict=company_position_dict.get("business_strategies", "").strip()
    else:
        company_position_dict = None  # Nếu không có dữ liệu, truyền None

    company_business_strategy=get_company_business_strategy(selected_stock)
    if not company_business_strategy.empty:
        company_business_strategy_dict = company_business_strategy.iloc[0].to_dict()  # Chuyển thành dictionary
        company_business_strategy_dict=company_business_strategy_dict.get("key_developments", "").strip()
    else:
        company_business_strategy_dict = None  # Nếu không có dữ liệu, truyền None


    stock_overview = get_stock_overview(selected_date, selected_stock)
    if not stock_overview.empty:
             stock_overview_dict = stock_overview.iloc[0].to_dict()  

        # Chuyển đổi số sang định dạng Mỹ
             for key in ["Value","Prev_Value", "H52W", "L52W", "Volume_Today", "MarketCap"]:
                if key in stock_overview_dict:
                    stock_overview_dict[key] = format_number_us(stock_overview_dict[key])
    else:
                stock_overview_dict = None
    
    stock_an = get_stock_overview(selected_date, selected_stock)
    stock_over=analyze_stock(stock_an, vnindex_percent_change)
   
    
    df_final=get_market_overview(selected_date, selected_stock)
    price_and_vnindex_chart,  price_and_volume_chart= plot_charts(df_final, selected_stock)

    financial_table= extract_financial_data(selected_stock)
    foreign_trading_chart=plot_foreign_trading(selected_stock, selected_date)
    balance_sheet=extract_balance_sheet(selected_stock)
    income_statement=extract_income_statement(selected_stock)
    trend_MA_chart=plot_trend_MA_chart(selected_date)

    df = load_data_NN("data/output.csv")
    base64_khop, base64_thoa_thuan= plot_investor_flow(df, selected_date)

    file_path = "data/output.csv"  # Đường dẫn đến file CSV
    df_GD = load_data_GD(file_path)
    df_filtered = investor_flow(df_GD, selected_date)
    # Danh sách các cột giao dịch (cần thay đổi cho đúng với dữ liệu thực tế)
    matching_columns = ['cá_nhân_tổng_gt_ròng', 'tổ_chức_trong_nước_tổng_gt_ròng', 'nước_ngoài_tổng_gt_ròng']
    matching_columns = list(matching_columns)
    # Hiển thị biểu đồ kết hợp
    combined_plot=create_combined_plot(df_filtered, matching_columns)
    df_KQKD = pd.read_csv("data/KQKD.csv")
    revenue_chart, profit_chart=vẽ_biểu_đồ_tăng_trưởng(df_KQKD, selected_stock)
    le_indicators=extract_le_data(selected_stock)
    li_indicators=extract_li_data(selected_stock)
    financial_analysis=extract_financial_data_from_api_and_analyze(selected_stock)
    co_cau_chart=bieu_do_no_von(selected_stock)
    doanh_loi_chart=bieu_do_doanh_loi(selected_stock)
    so_sanh_chart=bieu_do_so_sanh(selected_stock)
    pie_chart=ve_pie_chart_top5(selected_stock)
    top_chart=so_sanh_chi_so_tai_chinh(selected_stock)

    end_date = selected_date # Ngày kết thúc
    value_trends=plot_sector_value_trends(end_date, MERGED_DF, price, top_sectors)

       
    # Tạo báo cáo
    report_data = market_overview(selected_date, selected_stock)
    dinh_gia_indicators=extract_dinh_gia_data(selected_stock)
    report_date = selected_date.strftime('%Y-%m-%d')


    # Render HTML báo cáo
    rendered_html = render_template("report.html",stock_over=stock_over,vnindex_change=vnindex_percent_change,value_trends=value_trends,top_chart=top_chart,pie_chart=pie_chart,so_sanh_chart=so_sanh_chart,co_cau_chart=co_cau_chart,doanh_loi_chart=doanh_loi_chart,financial_analysis=financial_analysis,dinh_gia_indicators=dinh_gia_indicators,le_indicators=le_indicators, li_indicators=li_indicators,revenue_chart=revenue_chart, profit_chart=profit_chart, combined_plot=combined_plot, base64_khop=base64_khop, base64_thoa_thuan=base64_thoa_thuan, ma_chart=trend_MA_chart, income_statement= income_statement, balance_sheet=balance_sheet, foreign_trading_chart=foreign_trading_chart, financial_indicators=financial_table,price_and_vnindex_chart= price_and_vnindex_chart, price_and_volume_chart=price_and_volume_chart,
                           stock_price=stock_overview_dict, business_strategy=company_business_strategy_dict, stock_data=stock_data_dict,
                           company_profile = company_inf,company_position=company_position_dict, vnindex_summary = vnindex_summary,
                           market_cap_chart=market_cap_chart, vnindex_chart=vnindex_chart, market_overview = html_table, report_data=report_data,
                           report_date= report_date,company_code=selected_stock, company_name=company_name, industry_full_description=industry_full_description)

      # Chuyển HTML thành PDF
    pdf_file = io.BytesIO()
    HTML(string=rendered_html).write_pdf(pdf_file)
    pdf_file.seek(0)

    return send_file(
        pdf_file,
        as_attachment=True,
        download_name=f"{selected_stock}_{selected_date}.pdf",
        mimetype="application/pdf"
    )
if __name__ == "__main__":
    app.run(debug=True)
