# -*- coding: utf-8 -*-
# app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from prophet import Prophet
from datetime import timedelta, date
import numpy as np
import time
from flask_socketio import SocketIO, emit
from threading import Lock
from flask_sqlalchemy import SQLAlchemy
import logging
from logging.handlers import RotatingFileHandler
import traceback
import sys
import os

# Ensure templates directory exists
os.makedirs('templates', exist_ok=True)



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('proanz_app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///proanz.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=True, engineio_logger=True)

# Database Models
class SalesData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_name = db.Column(db.String(100), nullable=False)
    date = db.Column(db.Date, nullable=False)
    units_sold = db.Column(db.Float, nullable=False)
    price = db.Column(db.Float, nullable=False)

class DailySales(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, unique=True, nullable=False)
    units_sold = db.Column(db.Float, nullable=False)

@app.errorhandler(404)
def not_found(error):
    logger.warning(f'404 Not Found: {request.url}')
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f'500 Internal Server Error: {str(error)}')
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/')
def index():
    try:
        logger.info('Serving index page')
        return render_template('index.html')
    except Exception as e:
        logger.error(f'Error serving index: {str(e)}')
        return jsonify({'error': 'Failed to load page'}), 500

# Global variables for background thread
thread = None
thread_lock = Lock()
historical_sales = []  # To store historical daily sales for simulation (in memory for quick access)

@socketio.on('connect')
def connect():
    try:
        logger.info(f'Client connected: {request.sid}')
        emit('Connected', {'message': 'Welcome to real-time analytics!'}, to=request.sid)
    except Exception as e:
        logger.error(f'Error in connect event: {str(e)}')

@socketio.on('disconnect')
def disconnect():
    try:
        logger.info(f'Client disconnected: {request.sid}')
    except Exception as e:
        logger.error(f'Error in disconnect event: {str(e)}')

@socketio.on('start_streaming')
def start_streaming():
    global thread
    try:
        logger.info(f'Starting streaming for client: {request.sid}')
        with thread_lock:
            if thread is None:
                thread = socketio.start_background_task(background_thread)
        emit('streaming_started', {'status': 'Real-time sales data streaming initiated'})
    except Exception as e:
        logger.error(f'Error starting streaming: {str(e)}')
        emit('streaming_error', {'error': str(e)})

def background_thread():
    global historical_sales
    logger.info("Starting background thread for real-time sales simulation")
    try:
        if not historical_sales:
            # If no historical data, use default
            last_date = date.today()
            last_value = 100
            historical_sales = [{'date': last_date.strftime('%Y-%m-%d'), 'units': last_value}]
            logger.info("Initialized default historical sales data")
        
        while True:
            try:
                # Simulate new daily sales based on historical trend (simple random walk with trend)
                last_entry = historical_sales[-1]
                trend_factor = 1.01  # Slight upward trend
                change = np.random.normal(0, 10)  # Random variation
                new_units = max(0, last_entry['units'] * trend_factor + change)
                new_date = pd.to_datetime(last_entry['date']) + timedelta(days=1)
                
                new_data = {'date': new_date.strftime('%Y-%m-%d'), 'units': round(new_units)}
                historical_sales.append(new_data)
                
                # Store in database
                daily_entry = DailySales.query.filter_by(date=new_date.date()).first()
                if daily_entry:
                    daily_entry.units_sold = new_data['units']
                else:
                    daily_entry = DailySales(date=new_date.date(), units_sold=new_data['units'])
                    db.session.add(daily_entry)
                db.session.commit()
                
                # Emit to all clients
                emit('new_sales_data', new_data, broadcast=True)
                logger.info(f"Emitted new sales: {new_units} on {new_date}")
                
                time.sleep(5)  # Simulate every 5 seconds (adjust for demo)
            except Exception as e:
                logger.error(f'Error in background thread simulation: {str(e)}')
                time.sleep(10)  # Wait longer on error
    except Exception as e:
        logger.error(f'Fatal error in background thread: {str(e)}')
        traceback.print_exc()

@app.route('/upload', methods=['POST'])
def upload():
    global historical_sales
    logger.info('Received upload request')
    try:
        file = request.files.get('file')
        df = None
        if file and file.filename != '':
            logger.info(f'Uploading file: {file.filename}')
            # Read CSV
            df = pd.read_csv(file)
            required_columns = ['product_name', 'date', 'units_sold', 'price']
            if not all(col in df.columns for col in required_columns):
                error_msg = f'Missing required columns: {required_columns}'
                logger.warning(error_msg)
                return jsonify({'error': error_msg}), 400
        else:
            # Manual input (single or multi-row) - with debug logging
            product_names = request.form.getlist('product_name[]')
            date_strs = request.form.getlist('date[]')
            units_strs = request.form.getlist('units_sold[]')
            price_strs = request.form.getlist('price[]')
            
            # DEBUG: Log received data
            logger.info(f'Debug - Received: product_names len={len(product_names)} vals={product_names}')
            logger.info(f'Debug - Received: date_strs len={len(date_strs)} vals={date_strs}')
            logger.info(f'Debug - Received: units_strs len={len(units_strs)} vals={units_strs}')
            logger.info(f'Debug - Received: price_strs len={len(price_strs)} vals={price_strs}')
            
            if not all([product_names, date_strs, units_strs, price_strs]):
                error_msg = f'Missing required fields for manual input. Debug lens: names={len(product_names)}, dates={len(date_strs)}, units={len(units_strs)}, prices={len(price_strs)}'
                logger.warning(error_msg)
                return jsonify({'error': error_msg}), 400
            if len(product_names) != len(date_strs) or len(product_names) != len(units_strs) or len(product_names) != len(price_strs):
                error_msg = 'Mismatched number of fields across rows'
                logger.warning(error_msg)
                return jsonify({'error': error_msg}), 400
            try:
                units_sold = [float(u) for u in units_strs]
                prices = [float(p) for p in price_strs]
            except ValueError as ve:
                error_msg = f'Invalid number for units sold or price: {ve}'
                logger.warning(error_msg)
                return jsonify({'error': error_msg}), 400
            df = pd.DataFrame({
                'product_name': product_names,
                'date': pd.to_datetime(date_strs),
                'units_sold': units_sold,
                'price': prices
            })
            logger.info(f'Manual input processed: {len(df)} rows')
        
        # Data cleaning (applies to both CSV and manual)
        logger.info('Cleaning data')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])  # Drop invalid dates
        df['units_sold'] = pd.to_numeric(df['units_sold'], errors='coerce')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['units_sold', 'price'])

        if df.empty:
            error_msg = 'No valid data after cleaning'
            logger.warning(error_msg)
            return jsonify({'error': error_msg}), 400

        # For performance: Limit to last 1000 rows if very large
        original_len = len(df)
        if len(df) > 1000:
            df = df.tail(1000)
            logger.info(f"Data downsampled from {original_len} to last 1000 rows for performance")

        # Clear existing data in tables
        db.session.query(SalesData).delete()
        db.session.query(DailySales).delete()
        db.session.commit()
        logger.info('Cleared existing database tables')

        # Insert cleaned data into SalesData table
        logger.info('Inserting data into SalesData table')
        for index, row in df.iterrows():
            sale = SalesData(
                product_name=row['product_name'],
                date=row['date'].date(),
                units_sold=row['units_sold'],
                price=row['price']
            )
            db.session.add(sale)
        db.session.commit()

        # Aggregate product data from database
        logger.info('Aggregating product sales data from database')
        prod_sales = pd.read_sql(
            db.session.query(
                SalesData.product_name,
                db.func.sum(SalesData.units_sold).label('units_sold'),
                db.func.avg(SalesData.price).label('price')
            ).group_by(SalesData.product_name).statement,
            db.engine
        )

        # 1. Most Selling Products (Bar Chart) - Top 10 by total units sold
        most_selling = prod_sales.nlargest(10, 'units_sold')
        fig_most = px.bar(most_selling, x='product_name', y='units_sold', 
                          title='Most Selling Products (Top 10)', 
                          labels={'units_sold': 'Total Units Sold'})
        graph_most = fig_most.to_json()
        top_product = most_selling.iloc[0]['product_name']
        top_units = most_selling.iloc[0]['units_sold']
        note_most = f"Most trending product: {top_product} with {top_units:.0f} units sold overall."

        # 2. Low Selling Products (Bar Chart) - Bottom 10 by total units sold
        low_selling = prod_sales.nsmallest(10, 'units_sold')
        fig_low = px.bar(low_selling, x='product_name', y='units_sold', 
                         title='Low Selling Products (Bottom 10)', 
                         labels={'units_sold': 'Total Units Sold'})
        graph_low = fig_low.to_json()
        low_product = low_selling.iloc[0]['product_name']
        low_units = low_selling.iloc[0]['units_sold']
        note_low = f"Lowest selling product: {low_product} with only {low_units:.0f} units sold."

        # 3. High Cost but Most Sold (Scatter Plot: price vs sales for high-price products)
        median_price = prod_sales['price'].median()
        high_price_prods = prod_sales[prod_sales['price'] > median_price].nlargest(10, 'units_sold')
        fig_high_cost = px.scatter(high_price_prods, x='price', y='units_sold', size='units_sold', 
                                   hover_name='product_name', 
                                   title='High Cost but Most Sold Products',
                                   labels={'price': 'Average Price', 'units_sold': 'Total Units Sold'})
        graph_high_cost = fig_high_cost.to_json()
        if not high_price_prods.empty:
            top_high = high_price_prods.iloc[0]['product_name']
            note_high = f"Top high-cost seller: {top_high} at avg ${high_price_prods.iloc[0]['price']:.2f} with high sales volume."
        else:
            note_high = "No high-cost products with significant sales."

        # 4. Low Cost but Most Sold (Scatter Plot: price vs sales for low-price products)
        low_price_prods = prod_sales[prod_sales['price'] <= median_price].nlargest(10, 'units_sold')
        fig_low_cost = px.scatter(low_price_prods, x='price', y='units_sold', size='units_sold', 
                                  hover_name='product_name', 
                                  title='Low Cost but Most Sold Products',
                                  labels={'price': 'Average Price', 'units_sold': 'Total Units Sold'})
        graph_low_cost = fig_low_cost.to_json()
        if not low_price_prods.empty:
            top_low = low_price_prods.iloc[0]['product_name']
            note_low_cost = f"Top low-cost seller: {top_low} at avg ${low_price_prods.iloc[0]['price']:.2f} with high sales volume."
        else:
            note_low_cost = "No low-cost products with significant sales."

        # 5. Advanced Sales Prediction with Prophet (Line + Forecast)
        logger.info('Generating sales prediction with Prophet')
        daily_sales = pd.read_sql(
            db.session.query(
                SalesData.date,
                db.func.sum(SalesData.units_sold).label('units_sold')
            ).group_by(SalesData.date).order_by(SalesData.date).statement,
            db.engine
        )
        daily_sales['date'] = pd.to_datetime(daily_sales['date'])
        if len(daily_sales) < 2:
            # Handle single entry: just plot the point, no forecast
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=daily_sales['date'], y=daily_sales['units_sold'], 
                                          mode='markers', name='Actual Sales', 
                                          marker=dict(color='blue', size=20)))
            fig_pred.update_layout(title='Sales Data (Single Entry - No Prediction Available)', 
                                   xaxis_title='Date', yaxis_title='Units Sold')
            graph_pred = fig_pred.to_json()
            note_pred = "Single data point recorded. Provide more dates via CSV for full prediction analysis with Prophet model."
        else:
            # Insert into DailySales table
            for idx, row in daily_sales.iterrows():
                daily_entry = DailySales.query.filter_by(date=row['date'].date()).first()
                if not daily_entry:
                    daily_entry = DailySales(date=row['date'].date(), units_sold=row['units_sold'])
                    db.session.add(daily_entry)
            db.session.commit()
            logger.info('Inserted daily sales into DailySales table')
            
            # Prepare for Prophet
            prophet_df = daily_sales.rename(columns={'date': 'ds', 'units_sold': 'y'})
            model = Prophet(daily_seasonality=True, weekly_seasonality=True)
            model.fit(prophet_df)
            
            # Forecast next 30 days
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            
            # Plot with Plotly
            fig_pred = go.Figure()
            # Actual historical data
            fig_pred.add_trace(go.Scatter(x=daily_sales['date'], y=daily_sales['units_sold'], 
                                          mode='lines+markers', name='Actual Sales', line=dict(color='blue')))
            # Forecast
            fig_pred.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], 
                                          mode='lines', name='Predicted Sales', line=dict(color='red', dash='dot')))
            # Upper and lower bounds
            fig_pred.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], 
                                          fill=None, mode='lines', line_color='rgba(255,0,0,0)', showlegend=False))
            fig_pred.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], 
                                          fill='tonexty', mode='lines', line_color='rgba(255,0,0,0)', fillcolor='rgba(255,0,0,0.2)', name='Prediction Interval'))
            fig_pred.update_layout(title='Advanced Sales Prediction (Prophet Model, Next 30 Days)', 
                                   xaxis_title='Date', yaxis_title='Units Sold')
            graph_pred = fig_pred.to_json()
            avg_future = forecast['yhat'].tail(30).mean()
            note_pred = f"Predicted average daily sales for next 30 days: {avg_future:.0f} units. Prophet model fitted successfully."
        
        # For single entry, still insert to DailySales if not done
        if len(daily_sales) == 1:
            row = daily_sales.iloc[0]
            daily_entry = DailySales.query.filter_by(date=row['date'].date()).first()
            if not daily_entry:
                daily_entry = DailySales(date=row['date'].date(), units_sold=row['units_sold'])
                db.session.add(daily_entry)
            db.session.commit()
            logger.info('Inserted single daily sales into DailySales table')

        # Sync in-memory historical sales with database for streaming
        db_daily = pd.read_sql(DailySales.query.statement, db.engine)
        historical_sales = [{'date': d.strftime('%Y-%m-%d'), 'units': float(u)} for d, u in zip(db_daily['date'], db_daily['units_sold'])]
        logger.info('Updated historical sales from database for streaming')

        # 6. Product-Wise Detailed Report (Bar + Scatter combined)
        logger.info('Generating product-wise report')
        fig_report = go.Figure()
        fig_report.add_trace(go.Bar(x=prod_sales['product_name'], y=prod_sales['units_sold'], 
                                    name='Total Units Sold', marker_color='lightblue'))
        fig_report.add_trace(go.Scatter(x=prod_sales['product_name'], y=prod_sales['price'] * prod_sales['units_sold'], 
                                        mode='markers+lines', name='Revenue', yaxis='y2', line=dict(color='green')))
        fig_report.update_layout(title='Product-Wise Detailed Report (Bar: Sales, Line: Revenue)',
                                 xaxis_title='Product Name',
                                 yaxis=dict(title='Units Sold'),
                                 yaxis2=dict(title='Revenue', overlaying='y', side='right'))
        graph_report = fig_report.to_json()
        total_sales = prod_sales['units_sold'].sum()
        avg_sales = prod_sales['units_sold'].mean()
        note_report = f"Total sales across all products: {total_sales:.0f} units. Average per product: {avg_sales:.0f} units."

        logger.info('Upload and analysis completed successfully')
        return jsonify({
            'most_selling': {'graph': graph_most, 'note': note_most},
            'low_selling': {'graph': graph_low, 'note': note_low},
            'high_cost_high_sales': {'graph': graph_high_cost, 'note': note_high},
            'low_cost_high_sales': {'graph': graph_low_cost, 'note': note_low_cost},
            'sales_prediction': {'graph': graph_pred, 'note': note_pred},
            'product_report': {'graph': graph_report, 'note': note_report}
        })

    except pd.errors.EmptyDataError as e:
        error_msg = 'Empty CSV file or invalid format'
        logger.error(f'Pandas error in upload: {str(e)}')
        return jsonify({'error': error_msg}), 400
    except Exception as e:
        error_msg = f'Unexpected error during upload: {str(e)}'
        logger.error(f'Unexpected error in upload: {str(e)}\n{traceback.format_exc()}')
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        logger.info('Created database tables')
    try:
        logger.info('Starting ProAnz application')
        socketio.run(app, debug=False, host='0.0.0.0', port=3000, allow_unsafe_werkzeug=True)
    except Exception as e:
        logger.error(f'Failed to start application: {str(e)}')
        traceback.print_exc()

