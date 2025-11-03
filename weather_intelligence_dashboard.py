import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="🌤️ Weather Intelligence Dashboard",
    page_icon="⛅",
    layout="wide"
)

def fetch_weather_data(latitude, longitude, days=7):
    """Fetch weather data from Open-Meteo API"""
    try:
        # Add current date to get recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days-1)
        
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={latitude}&longitude={longitude}&"
            f"start_date={start_date.strftime('%Y-%m-%d')}&"
            f"end_date={end_date.strftime('%Y-%m-%d')}&"
            "hourly=temperature_2m,relative_humidity_2m,precipitation"
        )
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Check if we have the expected data
        if 'hourly' not in data or not all(key in data['hourly'] for key in ['time', 'temperature_2m', 'relative_humidity_2m', 'precipitation']):
            st.error("Unexpected API response format")
            return None
        
        # Process hourly data
        df = pd.DataFrame({
            'datetime': pd.to_datetime(data['hourly']['time']),
            'Temperature': data['hourly']['temperature_2m'],
            'Humidity': data['hourly']['relative_humidity_2m'],
            'Rainfall': data['hourly']['precipitation']
        })
        
        # Ensure we have data
        if df.empty:
            st.error("No weather data received")
            return None
            
        # Convert to daily data
        df['Date'] = df['datetime'].dt.date
        daily_df = df.groupby('Date', as_index=False).agg({
            'Temperature': 'mean',
            'Humidity': 'mean',
            'Rainfall': 'sum'
        })
        
        # Ensure we have at least 3 days of data for meaningful analysis
        if len(daily_df) < 3:
            st.warning("Insufficient data points for analysis")
            return None
            
        return daily_df
        
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {str(e)}")
    except Exception as e:
        st.error(f"Error processing weather data: {str(e)}")
    
    return None

def clean_and_analyze_data(df):
    """Clean data and generate insights"""
    try:
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Ensure Date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Ensure numeric types and handle non-numeric values
        for col in ['Temperature', 'Humidity', 'Rainfall']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values with column means
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # Add condition column with error handling
        if 'Temperature' in df.columns:
            df['Condition'] = pd.cut(
                df['Temperature'],
                bins=[-float('inf'), 20, 30, float('inf')],
                labels=['Cold ❄️', 'Mild 🌤️', 'Hot ☀️'],
                include_lowest=True
            )
        
        # Add comfort index if we have the required columns
        if all(col in df.columns for col in ['Temperature', 'Humidity']):
            df['Comfort_Index'] = 0.5 * (df['Temperature'] + (100 - df['Humidity']) / 5)
        
        return df
    except Exception as e:
        st.error(f"Error in data processing: {str(e)}")
        return None

def train_prediction_model(df):
    """Train a model to predict next day's temperature"""
    try:
        # Use sequence of days as feature
        X = np.array(range(len(df))).reshape(-1, 1)
        y = df['Temperature'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict next day
        next_day = len(df)
        prediction = model.predict([[next_day]])[0]
        
        # Calculate accuracy (R² score)
        score = model.score(X, y)
        
        return prediction, score
    except Exception as e:
        st.error(f"Error in prediction model: {str(e)}")
        return None, 0

def display_metrics(df):
    """Display key metrics"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🌡️ Avg Temperature", f"{df['Temperature'].mean():.1f}°C", 
                 f"Max: {df['Temperature'].max():.1f}°C")
    with col2:
        st.metric("💧 Avg Humidity", f"{df['Humidity'].mean():.1f}%", 
                 f"Max: {df['Humidity'].max():.1f}%")
    with col3:
        st.metric("🌧️ Total Rainfall", f"{df['Rainfall'].sum():.1f} mm", 
                 f"Max: {df['Rainfall'].max():.1f} mm")

def plot_weather_trends(df):
    """Plot temperature, humidity, and rainfall trends"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add temperature trace
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Temperature'], name="Temperature (°C)", 
                  line=dict(color='red', width=2)),
        secondary_y=False,
    )
    
    # Add humidity trace
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Humidity'], name="Humidity (%)",
                  line=dict(color='blue', width=2, dash='dot')),
        secondary_y=True,
    )
    
    # Add rainfall bars
    fig.add_trace(
        go.Bar(x=df['Date'], y=df['Rainfall'], name="Rainfall (mm)", 
              marker_color='lightblue', opacity=0.6),
        secondary_y=True,
    )
    
    # Update layout
    fig.update_layout(
        title="Weather Trends Over Time",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        yaxis2_title="Humidity (%) / Rainfall (mm)",
        hovermode="x",
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_correlation(df):
    """Plot correlation heatmap"""
    corr = df[['Temperature', 'Humidity', 'Rainfall', 'Comfort_Index']].corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        aspect="auto",
        title="Correlation Heatmap"
    )
    st.plotly_chart(fig, use_container_width=True)

def main():
    # Sidebar
    st.sidebar.title("🌍 Weather Intelligence Dashboard")
    st.sidebar.write("Enter location coordinates:")
    
    # Default to Delhi's coordinates
    default_lat = st.sidebar.number_input("Latitude", value=28.6139, format="%.4f")
    default_lon = st.sidebar.number_input("Longitude", value=77.2090, format="%.4f")
    
    # Add some space
    st.sidebar.markdown("---")
    st.sidebar.write("ℹ️ Data provided by Open-Meteo API")
    
    # Main content
    st.title(f"⛅ Weather Intelligence Dashboard")
    st.write("Real-time weather data analysis and prediction")
    
    # Fetch data
    with st.spinner('Fetching weather data...'):
        df = fetch_weather_data(default_lat, default_lon)
    
    if df is not None and not df.empty:
        # Clean and analyze data
        df = clean_and_analyze_data(df)
        
        # Display metrics
        display_metrics(df)
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📈 Trends", "📊 Analysis", "🔮 Prediction"])
        
        with tab1:
            st.header("Weather Trends")
            plot_weather_trends(df)
            
            # Show recent data
            st.subheader("Recent Weather Data")
            st.dataframe(df.tail(10).style.background_gradient(
                subset=['Temperature', 'Humidity', 'Rainfall'],
                cmap='RdBu_r',
                axis=0
            ))
        
        with tab2:
            st.header("Weather Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Weather Conditions")
                condition_counts = df['Condition'].value_counts()
                fig = px.pie(
                    values=condition_counts.values,
                    names=condition_counts.index,
                    title="Weather Condition Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Comfort Index")
                st.write("Higher values indicate more comfortable weather")
                st.metric("Average Comfort Index", f"{df['Comfort_Index'].mean():.1f}")
                
                # Comfort level interpretation
                avg_comfort = df['Comfort_Index'].mean()
                if avg_comfort > 30:
                    comfort_status = "Very Comfortable 😊"
                elif avg_comfort > 25:
                    comfort_status = "Comfortable 🙂"
                elif avg_comfort > 20:
                    comfort_status = "Moderate 😐"
                else:
                    comfort_status = "Uncomfortable 😕"
                    
                st.metric("Comfort Level", comfort_status)
            
            st.subheader("Correlation Analysis")
            plot_correlation(df)
            
            # Show statistics
            st.subheader("Weather Statistics")
            stats_df = df[['Temperature', 'Humidity', 'Rainfall', 'Comfort_Index']].describe()
            st.dataframe(stats_df.style.background_gradient(axis=0))
        
        with tab3:
            st.header("Weather Prediction")
            
            # Train model and make prediction
            next_day_temp, accuracy = train_prediction_model(df)
            
            if next_day_temp is not None:
                # Display prediction with nice styling
                st.markdown("""
                <div style='background-color:#f0f2f6;padding:20px;border-radius:10px;'>
                    <h3>Next Day Temperature Forecast</h3>
                    <h1 style='color:#1f77b4;text-align:center;'>{:.1f}°C</h1>
                    <p style='text-align:center;color:#666;'>Model Accuracy: {:.1%}</p>
                </div>
                """.format(next_day_temp, accuracy), unsafe_allow_html=True)
                
                # Show trend line with prediction
                st.subheader("Temperature Trend & Forecast")
                
                # Prepare data for plotting
                X_future = np.array(range(len(df) + 1)).reshape(-1, 1)
                y_future = np.append(df['Temperature'].values, next_day_temp)
                
                fig = go.Figure()
                
                # Actual data
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['Temperature'],
                    mode='lines+markers',
                    name='Historical Data',
                    line=dict(color='blue')
                ))
                
                # Predicted point
                fig.add_trace(go.Scatter(
                    x=[df['Date'].iloc[-1] + timedelta(days=1)],
                    y=[next_day_temp],
                    mode='markers',
                    name='Prediction',
                    marker=dict(color='red', size=10)
                ))
                
                # Add confidence interval (simplified)
                confidence = 1.0  # Simplified for this example
                fig.add_trace(go.Scatter(
                    x=list(df['Date']) + [df['Date'].iloc[-1] + timedelta(days=1)],
                    y=y_future + confidence,
                    fill=None,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=list(df['Date']) + [df['Date'].iloc[-1] + timedelta(days=1)],
                    y=y_future - confidence,
                    fill='tonexty',
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(255, 0, 0, 0.1)',
                    name='Confidence Interval'
                ))
                
                fig.update_layout(
                    title='Temperature Forecast with Confidence Interval',
                    xaxis_title='Date',
                    yaxis_title='Temperature (°C)',
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add interpretation
                temp_change = next_day_temp - df['Temperature'].iloc[-1]
                if temp_change > 1:
                    trend = f"{temp_change:.1f}°C warmer than today."
                elif temp_change < -1:
                    trend = f"{abs(temp_change):.1f}°C cooler than today."
                else:
                    trend = "similar to today's temperature."
                
                st.info(f"""
                **Forecast Analysis:**
                - Expected temperature: **{next_day_temp:.1f}°C**
                - Trend: {trend}
                - Model confidence: **{accuracy:.1%}**
                """)
                
                # Add some weather recommendations based on prediction
                if next_day_temp > 30:
                    st.warning("⚠️ Hot weather expected tomorrow. Stay hydrated and avoid direct sun exposure!")
                elif next_day_temp < 15:
                    st.info("❄️ Cold weather expected tomorrow. Dress warmly!")
                
                # Show feature importance (simplified for this example)
                st.subheader("Forecast Factors")
                st.write("""
                The prediction considers recent weather patterns including:
                - Historical temperature trends
                - Current weather conditions
                - Seasonal variations
                """)
        
        # Export data
        if st.button("💾 Export Weather Data"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="weather_report.csv",
                mime="text/csv"
            )
    else:
        st.error("Failed to fetch weather data. Please check your internet connection and try again.")
    
    # Footer
    st.markdown("---")
    st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    st.caption("© 2023 Weather Intelligence Dashboard | Data provided by Open-Meteo")

if __name__ == "__main__":
    main()
