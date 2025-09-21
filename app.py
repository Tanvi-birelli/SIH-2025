import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import io
import zipfile
import tempfile
from scipy.interpolate import UnivariateSpline
import os

# -------------------------
# Coverage & Deployment Functions
# -------------------------
def suggest_from_region(df, lat_min, lat_max, lon_min, lon_max, grid_size, k):
    suggestions = []

    # Ensure numeric lat/lon
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df = df.dropna(subset=['lat','lon'])

    # Create grid bins
    lat_bins = np.arange(lat_min, lat_max + 1e-6, grid_size)  # include max
    lon_bins = np.arange(lon_min, lon_max + 1e-6, grid_size)

    for lat in lat_bins:
        for lon in lon_bins:
            cell_df = df[(df['lat'] >= lat) & (df['lat'] <= lat + grid_size) &
                         (df['lon'] >= lon) & (df['lon'] <= lon + grid_size)]
            count = len(cell_df)
            if count > 0:
                # center of the cell
                suggestions.append({"lat": lat + grid_size / 2,
                                    "lon": lon + grid_size / 2,
                                    "count": count})

    if not suggestions:
        # fallback: return center of entire region
        return [{"lat": (lat_min + lat_max)/2, "lon": (lon_min + lon_max)/2}]

    # return top k by count
    suggestions = sorted(suggestions, key=lambda x: x["count"], reverse=True)[:k]
    return [{"lat": s["lat"], "lon": s["lon"]} for s in suggestions]


def suggest_from_center(df, lat_c, lon_c, radius_km, k):
    radius_deg = radius_km / 111.0
    candidates = df[(np.abs(df['lat'] - lat_c) <= radius_deg) &
                    (np.abs(df['lon'] - lon_c) <= radius_deg)]
    if len(candidates) == 0:
        return [{"lat": lat_c, "lon": lon_c}]
    candidates = candidates.sample(min(k, len(candidates)))
    return [{"lat": row["lat"], "lon": row["lon"]} for _, row in candidates.iterrows()]

def suggest_from_dataset(df, grid_size, k):
    # Ensure numeric
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df = df.dropna(subset=['lat','lon'])

    # Create bins
    lat_bins = np.arange(df['lat'].min(), df['lat'].max() + grid_size, grid_size)
    lon_bins = np.arange(df['lon'].min(), df['lon'].max() + grid_size, grid_size)

    # Assign each point to a bin
    df['lat_bin'] = pd.cut(df['lat'], bins=lat_bins, include_lowest=True, labels=False)
    df['lon_bin'] = pd.cut(df['lon'], bins=lon_bins, include_lowest=True, labels=False)

    df = df.dropna(subset=['lat_bin','lon_bin'])  # drop out-of-range

    # Group by bins and count points
    grouped = df.groupby(['lat_bin','lon_bin']).size().reset_index(name='count')

    # Select top k bins
    top_bins = grouped.nlargest(k, 'count')

    # Convert bin numbers to bin centers
    top_bins['lat'] = top_bins['lat_bin'].apply(lambda x: lat_bins[int(x)] + grid_size/2)
    top_bins['lon'] = top_bins['lon_bin'].apply(lambda x: lon_bins[int(x)] + grid_size/2)

    suggestions = top_bins[['lat','lon']].to_dict(orient='records')
    if not suggestions:
        # fallback
        suggestions = [{"lat": df['lat'].mean(), "lon": df['lon'].mean()}]
    return suggestions


# -------------------------
# Spline Interpolation
# -------------------------
def interpolate_multiple_columns(df, columns, spline_order):
    df_interp = df.copy()
    for col in columns:
        df_interp[col] = pd.to_numeric(df_interp[col], errors='coerce')
        x = np.arange(len(df_interp))
        y = df_interp[col].values
        mask = ~np.isnan(y)
        if mask.sum() > spline_order:
            spline = UnivariateSpline(x[mask], y[mask], k=spline_order, s=0)
            df_interp[col] = spline(x)
    return df_interp

# -------------------------
# Streamlit App Config
# -------------------------
st.set_page_config(page_title="ARGO Data Tools", layout="wide")
st.title("🌊 ARGO Data Tools")
st.markdown("A unified app with two features: **Coverage & Deployment Suggestion** and **Data Interpolation**.")

# -------------------------
# Sidebar feature selection & file upload
# -------------------------
feature = st.sidebar.radio("Select Feature", ["Coverage & Deployment Suggestion", "Data Interpolation"])

if feature == "Coverage & Deployment Suggestion":
    uploaded_file = st.sidebar.file_uploader("Upload zipped folder of floats", type="zip")
elif feature == "Data Interpolation":
    uploaded_file = st.sidebar.file_uploader("Upload single file (CSV or NetCDF)", type=["csv", "nc"])
else:
    uploaded_file = None

# -------------------------
# Load data
# -------------------------
@st.cache_data
def load_argo_data(uploaded_zip):
    all_profiles = []
    with tempfile.TemporaryDirectory() as extract_path:
        with zipfile.ZipFile(io.BytesIO(uploaded_zip.read())) as z:
            z.extractall(extract_path)
        for root, _, files in os.walk(extract_path):
            for f in files:
                if f.endswith(".csv"):
                    csv_path = os.path.join(root, f)
                    try:
                        df_float = pd.read_csv(csv_path)
                        lat_col = next((c for c in df_float.columns if c.lower() in ["lat", "latitude"]), None)
                        lon_col = next((c for c in df_float.columns if c.lower() in ["lon", "longitude"]), None)
                        if lat_col and lon_col:
                            df_float.rename(columns={lat_col: "lat", lon_col: "lon"}, inplace=True)
                            df_float["lat"] = pd.to_numeric(df_float["lat"], errors='coerce')
                            df_float["lon"] = pd.to_numeric(df_float["lon"], errors='coerce')
                            df_float = df_float.dropna(subset=['lat','lon'])
                            df_float["float_name"] = os.path.basename(os.path.dirname(csv_path))
                            all_profiles.append(df_float)
                    except Exception as e:
                        st.warning(f"Skipping {csv_path}: {e}")
                        continue
    if all_profiles:
        return pd.concat(all_profiles, ignore_index=True)
    return pd.DataFrame()

if uploaded_file:
    if feature == "Coverage & Deployment Suggestion":
        df = load_argo_data(uploaded_file)
    elif feature == "Data Interpolation":
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".nc"):
            import xarray as xr
            ds = xr.open_dataset(uploaded_file)
            df = ds.to_dataframe().reset_index()
        else:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    if df.empty:
        st.warning("⚠️ No usable data found.")
        st.stop()
    else:
        st.success(f"✅ Loaded {len(df)} rows")
        if st.checkbox("Show preview of data", value=True):
            st.dataframe(df.head(500))
else:
    st.warning("⬆️ Please upload a file to proceed.")
    st.stop()

# -------------------------
# Feature 1: Coverage & Deployment
# -------------------------
if feature == "Coverage & Deployment Suggestion":
    st.subheader("📍 Coverage & Deployment Suggestion")

    mode = st.selectbox("Suggestion Mode", ["region", "circle", "auto"])
    grid_size = st.slider("Grid Size (degrees)", 1, 10, 2)
    k = st.slider("Number of Suggestions (k)", 1, 10, 3)
    suggestions = []

    if mode == "region":
        lat_min = st.number_input("Latitude Min", value=float(df["lat"].min()))
        lat_max = st.number_input("Latitude Max", value=float(df["lat"].max()))
        lon_min = st.number_input("Longitude Min", value=float(df["lon"].min()))
        lon_max = st.number_input("Longitude Max", value=float(df["lon"].max()))
        if st.button("Run Region-Based Suggestion"):
            suggestions = suggest_from_region(df, lat_min, lat_max, lon_min, lon_max, grid_size, k)

    elif mode == "circle":
        lat_c = st.number_input("Center Latitude", value=float(df["lat"].mean()))
        lon_c = st.number_input("Center Longitude", value=float(df["lon"].mean()))
        radius_km = st.number_input("Radius (km)", value=500.0)
        if st.button("Run Circle-Based Suggestion"):
            suggestions = suggest_from_center(df, lat_c, lon_c, radius_km, k)

    elif mode == "auto":
        if st.button("Run Auto-Bounds Suggestion"):
            suggestions = suggest_from_dataset(df, grid_size, k)

    if suggestions:
        st.success("✅ Suggested Deployment Locations")
        st.table(pd.DataFrame(suggestions))

        MAX_POINTS = 2000
        df_plot = df.sample(n=min(MAX_POINTS, len(df)), random_state=42) if len(df) > MAX_POINTS else df

        fig = px.scatter_mapbox(
            df_plot, lat="lat", lon="lon",
            color_discrete_sequence=["blue"], zoom=2,
            title="ARGO Profiles & Suggested Deployment"
        )
        for s in suggestions:
            fig.add_scattermapbox(
                lat=[s["lat"]], lon=[s["lon"]],
                mode="markers",
                marker=dict(size=16, color="orange", symbol="circle"),
                name="Suggested Float"
            )
        fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Feature 2: Data Interpolation
# -------------------------
elif feature == "Data Interpolation":
    st.subheader("📈 Data Interpolation (Spline)")
    columns = st.multiselect("Select columns to interpolate", df.columns.tolist())
    spline_order = st.slider("Spline Order", 1, 5, 3)
    if st.button("Run Interpolation"):
        if not columns:
            st.warning("⚠️ Please select at least one column.")
        else:
            df_interp = interpolate_multiple_columns(df, columns, spline_order)
            st.success("✅ Interpolation complete")
            st.dataframe(df_interp.head(500))

            # Before vs After plot
            col = columns[0]
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=df[col].head(500), mode="lines+markers", name="Before", line=dict(color="red")))
            fig.add_trace(go.Scatter(y=df_interp[col].head(500), mode="lines", name="After", line=dict(color="blue")))
            fig.update_layout(title=f"{col}: Before vs After Interpolation")
            st.plotly_chart(fig, use_container_width=True)

            # Download button
            csv = df_interp.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download full interpolated data as CSV",
                data=csv,
                file_name="argo_interpolated.csv",
                mime="text/csv"
            )