# In dashboard.py
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from fault_logic import calculate_current_threshold, check_fault_status, determine_isolation_action

# --- Page Configuration ---
st.set_page_config(
    page_title="Power Grid Fault Monitoring",
    page_icon="⚡",
    layout="wide"
)

# --- CONFIGURATION ---
DATA_FILE_PATH = "synthetic_lt_break_dataset(1).csv"


# --- Data Loading & Threshold Calculation ---
@st.cache_data
def load_data_and_prepare(file_path):
    """Loads data, calculates thresholds, and prepares a placeholder graph."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Fatal Error: Data file not found at '{file_path}'. Please check the filename and path.")
        st.stop()
    except Exception as e:
        st.error(f"Error reading CSV file: {e}. Please ensure it's a valid CSV file.")
        st.stop()

    num_buses = len([col for col in df.columns if col.startswith('V_')])
    G = nx.random_geometric_graph(num_buses, 0.15, seed=42)
    pos = nx.spring_layout(G, seed=42)
    
    voltage_thresh = 0.90
    current_thresh = calculate_current_threshold(file_path)
    
    return df, G, pos, voltage_thresh, current_thresh

df, G, pos, V_THRESHOLD, I_THRESHOLD = load_data_and_prepare(DATA_FILE_PATH)

# --- Interactive Sidebar ---
st.sidebar.title("Simulation Controls")
st.sidebar.info(f"Using Calculated Thresholds:\n- Voltage < {V_THRESHOLD:.2f} p.u.\n- Current > {I_THRESHOLD:.2f} A")
scenario = st.sidebar.selectbox("Choose a scenario to simulate:", 
                                ("Normal Operation", "Line Fault Event", "Random Event"))

if st.sidebar.button("▶️ Run Simulation Step"):
    if scenario == "Normal Operation":
        sample = df[df['label'] == 'normal'].sample(1).iloc[0].to_dict()
    elif scenario == "Line Fault Event":
        fault_df = df[df['label'] == 'break']
        current_cols = [col for col in df.columns if col.startswith('I_')]
        voltage_cols = [col for col in df.columns if col.startswith('V_')]
        severe_faults_df = fault_df[
            (fault_df[current_cols].max(axis=1) > I_THRESHOLD) &
            (fault_df[voltage_cols].min(axis=1) < V_THRESHOLD)
        ]
        if not severe_faults_df.empty:
            sample = severe_faults_df.sample(1).iloc[0].to_dict()
        else:
            st.error("No faults in the dataset meet the detection criteria. Displaying a random 'break' sample instead.")
            sample = fault_df.sample(1).iloc[0].to_dict()
    else:
        sample = df.sample(1).iloc[0].to_dict()
    
    st.session_state['analysis'] = check_fault_status(sample, V_THRESHOLD, I_THRESHOLD)
    st.session_state['isolation'] = determine_isolation_action(st.session_state['analysis'], sample)

# --- Main Dashboard Display ---
st.title("⚡ Real-Time Grid Monitoring Dashboard")

if 'analysis' not in st.session_state:
    st.info("Click '▶️ Run Simulation Step' in the sidebar to begin.")
else:
    analysis = st.session_state['analysis']
    isolation = st.session_state['isolation']
    
    status_color = "green" if analysis['status'] == 'NORMAL' else "red"
    st.markdown(f"### System Status: <span style='color:{status_color};'>{analysis['status']}</span>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Minimum Voltage (p.u.)", f"{analysis['min_voltage']:.3f}")
    col2.metric("Maximum Current (A)", f"{analysis['max_current']:.2f}")
    col3.metric("Isolation Action", f"{isolation['action']}: {isolation['element_to_open'] or 'N/A'}")

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='gray'), hoverinfo='none', mode='lines')
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text',
        marker=dict(
            showscale=True, colorscale='YlGnBu', size=10, color=[],
            colorbar=dict(thickness=15, title=dict(text='Node Connections', side='right'), xanchor='left')
        )
    )
    
    node_adjacencies = [len(adj) for node, adj in G.adjacency()]
    node_colors = node_adjacencies
    
    if analysis['status'] == 'FAULT DETECTED':
        node_colors = ['#333F44'] * len(G.nodes())
        try:
            voltage_cols = [col for col in df.columns if col.startswith('V_')]
            bus_name_map = {name.split('_')[1]: i for i, name in enumerate(voltage_cols)}
            symptomatic_bus_indices = []
            for bus_key in analysis['symptomatic_buses']:
                bus_name = bus_key.split('_')[1]
                if bus_name in bus_name_map:
                    symptomatic_bus_indices.append(bus_name_map[bus_name])
            for i in symptomatic_bus_indices:
                if i < len(node_colors):
                    node_colors[i] = 'red'
        except Exception as e:
            st.warning(f"Could not parse bus names to highlight nodes. Error: {e}")

    node_trace.marker.color = node_colors
    node_text = [f'Bus #{i}<br># of connections: {adj}' for i, adj in enumerate(node_adjacencies)]
    node_trace.text = node_text

    # Create the figure object first
    fig = go.Figure(data=[edge_trace, node_trace])

    # *** NEW: Apply layout updates AFTER creation for robustness ***
    fig.update_layout(
        template='plotly_dark',  # Primary theme setting
        title=dict(text='Interactive Grid Topology', font=dict(size=20)),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        # Explicitly set background colors to prevent theme conflicts
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        # Keep the explicit hover label style as a final override
        hoverlabel=dict(
            bgcolor="black",
            font_size=14,
            font_family="Arial",
            font_color="white"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

