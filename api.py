import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import os
from collections import deque
from threading import Event
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pandas as pd
import plotly.express as px
from typing import List, Dict, Any, Tuple, Generator

# Signal timing constants
MIN_GREEN_DURATION = 10  # Minimum green light duration in seconds
MAX_GREEN_DURATION_DYNAMIC = 60  # Maximum green light duration in seconds
YELLOW_DURATION = 3  # Yellow light duration in seconds
STARVATION_THRESHOLD = 5  # Number of cycles before starvation occurs

# Vehicle classification constants
VEHICLE_CLASSES = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']
EMERGENCY_CLASSES = ['ambulance', 'fire_truck', 'police']


model = YOLO("yolov8m.pt")
vehicle_classes = [1, 2, 3, 5]  # car, motorbike, bus, truck

# Global control events
stop_event = Event()
pause_event = Event()

class TrafficProcessor:
    """Handles traffic analysis and processing for individual video feeds."""
    
    def __init__(self, model, tracker):
        self.model = model
        self.tracker = tracker
        self.heatmap_data = []
        self.vehicle_types = {}
        self.speed_violations = 0
        self.emergency_vehicles = 0
        
    def process_frame(self, frame, speed_limit):
        """Process a single frame and return analysis results."""
        results = self.model(frame)[0]
        vehicle_count = 0
        speed_violators = 0
        emergency_count = 0
        emergency_detected = False
        
        # Reset vehicle types for this frame
        self.vehicle_types = {}
        
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf > 0.5:  # Confidence threshold
                if cls in vehicle_classes:
                    vehicle_count += 1
                    class_name = self.model.names[cls]
                    self.vehicle_types[class_name] = self.vehicle_types.get(class_name, 0) + 1
                    
                    # Check for emergency vehicles
                    if class_name.lower() in [e.lower() for e in EMERGENCY_CLASSES]:
                        emergency_count += 1
                        emergency_detected = True
                        
                    # Simulate speed detection (in real implementation, use speed estimation)
                    speed_violators = np.random.randint(0, 3)  # Random for demo
        
        # Calculate traffic score based on vehicle count
        traffic_score = min(vehicle_count * 2, 100)  # Cap at 100
        
        # Update heatmap
        self.heatmap_data.append(vehicle_count)
        if len(self.heatmap_data) > 100:
            self.heatmap_data.pop(0)
            
        return vehicle_count, speed_violators, emergency_count, emergency_detected, traffic_score
        
    def render_heatmap(self):
        """Generate a simple heatmap visualization."""
        if not self.heatmap_data:
            return np.zeros((100, 100, 3), dtype=np.uint8)
            
        # Create a simple density visualization
        heatmap = np.zeros((100, 100, 3), dtype=np.uint8)
        density = np.mean(self.heatmap_data) if self.heatmap_data else 0
        
        # Color based on density (green to red)
        intensity = min(int(density * 2.55), 255)
        color = (0, 255-intensity, intensity)  # BGR format
        
        cv2.rectangle(heatmap, (10, 10), (90, 90), color, -1)
        cv2.putText(heatmap, f"Density: {density:.1f}", (15, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return heatmap

def load_models():
    """Load and initialize the YOLO model and DeepSORT tracker."""
    model = YOLO("yolov8m.pt")
    tracker = DeepSort(
        max_age=30,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=0.2,
        nn_budget=None,
        override_track_class=None,
        embedder="mobilenet",
        half=True,
        bgr=True,
        embedder_gpu=True,
        embedder_model_name=None,
        embedder_wts=None,
        polygon=False,
        today=None
    )
    return model, tracker

def calculate_dynamic_green_duration(current_vehicles, all_scores):
    """Calculate dynamic green duration based on traffic conditions."""
    if not all_scores or current_vehicles == 0:
        return MIN_GREEN_DURATION
    
    # Calculate relative traffic load
    total_vehicles = sum(all_scores)
    if total_vehicles == 0:
        return MIN_GREEN_DURATION
    
    traffic_ratio = current_vehicles / total_vehicles
    
    # Scale duration between min and max
    duration = MIN_GREEN_DURATION + (traffic_ratio * (MAX_GREEN_DURATION_DYNAMIC - MIN_GREEN_DURATION))
    
    # Ensure bounds
    return max(MIN_GREEN_DURATION, min(MAX_GREEN_DURATION_DYNAMIC, int(duration)))

def process_video(video_file, signal_state, speed_limit, processor):
    """Enhanced video processing generator with full traffic analysis."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(video_file.read())
        video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    buffer = deque(maxlen=30)

    try:
        while cap.isOpened() and not stop_event.is_set():
            if pause_event.is_set():
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Process frame with traffic processor
            vehicle_count, speed_violators, emergency_count, emergency_detected, score = processor.process_frame(frame, speed_limit)
            
            # Draw bounding boxes
            results = model(frame)[0]
            for box in results.boxes:
                cls = int(box.cls[0])
                if cls in vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = (0, 255, 0)  # Green for normal vehicles
                    if any(emergency.lower() in model.names[cls].lower() for emergency in EMERGENCY_CLASSES):
                        color = (0, 0, 255)  # Red for emergency vehicles
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, model.names[cls], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            buffer.append(vehicle_count)
            max_count = max(buffer) if buffer else 0

            frame = draw_traffic_light(frame, signal_state.value)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get heatmap visualization
            heatmap = processor.render_heatmap()
            
            # Get vehicle types
            vehicle_types = processor.vehicle_types

            yield frame, max_count, vehicle_count, speed_violators, emergency_count, emergency_detected, heatmap, score, vehicle_types

    finally:
        cap.release()
        os.unlink(video_path)

# Draw traffic signal
def draw_traffic_light(frame, state):
    height, width = frame.shape[:2]
    light_height = height // 3
    light_width = width // 10
    light = np.zeros((light_height, light_width, 3), dtype=np.uint8)

    if state == 'red':
        cv2.circle(light, (light_width//2, light_height//4), light_width//4, (255, 0, 0), -1)
    elif state == 'yellow':
        cv2.circle(light, (light_width//2, light_height//2), light_width//4, (255, 255, 0), -1)
    elif state == 'green':
        cv2.circle(light, (light_width//2, 3 * light_height // 4), light_width//4, (0, 255, 0), -1)

    frame[0:light_height, 0:light_width] = light
    return frame

# Signal state holder
class SignalState:
    def __init__(self):
        self.value = 'red'
        self.last_change = time.time()
        self.vehicle_count = 0
        self.paused_frame = None
        self.transition_count = 0
        self.current_green_duration = MIN_GREEN_DURATION

def render_sidebar() -> Tuple[List[Any], int, str]:
    """Sets up the sidebar with configuration options."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        num_videos = st.number_input("Number of Camera Feeds", 1, 6, 2, key="num_videos_input")
        video_files = [st.file_uploader(f"Upload Video {i+1}", type=["mp4", "mov", "avi"], key=f"upload_{i}") for i in range(num_videos)]
        st.markdown("---")
        st.header("🚦 Signal Parameters")
        speed_limit = st.slider("Speed Limit (km/h)", 20, 120, 60, key="speed_limit")
        override = st.selectbox("Manual Override", ["Auto", "Force All Green", "Force All Red"], key="override")
        st.markdown("---")
        st.info(f"**Dynamic Green:** {MIN_GREEN_DURATION}s - {MAX_GREEN_DURATION_DYNAMIC}s\n\n**Yellow:** {YELLOW_DURATION}s")
    return video_files, speed_limit, override

def render_dashboard_layout(num_videos: int) -> Tuple[Dict, Dict]:
    """Creates the placeholder layout for the live dashboard and analytics tab."""
    tab1, tab2 = st.tabs(["🚦 Live Management", "📊 Analytics Dashboard"])

    with tab2:
        st.header("📊 Real-time & Post-Session Analytics")
        analytics_placeholders = {
            "live_heatmaps": st.empty(),
            "vehicle_dist": st.empty(),
            "vehiclecount_live": st.empty(),
            "speedviolators_live": st.empty(),
            "emergencycount_live": st.empty(),
            "avg_vehiclecount": st.empty(),
            "avg_speedviolators": st.empty(),
            "avg_emergencycount": st.empty(),
        }
        st.markdown("---")
        st.subheader("🔥 Final Cumulative Heatmaps")
        st.info("These heatmaps show the total traffic density over the entire monitoring session.")
        analytics_placeholders['final_heatmaps'] = st.empty()

    with tab1:
        st.subheader("📷 Live Feeds & Metrics")
        live_placeholders = {}
        ui_grid = [st.columns(2) for _ in range((num_videos + 1) // 2)]
        for i in range(num_videos):
            row, col = i // 2, i % 2
            with ui_grid[row][col]:
                st.markdown(f"**Road {i+1}**")
                live_placeholders[i] = {
                    'video': st.empty(),
                    'status': st.empty(),
                    'metrics': st.columns(3),
                    'timer': st.empty()
                }

    return live_placeholders, analytics_placeholders

def update_analytics_charts(placeholders: Dict, processors: List, num_videos: int, is_final: bool = False) -> None:
    """Updates the analytics charts with the latest data from session state."""
    data = st.session_state.get('analytics_data', {})
    if not data or not data.get("Timestamp"):
        return

    df = pd.DataFrame(data)
    df['Time'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.drop(columns=['Timestamp']).set_index('Time')
    
    # Generate unique key suffix based on current timestamp
    import time
    key_suffix = str(int(time.time() * 1000))[-6:]
    
    # Live charts
    if not is_final:
        with placeholders['live_heatmaps'].container():
            st.markdown("### Live Congestion Heatmaps")
            heatmap_grid = [st.columns(2) for _ in range((num_videos + 1) // 2)]
            for i, proc in enumerate(processors):
                row, col = i // 2, i % 2
                with heatmap_grid[row][col]:
                    st.image(proc.render_heatmap(), caption=f"Road {i+1}", use_container_width=True)

        with placeholders['vehiclecount_live'].container():
            st.markdown("### Vehicle Count Over Time")
            cols = [f"Road {i+1}_VehicleCount" for i in range(num_videos) if f"Road {i+1}_VehicleCount" in df.columns]
            if cols and not df[cols].empty:
                st.line_chart(df[cols]) 
        
        with placeholders['vehicle_dist'].container():
            st.markdown("### Vehicle Distribution (Current)")
            total_vehicle_types = {cls: 0 for cls in VEHICLE_CLASSES + EMERGENCY_CLASSES}
            for proc in processors:
                for cls, count in proc.vehicle_types.items():
                    total_vehicle_types[cls] += count
            
            non_zero_types = {k: v for k, v in total_vehicle_types.items() if v > 0}
            if non_zero_types:
                pie_df = pd.DataFrame(list(non_zero_types.items()), columns=['Vehicle Type', 'Count'])
                fig = px.pie(pie_df, values='Count', names='Vehicle Type', title='Current Vehicle Distribution')
                placeholders['vehicle_dist'].plotly_chart(fig, use_container_width=True, key=f"vehicle_dist_pie_{key_suffix}")

    # Bar charts (final or continuously updated)
    for metric, title in [('VehicleCount', 'Vehicles'), ('SpeedViolators', 'Speed Violators'), ('EmergencyCount', 'Emergency Vehicles')]:
        cols = [f"Road {i+1}_{metric}" for i in range(num_videos) if f"Road {i+1}_{metric}" in df.columns]
        if cols and not df[cols].empty:
            avg_df = df[cols].mean().reset_index(name='Average')
            avg_df['Road'] = avg_df['index'].str.replace(f'_{metric}', '')
            fig = px.bar(avg_df, x="Road", y="Average", color="Road", title=f"Average {title} Per Road", text_auto='.2f')
            placeholders[f'avg_{metric.lower()}'].plotly_chart(fig, use_container_width=True, key=f"avg_{metric.lower()}_bar_{key_suffix}")


def initialize_session(num_videos: int) -> None:
    """Sets up the session state for a new monitoring run."""
    st.session_state.is_running = True
    st.session_state.stop_event = Event()
    model, tracker = load_models()
    st.session_state.processors = [TrafficProcessor(model, tracker) for _ in range(num_videos)]
    st.session_state.signal_states = [SignalState() for _ in range(num_videos)]
    st.session_state.current_green_idx = -1
    st.session_state.analytics_data = {"Timestamp": [], **{f"Road {i+1}_{m}": [] for i in range(num_videos) for m in ['VehicleCount', 'SpeedViolators', 'EmergencyCount']}}

def get_next_green_index(all_scores: list, signal_states: List[SignalState], current_green_idx: int) -> int:
    """Determines the next road to get a green light based on traffic logic."""
    eligible_starvation = [i for i, s in enumerate(signal_states) if s.transition_count >= STARVATION_THRESHOLD]
    eligible_all = [i for i, s in enumerate(signal_states) if s.value == 'red']
    if eligible_starvation:
        next_green_idx = max(eligible_starvation, key=lambda i: all_scores[i])
    elif eligible_all:
        next_green_idx = max(eligible_all, key=lambda i: all_scores[i] if i != current_green_idx else -1)
    else:
        next_green_idx = 0 if len(signal_states) > 0 else -1
    return next_green_idx


def run_simulation_loop(video_files: List, speed_limit: int, override: str, live_ph: Dict, analytics_ph: Dict) -> None:
    """Contains the main while loop for processing and rendering the simulation."""
    st.success("🟢 Monitoring in progress...")
    processors = st.session_state.processors
    signal_states = st.session_state.signal_states
    
    generators = [process_video(vf, signal_states[i], speed_limit, processors[i]) for i, vf in enumerate(video_files)]
    last_data = [None] * len(video_files)
    
    if st.session_state.current_green_idx == -1:
        st.session_state.current_green_idx = 0
        signal_states[0].value = 'green'
        signal_states[0].last_change = time.time()
        
    while not st.session_state.stop_event.is_set():
        current_time = time.time()
        all_scores, emergency_on_road = [], -1

        for i, gen in enumerate(generators):
            try:
                # Corrected: Unpack all 9 values yielded by the generator
                frame, _, veh, spd, emg_count, emg_flag, heatmap, score, vehicle_types = next(gen)
                last_data[i] = (frame, veh, spd, emg_count, emg_flag, heatmap, score, vehicle_types)
                if emg_flag:
                    emergency_on_road = i
                all_scores.append(score)
            except (StopIteration, RuntimeError):
                st.session_state.stop_event.set()
                st.error("A video stream ended unexpectedly. Stopping simulation.")
                break
        
        if st.session_state.stop_event.is_set():
            break

        current_green_idx = st.session_state.current_green_idx
        
        if override == "Auto":
            if emergency_on_road != -1:
                if current_green_idx != emergency_on_road:
                    signal_states[current_green_idx].value = 'yellow'
                    signal_states[current_green_idx].last_change = current_time
                    st.session_state.next_green_idx = emergency_on_road
                elif signal_states[current_green_idx].value == 'yellow' and current_time - signal_states[current_green_idx].last_change >= YELLOW_DURATION:
                    signal_states[current_green_idx].value = 'red'
                    signal_states[current_green_idx].last_change = current_time
                    
                    emergency_state = signal_states[emergency_on_road]
                    emergency_state.value = 'green'
                    emergency_state.last_change = current_time
                    emergency_state.transition_count = 0
                    emergency_state.current_green_duration = MAX_GREEN_DURATION_DYNAMIC
                    st.session_state.current_green_idx = emergency_on_road
                elif current_green_idx == emergency_on_road and signal_states[current_green_idx].value == 'green':
                    signal_states[current_green_idx].current_green_duration = MAX_GREEN_DURATION_DYNAMIC
            else:
                current_state = signal_states[current_green_idx]
                if current_state.value == 'green' and current_time - current_state.last_change >= current_state.current_green_duration:
                    current_state.value = 'yellow'
                    current_state.last_change = current_time
                    st.session_state.next_green_idx = get_next_green_index(all_scores, signal_states, current_green_idx)
                elif current_state.value == 'yellow' and current_time - current_state.last_change >= YELLOW_DURATION:
                    current_state.value = 'red'
                    current_state.last_change = current_time
                    for i, s in enumerate(signal_states):
                        if i != current_green_idx:
                            s.transition_count += 1
                    
                    next_green_idx = st.session_state.next_green_idx
                    if next_green_idx != -1:
                        next_state = signal_states[next_green_idx]
                        next_state.value = 'green'
                        next_state.last_change = current_time
                        next_state.transition_count = 0
                        next_state.current_green_duration = calculate_dynamic_green_duration(all_scores[next_green_idx], all_scores)
                        st.session_state.current_green_idx = next_green_idx
        elif override != "Auto":
            for i, state in enumerate(signal_states):
                if override == "Force All Green":
                    state.value = 'green'
                elif override == "Force All Red":
                    state.value = 'red'
                state.last_change = current_time

        # Update analytics data in place (keep only latest values)
        if "Timestamp" not in st.session_state.analytics_data:
            st.session_state.analytics_data["Timestamp"] = []
        
        # Keep only the last 100 data points to prevent memory issues
        max_data_points = 100
        st.session_state.analytics_data["Timestamp"].append(current_time)
        if len(st.session_state.analytics_data["Timestamp"]) > max_data_points:
            st.session_state.analytics_data["Timestamp"] = st.session_state.analytics_data["Timestamp"][-max_data_points:]
            
        for i, data in enumerate(last_data):
            if data:
                frame, veh, spd, emg_count, emg_flag, heatmap, score, vehicle_types = data
                state = signal_states[i].value
                
                # Ensure lists exist
                if f"Road {i+1}_VehicleCount" not in st.session_state.analytics_data:
                    st.session_state.analytics_data[f"Road {i+1}_VehicleCount"] = []
                if f"Road {i+1}_SpeedViolators" not in st.session_state.analytics_data:
                    st.session_state.analytics_data[f"Road {i+1}_SpeedViolators"] = []
                if f"Road {i+1}_EmergencyCount" not in st.session_state.analytics_data:
                    st.session_state.analytics_data[f"Road {i+1}_EmergencyCount"] = []
                
                # Update in place by replacing last value or appending
                if len(st.session_state.analytics_data[f"Road {i+1}_VehicleCount"]) >= max_data_points:
                    st.session_state.analytics_data[f"Road {i+1}_VehicleCount"][-1] = veh
                    st.session_state.analytics_data[f"Road {i+1}_SpeedViolators"][-1] = spd
                    st.session_state.analytics_data[f"Road {i+1}_EmergencyCount"][-1] = emg_count
                else:
                    st.session_state.analytics_data[f"Road {i+1}_VehicleCount"].append(veh)
                    st.session_state.analytics_data[f"Road {i+1}_SpeedViolators"].append(spd)
                    st.session_state.analytics_data[f"Road {i+1}_EmergencyCount"].append(emg_count)

                live_ph[i]['video'].image(draw_traffic_light(frame.copy(), state), use_container_width=True)
                
                time_elapsed = int(current_time - signal_states[i].last_change)
                if state == 'green':
                    time_remaining = int(max(0, signal_states[i].current_green_duration - time_elapsed))
                    live_ph[i]['status'].success(f"🟢 Signal: GREEN")
                    live_ph[i]['timer'].markdown(f"**Timer:** {time_remaining}s remaining")
                elif state == 'yellow':
                    time_remaining = int(max(0, YELLOW_DURATION - time_elapsed))
                    live_ph[i]['status'].warning(f"🟡 Signal: YELLOW")
                    live_ph[i]['timer'].markdown(f"**Timer:** {time_remaining}s remaining")
                else:
                    live_ph[i]['status'].error(f"🔴 Signal: RED")
                    live_ph[i]['timer'].markdown(f"**Timer:** Waiting...")
                
                live_ph[i]['metrics'][0].metric("Total Vehicles", veh)
                live_ph[i]['metrics'][1].metric("Speed Violators", spd)
                live_ph[i]['metrics'][2].metric("Emergency", emg_count)

        update_analytics_charts(analytics_ph, processors, len(processors))
        time.sleep(0.05)


def main() -> None:
    """Main function to run the Streamlit application."""
    st.title("🚦 Smart Traffic Signal Management System")
    st.markdown("An intelligent, real-time traffic control system using **YOLOv8**, **DeepSORT**, and dynamic signal logic.")

    video_files, speed_limit, override = render_sidebar()
    live_placeholders, analytics_placeholders = render_dashboard_layout(len(video_files))

    col1, col2 = st.columns(2)
    if col1.button("▶️ Start Monitoring", use_container_width=True, disabled=st.session_state.get('is_running', False)):
        if all(video_files):
            initialize_session(len(video_files))
            st.rerun()
        else:
            st.warning("Please upload all required video files to begin.")

    if col2.button("⏹️ Stop Monitoring", use_container_width=True, disabled=not st.session_state.get('is_running', False)):
        if 'stop_event' in st.session_state:
            st.session_state.stop_event.set()
        st.session_state.is_running = False
        st.rerun()

    if st.session_state.get('is_running', False):
        run_simulation_loop(video_files, speed_limit, override, live_placeholders, analytics_placeholders)
    
    elif not st.session_state.get('is_running', False) and 'processors' in st.session_state:
        st.info("✅ Monitoring stopped. Final analytics are now available in the 'Analytics Dashboard' tab.")
        if 'stop_event' in st.session_state:
            st.session_state.stop_event.clear()
        
        with analytics_placeholders['final_heatmaps'].container():
            processors = st.session_state.get('processors', [])
            if processors:
                heatmap_grid = [st.columns(2) for _ in range((len(processors) + 1) // 2)]
                for i, proc in enumerate(processors):
                    row, col = i // 2, i % 2
                    with heatmap_grid[row][col]:
                        st.markdown(f"**Road {i+1} Final Heatmap**")
                        st.image(proc.render_heatmap(), caption=f"Cumulative density for Road {i+1}", use_container_width=True)
        
        update_analytics_charts(analytics_placeholders, st.session_state.processors, len(st.session_state.processors), is_final=True)

if __name__ == "__main__":
    main()