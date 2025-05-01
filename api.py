import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import os
from collections import deque
from threading import Event
from ultralytics import YOLO
import pandas as pd
import plotly.express as px

# Initialize model and classes
model = YOLO("yolov8m.pt")
vehicle_classes = [1, 2, 3, 5]  # car, motorbike, bus, truck

# Global control events
stop_event = Event()
pause_event = Event()

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

# Processing generator for each video
def process_video(video_file, signal_state):
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

            results = model(frame)[0]
            count = 0
            for box in results.boxes:
                cls = int(box.cls[0])
                if cls in vehicle_classes:
                    count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            buffer.append(count)
            max_count = max(buffer)

            frame = draw_traffic_light(frame, signal_state.value)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            yield frame, max_count

    finally:
        cap.release()
        os.unlink(video_path)

# Signal state holder
class SignalState:
    def __init__(self):
        self.value = 'red'
        self.last_change = time.time()
        self.vehicle_count = 0
        self.paused_frame = None
        self.transition_count = 0

# Main app
def main():
    st.set_page_config(layout="wide", page_title="Smart Traffic Management")

    tab1, tab2 = st.tabs(["🚦 Traffic Management", "📊 Analytics"])

    with tab1:
        st.title("🚦 Smart Traffic Signal Management")

        num_videos = st.number_input("How many video feeds do you want to upload?", min_value=1, max_value=6, value=3)
        video_files = [st.file_uploader(f"Upload Video {i + 1}", type=["mp4", "avi"]) for i in range(num_videos)]
        video_files = [vf for vf in video_files if vf]

        if len(video_files) < num_videos:
            st.warning(f"Please upload {num_videos} video files.")
            return

        red_duration = st.slider("🔴 Red Light Duration (s)", 5, 60, 10)
        yellow_duration = st.slider("🟡 Yellow Light Duration (s)", 1, 5, 2)
        max_green_duration = 30

        signal_states = [SignalState() for _ in video_files]
        video_placeholders = []
        counter_placeholders = []

        st.markdown("### Live Video Feeds")
        rows = (len(video_files) + 1) // 2
        for i in range(rows):
            cols = st.columns(2)
            for j in range(2):
                idx = i * 2 + j
                if idx < len(video_files):
                    with cols[j]:
                        st.subheader(f"Road {idx + 1}")
                        video_placeholders.append(st.empty())
                        counter_placeholders.append(st.empty())

        col1, col2 = st.columns(2)
        start = col1.button("▶️ Start")
        stop = col2.button("⏹ Stop")

        analytics_data = {f"Road {i+1}": [] for i in range(len(video_files))}

        if start:
            stop_event.clear()
            pause_event.clear()
            generators = [process_video(vf, state) for vf, state in zip(video_files, signal_states)]
            last_frames = [next(gen) for gen in generators]

            current_green = max(range(len(last_frames)), key=lambda i: last_frames[i][1])
            signal_states[current_green].value = 'green'
            signal_states[current_green].last_change = time.time()

            while not stop_event.is_set():
                current_time = time.time()

                for i, (gen, ph, ch, state) in enumerate(zip(generators, video_placeholders, counter_placeholders, signal_states)):
                    if state.value != 'green':
                        frame, count = last_frames[i]
                        if state.paused_frame is None:
                            state.paused_frame = frame
                        state.vehicle_count = count
                        ch.text(f"Vehicles: {count}")
                        ph.image(draw_traffic_light(frame.copy(), state.value))
                        analytics_data[f"Road {i+1}"].append(count)
                    else:
                        try:
                            frame, count = next(gen)
                            last_frames[i] = (frame, count)
                            state.paused_frame = None
                            ph.image(frame)
                            analytics_data[f"Road {i+1}"].append(count)
                        except StopIteration:
                            st.warning(f"Video {i+1} ended.")

                if signal_states[current_green].value == 'green' and current_time - signal_states[current_green].last_change >= max_green_duration:
                    signal_states[current_green].value = 'yellow'
                    for _ in range(yellow_duration):
                        for i, ph in enumerate(video_placeholders):
                            ph.image(draw_traffic_light(last_frames[i][0].copy(), signal_states[i].value))
                        time.sleep(1)
                    signal_states[current_green].value = 'red'
                    signal_states[current_green].last_change = time.time()
                    current_green = (current_green + 1) % len(signal_states)

                elif current_time - signal_states[current_green].last_change >= red_duration:
                    for state in signal_states:
                        if state.value == 'red':
                            state.transition_count += 1

                    priority = [i for i, s in enumerate(signal_states) if s.transition_count >= 4]
                    next_green = max(priority, key=lambda i: signal_states[i].vehicle_count, default=None)

                    if next_green is None:
                        reds = [i for i, s in enumerate(signal_states) if s.value == 'red']
                        next_green = max(reds, key=lambda i: signal_states[i].vehicle_count, default=current_green)

                    signal_states[current_green].value = 'yellow'
                    for _ in range(yellow_duration):
                        for i, ph in enumerate(video_placeholders):
                            ph.image(draw_traffic_light(last_frames[i][0].copy(), signal_states[i].value))
                        time.sleep(1)

                    signal_states[current_green].value = 'red'
                    signal_states[next_green].value = 'yellow'
                    for _ in range(yellow_duration):
                        for i, ph in enumerate(video_placeholders):
                            ph.image(draw_traffic_light(last_frames[i][0].copy(), signal_states[i].value))
                        time.sleep(1)

                    current_green = next_green
                    signal_states[current_green].value = 'green'
                    signal_states[current_green].last_change = time.time()
                    signal_states[current_green].transition_count = 0

        if stop:
            stop_event.set()

    with tab2:
        st.title("📊 Traffic Analytics Dashboard")

        if not stop_event.is_set():
            st.info("Start and stop the traffic management to view analytics.")
        else:
            st.success("Analytics from previous session shown below.")
            df = pd.DataFrame(analytics_data)
            st.line_chart(df)

            st.subheader("📈 Average Vehicle Count per Road")
            avg_df = pd.DataFrame(df.mean()).reset_index()
            avg_df.columns = ['Road', 'Average Vehicles']
            fig = px.bar(avg_df, x="Road", y="Average Vehicles", color="Road", title="Average Traffic Per Road")
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
