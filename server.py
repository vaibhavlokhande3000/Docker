
import cv2
import numpy as np
import time
from PIL import Image
from gaze_tracking import GazeTracking
from fer import FER
from flask import Flask, render_template, request, session
from flask_socketio import SocketIO, emit, join_room
import platform

app = Flask(__name__)
app.config['SECRET_KEY'] = "wubba lubba dub dub"
socketio = SocketIO(app)

gaze = GazeTracking()
emotion_detector = FER()

users_data = {}
rooms_sid = {}
names_sid = {}

def analyze_attention_with_gaze(gaze):
    if not gaze.pupils_located:
        return 0.0
    if gaze.is_blinking():
        return 0.2
    elif gaze.is_right() or gaze.is_left():
        return 0.4
    elif gaze.is_center():
        return 0.8
    else:
        return 0.5

def analyze_behavior_with_emotion(frame):
    emotion_results = emotion_detector.detect_emotions(frame)
    if emotion_results:
        dominant_emotion = max(emotion_results[0]['emotions'], key=emotion_results[0]['emotions'].get)
        return map_emotion_to_behavior(dominant_emotion)
    return 'No face detected'

def map_emotion_to_behavior(emotion):
    emotion_behavior_map = {
        'angry': 'Agitated',
        'disgust': 'Disgusted',
        'fear': 'Fearful',
        'happy': 'Engaged',
        'sad': 'Distracted',
        'surprise': 'Alert',
        'neutral': 'Calm'
    }
    return emotion_behavior_map.get(emotion, 'Neutral')

@socketio.on("video-feed")
def on_video_feed(data):
    sid = request.sid
    room_id = rooms_sid.get(sid)
    frame_data = data.get("frame")

    if frame_data is None or room_id is None:
        return

    nparr = np.frombuffer(bytes(frame_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gaze.refresh(frame)
    attention_level = analyze_attention_with_gaze(gaze)
    behavior = analyze_behavior_with_emotion(frame)

    if sid not in users_data:
        users_data[sid] = {"attention_data": [], "behavior_data": []}
    users_data[sid]["attention_data"].append(attention_level)
    users_data[sid]["behavior_data"].append(behavior)

    emit("analysis-result", {"attention_level": attention_level, "behavior": behavior}, room=sid)

@socketio.on("join-room")
def on_join_room(data):
    sid = request.sid
    room_id = data["room_id"]
    display_name = data["display_name"]
    join_room(room_id)
    rooms_sid[sid] = room_id
    names_sid[sid] = display_name
    users_data[sid] = {"attention_data": [], "behavior_data": []}
    print(f"[{room_id}] New member joined: {display_name} ({sid})")

@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    room_id = rooms_sid.get(sid)
    if room_id:
        del users_data[sid]
        del rooms_sid[sid]
        del names_sid[sid]

@app.route("/generate-report/<sid>", methods=["GET"])
def generate_report(sid):
    if sid not in users_data:
        return "User data not found", 404
    attention_data = users_data[sid]["attention_data"]
    behavior_data = users_data[sid]["behavior_data"]
    score = calculate_score(attention_data, behavior_data)
    report_path = f"{names_sid.get(sid)}_meeting_report.pdf"
    generate_pdf_report(names_sid.get(sid), attention_data, behavior_data, score, report_path)
    return f"Report generated: {report_path}", 200

def calculate_score(attention_data, behavior_data):
    avg_attention = np.mean(attention_data)
    behavior_scores = {
        'Engaged': 1.0,
        'Alert': 0.8,
        'Calm': 0.7,
        'Neutral': 0.6,
        'Distracted': 0.4,
        'Agitated': 0.3,
        'Disgusted': 0.2,
        'Fearful': 0.1,
        'No face detected': 0.0
    }
    avg_behavior = np.mean([behavior_scores.get(b, 0.5) for b in behavior_data])
    return (avg_attention + avg_behavior) / 2 * 100

def generate_pdf_report(name, attention_data, behavior_data, score, report_path):
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(report_path)
    c.drawString(100, 800, f"Meeting Report for {name}")
    c.drawString(100, 780, f"Average Attention Level: {np.mean(attention_data):.2f}")
    c.drawString(100, 760, f"Overall Score: {score:.2f}/100")
    c.save()

if __name__ == "__main__":
    socketio.run(app, debug=True)
