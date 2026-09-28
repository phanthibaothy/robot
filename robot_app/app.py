import base64
import math
import os
import json
import time
import threading
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'robot_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# =========================================================
# 1. KHỞI TẠO YOLOV8 POSE AI & THƯ MỤC FEEDBACK DATASET
# =========================================================
yolo_model = YOLO('yolov8n-pose.pt')

FEEDBACK_DIR = "dataset_feedback"
os.makedirs(os.path.join(FEEDBACK_DIR, "correct"), exist_ok=True)
os.makedirs(os.path.join(FEEDBACK_DIR, "incorrect"), exist_ok=True)

last_raw_frame = None

# =========================================================
# 2. HÀM TÍNH KHOẢNG CÁCH GIỮA 2 ESP32 DỰA TRÊN BLE RSSI
# =========================================================
def calculate_distance_from_rssi(rssi, tx_power=-59, n=2.5):
    """
    Tính khoảng cách ước tính (mét) giữa ESP32 Robot và ESP32 Bệnh nhân
    sử dụng mô hình Log-distance Path Loss.
    """
    if rssi == 0 or rssi <= -100:
        return -1.0
    try:
        dist = math.pow(10.0, (tx_power - rssi) / (10.0 * n))
        return round(dist, 2)
    except Exception:
        return -1.0

# =========================================================
# 3. BỘ LỌC TRẠNG THÁI UỐNG THUỐC (YOLOV8 KEYPOINTS)
# =========================================================
drink_state = "IDLE"        
near_mouth_count = 0
done_display_timer = 0     
warmup_counter = 3
missing_frame_count = 0    
has_opened_tray_for_alarm = False  # Cờ đánh dấu đã mở khay thuốc cho đợt báo thức này

def reset_ai_state():
    global drink_state, near_mouth_count, done_display_timer, warmup_counter, missing_frame_count, has_opened_tray_for_alarm
    drink_state = "IDLE"
    near_mouth_count = 0
    done_display_timer = 0
    warmup_counter = 3
    missing_frame_count = 0
    has_opened_tray_for_alarm = False

def process_yolo_drinking(keypoints_data, frame_shape):
    global drink_state, near_mouth_count, done_display_timer, warmup_counter, missing_frame_count, has_opened_tray_for_alarm

    if warmup_counter > 0:
        warmup_counter -= 1
        return "Đang ổn định Camera AI..."

    img_h, img_w = frame_shape[:2]
    kpts = keypoints_data.cpu().numpy()

    nose = kpts[0]
    sh_l = kpts[5]
    sh_r = kpts[6]
    wrist_l = kpts[9]
    wrist_r = kpts[10]

    if nose[2] < 0.4:
        return "Không rõ khuôn mặt"

    nose_x, nose_y = nose[0] / img_w, nose[1] / img_h
    mouth_x, mouth_y = nose_x, nose_y + 0.035
    sh_y_avg = (sh_l[1] + sh_r[1]) / (2.0 * img_h) if (sh_l[2] > 0.3 or sh_r[2] > 0.3) else (mouth_y + 0.15)

    valid_dists = []
    for wrist in [wrist_l, wrist_r]:
        wx, wy, wconf = wrist[0] / img_w, wrist[1] / img_h, wrist[2]
        if wconf > 0.35:
            if (nose_y - 0.04) <= wy <= (sh_y_avg + 0.20):
                d = math.hypot(wx - mouth_x, wy - mouth_y)
                valid_dists.append(d)

    THRESHOLD_NEAR = 0.18
    THRESHOLD_FAR  = 0.28

    if not valid_dists:
        if drink_state == "NEAR_MOUTH":
            drink_state = "DONE"
            done_display_timer = 40  
            near_mouth_count = 0
            socketio.emit('cmd_to_pi', {'type': 'MEDICINE_DONE'})
            return "Đã uống thuốc xong 💊✅"
        elif drink_state == "DONE":
            done_display_timer -= 1
            if done_display_timer <= 0:
                reset_ai_state()
            return "Đã uống thuốc xong 💊✅"
        return "Bình thường (Chưa uống)"

    min_dist = min(valid_dists)

    if drink_state == "IDLE":
        if min_dist < THRESHOLD_NEAR:
            near_mouth_count += 1
            if near_mouth_count >= 1:
                drink_state = "NEAR_MOUTH"
        else:
            near_mouth_count = 0  

    elif drink_state == "NEAR_MOUTH":
        if min_dist > THRESHOLD_FAR:
            drink_state = "DONE"
            done_display_timer = 40  
            near_mouth_count = 0
            socketio.emit('cmd_to_pi', {'type': 'MEDICINE_DONE'})

    elif drink_state == "DONE":
        done_display_timer -= 1
        if done_display_timer <= 0:
            reset_ai_state()  
        return "Đã uống thuốc xong 💊✅"

    if drink_state == "NEAR_MOUTH":
        return "Đang đưa thuốc lên miệng..."
    else:
        return "Bình thường (Chưa uống)"

# =========================================================
# 4. TRẠNG THÁI HỆ THỐNG TOÀN CỤC
# =========================================================
pending_esp32_cmd = "STOP"
system_state = {
    "heart_rate": "--",
    "body_temp": "--",
    "cpu_temp": "0",
    "ai_res": "Chờ lệnh",
    "compass": 0,
    "hc04": 0,
    "servo_hc04_angle": 90,
    "motor_speed_l": 0,
    "motor_speed_r": 0,
    "motor_speed_set": 50,
    "mode": "MANUAL",
    "phone_number": "",
    "alarms": [],
    "alert": "",
    "battery": "--%",
    "relay1": True,
    "relay2": True,
    "patient_rssi": -100,
    "patient_dist": -1.0  
}

def get_next_med_time():
    if not system_state['alarms']:
        return "Chưa đặt"
    
    now_str = datetime.now().strftime("%H:%M")
    upcoming = [a for a in system_state['alarms'] if a > now_str]
    
    if upcoming:
        return min(upcoming)
    else:
        return min(system_state['alarms'])

# =========================================================
# 5. THREAD TỰ ĐỘNG KIỂM TRA LỊCH UỐNG THUỐC TRÊN SERVER
# =========================================================
completed_alarms_today = set()  # Lưu các mốc giờ đã uống xong trong ngày
last_trigger_time = ""

def alarm_checker_loop():
    global drink_state, last_trigger_time
    while True:
        try:
            now = datetime.now()
            now_hm = now.strftime("%H:%M")
            now_full = now.strftime("%H:%M:%S")

            # 1. Tự động làm sạch danh sách đã uống khi sang ngày mới (00:00:00)
            if now_full == "00:00:00":
                completed_alarms_today.clear()

            # 2. Kiểm tra từng lịch hẹn trong system_state['alarms']
            for alarm in system_state.get('alarms', []):
                if now_hm >= alarm and alarm not in completed_alarms_today:
                    
                    if drink_state == "DONE":
                        completed_alarms_today.add(alarm)
                        print(f"✅ [LỊCH THUỐC] Mốc giờ {alarm} đã hoàn thành uống thuốc!")
                    else:
                        if last_trigger_time != now_hm:
                            print(f"⏰ [CẢNH BÁO LỊCH] Đã đến/quá giờ uống thuốc ({alarm})! Đang gọi Pi...")
                            last_trigger_time = now_hm
                        
                        # Gửi lệnh SocketIO đánh thức Pi & nâng Tầng 3 lên 90° để quét tìm người
                        socketio.emit('cmd_to_pi', {'type': 'START_MEDICINE_TIME'})

        except Exception as e:
            print(f"❌ Lỗi Thread kiểm tra lịch: {e}")

        time.sleep(3)

# =========================================================
# 6. ROUTES API
# =========================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/esp32/status', methods=['POST'])
def receive_esp32_status():
    data = request.get_json() or {}
    system_state['compass'] = data.get('heading', 0)
    system_state['hc04'] = data.get('obstacle_dist', 0)
    system_state['servo_hc04_angle'] = data.get('servo_hc04', 90)
    system_state['motor_speed_l'] = data.get('speed_l', 0)
    system_state['motor_speed_r'] = data.get('speed_r', 0)
    
    if 'rssi' in data or 'patient_rssi' in data:
        rssi_val = data.get('rssi', data.get('patient_rssi', -100))
        system_state['patient_rssi'] = rssi_val
        if 'patient_dist' in data and data['patient_dist'] > 0:
            system_state['patient_dist'] = round(float(data['patient_dist']), 2)
        else:
            system_state['patient_dist'] = calculate_distance_from_rssi(rssi_val)

    if 'battery_volts' in data:
        system_state['battery'] = f"{float(data['battery_volts']):.1f} V"
    elif 'battery' in data:
        system_state['battery'] = f"{data['battery']}%"

    if data.get('battery_low'):
        system_state['alert'] = "CẢNH BÁO: Điện áp Pin Robot yếu!"
        
    socketio.emit('update_ui', system_state)
    return jsonify({"status": "ok"}), 200

@app.route('/api/esp32/command', methods=['GET'])
def send_esp32_command():
    global pending_esp32_cmd
    cmd = pending_esp32_cmd
    if pending_esp32_cmd not in ["STOP"]:
        pending_esp32_cmd = "STOP"
    return jsonify({
        "cmd": cmd,
        "mode": system_state["mode"],
        "speed": system_state["motor_speed_set"],
        "alarms": system_state["alarms"]
    }), 200

@app.route('/api/patient/health', methods=['POST'])
def receive_patient_health():
    data = request.get_json() or {}
    system_state['heart_rate'] = data.get('heart_rate', '--')
    system_state['body_temp'] = data.get('body_temp', '--')
    
    if 'rssi' in data:
        rssi_val = int(data['rssi'])
        system_state['patient_rssi'] = rssi_val
        system_state['patient_dist'] = calculate_distance_from_rssi(rssi_val)

    try:
        if float(data.get('body_temp', 0)) >= 41.0:
            system_state['alert'] = "CẢNH BÁO: Bệnh nhân sốt cao > 41°C!"
    except ValueError:
        pass
        
    socketio.emit('update_ui', system_state)

    return jsonify({
        "status": "ok",
        "next_med_time": get_next_med_time(),
        "rssi": system_state.get('patient_rssi', -100),
        "patient_dist": system_state.get('patient_dist', -1.0)
    }), 200

# =========================================================
# 7. SOCKETIO CONTROL & ACTIVE LEARNING FEEDBACK
# =========================================================
@socketio.on('connect')
def handle_connect():
    print("🔌 [SOCKET.IO] Thiết bị (Pi/Web) đã kết nối thành công!")
    # Tự đồng bộ: Khi Pi bật nguồn xong và vừa kết nối, nếu đúng giờ báo thức thì gửi ngay lệnh
    now_hm = datetime.now().strftime("%H:%M")
    for alarm in system_state.get('alarms', []):
        if now_hm == alarm and alarm not in completed_alarms_today:
            print(f"⏰ [SYNC BOOT] Pi vừa lên nguồn đúng giờ uống thuốc ({alarm})! Gửi ngay lệnh khởi động...")
            socketio.emit('cmd_to_pi', {'type': 'START_MEDICINE_TIME'})

@socketio.on('ui_feedback_ai')
def handle_ai_feedback(data):
    global last_raw_frame, system_state
    if last_raw_frame is None:
        system_state['alert'] = "Không tìm thấy ảnh để lưu feedback!"
        socketio.emit('update_ui', system_state)
        return

    is_correct = data.get('is_correct', True)
    folder = "correct" if is_correct else "incorrect"
    timestamp = int(time.time() * 1000)

    img_path = os.path.join(FEEDBACK_DIR, folder, f"frame_{timestamp}.jpg")
    json_path = os.path.join(FEEDBACK_DIR, folder, f"frame_{timestamp}.json")

    cv2.imwrite(img_path, last_raw_frame)

    meta_info = {
        "timestamp": timestamp,
        "ai_prediction": system_state.get('ai_res', ''),
        "user_confirmed_correct": is_correct,
        "notes": data.get('notes', '')
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)

    status_str = "ĐÚNG ✅" if is_correct else "SAI ❌"
    system_state['alert'] = f"Đã lưu mẫu ({status_str}) vào Dataset để Retrain!"
    socketio.emit('update_ui', system_state)

@socketio.on('ui_send_command')
def handle_ui_cmd(data):
    global pending_esp32_cmd
    cmd_type = data.get('type')
    payload = data.get('payload', {})

    if cmd_type == 'WAKE_UP':
        reset_ai_state()
        emit('cmd_to_pi', data, broadcast=True)
    elif cmd_type in ['SERVO_CONTROL', 'SAVE_CONFIG']:
        emit('cmd_to_pi', data, broadcast=True)
    elif cmd_type == 'MOTOR_MOVE':
        pending_esp32_cmd = payload.get('direction', 'STOP')
    elif cmd_type == 'CALIB_COMPASS':
        pending_esp32_cmd = "CALIB_QMC"
        system_state['alert'] = "Đang Calib La Bàn QMC! Vui lòng xoay Robot 360° trong 10 giây..."
    elif cmd_type == 'SET_SPEED':
        system_state['motor_speed_set'] = payload.get('speed', 50)
    elif cmd_type == 'TRAY_CONTROL':
        pending_esp32_cmd = f"TRAY_{payload.get('tray_id')}_{payload.get('action')}"
    elif cmd_type == 'TOGGLE_MODE':
        system_state['mode'] = payload.get('mode', 'MANUAL')
    elif cmd_type == 'RELAY_CONTROL':
        relay_id = payload.get('relay_id')
        action = payload.get('action')
        pending_esp32_cmd = f"RELAY{relay_id}_{action}"
        system_state[f'relay{relay_id}'] = (action == 'ON')
    elif cmd_type == 'ADD_ALARM':
        alarm_time = payload.get('time')
        if alarm_time and alarm_time not in system_state['alarms']:
            system_state['alarms'].append(alarm_time)
            system_state['alarms'].sort()
    elif cmd_type == 'DELETE_ALARM':
        alarm_time = payload.get('time')
        if alarm_time in system_state['alarms']:
            system_state['alarms'].remove(alarm_time)
    elif cmd_type == 'SAVE_PHONE':
        system_state['phone_number'] = payload.get('phone', '')
    elif cmd_type == 'CALL_PHONE':
        phone_to_call = payload.get('phone', system_state['phone_number'])
        if phone_to_call:
            pending_esp32_cmd = f"CALL_{phone_to_call}"
            system_state['alert'] = f"Đang thực hiện cuộc gọi thử tới: {phone_to_call}"
    elif cmd_type == 'SEND_ALERT':
        system_state['alert'] = payload.get('message', 'Cảnh báo khẩn cấp!')

    socketio.emit('update_ui', system_state)

@socketio.on('pi_data_upload')
def handle_pi_data(data):
    global missing_frame_count, last_raw_frame, has_opened_tray_for_alarm
    img_b64 = data.get('camera_frame', '')
    ai_status = "Không có luồng ảnh"

    if img_b64:
        try:
            img_bytes = base64.b64decode(img_b64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
                last_raw_frame = frame.copy()

                results = yolo_model(frame, conf=0.5, verbose=False)

                person_found = False
                for r in results:
                    if r.keypoints is not None and len(r.keypoints) > 0:
                        person_found = True
                        missing_frame_count = 0

                        # === KÍCH HOẠT XOAY SERVO (0°, 0°, 50°) KHI YOLO THẤY NGƯỜI ===
                        now_hm = datetime.now().strftime("%H:%M")
                        if (now_hm in system_state.get('alarms', [])) and not has_opened_tray_for_alarm:
                            print("🤖 [YOLO DETECTED] Đã thấy người! Gửi lệnh ARRIVED_PATIENT để Pi mở khay (0°, 0°, 50°)...")
                            socketio.emit('cmd_to_pi', {'type': 'ARRIVED_PATIENT'})
                            has_opened_tray_for_alarm = True

                        kpts_data = r.keypoints.data[0]
                        ai_status = process_yolo_drinking(kpts_data, frame.shape)
                        frame = r.plot() 
                        break

                if not person_found:
                    missing_frame_count += 1
                    if missing_frame_count > 10:
                        reset_ai_state()
                        ai_status = "Không phát hiện người"
                    else:
                        ai_status = system_state.get('ai_res', 'Đang xử lý...')

                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                img_b64 = base64.b64encode(buffer).decode('utf-8')

        except Exception:
            ai_status = "Đang xử lý ảnh..."

    system_state['cpu_temp'] = data.get('cpu_temp', '0')
    system_state['ai_res'] = ai_status

    data_to_send = dict(system_state)
    data_to_send['camera_frame'] = img_b64
    emit('update_ui', data_to_send, broadcast=True)

# =========================================================
# 8. KÍCH HOẠT SERVER VÀ THREAD NGẦM
# =========================================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
