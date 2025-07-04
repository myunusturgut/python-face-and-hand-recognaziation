import cv2
import mediapipe as mp

# Modülleri hazırla
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# El ve yüz modellerini başlat
hands = mp_hands.Hands(max_num_hands=2)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Kamera
cap = cv2.VideoCapture(0)

# Duygu eşikleri
MUTLU_THRESHOLD = 10
UZGUN_THRESHOLD = -3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü RGB'ye çevir
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # El tespiti
    hand_result = hands.process(rgb)
    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )
            cv2.putText(frame, "El Algilandi", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Yüz iskeleti tespiti ve duygu tahmini
    face_result = face_mesh.process(rgb)
    emotion = "Yuz Yok"
    if face_result.multi_face_landmarks:
        for face_landmarks in face_result.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)
            )

            h, w, _ = frame.shape
            landmarks = face_landmarks.landmark

            # Üst ve alt dudak noktaları (13 ve 14)
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]
            y_diff = (lower_lip.y - upper_lip.y) * h

            if y_diff > MUTLU_THRESHOLD:
                emotion = "Mutlu"
            elif y_diff < UZGUN_THRESHOLD:
                emotion = "Uzgun"
            else:
                emotion = "Normal"

    # Ekrana duygu yazısı
    cv2.putText(frame, f"Duygu: {emotion}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)

    # Göster
    cv2.imshow("El ve Yuz Tespiti", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
