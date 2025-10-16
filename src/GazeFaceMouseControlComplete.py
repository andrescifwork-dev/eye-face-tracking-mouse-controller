#ANDRES FABIAN CIFUENTES MUCIÑO.
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import threading
import traceback

# Intento de import de gpiozero; si no estás en una Raspberry Pi,
# usamos un mock ligero para no romper la ejecución.
try:
    from gpiozero import DigitalOutputDevice
except Exception:
    class DigitalOutputDevice:
        def _init_(self, pin):
            self.pin = pin
            self._state = False
        def on(self): self._state = True
        def off(self): self._state = False
        def close(self): pass
        def _repr_(self): return f"<MockPin {self.pin} state={self._state}>"

# ------------------ Ajustes principales ------------------
radius_scale = 0.60
min_radius   = 11
max_radius   = 40
center_offset_x = 0   # <-- poner 0 para evitar sesgo por defecto; calibrar si hace falta
center_offset_y = -2
mask_margin = 2

# GPIO pins
pin_arriba = DigitalOutputDevice(17) #era 4
pin_abajo = DigitalOutputDevice(4)#era 17
pin_izquierda = DigitalOutputDevice(22) #era 27
pin_derecha = DigitalOutputDevice(27) #era 22
pin_clickI = DigitalOutputDevice(5)
pin_ScrollUp = DigitalOutputDevice(19)
pin_ScrollDwn = DigitalOutputDevice(26)

pin_modo_2 = DigitalOutputDevice(6)
pin_modo_3 = DigitalOutputDevice(13)

pin_led_azul = DigitalOutputDevice(18)
pin_led_verde = DigitalOutputDevice(23)
pin_led_rojo = DigitalOutputDevice(24)

# tiempos para selección de modo por apertura de boca
preview_t1 = 1.0
preview_t2 = 3.0
preview_t3 = 5.0

def set_mode_leds(mode):
    if mode == 1:
        pin_led_rojo.on();  pin_led_verde.off(); pin_led_azul.off()
    elif mode == 2:
        pin_led_verde.on(); pin_led_rojo.off();  pin_led_azul.off()
    elif mode == 3:
        pin_led_azul.on();  pin_led_rojo.off();  pin_led_verde.off()
    else:
        pin_led_rojo.off(); pin_led_verde.off(); pin_led_azul.off()

# MediaPipe init (creado globalmente y cerrado al final)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Camera
cap = cv2.VideoCapture(0)
print("DEBUG: cap.isOpened() ->", cap.isOpened())

zoom_level = 2.9

# Click params
radio_circulo_superior = 11
radio_circulo_inferior = 11
click_hold_time   = 1.0
click_pulse_time  = 0.10
click_threshold_px = radio_circulo_superior + radio_circulo_inferior

# Estado global inicializado
eye_position = None
update_time = time.time()
mouth_open_threshold = 0.03

start_time_click_right = None
click_performed = False
click_time = None

mouth_was_open = False
mouth_open_start_time = None

current_mode = 1
submode1_active = False  # submodo 1.2: mapea izquierda->arriba, derecha->abajo

# Buffers / compatibility
diff_buffer = []
buffer_size = 6
movement_thr = 0.12
dead_zone = 0.10

# Hysteresis thresholds (enter/exit) para estabilidad LR
enter_threshold = 0.20
exit_threshold  = 0.12

# --- PARÁMETROS GRILLA ---
grid_rows = 10 #era 7x7
grid_cols = 10#era 7X7
min_cell_pixels = 6
activation_delta_thresh = 0.08
activation_abs_thresh = 0.18
quorum_pct = 0.90
# umbrales por lado (número mínimo de celdas activas requeridas para aceptar esa dirección)
min_active_cells = 0
left_min_active_cells = 25   # ajustar según pruebas (1..4 típico)
right_min_active_cells = 25  # ajustar según pruebas (1..4 típico)
debounce_frames = 4
debug_overlay = True

# Debounce state
last_candidate = "centro"
candidate_count = 0
stable_gaze = "centro"

# Landmarks para contorno completo del ojo izquierdo (MediaPipe FaceMesh indices)
LEFT_EYE_OUTLINE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

def decide_gaze_lr_using_ratios(left_active, left_valid, right_active, right_valid, current_stable):
    """
    Decide entre 'izquierda', 'derecha' o 'centro' usando ratios normalizadas
    y aplicando hysteresis (enter/exit thresholds).
    """
    # calcular ratios (evitar division por cero)
    left_ratio = (left_active / left_valid) if left_valid > 0 else 0.0
    right_ratio = (right_active / right_valid) if right_valid > 0 else 0.0
    diff = left_ratio - right_ratio  # positivo -> izquierda

    # Hysteresis: si estado actual es centro usamos enter_threshold para activar,
    # si ya estamos en dirección usamos exit_threshold para volver a centro.
    if current_stable == "centro":
        if diff > enter_threshold:
            return "izquierda"
        elif diff < -enter_threshold:
            return "derecha"
        else:
            return "centro"
    else:
        if diff > exit_threshold:
            return "izquierda"
        elif diff < -exit_threshold:
            return "derecha"
        else:
            return "centro"

def process_frame():
    global eye_position, diff_buffer, update_time, mouth_was_open, mouth_open_start_time
    global start_time_click_right, click_performed, click_time, current_mode, submode1_active
    global last_candidate, candidate_count, stable_gaze

    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                print("DEBUG: no frame")
                break

            # zoom crop
            height, width, _ = frame.shape
            cx_img, cy_img = width // 2, height // 2
            try:
                frame = frame[int(cy_img - height // (2 * zoom_level)):int(cy_img + height // (2 * zoom_level)),
                              int(cx_img - width // (2 * zoom_level)):int(cx_img + width // (2 * zoom_level))]
                frame = cv2.resize(frame, (width, height))
            except Exception:
                pass

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if not results.multi_face_landmarks:
                cv2.imshow('Face Landmarks', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return
                continue

            for face_landmarks in results.multi_face_landmarks:
                # dibujar face mesh opcional
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

                # obtener coords ojo izquierdo outline
                left_eye_coords = []
                for idx in LEFT_EYE_OUTLINE:
                    lm = face_landmarks.landmark[idx]
                    left_eye_coords.append((int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])))

                xs = [p[0] for p in left_eye_coords]
                ys = [p[1] for p in left_eye_coords]
                if not xs or not ys:
                    continue
                min_x_eye = min(xs); max_x_eye = max(xs)
                min_y_eye = min(ys); max_y_eye = max(ys)

                center_x_left = sum(xs) // len(xs)
                center_y_left = sum(ys) // len(ys)
                center_x_left += center_offset_x
                center_y_left += center_offset_y

                pad = 2 + mask_margin
                x1 = max(0, min_x_eye - pad); x2 = min(frame.shape[1], max_x_eye + pad)
                y1 = max(0, min_y_eye - pad); y2 = min(frame.shape[0], max_y_eye + pad)
                roi_crop = frame[y1:y2, x1:x2]
                gaze = "centro"

                if roi_crop.size == 0:
                    continue

                gray_crop = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY) if roi_crop.ndim == 3 else roi_crop.copy()
                h, w = gray_crop.shape

                # máscara del ojo (poly -> fillPoly para robustez)
                poly_pts = []
                for (px, py) in left_eye_coords:
                    rx = px - x1
                    ry = py - y1
                    rx = int(np.clip(rx, 0, w-1))
                    ry = int(np.clip(ry, 0, h-1))
                    poly_pts.append([rx, ry])
                poly_np = np.array(poly_pts, dtype=np.int32)
                mask_eye = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask_eye, [poly_np], 255)

                # fallback círculo si máscara muy pequeña
                if cv2.countNonZero(mask_eye) < 10:
                    cx_rel = int((center_x_left - x1))
                    cy_rel = int((center_y_left - y1))
                    radius_rel = int(max((max_x_eye - min_x_eye), (max_y_eye - min_y_eye)) * 0.5) + mask_margin
                    radius_rel = max(1, min(radius_rel, min(w//2-1, h//2-1)))
                    mask_eye = np.zeros((h, w), dtype=np.uint8)
                    cv2.circle(mask_eye, (cx_rel, cy_rel), radius_rel, 255, -1)

                # binarizar (Otsu) y enmascarar
                blur = cv2.GaussianBlur(gray_crop, (5,5), 0)
                _, binary_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                # limpieza morfológica pequeña para reducir ruido lateral
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                binary_otsu = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, kernel)
                masked_binary = cv2.bitwise_and(binary_otsu, binary_otsu, mask=mask_eye)

                # construir grilla y calcular proporciones por celda
                sensor_matrix = np.zeros((grid_rows, grid_cols), dtype=np.float32)
                mask_counts = np.zeros((grid_rows, grid_cols), dtype=np.int32)
                for i in range(grid_rows):
                    y1c = int(round(i * h / grid_rows))
                    y2c = int(round((i+1) * h / grid_rows))
                    if y2c <= y1c: y2c = min(h, y1c + 1)
                    for j in range(grid_cols):
                        x1c = int(round(j * w / grid_cols))
                        x2c = int(round((j+1) * w / grid_cols))
                        if x2c <= x1c: x2c = min(w, x1c + 1)

                        cell_mask = mask_eye[y1c:y2c, x1c:x2c]
                        cell_bin = masked_binary[y1c:y2c, x1c:x2c]
                        mask_count = int(cv2.countNonZero(cell_mask))
                        mask_counts[i,j] = mask_count
                        if mask_count >= min_cell_pixels:
                            dark_count = int(cv2.countNonZero(cell_bin))
                            sensor_matrix[i,j] = dark_count / mask_count if mask_count > 0 else 0.0
                        else:
                            sensor_matrix[i,j] = 0.0

                valid_cells = mask_counts >= min_cell_pixels
                num_valid = int(np.count_nonzero(valid_cells))
                if num_valid < 4:
                    valid_cells = mask_counts > 0
                    num_valid = int(np.count_nonzero(valid_cells))

                # Si no hay celdas válidas, fallback por mitad
                if num_valid == 0:
                    mid = w // 2
                    left_mask = int(cv2.countNonZero(mask_eye[:, :mid]))
                    right_mask = int(cv2.countNonZero(mask_eye[:, mid:]))
                    left_dark = int(cv2.countNonZero(masked_binary[:, :mid]))
                    right_dark = int(cv2.countNonZero(masked_binary[:, mid:]))
                    left_ratio = (left_dark / left_mask) if left_mask > 0 else 0.0
                    right_ratio = (right_dark / right_mask) if right_mask > 0 else 0.0
                    diff = left_ratio - right_ratio
                    diff_buffer.append(diff)
                    if len(diff_buffer) > buffer_size: diff_buffer.pop(0)
                    avg_diff = sum(diff_buffer) / len(diff_buffer) if diff_buffer else 0.0

                    # aplicar thresholds (hysteresis) usando stable_gaze actual
                    gaze_candidate = None
                    if stable_gaze == "centro":
                        if avg_diff > enter_threshold:
                            gaze_candidate = "izquierda"
                        elif avg_diff < -enter_threshold:
                            gaze_candidate = "derecha"
                        else:
                            gaze_candidate = "centro"
                    else:
                        if avg_diff > exit_threshold:
                            gaze_candidate = "izquierda"
                        elif avg_diff < -exit_threshold:
                            gaze_candidate = "derecha"
                        else:
                            gaze_candidate = "centro"

                else:
                    mean_ratio = np.mean(sensor_matrix[valid_cells]) if np.any(valid_cells) else 0.0
                    delta_matrix = sensor_matrix - mean_ratio
                    active = (delta_matrix > activation_delta_thresh) | (sensor_matrix > activation_abs_thresh)
                    active = active & valid_cells

                    total_valid = int(np.count_nonzero(valid_cells))
                    total_active = int(np.count_nonzero(active))

                    # conteo/valid por regiones horizontales, manejando columna central si cols impares
                    mid_col = grid_cols // 2
                    if grid_cols % 2 == 1:
                        # ignorar columna central en la comparación L/R
                        left_valid = int(np.count_nonzero(valid_cells[:, :mid_col]))
                        right_valid = int(np.count_nonzero(valid_cells[:, mid_col+1:]))
                        left_active = int(np.count_nonzero(active[:, :mid_col]))
                        right_active = int(np.count_nonzero(active[:, mid_col+1:]))
                    else:
                        left_valid = int(np.count_nonzero(valid_cells[:, :mid_col]))
                        right_valid = int(np.count_nonzero(valid_cells[:, mid_col:]))
                        left_active = int(np.count_nonzero(active[:, :mid_col]))
                        right_active = int(np.count_nonzero(active[:, mid_col:]))

                    # si hay muy pocas activas, fallback a medias por región (pero normalizando por valid)
                    if total_active < min_active_cells:
                        left_vals = []
                        right_vals = []
                        for i in range(grid_rows):
                            for j in range(grid_cols):
                                if not valid_cells[i,j]: continue
                                if grid_cols % 2 == 1 and j == mid_col:
                                    continue  # ignorar columna central
                                if j < mid_col:
                                    left_vals.append(sensor_matrix[i,j])
                                else:
                                    # cuando cols impares y j>mid_col será derecha; cuando pares, mid_col.. es derecha
                                    if grid_cols % 2 == 1:
                                        if j > mid_col:
                                            right_vals.append(sensor_matrix[i,j])
                                    else:
                                        right_vals.append(sensor_matrix[i,j])
                        left_avg = np.mean(left_vals) if left_vals else 0.0
                        right_avg = np.mean(right_vals) if right_vals else 0.0
                        diff = left_avg - right_avg
                        diff_buffer.append(diff)
                        if len(diff_buffer) > buffer_size: diff_buffer.pop(0)
                        avg_diff = sum(diff_buffer)/len(diff_buffer) if diff_buffer else 0.0

                        # convertir a pseudo-activations usando proporciones de valid cells (si existen)
                        # para mantener coherencia con decide_gaze_lr_using_ratios
                        # tratar left_valid/right_valid como conteo de celdas válidas
                        gaze_candidate = decide_gaze_lr_using_ratios(
                            left_active = int(avg_diff > 0), left_valid = max(1, left_valid),
                            right_active = int(avg_diff < 0), right_valid = max(1, right_valid),
                            current_stable = stable_gaze
                        )
                    else:
                        # decisión directa con ratios normalizadas
                        gaze_candidate = decide_gaze_lr_using_ratios(
                            left_active = left_active, left_valid = max(1, left_valid),
                            right_active = right_active, right_valid = max(1, right_valid),
                            current_stable = stable_gaze
                        )
                        
                # --- Aplicar umbral mínimo por lado para validar gaze_candidate ---
                # Determinar métricas a usar para la comprobación:
                # preferir contar activas reales (left_active/right_active); si son pseudo (0/1) 
                # o no confiables, puedes usar left_valid/right_valid como fallback.
                try:
                    # si left_active/right_active existen y son enteros usarlos directamente
                    left_metric = int(left_active)
                    right_metric = int(right_active)
                except Exception:
                    left_metric = int(left_valid) if 'left_valid' in locals() else 0
                    right_metric = int(right_valid) if 'right_valid' in locals() else 0

                # Si la dirección candidata no cumple el mínimo de celdas activas del lado, cancelarla
                if gaze_candidate == "izquierda" and left_metric < left_min_active_cells:
                    gaze_candidate = "centro"
                elif gaze_candidate == "derecha" and right_metric < right_min_active_cells:
                    gaze_candidate = "centro"

                # debounce
                if gaze_candidate == last_candidate:
                    candidate_count += 1
                else:
                    last_candidate = gaze_candidate
                    candidate_count = 1
                if candidate_count >= debounce_frames:
                    stable_gaze = gaze_candidate
                    gaze = stable_gaze
                else:
                    gaze = stable_gaze  # mantener anterior hasta confirmar

                # debug overlay: dibujar grilla y activaciones (solo L/R info)
                if debug_overlay:
                    for i in range(grid_rows):
                        y_line = int(round(i * h / grid_rows))
                        cv2.line(roi_crop, (0,y_line), (w,y_line), (100,255,100), 1)
                    for j in range(grid_cols):
                        x_line = int(round(j * w / grid_cols))
                        cv2.line(roi_crop, (x_line,0), (x_line,h), (100,255,100), 1)
                    for i in range(grid_rows):
                        for j in range(grid_cols):
                            x1c = int(round(j * w / grid_cols))
                            x2c = int(round((j+1) * w / grid_cols))
                            y1c = int(round(i * h / grid_rows))
                            y2c = int(round((i+1) * h / grid_rows))
                            if valid_cells[i,j]:
                                # active puede no existir en fallback; proteger
                                is_act = False
                                try:
                                    is_act = bool(active[i,j])
                                except Exception:
                                    is_act = False
                                color = (0,200,0) if is_act else (0,0,200)
                                cv2.rectangle(roi_crop, (x1c,y1c), (x2c,y2c), color, 1)
                    # mostrar ratios si disponibles
                    try:
                        info_text = f"Gaze:{gaze} stable:{stable_gaze}"
                        cv2.putText(frame, info_text, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
                    except Exception:
                        pass

                # aplicar pines segun gaze y modo (solo L/R y submodo1 mapping)
                if current_mode == 1:
                    # reset pines relevantes
                    pin_arriba.off(); pin_abajo.off(); pin_izquierda.off(); pin_derecha.off()
                    if submode1_active:
                        # submodo 1.2: izquierda->arriba, derecha->abajo
                        if gaze == "izquierda":
                            pin_arriba.on(); pin_abajo.off()
                        elif gaze == "derecha":
                            pin_abajo.on(); pin_arriba.off()
                        else:
                            pin_arriba.off(); pin_abajo.off()
                    else:
                        if gaze == "izquierda":
                            pin_izquierda.on(); pin_derecha.off()
                        elif gaze == "derecha":
                            pin_derecha.on(); pin_izquierda.off()
                        else:
                            pin_izquierda.off(); pin_derecha.off()

                elif current_mode == 2:
                    if gaze == "izquierda":
                        pin_izquierda.on(); pin_derecha.off()
                    elif gaze == "derecha":
                        pin_derecha.on(); pin_izquierda.off()
                    else:
                        pin_izquierda.off(); pin_derecha.off()
                    if time.time() - update_time >= 0.2:
                        if gaze == "izquierda": pin_izquierda.on()
                        elif gaze == "derecha": pin_derecha.on()
                        else: pin_izquierda.off(); pin_derecha.off()
                        update_time = time.time()
                elif current_mode == 3:
                    if gaze == "izquierda":
                        pin_ScrollDwn.on(); pin_ScrollUp.off()
                    elif gaze == "derecha":
                        pin_ScrollUp.on(); pin_ScrollDwn.off()
                    else:
                        pin_ScrollUp.off(); pin_ScrollDwn.off()
                    if time.time() - update_time >= 0.2:
                        if gaze == "izquierda": pin_ScrollDwn.on()
                        elif gaze == "derecha": pin_ScrollUp.on()
                        else: pin_ScrollUp.off(); pin_ScrollDwn.off()
                        update_time = time.time()

                # Detección de boca (preview y asignación)
                top_lip = face_landmarks.landmark[13]
                bottom_lip = face_landmarks.landmark[14]
                mouth_open = bottom_lip.y - top_lip.y > mouth_open_threshold
                if mouth_open:
                    if not mouth_was_open:
                        mouth_was_open = True
                        mouth_open_start_time = time.time()
                    mouth_open_duration = time.time() - (mouth_open_start_time or time.time())
                    if mouth_open_duration >= preview_t3:
                        set_mode_leds(3)
                    elif mouth_open_duration >= preview_t2:
                        set_mode_leds(2)
                    elif mouth_open_duration >= preview_t1:
                        set_mode_leds(1)
                    else:
                        set_mode_leds(0)
                else:
                    if mouth_open_start_time is not None:
                        mouth_open_duration = time.time() - mouth_open_start_time
                        mouth_open_start_time = None
                        # apertura corta (< preview_t1) togglea submodo1
                        if mouth_open_duration < preview_t1:
                            submode1_active = not submode1_active
                            print("Submodo 1.2 activado" if submode1_active else "Submodo 1.2 desactivado")
                        elif preview_t1 <= mouth_open_duration < preview_t2:
                            current_mode = 1; pin_modo_2.off(); pin_modo_3.off(); print("Modo 1 activado")
                        elif preview_t2 <= mouth_open_duration < preview_t3:
                            current_mode = 2; pin_modo_2.on(); pin_modo_3.off(); print("Modo 2 activado")
                        elif mouth_open_duration >= preview_t3:
                            current_mode = 3; pin_modo_2.off(); pin_modo_3.on(); print("Modo 3 activado")
                        set_mode_leds(current_mode)
                    mouth_was_open = False

                # ojo derecho click
                right_eye_landmarks_top = [386, 374]
                right_eye_landmarks_bottom = [390, 382]
                right_eye_coordinates_top = [(int(face_landmarks.landmark[pt].x * frame.shape[1]),
                                              int(face_landmarks.landmark[pt].y * frame.shape[0]))
                                             for pt in right_eye_landmarks_top]
                right_eye_coordinates_bottom = [(int(face_landmarks.landmark[pt].x * frame.shape[1]),
                                                 int(face_landmarks.landmark[pt].y * frame.shape[0]))
                                                for pt in right_eye_landmarks_bottom]
                center_x_right_top = sum(c[0] for c in right_eye_coordinates_top) // len(right_eye_coordinates_top)
                min_y_right_top = min(c[1] for c in right_eye_coordinates_top)
                center_x_right_bottom = sum(c[0] for c in right_eye_coordinates_bottom) // len(right_eye_coordinates_bottom)
                max_y_right_bottom = max(c[1] for c in right_eye_coordinates_bottom)
                cv2.circle(frame, (center_x_right_top, min_y_right_top), 6, (0,0,255), -1)
                cv2.circle(frame, (center_x_right_bottom, max_y_right_bottom), 6, (0,0,255), -1)
                distance = math.sqrt((center_x_right_bottom - center_x_right_top)*2 + (max_y_right_bottom - min_y_right_top)*2)
                if distance <= click_threshold_px:
                    if start_time_click_right is None:
                        start_time_click_right = time.time()
                    else:
                        elapsed_time_click = time.time() - start_time_click_right
                        if elapsed_time_click >= click_hold_time and not click_performed:
                            print("Click Izquierdo")
                            pin_clickI.on()
                            click_time = time.time()
                            click_performed = True
                else:
                    start_time_click_right = None
                if click_time is not None and (time.time() - click_time) >= click_pulse_time:
                    pin_clickI.off(); click_time = None
                if distance > click_threshold_px and click_performed:
                    click_performed = False

                # mostrar y salida
                cv2.imshow('Face Landmarks', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return

        except Exception as e:
            print("ERROR in process_frame:", e)
            traceback.print_exc()
            break

    print("DEBUG: process_frame exiting.")

def gpio_cleanup():
    # cerrar todos los pines usados, con protección
    for p in [pin_arriba, pin_abajo, pin_izquierda, pin_derecha,
              pin_modo_2, pin_modo_3, pin_clickI,
              pin_led_azul, pin_led_verde, pin_led_rojo,
              pin_ScrollUp, pin_ScrollDwn]:
        try:
            p.close()
        except Exception:
            pass

if _name_ == '_main_':
    set_mode_leds(current_mode)
    try:
        frame_thread = threading.Thread(target=process_frame)
        frame_thread.start()
        frame_thread.join()
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        try: cap.release()
        except: pass
        try: cv2.destroyAllWindows()
        except: pass
        # cerrar media-pipe
        try:
            face_mesh.close()
        except Exception:
            pass
        gpio_cleanup()
        print("Exiting cleanly.")