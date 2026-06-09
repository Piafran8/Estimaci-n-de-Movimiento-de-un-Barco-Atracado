import cv2
import numpy as np
import pandas as pd
import glob
import os

try:
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# --- CONFIGURACIÓN ---
carpeta_principal = "DATASET"

carpetas_entrada = [
    #"Dataset_Fase1_Jonswap_12pt", 
    #"Dataset_Fase2_Jonswap_12pt",
    #"Dataset_Fase3_Jonswap_12pt",
    #"Dataset_Fase4_sinusoidal_12pt"
     # --- FASE 1 ---
    f"{carpeta_principal}/Dataset_Fase1_Tp6",
    f"{carpeta_principal}/Dataset_Fase1_Tp8",
    f"{carpeta_principal}/Dataset_Fase1_Tp10",
    f"{carpeta_principal}/Dataset_Fase1_Tp12",
    # --- FASE 2 ---
    f"{carpeta_principal}/Dataset_Fase2_Tp6",
    f"{carpeta_principal}/Dataset_Fase2_Tp8",
    f"{carpeta_principal}/Dataset_Fase2_Tp10",
    f"{carpeta_principal}/Dataset_Fase2_Tp12",
    # --- FASE 3 ---
    *[f"{carpeta_principal}/Dataset_Fase3_Hs{hs}_Tp{tp}_H{h}" 
      for hs in [0.5, 1.5, 3.0, 5.0] 
      for tp in [6, 8, 10, 12, 14] 
      for h  in [0, 90, 135, 180]],
    # --- FASE 4 ---
  f"{carpeta_principal}/Dataset_Fase4_Tp7",
    f"{carpeta_principal}/Dataset_Fase4_Tp9",
    f"{carpeta_principal}/Dataset_Fase4_Tp11",
    f"{carpeta_principal}/Dataset_Fase4_Tp13"
]
carpeta_resultados = "Tracking_Features_ML_Color"
os.makedirs(carpeta_resultados, exist_ok=True)

def ordenar_12_puntos_frame0(puntos):
    """ Ordena de izquierda a derecha, arriba a abajo SOLO para el primer frame """
    puntos_planos = [np.array(p).ravel() for p in puntos]
    puntos_ordenados_y = sorted(puntos_planos, key=lambda p: p[1])
    filas = []
    fila_actual = [puntos_ordenados_y[0]]
    
    for p in puntos_ordenados_y[1:]:
        if abs(p[1] - fila_actual[-1][1]) < 25:
            fila_actual.append(p)
        else:
            filas.append(sorted(fila_actual, key=lambda px: px[0]))
            fila_actual = [p]
    filas.append(sorted(fila_actual, key=lambda px: px[0]))
    
    puntos_finales = []
    for f in filas: puntos_finales.extend(f)
    return np.array(puntos_finales, dtype=np.float32)

def emparejar_puntos(viejos, nuevos):
   
    viejos = np.array(viejos)
    nuevos = np.array(nuevos)
    
    if HAS_SCIPY:
        # Método 1
        dist_matrix = cdist(viejos, nuevos)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        return nuevos[col_ind]
    else:
        # Método alternativo 
        nuevos_ordenados = np.zeros_like(viejos)
        nuevos_list = list(nuevos)
        for i, p_viejo in enumerate(viejos):
            distancias = [np.linalg.norm(p_viejo - p_nuevo) for p_nuevo in nuevos_list]
            mejor_idx = np.argmin(distancias)
            nuevos_ordenados[i] = nuevos_list.pop(mejor_idx)
        return nuevos_ordenados

for carpeta in carpetas_entrada:
    if not os.path.exists(carpeta): continue
    lista_videos = glob.glob(os.path.join(carpeta, "*.mp4"))
    
    for video_path in lista_videos:
        nombre_base = os.path.basename(video_path).replace('.mp4', '')
        csv_salida = os.path.join(carpeta_resultados, f"feat_{nombre_base}.csv")

        if os.path.exists(csv_salida):
            print(f" Saltando {nombre_base}")
            continue

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dt = 1.0 / fps
        
        raw_data = []
        frame_idx = 0
        puntos_viejos = None
        video_valido = True  
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # busca manchas por coloor (rojas) en el frame actual
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])) + \
                   cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
            
            contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            puntos_y_areas = []
            for c in contornos:
                M = cv2.moments(c)
                if M["m00"] > 0: 
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    area = M["m00"]
                    puntos_y_areas.append((cx, cy, area))
            
            puntos_y_areas.sort(key=lambda x: x[2], reverse=True)
            pts_frame = [(p[0], p[1]) for p in puntos_y_areas[:12]]
            
            # ALTO oclusion
            if len(pts_frame) != 12:
                print(f"⚠️ Oclusión detectada en frame {frame_idx}. Descartando video: {nombre_base}")
                video_valido = False
                break
            
            
            if frame_idx == 0:
                
                puntos_ordenados = ordenar_12_puntos_frame0(pts_frame)
            else:
                
                puntos_ordenados = emparejar_puntos(puntos_viejos, pts_frame)
            
            puntos_viejos = puntos_ordenados.copy()
            
            fila = {'tiempo': frame_idx * dt}
            for i, pt in enumerate(puntos_ordenados):
                x, y = pt.ravel()
                fila[f'p{i+1}_x'] = x
                fila[f'p{i+1}_y'] = y

                # dibujo para comprobación y seguimiento
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
                cv2.putText(frame, str(i+1), (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            cv2.imshow('Tracking por Proximidad', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                video_valido = False
                break
            
            raw_data.append(fila)
            frame_idx += 1
            
        # --- CÁLCULO DE DINÁMICA ---
        
        if video_valido and len(raw_data) > 0:
            df = pd.DataFrame(raw_data)
            for i in range(1, 13):
                for eje in ['x', 'y']:
                    col = f'p{i}_{eje}'
                    df[f'v_{col}'] = df[col].diff() / dt
                    df[f'a_{col}'] = df[f'v_{col}'].diff() / dt
            
            df.dropna().to_csv(csv_salida, index=False)
            print(f"OK, Todo bien, Procesado: {nombre_base}")

        cap.release()

cv2.destroyAllWindows()
