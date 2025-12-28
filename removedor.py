import cv2
import numpy as np
import os
import FreeSimpleGUI as sg
from concurrent.futures import ProcessPoolExecutor
from moviepy.editor import VideoFileClip
import multiprocessing
import time

# --- CONFIGURAÇÕES ---
NUM_PROCESSOS = max(1, os.cpu_count() - 1)
SCALE_FACTOR = 0.5 

# Tamanho do lote de quadros para analisar o futuro (8 frames = ~0.26s em 30fps)
BATCH_SIZE = 8

# --- ÁREA DE APAGAMENTO ---
FATOR_APAGAR_ABAIXO = 0.60    
FATOR_APAGAR_ESQUERDA = 0.15  
FATOR_APAGAR_DIREITA = 0.02   
FATOR_APAGAR_CIMA = 0.02      

def contar_frames_totais(pasta_entrada, stop_event):
    arquivos = [f for f in os.listdir(pasta_entrada) if f.endswith(('.mp4', '.avi', '.mov'))]
    total_frames = 0; validos = []
    for arq in arquivos:
        if stop_event.is_set(): return 0, [] 
        cap = cv2.VideoCapture(os.path.join(pasta_entrada, arq))
        if cap.isOpened():
            total_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            validos.append(arq)
            cap.release()
    return total_frames, validos

def validar_geometria_inteligente(w_detectado, h_detectado, w_ref, h_ref, largura_video_small):
    if w_detectado > (largura_video_small * 0.35): return False
    ratio_ref = w_ref / h_ref; ratio_det = w_detectado / h_detectado
    if abs(ratio_det - ratio_ref) / ratio_ref > 0.4: return False
    return True

def detect_single_frame(frame, sift, flann, kp_logo, des_logo, h_ref, w_ref, h_small, w_small):
    """Função auxiliar para detectar logo em um frame específico"""
    frame_small = cv2.resize(frame, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    frame_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = sift.detectAndCompute(frame_gray, None)

    if des_frame is None or len(kp_frame) < 10:
        return None

    matches = flann.knnMatch(des_logo, des_frame, k=2)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]

    if len(good) <= 10:
        return None

    src_pts = np.float32([kp_logo[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
    
    if M is None:
        return None

    pts_ref = np.float32([[0, 0], [0, h_ref], [w_ref, h_ref], [w_ref, 0]]).reshape(-1, 1, 2)
    dst_pts_box = cv2.transform(pts_ref, M)
    x, y, w, h = cv2.boundingRect(dst_pts_box)
    
    if not validar_geometria_inteligente(w, h, w_ref, h_ref, w_small):
        return None

    # Calcula coordenadas finais da máscara
    x_new = int(x - (w * FATOR_APAGAR_ESQUERDA))
    y_new = int(y - (h * FATOR_APAGAR_CIMA))
    w_new = int(w + (w * FATOR_APAGAR_ESQUERDA) + (w * FATOR_APAGAR_DIREITA))
    h_new = int(h + (h * FATOR_APAGAR_ABAIXO))
    
    x_real = int(x_new / SCALE_FACTOR)
    y_real = int(y_new / SCALE_FACTOR)
    w_real = int(w_new / SCALE_FACTOR)
    h_real = int(h_new / SCALE_FACTOR)

    return (x_real, y_real, w_real, h_real)

def worker_process_video(args):
    nome_arquivo, pasta_entrada, pasta_saida, logo_path, queue_progresso, stop_event, modo_debug = args
    if stop_event.is_set(): return

    caminho_entrada = os.path.join(pasta_entrada, nome_arquivo)
    caminho_final = os.path.join(pasta_saida, nome_arquivo)
    caminho_temp = os.path.join(pasta_saida, f"temp_{nome_arquivo}")

    try:
        if not logo_path or not os.path.exists(logo_path):
            queue_progresso.put(('erro', "Logo não encontrada!")); return

        # Setup Logo e SIFT
        img_logo = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)
        altura_orig, largura_orig = img_logo.shape
        fator_norm = 200.0 / largura_orig
        img_logo_norm = cv2.resize(img_logo, None, fx=fator_norm, fy=fator_norm)
        h_ref, w_ref = img_logo_norm.shape

        sift = cv2.SIFT_create(contrastThreshold=0.03)
        kp_logo, des_logo = sift.detectAndCompute(img_logo_norm, None)
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

        cap = cv2.VideoCapture(caminho_entrada)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(caminho_temp, fourcc, fps, (width, height))

        # Variáveis de Estado
        last_valid_detection = None # (box, 'TOPO'/'BAIXO')
        frames_since_last = 0
        LIMIT_PERSISTENCIA = 120

        while True:
            if stop_event.is_set(): cap.release(); out.release(); return
            
            # 1. LER O LOTE (BATCH) DO FUTURO
            frames_buffer = []
            for _ in range(BATCH_SIZE):
                ret, frame = cap.read()
                if not ret: break
                frames_buffer.append(frame)
            
            if not frames_buffer: break # Fim do vídeo

            # 2. DETECTAR EM TODO O LOTE
            detections_buffer = [] # Vai guardar (box, zona) ou None
            mid_y = int((height * SCALE_FACTOR) / 2)

            for frame in frames_buffer:
                h_small = int(height * SCALE_FACTOR)
                w_small = int(width * SCALE_FACTOR)
                
                res = detect_single_frame(frame, sift, flann, kp_logo, des_logo, h_ref, w_ref, h_small, w_small)
                
                if res:
                    x, y, w, h = res
                    # Determina zona
                    cy = (y * SCALE_FACTOR) + ((h * SCALE_FACTOR) // 2)
                    zona = 'TOPO' if cy < mid_y else 'BAIXO'
                    detections_buffer.append((res, zona))
                else:
                    detections_buffer.append(None)

            # 3. HEURÍSTICA DE PREENCHIMENTO (TIME TRAVEL)
            processed_buffer = []
            
            for i in range(len(frames_buffer)):
                current_det = detections_buffer[i]
                final_boxes_to_apply = [] # Lista de caixas para desenhar neste frame
                status_color = (0, 255, 0) # Verde por padrão

                if current_det:
                    # Detecção sólida no presente
                    final_boxes_to_apply.append(current_det[0])
                    last_valid_detection = current_det
                    frames_since_last = 0
                else:
                    # Falha de detecção: Precisamos preencher o buraco
                    
                    # Olhar para o Futuro (Lookahead)
                    future_det = None
                    for j in range(i + 1, len(frames_buffer)):
                        if detections_buffer[j]:
                            future_det = detections_buffer[j]
                            break
                    
                    # Decisão Lógica
                    if last_valid_detection and future_det:
                        # Estamos num buraco entre duas detecções
                        prev_box, prev_zona = last_valid_detection
                        next_box, next_zona = future_det
                        
                        if prev_zona == next_zona:
                            # Mesma zona: Mantém a anterior (estabilidade)
                            final_boxes_to_apply.append(prev_box)
                        else:
                            # PULO DETECTADO! (Zona mudou)
                            # Aplica AMBAS para garantir cobertura total na transição
                            final_boxes_to_apply.append(prev_box)
                            final_boxes_to_apply.append(next_box)
                            status_color = (0, 255, 255) # Amarelo (Dual Mask)
                            
                    elif future_det:
                        # Sem passado, mas com futuro (Início do vídeo ou logo aparecendo)
                        # Aplica o futuro retroativamente!
                        final_boxes_to_apply.append(future_det[0])
                        status_color = (255, 0, 255) # Roxo (Backfill)
                        
                    elif last_valid_detection and frames_since_last < LIMIT_PERSISTENCIA:
                        # Sem futuro, usa memória
                        final_boxes_to_apply.append(last_valid_detection[0])
                        frames_since_last += 1
                        status_color = (0, 0, 255) # Vermelho (Memória)

                # 4. APLICA AS MÁSCARAS ACUMULADAS
                frame_out = frames_buffer[i]
                
                if modo_debug:
                    for box in final_boxes_to_apply:
                        bx, by, bw, bh = box
                        cv2.rectangle(frame_out, (bx, by), (bx+bw, by+bh), status_color, 2)
                        cv2.putText(frame_out, "X", (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                else:
                    if final_boxes_to_apply:
                        mask_accum = np.zeros(frame_out.shape[:2], dtype=np.uint8)
                        for box in final_boxes_to_apply:
                            bx, by, bw, bh = box
                            cv2.rectangle(mask_accum, (bx, by), (bx+bw, by+bh), 255, -1)
                        
                        mask_accum = cv2.dilate(mask_accum, None, iterations=2)
                        frame_out = cv2.inpaint(frame_out, mask_accum, 3, cv2.INPAINT_TELEA)

                processed_buffer.append(frame_out)

            # 5. ESCREVE O LOTE
            for p_frame in processed_buffer:
                out.write(p_frame)
                queue_progresso.put(('progresso', 1))

        cap.release(); out.release()

        if stop_event.is_set(): return
        queue_progresso.put(('status_fase', f"Salvando: {nome_arquivo}..."))
        
        try:
            clip_orig = VideoFileClip(caminho_entrada)
            clip_proc = VideoFileClip(caminho_temp)
            final = clip_proc.set_audio(clip_orig.audio) if clip_orig.audio else clip_proc
            if stop_event.is_set(): clip_orig.close(); clip_proc.close(); return
            
            final.write_videofile(caminho_final, codec='libx264', audio_codec='aac', logger=None, threads=1, preset='superfast', ffmpeg_params=['-crf', '18'])
            clip_orig.close(); clip_proc.close(); final.close()
            if os.path.exists(caminho_temp): os.remove(caminho_temp)
            if not stop_event.is_set(): queue_progresso.put(('concluido', nome_arquivo))
        except Exception as e:
            if not stop_event.is_set(): queue_progresso.put(('erro', f"Erro ffmpeg: {str(e)}"))
    except Exception as e:
        if not stop_event.is_set(): queue_progresso.put(('erro', f"Erro Fatal: {str(e)}"))

def main_gui():
    sg.theme('GrayGrayGray')
    sg.set_options(background_color='#1c1c1c', text_element_background_color='#1c1c1c', element_background_color='#1c1c1c', input_elements_background_color='#333333', input_text_color='white', text_color='white')
    
    layout_inputs = [
        [sg.Text('Pasta Origem:', size=(12,1), justification='right'), sg.Input(key='-IN-', default_text='videos_originais', size=(40,1)), sg.FolderBrowse('...')],
        [sg.Text('Pasta Destino:', size=(12,1), justification='right'), sg.Input(key='-OUT-', default_text='videos_prontos', size=(40,1)), sg.FolderBrowse('...')],
        [sg.Text('Arquivo Logo:', size=(12,1), justification='right'), sg.Input(key='-LOGO-', default_text='logo.png', size=(40,1)), sg.FileBrowse('...')],
        [sg.Text('', size=(12,1)), sg.Checkbox('Modo Debug (Ver caixas de previsão)', key='-DEBUG-', text_color='#00FF00', default=True)]
    ]

    layout = [
        [sg.Push(), sg.Text('Removedor v23 (Buffer do Futuro)', font=('Segoe UI', 18, 'bold'), pad=((0,0),(20,20))), sg.Push()],
        [sg.Push(), sg.Frame('Configurações', layout_inputs, font=('Segoe UI', 10, 'bold'), pad=((0,0),(0,20)), element_justification='center', title_color='white'), sg.Push()],
        [sg.Push(), sg.Text('Aguardando Início', key='-STATUS-', font=('Segoe UI', 11, 'italic'), text_color='#FFD700'), sg.Push()],
        [sg.Push(), sg.ProgressBar(1000, orientation='h', size=(60, 20), key='-PROG-', bar_color=('#4CAF50', '#333333')), sg.Push()],
        [sg.Push(), sg.Text('0%', key='-PERCENT-', font=('Segoe UI', 12, 'bold')), sg.Push()],
        [sg.Push(), sg.Multiline(size=(80, 8), key='-LOG-', autoscroll=True, disabled=True, font=('Consolas', 10), background_color='#111111', text_color='#eeeeee', border_width=0), sg.Push()],
        [sg.Button('INICIAR', key='INICIAR', size=(15, 2), button_color=('white', '#2E7D32'), font=('Segoe UI', 10, 'bold'), pad=((20, 5), (20, 20))),
         sg.Button('CANCELAR', key='CANCELAR', size=(15, 2), button_color=('white', '#C62828'), font=('Segoe UI', 10, 'bold'), disabled=True, pad=((5, 5), (20, 20))),
         sg.Push(), 
         sg.Button('SAIR', size=(10, 2), button_color=('white', '#444444'), pad=((5, 20), (20, 20)))]
    ]

    window = sg.Window('Removedor v23', layout, finalize=True, element_justification='c')
    manager = multiprocessing.Manager(); queue_progresso = manager.Queue(); stop_event = manager.Event()
    executor = None; futures = []; processing = False
    total_frames_global = 0; frames_processados_atual = 0
    
    while True:
        event, values = window.read(timeout=50)
        if event in (sg.WIN_CLOSED, 'SAIR'):
            if processing: stop_event.set()
            break

        if event == 'INICIAR':
            pasta_in = values['-IN-']; pasta_out = values['-OUT-']; logo_file = values['-LOGO-']; debug_mode = values['-DEBUG-']
            if not os.path.exists(pasta_in) or not os.path.exists(logo_file): sg.popup_error("Arquivos não encontrados!"); continue
            if not os.path.exists(pasta_out): os.makedirs(pasta_out)

            window['-STATUS-'].update("Analisando Futuro...", text_color='#FFD700')
            window['INICIAR'].update(disabled=True); window['CANCELAR'].update(disabled=False)
            window['-LOG-'].update(""); window.refresh(); stop_event.clear()
            
            total_frames_global, lista_arquivos = contar_frames_totais(pasta_in, stop_event)
            if stop_event.is_set() or len(lista_arquivos) == 0:
                window['INICIAR'].update(disabled=False); window['CANCELAR'].update(disabled=True); continue

            window['-LOG-'].print(f"Vídeos: {len(lista_arquivos)} | Frames: {total_frames_global}")
            if debug_mode: window['-LOG-'].print("⚠ DEBUG: Roxo=Futuro, Amarelo=Transição Dupla", text_color='#00FF00')

            frames_processados_atual = 0; processing = True
            executor = ProcessPoolExecutor(max_workers=NUM_PROCESSOS)
            futures = []
            for arq in lista_arquivos:
                futures.append(executor.submit(worker_process_video, (arq, pasta_in, pasta_out, logo_file, queue_progresso, stop_event, debug_mode)))

        if event == 'CANCELAR' and processing:
            if sg.popup_yes_no("Parar?") == 'Yes': stop_event.set()

        if processing:
            while not queue_progresso.empty():
                try:
                    tipo, valor = queue_progresso.get_nowait()
                    if tipo == 'progresso': frames_processados_atual += valor
                    elif tipo == 'status_fase': window['-STATUS-'].update(valor, text_color='#FFA000') 
                    elif tipo == 'concluido': window['-LOG-'].print(f"✔ Pronto: {valor}", text_color='#4CAF50')
                    elif tipo == 'erro': window['-LOG-'].print(f"✘ Erro: {valor}", text_color='#C62828')
                except: break
            
            if total_frames_global > 0: ratio = frames_processados_atual / total_frames_global
            else: ratio = 0
            window['-PROG-'].update(int(ratio * 1000)); window['-PERCENT-'].update(f"{ratio * 100:.1f}%")

            if all(f.done() for f in futures):
                msg = "CANCELADO" if stop_event.is_set() else "CONCLUÍDO"
                window['-STATUS-'].update(msg, text_color='#C62828' if stop_event.is_set() else '#4CAF50')
                if not stop_event.is_set():
                    window['-PROG-'].update(1000); window['-PERCENT-'].update("100%"); sg.popup("Fim!", title="Sucesso")
                processing = False; executor.shutdown(wait=False); executor = None
                window['INICIAR'].update(disabled=False); window['CANCELAR'].update(disabled=True)
    window.close()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main_gui()