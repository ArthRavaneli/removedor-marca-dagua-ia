import cv2
import numpy as np
import os
import FreeSimpleGUI as sg
from concurrent.futures import ProcessPoolExecutor
from moviepy.editor import VideoFileClip
import multiprocessing
import time

# --- CONFIGURAÇÕES TURBO & ÁREA ---
MIN_MATCH_COUNT = 10 
NUM_PROCESSOS = os.cpu_count()

# FATOR DE ACELERAÇÃO (Busca em meia resolução)
SCALE_FACTOR = 0.5 

# EXPANSÃO DA ÁREA DE BORRÃO
# Quanto expandir para BAIXO (para cobrir o @usuario)
# 0.9 = 90% da altura da logo para baixo
FATOR_APAGAR_ABAIXO = 0.9 

# NOVO: Quanto expandir para a ESQUERDA (para cobrir o início do @)
# 0.2 = 20% da largura da logo para a esquerda
FATOR_APAGAR_ESQUERDA = 0.1

def contar_frames_totais(pasta_entrada, stop_event):
    arquivos = [f for f in os.listdir(pasta_entrada) if f.endswith(('.mp4', '.avi', '.mov'))]
    total_frames = 0
    validos = []
    for arq in arquivos:
        if stop_event.is_set(): return 0, [] 
        cap = cv2.VideoCapture(os.path.join(pasta_entrada, arq))
        if cap.isOpened():
            total_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            validos.append(arq)
            cap.release()
    return total_frames, validos

def worker_process_video(args):
    nome_arquivo, pasta_entrada, pasta_saida, logo_full_path, queue_progresso, stop_event = args
    
    if stop_event.is_set(): return

    caminho_entrada = os.path.join(pasta_entrada, nome_arquivo)
    caminho_final = os.path.join(pasta_saida, nome_arquivo)
    caminho_temp = os.path.join(pasta_saida, f"temp_{nome_arquivo}")

    try:
        # --- 1. PREPARAÇÃO VISUAL ---
        img_logo = cv2.imread(logo_full_path, cv2.IMREAD_GRAYSCALE)
        if img_logo is None: 
            queue_progresso.put(('erro', f"Logo não encontrada: {nome_arquivo}"))
            return

        # Redimensiona a logo de referência
        img_logo_small = cv2.resize(img_logo, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)

        sift = cv2.SIFT_create()
        kp_logo, des_logo = sift.detectAndCompute(img_logo_small, None)
        
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        cap = cv2.VideoCapture(caminho_entrada)
        if not cap.isOpened(): 
            queue_progresso.put(('erro', f"Erro ao abrir: {nome_arquivo}"))
            return

        largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(caminho_temp, fourcc, fps, (largura, altura))

        ultimo_mask = None
        frames_lost = 0
        LIMIT_PERSISTENCIA = 120
        frame_count_local = 0

        while True:
            if stop_event.is_set():
                cap.release()
                out.release()
                if os.path.exists(caminho_temp): 
                    try: os.remove(caminho_temp)
                    except: pass
                return

            ret, frame = cap.read()
            if not ret: break
            
            frame_count_local += 1
            if frame_count_local % 10 == 0:
                queue_progresso.put(('progresso', 10))

            # --- VISÃO COMPUTACIONAL TURBO ---
            frame_small = cv2.resize(frame, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            frame_gray_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            kp_frame, des_frame = sift.detectAndCompute(frame_gray_small, None)

            found = False
            if des_frame is not None and len(kp_frame) >= MIN_MATCH_COUNT:
                matches = flann.knnMatch(des_logo, des_frame, k=2)
                good = [m for m, n in matches if m.distance < 0.7 * n.distance]

                if len(good) > MIN_MATCH_COUNT:
                    src_pts = np.float32([kp_logo[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if M is not None:
                        found = True
                        frames_lost = 0
                        
                        # --- CÁLCULO DA MÁSCARA ESTENDIDA (ESQUERDA + BAIXO) ---
                        h_small, w_small = img_logo_small.shape
                        
                        # Calcula margem esquerda em pixels
                        margem_esq = w_small * FATOR_APAGAR_ESQUERDA
                        
                        # Define a caixa usando coordenadas negativas para a esquerda
                        pts_small = np.float32([
                            [-margem_esq, 0], # Topo-Esquerda expandido
                            [-margem_esq, h_small - 1 + (h_small * FATOR_APAGAR_ABAIXO)], # Baixo-Esquerda expandido
                            [w_small - 1, h_small - 1 + (h_small * FATOR_APAGAR_ABAIXO)],
                            [w_small - 1, 0]
                        ]).reshape(-1, 1, 2)
                        
                        dst_small = cv2.perspectiveTransform(pts_small, M)
                        
                        # Amplia para o tamanho real
                        dst_big = dst_small * (1.0 / SCALE_FACTOR)

                        mask_overlay = np.zeros(frame.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(mask_overlay, [np.int32(dst_big)], 255)
                        # Dilatação extra para garantir bordas suaves
                        mask_overlay = cv2.dilate(mask_overlay, None, iterations=4)
                        ultimo_mask = mask_overlay
                        
                        frame = cv2.inpaint(frame, mask_overlay, 3, cv2.INPAINT_TELEA)

            if not found and ultimo_mask is not None and frames_lost < LIMIT_PERSISTENCIA:
                frame = cv2.inpaint(frame, ultimo_mask, 3, cv2.INPAINT_TELEA)
                frames_lost += 1
            
            out.write(frame)

        sobra = frame_count_local % 10
        if sobra > 0: queue_progresso.put(('progresso', sobra))

        cap.release()
        out.release()

        # --- PÓS-PROCESSAMENTO ---
        if stop_event.is_set(): return

        try:
            queue_progresso.put(('status_fase', f"Finalizando compressão: {nome_arquivo}..."))
            
            clip_original = VideoFileClip(caminho_entrada)
            clip_processado = VideoFileClip(caminho_temp)
            
            if clip_original.audio:
                video_final = clip_processado.set_audio(clip_original.audio)
            else:
                video_final = clip_processado
            
            if stop_event.is_set(): 
                clip_original.close(); clip_processado.close()
                return

            video_final.write_videofile(
                caminho_final, 
                codec='libx264', 
                audio_codec='aac', 
                logger=None, 
                threads=1, 
                preset='superfast', # Velocidade máxima de escrita
                ffmpeg_params=['-crf', '18'] # Qualidade visual máxima
            )
            
            clip_original.close()
            clip_processado.close()
            video_final.close()
            if os.path.exists(caminho_temp): os.remove(caminho_temp)
            
            if not stop_event.is_set():
                queue_progresso.put(('concluido', nome_arquivo))
            
        except Exception as e:
            if not stop_event.is_set(): queue_progresso.put(('erro', f"Erro export: {str(e)}"))

    except Exception as e:
        if not stop_event.is_set(): queue_progresso.put(('erro', f"Fatal: {str(e)}"))

def main_gui():
    sg.theme('GrayGrayGray')
    sg.set_options(background_color='#1c1c1c', 
                   text_element_background_color='#1c1c1c', 
                   element_background_color='#1c1c1c',
                   input_elements_background_color='#333333',
                   input_text_color='white',
                   text_color='white')
    
    fonte_titulo = ('Segoe UI', 18, 'bold')
    fonte_padrao = ('Segoe UI', 11)
    fonte_console = ('Consolas', 10)

    # Layout de Entradas
    layout_inputs = [
        [sg.Text('Pasta Origem:', size=(12,1), justification='right'), 
         sg.Input(key='-IN-', default_text='videos_originais', size=(40,1)), sg.FolderBrowse('...')],
        
        [sg.Text('Pasta Destino:', size=(12,1), justification='right'), 
         sg.Input(key='-OUT-', default_text='videos_prontos', size=(40,1)), sg.FolderBrowse('...')],
        
        [sg.Text('Arquivo Logo:', size=(12,1), justification='right'), 
         sg.Input(key='-LOGO-', default_text='logo.png', size=(40,1)), sg.FileBrowse('...')]
    ]

    layout = [
        # Título
        [sg.Push(), sg.Text('Removedor Automático de Marcas (TURBO)', font=fonte_titulo, pad=((0,0),(20,20))), sg.Push()],
        
        # Frame de Configurações
        [sg.Push(), sg.Frame('Configurações', layout_inputs, 
                             font=('Segoe UI', 10, 'bold'), 
                             pad=((0,0),(0,20)), 
                             element_justification='center',
                             title_color='white'), sg.Push()],
        
        # Status
        [sg.Push(), sg.Text('Aguardando Início', key='-STATUS-', font=('Segoe UI', 11, 'italic'), text_color='#FFD700'), sg.Push()],
        
        # Barra de Progresso
        [sg.Push(), sg.ProgressBar(1000, orientation='h', size=(60, 20), key='-PROG-', bar_color=('#4CAF50', '#333333')), sg.Push()],
        
        # Porcentagem
        [sg.Push(), sg.Text('0%', key='-PERCENT-', font=('Segoe UI', 12, 'bold')), sg.Push()],
        
        # Log
        [sg.Push(), sg.Multiline(size=(80, 8), key='-LOG-', autoscroll=True, disabled=True, font=fonte_console, 
                      background_color='#111111', text_color='#eeeeee', border_width=0), sg.Push()],
        
        # Botões
        [sg.Button('INICIAR', key='INICIAR', size=(15, 2), button_color=('white', '#2E7D32'), font=('Segoe UI', 10, 'bold'), pad=((20, 5), (20, 20))),
         sg.Button('CANCELAR', key='CANCELAR', size=(15, 2), button_color=('white', '#C62828'), font=('Segoe UI', 10, 'bold'), disabled=True, pad=((5, 5), (20, 20))),
         sg.Push(), 
         sg.Button('SAIR', size=(10, 2), button_color=('white', '#444444'), pad=((5, 20), (20, 20)))]
    ]

    window = sg.Window('Removedor Automático', layout, finalize=True, element_justification='c')
    
    manager = multiprocessing.Manager()
    queue_progresso = manager.Queue()
    stop_event = manager.Event()
    executor = None
    
    futures = []
    processing = False
    total_frames_global = 0
    frames_processados_atual = 0
    
    while True:
        event, values = window.read(timeout=50)

        if event in (sg.WIN_CLOSED, 'SAIR'):
            if processing:
                if sg.popup_yes_no("O processamento está rodando. Deseja sair?") == 'Yes':
                    stop_event.set()
                    if executor: executor.shutdown(wait=False)
                    break
            else:
                break

        if event == 'INICIAR':
            pasta_in = values['-IN-']
            pasta_out = values['-OUT-']
            logo_file = values['-LOGO-']

            if not os.path.exists(pasta_in) or not os.path.exists(logo_file):
                sg.popup_error("Arquivos não encontrados!")
                continue
            if not os.path.exists(pasta_out): os.makedirs(pasta_out)

            window['-STATUS-'].update("Analisando vídeos...", text_color='#FFD700')
            window['INICIAR'].update(disabled=True)
            window['CANCELAR'].update(disabled=False)
            window['-LOG-'].update("")
            window.refresh()
            
            stop_event.clear()
            total_frames_global, lista_arquivos = contar_frames_totais(pasta_in, stop_event)
            
            if stop_event.is_set() or len(lista_arquivos) == 0:
                window['INICIAR'].update(disabled=False)
                window['CANCELAR'].update(disabled=True)
                window['-STATUS-'].update("Aguardando...", text_color='#FFD700')
                continue

            window['-LOG-'].print(f"Vídeos: {len(lista_arquivos)} | Frames Totais: {total_frames_global}\n")
            window['-STATUS-'].update(f"Processando com {NUM_PROCESSOS} núcleos (Modo Turbo)...", text_color='#4CAF50')
            
            frames_processados_atual = 0
            window['-PROG-'].update(0)
            window['-PERCENT-'].update("0%")
            
            processing = True
            executor = ProcessPoolExecutor(max_workers=NUM_PROCESSOS)
            futures = []
            
            for arq in lista_arquivos:
                fut = executor.submit(worker_process_video, (arq, pasta_in, pasta_out, logo_file, queue_progresso, stop_event))
                futures.append(fut)

        if event == 'CANCELAR' and processing:
            if sg.popup_yes_no("Parar tudo?") == 'Yes':
                window['-STATUS-'].update("Parando processos...", text_color='#C62828')
                stop_event.set()

        if processing:
            while not queue_progresso.empty():
                try:
                    tipo, valor = queue_progresso.get_nowait()
                    if tipo == 'progresso': frames_processados_atual += valor
                    elif tipo == 'status_fase': window['-STATUS-'].update(valor, text_color='#FFA000') 
                    elif tipo == 'concluido': window['-LOG-'].print(f"✔ Pronto: {valor}", text_color='#4CAF50')
                    elif tipo == 'erro': window['-LOG-'].print(f"✘ Erro: {valor}", text_color='#C62828')
                except: break
            
            if total_frames_global > 0:
                ratio = frames_processados_atual / total_frames_global
                if ratio > 1: ratio = 1
                window['-PROG-'].update(int(ratio * 1000))
                window['-PERCENT-'].update(f"{ratio * 100:.1f}%")

            if all(f.done() for f in futures):
                msg_final = "CANCELADO" if stop_event.is_set() else "CONCLUÍDO"
                cor_final = '#C62828' if stop_event.is_set() else '#4CAF50'
                
                window['-STATUS-'].update(msg_final, text_color=cor_final)
                if not stop_event.is_set():
                    window['-PROG-'].update(1000); window['-PERCENT-'].update("100%")
                    sg.popup("Finalizado com sucesso!", title="Fim")

                processing = False
                if executor: executor.shutdown(wait=False); executor = None
                window['INICIAR'].update(disabled=False)
                window['CANCELAR'].update(disabled=True)

    window.close()
    if executor: executor.shutdown(wait=False)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main_gui()