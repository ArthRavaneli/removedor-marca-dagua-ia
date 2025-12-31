import cv2
import numpy as np
import os
import FreeSimpleGUI as sg
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import sys
import shutil
import traceback
import subprocess
import importlib
from types import ModuleType
import io
import webbrowser
import time

# --- CONFIGURAÇÕES GERAIS ---
cv2.setNumThreads(1)

# --- BYPASS FLASH_ATTN ---
fake_flash = ModuleType("flash_attn")
fake_flash.__spec__ = object()
fake_flash.__path__ = []
fake_flash.__version__ = "2.5.6"
fake_flash.flash_attn_func = lambda *args, **kwargs: None
fake_flash.flash_attn_varlen_func = lambda *args, **kwargs: None
fake_flash.flash_attn_qkvpacked_func = lambda *args, **kwargs: None
sys.modules["flash_attn"] = fake_flash

# --- IMPORTAÇÕES DO IOPAINT ---
try:
    import iopaint
    from iopaint.model_manager import ModelManager
    from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config
except ImportError:
    sg.popup_error("Erro Crítico: Biblioteca 'iopaint' não encontrada.\nPor favor, execute: pip install iopaint")
    sys.exit()

# --- HELPER: FFMPEG ---
def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def get_ffmpeg_path():
    base_path = get_base_path()
    local_ffmpeg = os.path.join(base_path, 'ffmpeg.exe')
    if os.path.exists(local_ffmpeg):
        return local_ffmpeg
    return None

# --- WIZARD DE DEPENDÊNCIA ---
ffmpeg_exe_path = get_ffmpeg_path()
while ffmpeg_exe_path is None:
    sg.theme('GrayGrayGray')
    sg.set_options(background_color='#1c1c1c', text_element_background_color='#1c1c1c', element_background_color='#1c1c1c', input_text_color='white', text_color='white')
    layout_setup = [
        [sg.Text("⚠️ Componente FFmpeg Ausente", text_color='#FFD700', font=('Segoe UI', 16, 'bold'))],
        [sg.Text("Baixe o arquivo ffmpeg.exe e coloque na pasta do programa.", font=('Segoe UI', 10))],
        [sg.Button("Baixar (GitHub)", key='-DL-', size=(20,1)), sg.Button("Abrir Pasta", key='-OPEN-', size=(20,1))],
        [sg.Button("Tentar Novamente", key='-RETRY-', button_color=('white', '#2E7D32')), sg.Button("Sair", key='-EXIT-')]
    ]
    window_err = sg.Window("Setup", layout_setup, element_justification='c', keep_on_top=True)
    ev, val = window_err.read()
    window_err.close()
    if ev in (sg.WIN_CLOSED, '-EXIT-'): sys.exit()
    if ev == '-DL-': webbrowser.open("https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip")
    if ev == '-OPEN-': 
        try: os.startfile(get_base_path())
        except: subprocess.Popen(['explorer', get_base_path()])
    if ev == '-RETRY-': ffmpeg_exe_path = get_ffmpeg_path()

# --- FUNÇÕES IMAGEM ---
def criar_imagem_vazia(tamanho=(80, 80)):
    try:
        blank_image = Image.new('RGBA', tamanho, (0, 0, 0, 0))
        bio = io.BytesIO()
        blank_image.save(bio, format="PNG")
        return bio.getvalue()
    except: return None

def carregar_imagem_redimensionada(caminho_img, tamanho=(80, 80)):
    try:
        if not os.path.exists(caminho_img): return None
        pil_image = Image.open(caminho_img)
        pil_image = pil_image.resize(tamanho, Image.Resampling.LANCZOS)
        bio = io.BytesIO()
        pil_image.save(bio, format="PNG")
        return bio.getvalue()
    except: return None

# --- MOTORES IA ---
def download_lama_model():
    try:
        subprocess.run([sys.executable, "-m", "iopaint", "download", "--model", "lama"], check=True)
        return True
    except: return False

class FlorenceDetector:
    def __init__(self):
        self.model_id = 'microsoft/Florence-2-base'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True).to(self.device).eval()
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        except: self.model = None

    def detect(self, frame_cv2):
        if self.model is None: return []
        image = Image.fromarray(cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB))
        task_prompt = '<OPEN_VOCABULARY_DETECTION>'
        text_input = "watermark"
        try:
            inputs = self.processor(text=task_prompt + text_input, images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                generated_ids = self.model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, do_sample=False, num_beams=3)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = self.processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
            bboxes = parsed.get(task_prompt, {}).get('bboxes', [])
            final_boxes = []
            for box in bboxes:
                x1, y1, x2, y2 = map(int, box)
                final_boxes.append((x1, y1, x2-x1, y2-y1))
            return final_boxes
        except: return []

class LaMaInpainter:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.model_manager = ModelManager(name="lama", device=self.device)
        except:
            if download_lama_model():
                importlib.reload(iopaint.model)
                self.model_manager = ModelManager(name="lama", device=self.device)
            else: self.model_manager = None

    def inpaint(self, image, mask):
        if self.model_manager is None: return image
        # Converte Entrada BGR (OpenCV) para RGB (IA)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        config = Config(ldm_steps=20, ldm_sampler=LDMSampler.ddim, hd_strategy=HDStrategy.CROP, hd_strategy_crop_margin=32)
        res = self.model_manager(image_rgb, mask, config)
        
        # CORREÇÃO PONTO 1: Não inverter a saída. 
        # Se estava azul antes, significa que 'res' já é BGR (ou compatível).
        return res 

# --- WORKER LÓGICO ---
detector_engine = None
lama_engine = None

def contar_frames_totais(pasta_entrada):
    arquivos = [f for f in os.listdir(pasta_entrada) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    total = 0; validos = []
    for arq in arquivos:
        cap = cv2.VideoCapture(os.path.join(pasta_entrada, arq))
        if cap.isOpened():
            f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if f > 0: total += f; validos.append(arq)
            cap.release()
    return total, validos

def is_watermark_location(box, w_img, h_img):
    x, y, w, h = box
    cx, cy = x + w//2, y + h//2
    if (w * h) > (w_img * h_img * 0.30): return False
    margin_x, margin_y = w_img * 0.20, h_img * 0.20
    if cx < margin_x or cx > (w_img - margin_x) or cy < margin_y or cy > (h_img - margin_y): return True
    return False

def create_master_mask(boxes, w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    if not boxes: return None
    for (bx, by, bw, bh) in boxes:
        pad = 10 
        x1, y1 = max(0, bx - pad), max(0, by - pad)
        x2, y2 = min(w, bx + bw + pad), min(h, by + bh + pad)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

def worker_process_video(args):
    nome_arquivo, pasta_entrada, pasta_saida, _, queue, stop_event, debug, batch_size = args
    if stop_event.is_set(): return

    global detector_engine, lama_engine
    if detector_engine is None:
        queue.put(('terminal', "[SYS] Carregando Modelos IA..."))
        detector_engine = FlorenceDetector()
        lama_engine = LaMaInpainter()
        queue.put(('terminal', "[SYS] Modelos Prontos."))

    try:
        caminho_entrada = os.path.join(pasta_entrada, nome_arquivo)
        caminho_final = os.path.join(pasta_saida, nome_arquivo)
        temp_out = os.path.join(pasta_saida, f"temp_{nome_arquivo}")

        cap = cv2.VideoCapture(caminho_entrada)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        queue.put(('terminal', f"[INIT] Iniciando: {nome_arquivo}"))
        queue.put(('status_fase', f"Scaneando: {nome_arquivo}..."))

        scan_indices = np.linspace(0, total_frames - 5, 20, dtype=int)
        detected_boxes = []
        
        for idx in scan_indices:
            if stop_event.is_set(): cap.release(); return
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: continue
            
            raw_boxes = detector_engine.detect(frame)
            valid_boxes = [b for b in raw_boxes if is_watermark_location(b, w, h)]
            detected_boxes.extend(valid_boxes)
            if len(valid_boxes) > 0:
                queue.put(('terminal', f"[SCAN] Frame {idx}: Encontrado."))

        master_mask = create_master_mask(detected_boxes, w, h)
        if master_mask is not None:
             queue.put(('terminal', f"[SCAN] Máscara Mestra Criada e Dilatada."))
        else:
             queue.put(('terminal', f"[AVISO] Scan não encontrou marcas."))

        queue.put(('status_fase', f"Processando: {nome_arquivo}"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        out = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        batch_frames = []
        batch_count = 0
        total_batches = (total_frames // batch_size) + 1
        
        while True:
            if stop_event.is_set(): break
            ret, frame = cap.read()
            if not ret:
                if batch_frames:
                    for f in batch_frames:
                        if master_mask is not None:
                            if debug:
                                contours, _ = cv2.findContours(master_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                cv2.drawContours(f, contours, -1, (0, 0, 255), 2)
                                out.write(f)
                            else:
                                out.write(lama_engine.inpaint(f, master_mask))
                        else: out.write(f)
                    queue.put(('progresso', len(batch_frames)))
                break
            
            batch_frames.append(frame)
            
            if len(batch_frames) >= batch_size:
                batch_count += 1
                if batch_count % 5 == 0:
                    queue.put(('terminal', f"[GPU] Processando Lote {batch_count}/{total_batches}..."))
                
                for f in batch_frames:
                    if master_mask is not None:
                        if debug:
                            contours, _ = cv2.findContours(master_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(f, contours, -1, (0, 0, 255), 2)
                            out.write(f)
                        else:
                            out.write(lama_engine.inpaint(f, master_mask))
                    else:
                        out.write(f)
                
                queue.put(('progresso', len(batch_frames)))
                batch_frames = []

        cap.release(); out.release()
        if stop_event.is_set():
            if os.path.exists(temp_out): os.remove(temp_out)
            return

        queue.put(('status_fase', f"Finalizando MP4: {nome_arquivo}..."))
        queue.put(('terminal', f"[IO] Iniciando codificação FFmpeg..."))
        
        cmd = [
            get_ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
            "-i", temp_out, "-i", caminho_entrada,
            "-c:v", "libx264", "-crf", "23", "-preset", "ultrafast",
            "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
            "-shortest", caminho_final
        ]
        
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        proc = subprocess.run(cmd, capture_output=True, startupinfo=startupinfo)
        
        if os.path.exists(temp_out): os.remove(temp_out)
        
        if proc.returncode == 0:
            queue.put(('concluido', nome_arquivo))
        else:
            queue.put(('erro', f"Falha no FFmpeg: {nome_arquivo}"))

    except Exception as e:
        traceback.print_exc()
        queue.put(('erro', str(e)))

# --- GUI PRINCIPAL ---
def main_gui():
    sg.theme('GrayGrayGray')
    sg.set_options(background_color='#1c1c1c', text_element_background_color='#1c1c1c', element_background_color='#1c1c1c', 
                   input_elements_background_color='#333333', input_text_color='white', text_color='white')
    
    layout_inputs = [
        [sg.Text('Pasta Origem:', size=(15,1), justification='right'), sg.Input(key='-IN-', default_text='videos_originais', size=(45,1)), sg.FolderBrowse('...')],
        [sg.Text('Pasta Destino:', size=(15,1), justification='right'), sg.Input(key='-OUT-', default_text='videos_prontos', size=(45,1)), sg.FolderBrowse('...')],
        [
            sg.Text('Modo:', size=(6,1), justification='right'), 
            sg.Combo(['AUTO (Multi-Scan)'], default_value='AUTO (Multi-Scan)', key='-PROMPT-', size=(20,1), readonly=True),
            sg.Text('GPU Workers:', size=(12,1), justification='right'), 
            sg.Combo(['1', '2'], default_value='1', key='-WORKERS-', size=(4,1), readonly=True), 
            # REMOVIDO: Opção de Smart Batch. Agora é fixo em 10.
        ]
    ]

    tamanho_icone = (80, 80)
    logo_data = carregar_imagem_redimensionada("icone.ico", tamanho=tamanho_icone)
    blank_data = criar_imagem_vazia(tamanho=tamanho_icone)

    layout = [
        [sg.Image(data=blank_data, background_color='#1c1c1c', pad=(0,0)) if blank_data else sg.Text("", size=(11,1)),
         sg.Push(), sg.Text('Removedor Automático', font=('Segoe UI', 20, 'bold'), pad=((0,0),(20,20))), sg.Push(),
         sg.Image(data=logo_data, key='-LOGO-', background_color='#1c1c1c', pad=(0,0)) if logo_data else sg.Text("", size=(11,1))],
        [sg.Push(), sg.Frame('Configurações', layout_inputs, font=('Segoe UI', 10, 'bold'), pad=((0,0),(0,20)), element_justification='c', title_color='white'), sg.Push()],
        [sg.Push(), sg.Checkbox('Modo Debug (Ver caixas vermelhas)', key='-DEBUG-', text_color='#00FF00'), sg.Push()],
        [sg.Push(), sg.Text('Aguardando Início', key='-STATUS-', font=('Segoe UI', 11, 'italic'), text_color='#FFD700'), sg.Push()],
        [sg.Push(), sg.ProgressBar(1000, orientation='h', size=(60, 20), key='-PROG-', bar_color=('#4CAF50', '#333333')), sg.Push()],
        [sg.Push(), sg.Text('0%', key='-PERCENT-', font=('Segoe UI', 12, 'bold')), sg.Push()],
        [sg.Text("Histórico:", font=('Segoe UI', 9, 'bold'), text_color='#888888')],
        [sg.Push(), sg.Multiline(size=(90, 6), key='-LOG-', autoscroll=True, disabled=True, font=('Consolas', 9), background_color='#111111', text_color='#eeeeee', border_width=0), sg.Push()],
        [sg.Text("Terminal / Backend:", font=('Segoe UI', 9, 'bold'), text_color='#888888')],
        [sg.Push(), sg.Multiline(size=(90, 8), key='-TERM-', autoscroll=True, disabled=True, font=('Consolas', 8), background_color='#000000', text_color='#00FF00', border_width=0), sg.Push()],
        [sg.Button('INICIAR', key='INICIAR', size=(15, 2), button_color=('white', '#2E7D32'), font=('Segoe UI', 10, 'bold'), pad=((20, 5), (20, 10))),
         sg.Button('CANCELAR', key='CANCELAR', size=(15, 2), button_color=('white', '#C62828'), font=('Segoe UI', 10, 'bold'), disabled=True, pad=((5, 5), (20, 10))),
         sg.Push(), sg.Button('SAIR', size=(10, 2), button_color=('white', '#444444'), pad=((5, 20), (20, 10)))],
        [sg.Push(), sg.Text('v68.2 (Fixed Batch 10)', font=('Segoe UI', 8), text_color='#555555', pad=(10,0))]
    ]

    window = sg.Window('Removedor Automático', layout, finalize=True, element_justification='c')
    
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    stop = manager.Event()
    executor = None
    processing = False
    
    total_frames_global = 0
    processed_global = 0

    while True:
        event, values = window.read(timeout=50)
        
        if event in (sg.WIN_CLOSED, 'SAIR'):
            if processing: stop.set()
            break
            
        if event == 'INICIAR':
            if not os.path.exists(values['-IN-']): continue
            if not os.path.exists(values['-OUT-']): os.makedirs(values['-OUT-'])

            window['INICIAR'].update(disabled=True)
            window['CANCELAR'].update(disabled=False)
            stop.clear()
            window['-LOG-'].update("")
            window['-TERM-'].update("")
            window.refresh()
            
            total_frames_global, lista = contar_frames_totais(values['-IN-'])
            if total_frames_global == 0:
                window['-STATUS-'].update("Nenhum vídeo encontrado.")
                window['INICIAR'].update(disabled=False)
                continue

            # CORREÇÃO PONTO 2: Adicionado \n no final
            header_msg = (
                f"=== INICIANDO LOTE ===\n"
                f"JOB: {len(lista)} vídeos | Total Frames: {total_frames_global} | "
                f"Config: {values['-WORKERS-']} Workers / Batch: Auto (10)\n"
                f"--------------------------------------------------\n" 
            )
            window['-LOG-'].update(header_msg)

            ctx = multiprocessing.get_context('spawn')
            executor = ProcessPoolExecutor(max_workers=int(values['-WORKERS-']), mp_context=ctx)
            futures = []
            
            for arq in lista:
                futures.append(executor.submit(worker_process_video, (
                    arq, values['-IN-'], values['-OUT-'], 
                    values['-PROMPT-'], queue, stop, 
                    values['-DEBUG-'], 10 # Batch fixo em 10
                )))
            
            processing = True
            processed_global = 0

        if event == 'CANCELAR' and processing:
            if sg.popup_yes_no("Parar?") == 'Yes': stop.set()

        if processing:
            while not queue.empty():
                try:
                    t, v = queue.get_nowait()
                    if t == 'progresso': 
                        processed_global += v
                    elif t == 'concluido': 
                        window['-LOG-'].print(f"✔ Pronto: {v}", text_color='#4CAF50')
                        window['-STATUS-'].update("Aguardando próximo...", text_color='#888888')
                    elif t == 'erro': 
                        window['-LOG-'].print(f"✘ Erro: {v}", text_color='#C62828')
                    elif t == 'status_fase': 
                        window['-STATUS-'].update(v, text_color='#FFD700')
                    elif t == 'terminal': 
                        window['-TERM-'].print(v)
                except: break
            
            if total_frames_global > 0:
                ratio = processed_global / total_frames_global
                window['-PROG-'].update(int(ratio * 1000))
                window['-PERCENT-'].update(f"{ratio * 100:.1f}%")

            if all(f.done() for f in futures):
                processing = False
                msg = "CONCLUÍDO" if not stop.is_set() else "CANCELADO"
                color = '#4CAF50' if not stop.is_set() else '#C62828'
                window['-STATUS-'].update(msg, text_color=color)
                window['INICIAR'].update(disabled=False)
                window['CANCELAR'].update(disabled=True)
                window['-TERM-'].print(">>> FIM DO PROCESSO.")

    window.close()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main_gui()