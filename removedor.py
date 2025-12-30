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
from collections import deque
import webbrowser
import io
import tempfile # Usado para criar o ícone temporário

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
    sg.popup_error("Erro Crítico: Biblioteca 'iopaint' não encontrada.\n\nPor favor, execute no terminal:\npip install iopaint")
    sys.exit()

# --- HELPER: DETECÇÃO ESTRITA DE FFMPEG ---
def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def get_ffmpeg_path():
    base_path = get_base_path()
    local_ffmpeg = os.path.join(base_path, 'ffmpeg.exe')
    
    # MODO ESTRITO: Só aceita se estiver na pasta do programa.
    if os.path.exists(local_ffmpeg):
        return local_ffmpeg
    
    return None

# --- WIZARD DE DEPENDÊNCIA (V55) ---
ffmpeg_exe_path = get_ffmpeg_path()

while ffmpeg_exe_path is None:
    sg.theme('GrayGrayGray')
    sg.set_options(
        background_color='#1c1c1c', 
        text_element_background_color='#1c1c1c', 
        element_background_color='#1c1c1c', 
        input_elements_background_color='#333333', 
        input_text_color='white', 
        text_color='white'
    )
    
    base_folder = get_base_path()
    
    layout_setup = [
        [sg.Text("⚠️ Componente de Vídeo Ausente", text_color='#FFD700', font=('Segoe UI', 16, 'bold'), pad=((0,0),(20,10)))],
        [sg.Text("Este é um software portátil. O motor de vídeo (FFmpeg) precisa estar na pasta.", font=('Segoe UI', 10), text_color='#DDDDDD')],
        [sg.Text(f"Siga os passos para configurar:", font=('Segoe UI', 11, 'bold'), text_color='white', pad=((0,0),(10,10)))],
        [sg.HorizontalSeparator()],
        
        [sg.Text("1. Baixe o arquivo ZIP (Versão Master ~190MB):", text_color='#AAAAAA', pad=((0,0),(10,5)))],
        
        [sg.Button("Baixar FFmpeg (Mirror GitHub Rápido)", key='-DL-', size=(35,2), button_color=('white', '#0078D7'), font=('Segoe UI', 10, 'bold'))],
        
        # --- INSTRUÇÃO PRECISA ---
        [sg.Text("2. Abra o ZIP e entre na pasta interna (ffmpeg-master-latest-win64-gpl).", text_color='#AAAAAA', pad=((0,0),(15,2)))],
        [sg.Text("3. Entre na pasta 'bin' e copie APENAS o arquivo 'ffmpeg.exe'.", text_color='#FFD700', font=('Segoe UI', 9, 'bold'), pad=((0,0),(0,5)))],
        
        [sg.Text("4. Cole o arquivo nesta pasta aqui:", text_color='#AAAAAA', pad=((0,0),(10,5)))],
        
        # Campo com contraste corrigido
        [sg.Input(base_folder, readonly=True, size=(50,1), text_color='black', background_color='#DDDDDD'), 
         sg.Button("Abrir Pasta", key='-OPEN-', button_color=('white', '#505050'))],
        
        [sg.HorizontalSeparator(pad=((0,0),(20,20)))],
        [sg.Button("Já colei o arquivo! Iniciar", key='-RETRY-', size=(30,2), button_color=('white', '#2E7D32'), font=('Segoe UI', 11, 'bold')), 
         sg.Button("Sair", key='-EXIT-', size=(10,2), button_color='#C62828')]
    ]
    
    window_err = sg.Window("Instalação Portátil", layout_setup, element_justification='c', keep_on_top=True, finalize=True)
    
    ev, val = window_err.read()
    window_err.close()
    
    if ev in (sg.WIN_CLOSED, '-EXIT-'):
        sys.exit()
        
    if ev == '-DL-':
        webbrowser.open("https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip")
        
    if ev == '-OPEN-':
        try: os.startfile(base_folder)
        except: subprocess.Popen(['explorer', base_folder])
        
    if ev == '-RETRY-':
        ffmpeg_exe_path = get_ffmpeg_path()
        if ffmpeg_exe_path is None:
            sg.theme('GrayGrayGray')
            sg.popup_error("Ainda não encontrei o 'ffmpeg.exe'.\n\nVerifique se você copiou o arquivo da pasta 'bin' dentro do ZIP.", title="Não encontrado", background_color='#1c1c1c', text_color='white')

# --- FIM DO WIZARD ---

# --- FUNÇÕES DE DOWNLOAD ---
def download_lama_model():
    print(">>> Baixando modelo LaMA (~200MB)... Isso pode demorar um pouco. <<<")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "iopaint", "download", "--model", "lama"],
            capture_output=False,
            text=True
        )
        if result.returncode != 0:
            print("Falha ao baixar modelo LaMA.")
            return False
        print(">>> Modelo LaMA baixado com sucesso! <<<")
        return True
    except Exception as e:
        print(f"Erro no download: {e}")
        return False

# --- ESTABILIZADOR ESTATÍSTICO (V46) ---
class StatisticalStabilizer:
    def __init__(self, history_size=5, teleport_threshold=100):
        self.history = deque(maxlen=history_size)
        self.teleport_threshold = teleport_threshold

    def update(self, new_boxes):
        if not new_boxes:
            if len(self.history) > 0:
                return [np.mean(self.history, axis=0).astype(int)]
            return []

        current_box = sorted(new_boxes, key=lambda b: b[2]*b[3], reverse=True)[0]
        cx, cy, cw, ch = current_box
        
        if len(self.history) > 0:
            avg_box = np.mean(self.history, axis=0)
            ax, ay, aw, ah = avg_box
            dist = ((cx - ax)**2 + (cy - ay)**2)**0.5
            
            if dist > self.teleport_threshold:
                self.history.clear()
                self.history.append(current_box)
                return [current_box]
            else:
                self.history.append(current_box)
                stable_box = np.mean(self.history, axis=0).astype(int)
                return [stable_box]
        else:
            self.history.append(current_box)
            return [current_box]

# --- DETECTOR ---
class FlorenceDetector:
    def __init__(self):
        self.model_id = 'microsoft/Florence-2-base'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True, attn_implementation="eager").to(self.device).eval()
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        except Exception as e:
            self.model = None

    def detect(self, frame_cv2):
        if self.model is None: return []
        image = Image.fromarray(cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB))
        final_boxes = []

        try:
            task_ocr = '<OCR_WITH_REGION>'
            inputs = self.processor(text=task_ocr, images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, do_sample=False, num_beams=3)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = self.processor.post_process_generation(generated_text, task=task_ocr, image_size=(image.width, image.height))
            ocr_data = parsed.get(task_ocr, {})
            quad_boxes = ocr_data.get('quad_boxes', [])
            for quad in quad_boxes:
                xs = [quad[0], quad[2], quad[4], quad[6]]
                ys = [quad[1], quad[3], quad[5], quad[7]]
                x1 = min(xs); x2 = max(xs); y1 = min(ys); y2 = max(ys)
                final_boxes.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
        except: pass
        
        return final_boxes

class LaMaInpainter:
    def __init__(self):
        self.model_manager = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            try:
                self.model_manager = ModelManager(name="lama", device=self.device)
            except Exception as e:
                if "Unsupported model" in str(e) or "not found" in str(e).lower():
                    if download_lama_model():
                        importlib.reload(iopaint.model)
                        self.model_manager = ModelManager(name="lama", device=self.device)
                    else:
                        raise RuntimeError("Falha ao baixar o modelo LaMA.")
                else:
                    raise e
        except Exception as e:
            self.model_manager = None

    def inpaint(self, image, mask):
        if self.model_manager is None: return image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # --- CORREÇÃO DA SINTAXE (V55) ---
        config = Config(
            ldm_steps=50,
            ldm_sampler=LDMSampler.ddim,
            hd_strategy=HDStrategy.CROP,
            hd_strategy_crop_margin=64,
            hd_strategy_crop_trigger_size=800, # Agora está correto (apenas 1 vez)
            hd_strategy_resize_limit=1600,
        )
        # ---------------------------------
        
        res = self.model_manager(image_rgb, mask, config)
        if res.dtype in [np.float64, np.float32]:
            res = np.clip(res, 0, 255).astype(np.uint8)
        return res

detector_engine = None
lama_engine = None

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

def is_watermark_location(box, w_img, h_img):
    x, y, w, h = box
    cx = x + w // 2; cy = y + h // 2
    if (w * h) > (w_img * h_img * 0.35): return False 
    margin_x = w_img * 0.25; margin_y = h_img * 0.25
    is_left = cx < margin_x; is_right = cx > (w_img - margin_x)
    is_top = cy < margin_y; is_bottom = cy > (h_img - margin_y)
    if (is_top or is_bottom) and (is_left or is_right): return True
    if cy < (h_img * 0.12) or cy > (h_img * 0.88): return True 
    return False

def create_mask_from_boxes(boxes, w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    if not boxes: return None
    for (bx, by, bw, bh) in boxes:
        pad_right = int(bw * 0.02) + 5
        pad_bottom = int(bh * 0.05) + 5
        pad_top = int(bh * 0.05) + 5
        pad_left = int(bh * 0.6) 
        
        x1 = max(0, bx - pad_left)
        y1 = max(0, by - pad_top)
        x2 = min(w, bx + bw + pad_right)
        y2 = min(h, by + bh + pad_bottom)
        
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask

def worker_process_video(args):
    try:
        nome_arquivo, pasta_entrada, pasta_saida, prompt_txt, lama_path, queue, stop, debug, batch_size = args
        if stop.is_set(): return

        queue.put(('terminal', f"[INIT] Processo iniciado para: {nome_arquivo}"))

        global detector_engine, lama_engine
        if detector_engine is None: 
            queue.put(('status_fase', "Carregando Florence-2 na GPU..."))
            detector_engine = FlorenceDetector()
            queue.put(('terminal', f"[GPU] Florence-2 Carregado: {detector_engine.device}"))
            
        if lama_engine is None: 
            queue.put(('status_fase', "Carregando LaMA Inpaint..."))
            lama_engine = LaMaInpainter()
            queue.put(('terminal', f"[GPU] LaMA Carregado: {lama_engine.device}"))

        if lama_engine.model_manager is None:
            queue.put(('erro', "Falha ao carregar modelos."))
            return

        caminho_entrada = os.path.join(pasta_entrada, nome_arquivo)
        caminho_final = os.path.join(pasta_saida, nome_arquivo)
        temp_out = os.path.join(pasta_saida, f"temp_{nome_arquivo}")
        
        cap = cv2.VideoCapture(caminho_entrada)
        if not cap.isOpened():
            queue.put(('erro', f"Erro abrir: {nome_arquivo}"))
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        frame_buffer = []
        stabilizer = StatisticalStabilizer() 
        
        frame_count_global = 0

        while True:
            if stop.is_set(): break
            
            while len(frame_buffer) < batch_size:
                ret, fr = cap.read()
                if not ret: break
                frame_buffer.append(fr)
            
            if not frame_buffer: break 
            
            frame_count_global += len(frame_buffer)
            queue.put(('progresso', len(frame_buffer)))

            last_frame_in_batch = frame_buffer[-1]
            current_raw_boxes = []
            
            try:
                raw = detector_engine.detect(last_frame_in_batch)
                current_raw_boxes = [b for b in raw if "AUTO" not in prompt_txt or is_watermark_location(b, w, h)]
            except: pass

            stabilized_boxes = stabilizer.update(current_raw_boxes)
            mask = create_mask_from_boxes(stabilized_boxes, w, h)

            for frm in frame_buffer:
                if mask is not None:
                    if debug:
                        final_frm = frm.copy()
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(final_frm, contours, -1, (0, 255, 0), 2)
                        out.write(final_frm)
                    else:
                        out.write(lama_engine.inpaint(frm, mask))
                else:
                    out.write(frm)
            
            frame_buffer = [] 

        cap.release(); out.release()
        if stop.is_set(): 
            if os.path.exists(temp_out): os.remove(temp_out)
            return
            
        queue.put(('status_fase', f"Encoding Final (CPU): {nome_arquivo}..."))
        queue.put(('terminal', f"[FFMPEG] Comprimindo {nome_arquivo} (libx264/crf23)..."))
        
        try:
            # Usa o FFmpeg detectado
            global ffmpeg_exe_path 
            
            cmd = [
                ffmpeg_exe_path, "-y", "-hide_banner", "-loglevel", "error",
                "-i", temp_out,          
                "-i", caminho_entrada,   
                "-c:v", "libx264",       
                "-crf", "23",            
                "-preset", "veryfast",   
                "-c:a", "aac",           
                "-map", "0:v:0",         
                "-map", "1:a:0",         
                "-shortest",             
                caminho_final
            ]
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            result = subprocess.run(cmd, capture_output=True, startupinfo=startupinfo)
            
            if result.returncode == 0: 
                queue.put(('concluido', nome_arquivo))
                queue.put(('terminal', f"[OK] Arquivo salvo: {caminho_final}"))
            else:
                shutil.move(temp_out, caminho_final)
                queue.put(('concluido', f"{nome_arquivo} (Mudo)"))
                queue.put(('terminal', f"[WARN] Salvo sem áudio (Falha no Mux)"))

        except Exception as e_ffmpeg:
             shutil.move(temp_out, caminho_final)
             queue.put(('erro', f"Erro FFmpeg: {str(e_ffmpeg)}"))
             queue.put(('terminal', f"[ERRO] {str(e_ffmpeg)}"))

        if os.path.exists(temp_out): os.remove(temp_out)

    except Exception as e_critico:
        traceback.print_exc()
        queue.put(('erro', f"CRASH: {str(e_critico)}"))
        queue.put(('terminal', f"[CRASH] {str(e_critico)}"))

# --- HELPERS DE IMAGEM (NOVOS/ALTERADOS) ---
def criar_imagem_vazia(tamanho=(80, 80)):
    """Cria uma imagem transparente vazia para balanceamento de layout."""
    try:
        blank_image = Image.new('RGBA', tamanho, (0, 0, 0, 0)) # Transparente
        bio = io.BytesIO()
        blank_image.save(bio, format="PNG")
        return bio.getvalue()
    except Exception as e:
        print(f"Erro ao criar imagem vazia: {e}")
        return None

def carregar_imagem_redimensionada(caminho_img, tamanho=(80, 80)):
    """Carrega imagem (PNG/ICO), redimensiona e converte para bytes pro FreeSimpleGUI"""
    try:
        if not os.path.exists(caminho_img):
            return None
        
        pil_image = Image.open(caminho_img)
        pil_image = pil_image.resize(tamanho, Image.Resampling.LANCZOS)
        
        # Converte para bytes (PNG na memória)
        bio = io.BytesIO()
        pil_image.save(bio, format="PNG")
        return bio.getvalue()
    except Exception as e:
        print(f"Erro ao carregar logo: {e}")
        return None

def main_gui():
    sg.theme('GrayGrayGray')
    sg.set_options(background_color='#1c1c1c', text_element_background_color='#1c1c1c', element_background_color='#1c1c1c', input_elements_background_color='#333333', input_text_color='white', text_color='white')
    
    layout_inputs = [
        [sg.Text('Pasta Origem:', size=(15,1), justification='right'), sg.Input(key='-IN-', default_text='videos_originais', size=(45,1)), sg.FolderBrowse('...')],
        [sg.Text('Pasta Destino:', size=(15,1), justification='right'), sg.Input(key='-OUT-', default_text='videos_prontos', size=(45,1)), sg.FolderBrowse('...')],
        [
            sg.Text('Modo:', size=(6,1), justification='right'), 
            sg.Combo(['AUTO (Corners)'], default_value='AUTO (Corners)', key='-PROMPT-', size=(20,1), readonly=True),
            
            sg.Text('GPU Workers:', size=(12,1), justification='right'), 
            sg.Combo(['1', '2'], default_value='1', key='-WORKERS-', size=(4,1), readonly=True), 
            
            sg.Text('Smart Batch:', size=(12,1), justification='right', tooltip="Tamanho do lote."), 
            sg.Spin(values=[5, 10, 15, 20, 30], initial_value=10, key='-BATCH-', size=(4,1))
        ]
    ]

    # --- CARREGA O LOGO E O ESPAÇADOR PARA CENTRALIZAÇÃO ---
    tamanho_icone = (80, 80) # Tamanho fixo para garantir balanço
    logo_data = carregar_imagem_redimensionada("icone.ico", tamanho=tamanho_icone)
    blank_data = criar_imagem_vazia(tamanho=tamanho_icone) # Espaçador esquerdo
    # -------------------------------------------------------

    layout = [
        # --- LINHA DO TÍTULO PERFEITAMENTE CENTRALIZADO ---
        [
            # 1. Espaçador invisível na esquerda (mesmo tamanho do logo)
            sg.Image(data=blank_data, background_color='#1c1c1c', pad=(0,0)) if blank_data else sg.Text("", size=(11,1)),
            
            # 2. Título centralizado com Pushes
            sg.Push(),
            sg.Text('Removedor Automático', font=('Segoe UI', 20, 'bold'), pad=((0,0),(20,20))),
            sg.Push(),
            
            # 3. Logo na direita
            sg.Image(data=logo_data, key='-LOGO-', background_color='#1c1c1c', pad=(0,0)) if logo_data else sg.Text("", size=(11,1))
        ],
        # --------------------------------------------------
        
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
         
        [sg.Push(), sg.Text('v56.0 (Auto-Create Folders)', font=('Segoe UI', 8), text_color='#555555', pad=(10,0))]
    ]

    window = sg.Window('Removedor Automático', layout, finalize=True, element_justification='c')
    
    # ... (resto do código da main_gui permanece igual) ...
    manager = multiprocessing.Manager(); queue = manager.Queue(); stop = manager.Event()
    executor = None; processing = False; total_frames = 0; processed = 0
    
    while True:
        event, values = window.read(timeout=50)
        if event in (sg.WIN_CLOSED, 'SAIR'):
            if processing: stop.set()
            break
        if event == 'INICIAR':
            # --- BLINDAGEM DE PASTAS (NOVO NA V56) ---
            folder_in = values['-IN-']
            folder_out = values['-OUT-']

            # 1. Valida Entrada
            if not os.path.exists(folder_in):
                sg.popup_error(f"A pasta de origem não existe:\n{folder_in}\n\nPor favor, selecione uma pasta válida.", title="Erro de Pasta")
                continue # Volta para o loop, não inicia nada
            
            # 2. Cria Saída Automática
            if not os.path.exists(folder_out):
                try:
                    os.makedirs(folder_out)
                    window['-TERM-'].print(f">>> Pasta criada automaticamente: {folder_out}")
                except Exception as e:
                    sg.popup_error(f"Não foi possível criar a pasta de destino:\n{e}", title="Erro de Permissão")
                    continue
            # -----------------------------------------

            window['-STATUS-'].update("Inicializando Sistema...", text_color='#FFD700')
            window['INICIAR'].update(disabled=True); window['CANCELAR'].update(disabled=False)
            stop.clear(); window['-LOG-'].update(""); window['-TERM-'].update(""); window.refresh()
            
            total_frames, lista = contar_frames_totais(folder_in, stop)
            
            if total_frames == 0:
                window['-TERM-'].print(">>> NENHUM VÍDEO ENCONTRADO NA PASTA DE ORIGEM.")
                sg.popup_ok("Nenhum vídeo (.mp4, .avi, .mov) encontrado na pasta de origem!", title="Aviso")
                window['INICIAR'].update(disabled=False); window['CANCELAR'].update(disabled=True)
                window['-STATUS-'].update("Aguardando...", text_color='#FFD700')
                continue

            try: num_workers = int(values['-WORKERS-'])
            except: num_workers = 1
            
            window['-TERM-'].print(f">>> START JOB: {len(lista)} arquivos detectados.")
            window['-TERM-'].print(f">>> WORKERS: {num_workers} | BATCH SIZE: {values['-BATCH-']}")
            window['-LOG-'].print(f"Vídeos: {len(lista)} | Frames Totais: {total_frames}\n")
            
            ctx = multiprocessing.get_context('spawn')
            executor = ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx)
            futures = []
            for arq in lista:
                futures.append(executor.submit(worker_process_video, (
                    arq, folder_in, folder_out, values['-PROMPT-'], 
                    None, queue, stop, values['-DEBUG-'], int(values['-BATCH-'])
                )))
            processing = True; processed = 0

        if event == 'CANCELAR' and processing:
            if sg.popup_yes_no("Parar?") == 'Yes': stop.set()
        if processing:
            while not queue.empty():
                try:
                    t, v = queue.get_nowait()
                    if t == 'progresso': processed += v
                    elif t == 'concluido': window['-LOG-'].print(f"✔ Pronto: {v}", text_color='#4CAF50')
                    elif t == 'erro': window['-LOG-'].print(f"✘ Erro: {v}", text_color='#C62828')
                    elif t == 'status_fase': window['-STATUS-'].update(v, text_color='#FFA000')
                    elif t == 'terminal':
                        cor_texto = '#00FF00' 
                        if '[OK]' in v: cor_texto = '#FFD700' 
                        elif '[ERRO]' in v or '[CRASH]' in v: cor_texto = '#FF5555' 
                        elif '[WARN]' in v: cor_texto = '#FFA500' 
                        window['-TERM-'].print(v, text_color=cor_texto)
                except: break
            if total_frames > 0:
                ratio = processed / total_frames
                window['-PROG-'].update(int(ratio * 1000))
                window['-PERCENT-'].update(f"{ratio * 100:.1f}%")
            if all(f.done() for f in futures):
                processing = False
                msg = "CONCLUÍDO" if not stop.is_set() else "CANCELADO"
                color = '#4CAF50' if not stop.is_set() else '#C62828'
                window['-STATUS-'].update(msg, text_color=color)
                window['-TERM-'].print(">>> JOB FINISHED.")
                if not stop.is_set(): sg.popup("Fim do Lote!")
                window['INICIAR'].update(disabled=False); window['CANCELAR'].update(disabled=True)
    window.close()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main_gui()