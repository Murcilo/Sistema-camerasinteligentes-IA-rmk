import os
import threading
import time
from collections import deque
from datetime import datetime
import cv2
import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from ultralytics import YOLO


class DetectorComBoxes:
    def __init__(self, video_source=0):
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)

        print("🔄 Carregando YOLO...")
        self.yolo = YOLO('yolov8n.pt')
        print("✅ YOLO carregado!\n")

        print("🔄 Carregando VideoMAE...")
        model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_name).to(self.device)

        print(f"✅ VideoMAE em '{self.device}'.\n")

        self.frame_buffer = deque(maxlen=16)

        # Gravação
        self.gravando = False
        self.inicio_gravacao = None
        self.video_writer = None
        self.duracao_gravacao = 10
        self.pasta_videos = "videos_anomalias"
        if not os.path.exists(self.pasta_videos):
            os.makedirs(self.pasta_videos)

        # Threading
        self.analisando = False
        self.thread_analise = None

        # Flask
        self.ultimo_frame = None
        self.ultima_deteccao = None
        self.historico_videos = []
        self.frame_anterior = None
        self.rodando = False

        # YOLO otimizado - só a cada 10 frames!
        self.contador_frames = 0
        self.intervalo_yolo = 10  # Aumentado de 5 para 10
        self.num_pessoas = 0
        self.boxes_yolo = []

    def detectar_movimento(self, frame1, frame2, limiar_area=2000):
        """Detecta movimento significativo entre dois frames."""
        if frame1 is None or frame2 is None:
            return False
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contornos, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contornos:
            if cv2.contourArea(c) > limiar_area:
                return True
        return False

    def processar_yolo_rapido(self, frame):
        """YOLO ultra-otimizado - só conta pessoas"""
        results = self.yolo(frame, verbose=False, conf=0.6)  # Confiança maior = mais rápido
        pessoas = 0
        boxes = []

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:  # Classe pessoa
                    pessoas += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append((x1, y1, x2, y2))

        return pessoas, boxes

    def desenhar_deteccoes(self, frame):
        """Desenha boxes de forma SUPER rápida - direto no frame original"""
        for i, (x1, y1, x2, y2) in enumerate(self.boxes_yolo, 1):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'P{i}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def classificar_video(self, video_clip):
        """Classifica clipe de vídeo."""
        inputs = self.processor(list(video_clip), return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            pred_idx = logits.argmax(-1).item()
            label = self.model.config.id2label[pred_idx]

        return label.lower()

    def analisar_em_background(self, video_clip, frame_atual):
        """Análise em thread separada - IGUAL ao PROJETO_CAMERA."""
        try:
            print("📸 Analisando...")

            label = self.classificar_video(video_clip)

            print(f"🔎 IA: '{label}'")

            evento = None
            if any(palavra in label for palavra in ["fight", "punch", "kick", "hit"]):
                evento = "violencia_detectada"
            elif any(palavra in label for palavra in ["running", "jumping", "falling", "climbing"]):
                evento = "comportamento_suspeito"
            elif any(palavra in label for palavra in ["robbery", "burglary", "stealing"]):
                evento = "atividade_ilicita"

            self.ultima_deteccao = {
                'acao': label,
                'evento': evento,
                'timestamp': datetime.now()
            }

            if evento:
                print(f"🚨 Evento: {evento}")
                self.iniciar_gravacao(frame_atual, evento)

        finally:
            self.analisando = False

    def iniciar_gravacao(self, frame, evento_detectado):
        """Inicia gravação - SIMPLIFICADO."""
        if self.gravando:
            return

        altura, largura = frame.shape[:2]
        timestamp = int(time.time())
        nome = f"{evento_detectado}_{timestamp}.mp4"
        caminho = os.path.join(self.pasta_videos, nome)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(caminho, fourcc, 20.0, (largura, altura))
        self.gravando = True
        self.inicio_gravacao = time.time()

        self.historico_videos.append({
            'nome': nome,
            'caminho': caminho,
            'evento': evento_detectado,
            'timestamp': datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        })

        print(f"🎥 Gravando: {nome}")

        # Grava frames do buffer
        for buffered_frame in self.frame_buffer:
            self.video_writer.write(buffered_frame)

    def processar_frame(self):
        """Processa frame - SIMPLIFICADO como PROJETO_CAMERA."""
        if not self.rodando:
            return None

        ret, frame_atual = self.cap.read()
        if not ret:
            return None

        # YOLO apenas a cada 10 frames (mais rápido!)
        self.contador_frames += 1
        if self.contador_frames % self.intervalo_yolo == 0:
            self.num_pessoas, self.boxes_yolo = self.processar_yolo_rapido(frame_atual)

        # Desenha boxes DIRETO no frame (sem cópia!)
        if self.boxes_yolo:
            self.desenhar_deteccoes(frame_atual)

        # Adiciona ao buffer
        self.frame_buffer.append(frame_atual)

        # Detecta movimento
        if self.frame_anterior is not None:
            movimento = self.detectar_movimento(self.frame_anterior, frame_atual)

            if movimento and not self.gravando and len(self.frame_buffer) == 16 and not self.analisando:
                self.analisando = True

                video_clip_copy = list(self.frame_buffer)
                frame_copy = frame_atual.copy()

                self.thread_analise = threading.Thread(
                    target=self.analisar_em_background,
                    args=(video_clip_copy, frame_copy),
                    daemon=True
                )
                self.thread_analise.start()

        # Gravação
        if self.gravando:
            self.video_writer.write(frame_atual)
            if time.time() - self.inicio_gravacao >= self.duracao_gravacao:
                self.video_writer.release()
                self.gravando = False
                print("💾 Gravação finalizada")

        self.frame_anterior = frame_atual.copy()
        self.ultimo_frame = frame_atual

        return frame_atual

    def iniciar(self):
        """Inicia o sistema."""
        print("🔵 Iniciando...")

        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.video_source)
            time.sleep(0.5)

        self.rodando = True
        ret, frame = self.cap.read()

        if frame is not None:
            print("✅ Sistema iniciado!")
            for _ in range(16):
                self.frame_buffer.append(frame)
            self.frame_anterior = frame
        else:
            print("❌ Erro ao capturar frame")

    def parar(self):
        """Para o sistema."""
        print("🔴 Parando...")
        self.rodando = False

        if self.gravando and self.video_writer:
            self.video_writer.release()
            self.gravando = False

        if self.cap.isOpened():
            self.cap.release()

    def get_historico_videos(self):
        """Retorna lista de vídeos."""
        return sorted(self.historico_videos, key=lambda x: x['timestamp'], reverse=True)
class DetectorComBoxes:
    def __init__(self, video_source=0):
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)

        print("🔄 Carregando modelo YOLO...")
        self.yolo = YOLO('yolov8n.pt')
        print("✅ YOLO carregado!\n")

        print("🔄 Carregando modelo VideoMAE...")
        model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_name).to(self.device)

        print(f"✅ VideoMAE carregado em '{self.device}'.\n")

        self.frame_buffer = deque(maxlen=16)

        # Estado do sistema
        self.gravando = False
        self.inicio_gravacao = None
        self.video_writer = None
        self.duracao_gravacao = 10
        self.pasta_videos = "videos_anomalias"
        if not os.path.exists(self.pasta_videos):
            os.makedirs(self.pasta_videos)

        # Threading
        self.analisando = False
        self.thread_analise = None

        # Atributos Flask
        self.ultimo_frame = None
        self.ultima_deteccao = None
        self.historico_videos = []
        self.frame_anterior = None
        self.rodando = False

        # OTIMIZAÇÃO: YOLO só a cada N frames
        self.contador_frames = 0
        self.intervalo_yolo = 5  # A cada 5 frames (mais rápido)
        self.boxes_cache = []
        self.num_pessoas = 0

        # Buffer para gravação (frames limpos)
        self.buffer_gravacao = deque(maxlen=60)

    def detectar_movimento(self, frame1, frame2, limiar_area=2000):
        """Detecta movimento significativo entre dois frames."""
        if frame1 is None or frame2 is None:
            return False
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contornos, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contornos:
            if cv2.contourArea(c) > limiar_area:
                return True
        return False

    def detectar_pessoas_yolo(self, frame):
        """Detecta pessoas com YOLO (chamado esporadicamente)"""
        results = self.yolo(frame, verbose=False)
        boxes_detectadas = []
        pessoas = 0

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == 0 and conf > 0.5:
                    pessoas += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes_detectadas.append({
                        'coords': (x1, y1, x2, y2),
                        'conf': conf
                    })

        return boxes_detectadas, pessoas

    def desenhar_boxes(self, frame, boxes):
        """Desenha boxes no frame (operação rápida)"""
        for i, box_info in enumerate(boxes, 1):
            x1, y1, x2, y2 = box_info['coords']
            conf = box_info['conf']

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f'P{i} {conf:.0%}'
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 18), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return frame

    def classificar_video(self, video_clip):
        """Classifica um clipe de vídeo usando VideoMAE."""
        inputs = self.processor(list(video_clip), return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            pred_idx = logits.argmax(-1).item()
            confianca = probs[0][pred_idx].item()
            label = self.model.config.id2label[pred_idx]

        return label.lower(), confianca

    def analisar_em_background(self, video_clip):
        """Analisa o vídeo em thread separada."""
        try:
            print("📸 Analisando movimento...")

            label, confianca = self.classificar_video(video_clip)

            print(f"🔎 IA: '{label}' ({confianca:.0%})")

            evento = None
            if any(palavra in label for palavra in ["fight", "punch", "kick", "hit"]):
                evento = "violencia_detectada"
            elif any(palavra in label for palavra in ["running", "jumping", "falling", "climbing"]):
                evento = "comportamento_suspeito"
            elif any(palavra in label for palavra in ["robbery", "burglary", "stealing"]):
                evento = "atividade_ilicita"

            self.ultima_deteccao = {
                'acao': label,
                'confianca': f"{confianca:.0%}",
                'evento': evento,
                'timestamp': datetime.now()
            }

            if evento:
                print(f"🚨 Evento: {evento}")
                self.iniciar_gravacao(evento)

        finally:
            self.analisando = False

    def iniciar_gravacao(self, evento_detectado):
        """Inicia gravação usando buffer de frames limpos."""
        if self.gravando or len(self.buffer_gravacao) == 0:
            return

        primeiro_frame = self.buffer_gravacao[0]
        altura, largura = primeiro_frame.shape[:2]

        timestamp = int(time.time())
        nome = f"{evento_detectado}_{timestamp}.mp4"
        caminho = os.path.join(self.pasta_videos, nome)

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        self.video_writer = cv2.VideoWriter(caminho, fourcc, 20.0, (largura, altura))

        if not self.video_writer.isOpened():
            print("❌ Erro ao criar vídeo")
            return

        self.gravando = True
        self.inicio_gravacao = time.time()

        self.historico_videos.append({
            'nome': nome,
            'caminho': caminho,
            'evento': evento_detectado,
            'timestamp': datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        })

        print(f"🎥 Gravando: {nome}")

        # Grava frames do buffer
        for frame in self.buffer_gravacao:
            self.video_writer.write(frame)

    def processar_frame(self):
        """Processa um frame (chamado pelo Flask)."""
        if not self.rodando:
            return None

        ret, frame_limpo = self.cap.read()
        if not ret:
            return None

        # Adiciona ao buffer de gravação
        self.buffer_gravacao.append(frame_limpo)

        # YOLO apenas a cada 5 frames
        self.contador_frames += 1
        if self.contador_frames % self.intervalo_yolo == 0:
            self.boxes_cache, self.num_pessoas = self.detectar_pessoas_yolo(frame_limpo)

        # Desenha boxes (rápido)
        frame_display = frame_limpo.copy()
        if self.boxes_cache:
            frame_display = self.desenhar_boxes(frame_display, self.boxes_cache)

        # Info no frame
        cv2.putText(frame_display, f"Pessoas: {self.num_pessoas}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if self.analisando:
            cv2.putText(frame_display, "ANALISANDO", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if self.gravando:
            tempo = int(self.duracao_gravacao - (time.time() - self.inicio_gravacao))
            cv2.putText(frame_display, f"REC {tempo}s", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Adiciona ao buffer de análise
        self.frame_buffer.append(frame_limpo)

        # Detecção de movimento
        if self.frame_anterior is not None:
            movimento = self.detectar_movimento(self.frame_anterior, frame_limpo)

            if (movimento and not self.gravando and
                    len(self.frame_buffer) == 16 and not self.analisando):
                self.analisando = True
                video_clip_copy = list(self.frame_buffer)

                self.thread_analise = threading.Thread(
                    target=self.analisar_em_background,
                    args=(video_clip_copy,),
                    daemon=True
                )
                self.thread_analise.start()

        # Gravação
        if self.gravando:
            self.video_writer.write(frame_limpo)

            if time.time() - self.inicio_gravacao >= self.duracao_gravacao:
                self.video_writer.release()
                self.gravando = False
                print("💾 Gravação finalizada")

        self.frame_anterior = frame_limpo.copy()
        self.ultimo_frame = frame_display

        return frame_display

    def iniciar(self):
        """Inicia o sistema."""
        print("🔵 Iniciando sistema...")

        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.video_source)
            time.sleep(0.5)

        self.rodando = True
        ret, frame = self.cap.read()

        if frame is not None:
            print("✅ Sistema iniciado!")
            for _ in range(16):
                self.frame_buffer.append(frame)
                self.buffer_gravacao.append(frame)
            self.frame_anterior = frame
        else:
            print("❌ Erro ao capturar frame")

    def parar(self):
        """Para o sistema."""
        print("🔴 Parando sistema...")
        self.rodando = False

        if self.gravando and self.video_writer:
            self.video_writer.release()
            self.gravando = False

        if self.cap.isOpened():
            self.cap.release()

    def get_historico_videos(self):
        """Retorna lista de vídeos gravados."""
        return sorted(self.historico_videos, key=lambda x: x['timestamp'], reverse=True)