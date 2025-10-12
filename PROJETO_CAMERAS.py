import cv2
import time
import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from PIL import Image
import numpy as np
import os
from collections import deque  # ### MUDANÇA 1: Importar deque para criar o buffer ###


class DetectorAnomalias:
    def __init__(self, video_source=0):
        self.cap = cv2.VideoCapture(video_source)

        # ### MUDANÇA 2: Unificar para um único e poderoso modelo de VÍDEO ###
        print("🔄 Carregando modelo de análise de vídeo do Hugging Face...")
        # Usaremos o VideoMAE treinado no Kinetics-400, que conhece centenas de ações.
        model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_name).to(self.device)

        print(f"✅ Modelo carregado com sucesso em '{self.device}'.\n")

        # ### MUDANÇA 3: Criar um buffer para acumular frames para o clipe ###
        # O modelo espera um clipe de 16 frames.
        self.frame_buffer = deque(maxlen=16)

        # Parâmetros de gravação (sem alterações)
        self.gravando = False
        self.inicio_gravacao = None
        self.video_writer = None
        self.duracao_gravacao = 10  # segundos
        self.pasta_videos = "videos_anomalias"
        if not os.path.exists(self.pasta_videos):
            os.makedirs(self.pasta_videos)

    def detectar_movimento(self, frame1, frame2, limiar_area=2000):
        """Detecta movimento significativo entre dois frames. (Sem alterações)"""
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

    # ### MUDANÇA 4: Função de classificação agora processa um VÍDEO (lista de frames) ###
    def classificar_video(self, video_clip):
        """Classifica um clipe de vídeo (lista de frames) usando o modelo VideoMAE."""
        # Prepara o vídeo para o modelo
        inputs = self.processor(list(video_clip), return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            pred_idx = logits.argmax(-1).item()
            label = self.model.config.id2label[pred_idx]

        return label.lower()

    def iniciar_gravacao(self, frame, evento_detectado):
        """Inicia gravação do vídeo. (Adicionado o nome do evento ao arquivo)"""
        altura, largura = frame.shape[:2]
        timestamp = int(time.time())
        # Adiciona o tipo de evento ao nome do arquivo para fácil identificação
        nome = f"{evento_detectado.replace(' ', '_')}_{timestamp}.mp4"
        caminho = os.path.join(self.pasta_videos, nome)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(caminho, fourcc, 20.0, (largura, altura))
        self.gravando = True
        self.inicio_gravacao = time.time()
        print(f"🎥 Gravando vídeo: {nome}")

        # Salva também os frames do buffer que levaram à detecção
        for buffered_frame in self.frame_buffer:
            self.video_writer.write(buffered_frame)

    def processar(self):
        """Loop principal com a nova lógica de buffer."""
        ret, frame_anterior = self.cap.read()

        if frame_anterior is not None:
            # Inicializa o buffer com o primeiro frame
            for _ in range(16):
                self.frame_buffer.append(frame_anterior)

        while True:
            ret, frame_atual = self.cap.read()
            if not ret:
                break

            # Adiciona o frame atual ao buffer de forma contínua
            self.frame_buffer.append(frame_atual)

            movimento = self.detectar_movimento(frame_anterior, frame_atual)

            # ### MUDANÇA 5: Lógica de análise integrada ao buffer ###
            # Se detectou movimento, não está gravando E o buffer está cheio
            if movimento and not self.gravando and len(self.frame_buffer) == 16:
                print("📸 Movimento detectado, enviando clipe para análise da IA...")

                # Classifica o clipe inteiro que está no buffer
                label = self.classificar_video(self.frame_buffer)

                print(f"🔎 IA detectou a ação: '{label}'")

                evento = None
                # Condições de gravação baseadas na ação detectada
                # As palavras-chave são baseadas nas classes do dataset Kinetics
                if any(palavra in label for palavra in ["fight", "punch", "kick", "hit"]):
                    evento = "violencia detectada"
                elif any(palavra in label for palavra in ["running", "jumping", "falling", "climbing"]):
                    evento = "comportamento suspeito"
                elif any(palavra in label for palavra in ["robbery", "burglary", "stealing"]):
                    evento = "atividade ilicita"

                if evento:
                    print(f"🚨 Evento anômalo confirmado: {evento}")
                    self.iniciar_gravacao(frame_atual, evento)

            if self.gravando:
                self.video_writer.write(frame_atual)
                if time.time() - self.inicio_gravacao >= self.duracao_gravacao:
                    self.video_writer.release()
                    self.gravando = False
                    print("💾 Gravação finalizada.")

            cv2.imshow("Detecção Inteligente", frame_atual)
            frame_anterior = frame_atual.copy()

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("🛑 Sistema encerrado.")


if __name__ == "__main__":
    # Use 0 para webcam ou "caminho/para/video.mp4" para um arquivo
    detector = DetectorAnomalias(video_source=0)
    detector.processar()