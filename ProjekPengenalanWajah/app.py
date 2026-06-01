import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("Projek Deteksi Wajah Real-time (LIVE)")
st.write("Aplikasi akan mendeteksi wajahmu secara otomatis dari video kamera.")

# Memanggil model wajah bawaan OpenCV
cascPath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        # Mengubah format gambar dari video stream ke bentuk array OpenCV
        img = frame.to_ndarray(format="bgr24")
        
        # Mengubah ke abu-abu untuk proses deteksi
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Proses deteksi wajah
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Menghitung jumlah wajah
        jumlah_wajah = len(faces)

        # Menggambar KOTAK HIJAU di setiap wajah
        for (x, y, w, h) in faces:        
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
        # Menampilkan teks jumlah wajah langsung melayang di atas video stream
        cv2.putText(img, f"Jumlah Wajah: {jumlah_wajah}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        return img

# Menjalankan modul kamera live video di halaman web
webrtc_streamer(
    key="deteksi-wajah-live", 
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False}
)
