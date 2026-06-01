import cv2
import streamlit as st
import numpy as np

st.title("Projek Deteksi Wajah")
st.write("Silakan ambil foto di bawah, hasil deteksi kotak hijau akan langsung muncul otomatis!")

# Memanggil model wajah bawaan OpenCV
cascPath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

# HANYA ADA SATU KAMERA RESMI DI SINI
camera = st.camera_input("Klik tombol di bawah untuk potret wajah")

if camera is not None:
    # Mengubah format gambar dari browser agar bisa dibaca OpenCV
    file_bytes = np.asarray(bytearray(camera.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    
    # Mengubah ke abu-abu untuk proses deteksi AI
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Proses deteksi wajah
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Menghitung jumlah wajah
    jumlah_wajah = len(faces)
    st.success(f"Jumlah Wajah Terdeteksi: {jumlah_wajah}")

    # Menggambar KOTAK HIJAU di setiap wajah yang ketemu
    for (x, y, w, h) in faces:        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
       
    # Mengubah kembali format warna ke RGB agar pas ditampilkan di web
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # LANGSUNG MENAMPILKAN HASIL DETEKSI YANG SUDAH ADA KOTAK HIJAUNYA
    st.image(frame, caption="Hasil Analisis AI", use_container_width=True)
