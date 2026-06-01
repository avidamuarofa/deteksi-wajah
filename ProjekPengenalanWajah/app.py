import cv2
import streamlit as st
import numpy as np

# Judul utama di halaman Website
st.title("Projek Deteksi Wajah Real-time")
st.write("Aplikasi ini mendeteksi wajah langsung melalui kamera HP/Laptop kamu.")

# Memanggil model wajah bawaan OpenCV
cascPath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

# Mengambil input kamera langsung dari browser HP/Laptop
camera = st.camera_input("Ambil Foto untuk Deteksi Wajah")

if camera is not None:
    # Mengubah format gambar dari browser agar bisa dibaca OpenCV
    file_bytes = np.asarray(bytearray(camera.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    
    # Mengubah ke abu-abu untuk proses deteksi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Proses deteksi wajah
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # NOMOR 1: Menghitung jumlah wajah
    jumlah_wajah = len(faces)
    st.success(f"Jumlah Wajah Terdeteksi: {jumlah_wajah}")

    # NOMOR 3: Menggambar KOTAK HIJAU di setiap wajah
    for (x, y, w, h) in faces:        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
       
    # Mengubah kembali format warna ke RGB agar pas ditampilkan di web
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Tampilkan hasil foto yang sudah diberi kotak hijau ke halaman web
    st.image(frame, caption="Hasil Deteksi Wajah", use_container_width=True)
