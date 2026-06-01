import cv2
import streamlit as st
import numpy as np

st.title("Projek Deteksi Wajah Otomatis")
st.write("Centang saklar di bawah untuk menyalakan kamera dan mendeteksi wajah secara otomatis!")

# Memanggil model wajah bawaan OpenCV
cascPath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

# Membuat SAKLAR AKTIFKAN KAMERA (Tanpa tombol potret)
run = st.checkbox('Nyalakan Kamera')

# Tempat untuk menampilkan video stream
FRAME_WINDOW = st.image([])

# Menggunakan webcam bawaan Streamlit secara looping otomatis jika saklar ON
camera = st.camera_input("Kamera Terhubung", label_visibility="collapsed")

if run and camera is not None:
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

    # Menghitung jumlah wajah
    jumlah_wajah = len(faces)
    st.success(f"Jumlah Wajah Terdeteksi: {jumlah_wajah}")

    # Menggambar KOTAK HIJAU di setiap wajah
    for (x, y, w, h) in faces:        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
       
    # Mengubah kembali format warna ke RGB agar pas ditampilkan di web
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Tampilkan hasil gambar yang sudah dikotaki ke layar web secara real-time
    FRAME_WINDOW.image(frame)
else:
    st.info("Kamera dinonaktifkan. Silakan centang 'Nyalakan Kamera' di atas.")
