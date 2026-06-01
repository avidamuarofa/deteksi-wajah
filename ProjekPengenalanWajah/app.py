import cv2

# Menggunakan model wajah bawaan OpenCV langsung
cascPath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

# Membuka kamera utama
video_capture = cv2.VideoCapture(0)

while True:
    # Ambil gambar frame demi frame
    rect, frame = video_capture.read()    
    
    # Antisipasi jika kamera macet
    if not rect or frame is None:
        print("Peringatan: Kamera gagal merespon.")
        break
        
    # Mengubah ke abu-abu untuk proses deteksi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Proses deteksi wajah
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # NOMOR 1: Menghitung jumlah wajah yang terdeteksi
    # len(faces) akan menghitung jumlah kotak wajah yang ditemukan oleh OpenCV
    jumlah_wajah = len(faces)
    
    # Menampilkan teks jumlah wajah di layar kamera (Pojok kiri atas)
    # Parameter: (frame, teks, koordinat_xy, jenis_font, ukuran_font, warna_BGR, ketebalan)
    cv2.putText(frame, f"Jumlah Wajah: {jumlah_wajah}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # NOMOR 3: Menggambar KOTAK HIJAU di setiap wajah yang terdeteksi
    for (x, y, w, h) in faces:        
        # Menggunakan cv2.rectangle dengan warna hijau terang (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
       
    # Menampilkan hasil video
    cv2.imshow('Projek Deteksi Wajah - Tekan Q untuk Keluar', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()