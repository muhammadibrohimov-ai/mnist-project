MNIST-Project

Oddiy konvolyutsion neyron tarmoq yordamida handwriting (qo‘l bilan yozilgan) raqamlarni tanish loyihasi.

📌 Maqsad

MNIST ma’lumotlar to‘plamidagi qo‘l bilan yozilgan raqamlarni tasniflash uchun neyron tarmoq yaratish.

Modelni o‘qitish (training) va keyin yangi rasmlar bilan test qilish (prediction) imkonini taqdim etish.

Loyihani kodni tushunish oson tarzda tuzish hamda predict.py orqali real vaqt (yoki oldindan tayyorlangan rasm) bilan natijalar ko‘rish.

🧰 Loyihaning tuzilishi
mnist-project/
├── images/                  # Qo‘l bilan yozilgan raqamlar namunasi rasmlar
├── .gitignore  
├── drawing_image.py         # Rasm chizish/yaratish uchun skript  
├── predict.py               # Tayyor model bilan bashorat qilish skripti  
├── requirements.txt         # Loyihaga kerakli Python kutubxonalar  
├── simplecnn_state.pth      # O‘qitilgan model og‘irliklari fayli  
└── train.ipynb              # Jupyter Notebook: modelni o‘qitish bo‘yicha kod  

🚀 Ishga tushirish

Quyidagi bosqichlarni bajarish orqali loyiha bilan ishlashingiz mumkin:

Loyihani klon qilish:

git clone https://github.com/muhammadibrohimov-ai/mnist-project.git
cd mnist-project


Kerakli kutubxonalarni o‘rnatish:

pip install -r requirements.txt


(Majburiy emas, faqat agar siz modelni qayta o‘qitmoqchi bo‘lsangiz) train.ipynb faylini ochib, trening jarayonini bajarish.

Modelni ishlatish uchun:

python predict.py


Bu skript sizdan rasm faylini so‘rashi mumkin yoki drawing_image.py orqali oddiy rasm chizish imkoniyatini beradi.

🧠 Texnik tafsilotlari

Model arxitekturasi: Konvolyutsion neyron tarmoq (CNN).

Ma’lumotlar: MNIST dataset — 28×28 piksel, 10 sinf (0-9 raqamlar).

Kutubxonalar: Python, PyTorch (yoki boshqa foydalanilgan kutubxona) (buni requirements.txt faylida tekshirishingiz mumkin).

Model og‘irliklari simplecnn_state.pth faylida saqlanadi.

📈 Natijalar

Treningdan so‘ng model o‘rganilgan va bashorat qila oladigan holatda.

predict.py orqali siz chizilgan raqamni model yordamida aniqlay olasiz.

Loyihada klassik “hello world” darajasidagi raqamlarni tanish demo amalga oshirilgan.

✅ Qanday foydalanish mumkin

Ta’lim maqsadida: CNN strukturasini o‘rganish, PyTorch bilan tajriba qilish.

Loyihani kengaytirish: boshqa ma’lumotlar to‘plami bilan qo‘llash (masalan: EMNIST), model arxitekturasini chuqurlashtirish, mobil ilovaga integratsiya qilish.

Illyustratsiya yoki vizualizatsiya ko‘nikmalarini kengaytirish: train jarayonidagi grafiklar, bashorat natijalarini ko‘rsatish.

🧩 Kelajakdagi rejalar

Modelni mobil qurilmaga eksport qilish (ONNX yoki TensorFlow Lite).

Veb-ilova tarzida chizish va real vaqtli tanish imkoniyati.

Yana murakkab raqamlar yoki qo‘l yozuvi bilan ishlaydigan datasetlar bilan tajriba (masalan: lotin harflari, alohida yozuvlar).

Foydalanuvchi interfeysi qo‘shish (grafik UI yoki web front-end).

📄 Litsenziya

Bu loyiha ochiq manbali bo‘lib, xohlagan maqsadda foydalanish, o‘zgartirish va tarqatish mumkin.
