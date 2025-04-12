# CapsGNN (Capsule Graph Neural Network)

CapsGNN, grafik verisi üzerinde çalışan gelişmiş bir derin öğrenme modelidir. Capsule Network (CapsNet) ve Graph Neural Network (GNN) mimarilerinin birleşiminden doğmuştur.

---

## 1. Capsule Networks (CapsNet)

Capsule Networks, Geoffrey Hinton ve ekibi tarafından geliştirilen bir derin öğrenme modelidir. Geleneksel CNN'lere alternatif olarak geliştirilmiş olup, uzamsal bilgi ve nesne ilişkilerini daha iyi öğrenmeyi amaçlar.

### 🔹 Capsules (Kapsüller)

Kapsüller, bir grup nörondan oluşur ve nesnenin konumu, yönü, boyutu gibi önemli özellikleri daha etkili bir şekilde temsil eder. Bu yapılar nesnelerin daha anlamlı tanınmasını sağlar.

### 🔹 Dinamik Routing

CapsNet, CNN’lerin aksine sadece ileri propagasyon yapmaz. Katmanlar arasındaki etkileşimi sağlayarak kapsüllerin birbirleriyle iletişim kurmasını sağlar. Bu sayede model, daha az veriyle daha doğru sonuçlar verebilir.

---

## 2. Graph Neural Networks (GNN)

Graph Neural Networks, düğümler (nodes) ve kenarlardan (edges) oluşan grafik verilerini analiz edebilen yapay zeka modelleridir. Özellikle ilişkisel verilerde güçlü sonuçlar verir.

### 🔹 Grafikler

Gerçek dünyadaki birçok problem grafiklerle temsil edilebilir: sosyal ağlar, moleküler yapılar, yol ağları vb.

### 🔹 GNN Temelleri

GNN'ler, **message passing** (mesaj iletimi) yöntemiyle çalışır. Her düğüm, kendi özelliklerini ve komşu düğümlerden gelen bilgileri birleştirerek öğrenme sürecini gerçekleştirir.

---

## 📌 CapsGNN (Capsule GNN)

CapsGNN, CapsNet'in uzamsal öğrenme gücü ile GNN'nin bağlantısal analiz yeteneğini bir araya getirir.

### ✔️ Temel Avantajları

- **Grafik Verisiyle Çalışabilme**: Düğümler arasındaki karmaşık ilişkileri öğrenebilir.
- **Kapsül Temelli Öğrenme**: Daha az parametre ile daha anlamlı temsil öğrenme.
- **Uzamsal ve Bağlantısal Öğrenme**: Nesnelerin özellikleri ve grafik yapısı aynı anda öğrenilir.

---

## 🎯 Uygulama Alanları

- **Sosyal Ağ Analizi**: Kullanıcı etkileşimlerinin modellenmesi
- **Kimya ve Biyoloji**: Molekül yapılarının analizi
- **Yol Ağı ve Navigasyon**: Trafik optimizasyonu
- **Görüntü İşleme ve Bilgisayarla Görme**: Nesne tanıma ve segmentasyon

---

## 🧠 Derin Öğrenme ile İlişkisi

- **CapsGNN**, derin öğrenmenin bir alt alanıdır.
- **CapsNet**, CNN’lere alternatif olarak daha az veri ile daha güçlü öğrenme sunar.
- GNN'lerde her katman, düğümler arası mesajlaşmayla çalışır.
- **End-to-End Öğrenme** yapılabilir; tüm süreç geri besleme ile optimize edilir.

---

## 🛠️ Kurulum Talimatları

### 1. PyTorch ve torchvision Yükleme

```bash
pip install torch torchvision
