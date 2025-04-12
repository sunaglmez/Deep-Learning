# Kapsül Ağları (Capsule Networks - CapsNet)

Standart bir Evrişimli Sinir Ağı’nda (CNN) özellik haritalarının boyutlarını azaltmak ve işlem gücünü düşürmek amacıyla **ortaklama (pooling)** katmanları kullanılır. Ancak bu katmanlar, her uygulamada nesne ile ilgili bazı önemli bilgileri kaybetmeye başlar.

Evrişimli Sinir Ağları, nesne sınıflandırma problemlerinde yüksek başarı göstermiş olsa da, günümüz problemlerinde **yetersiz** kalabilmektedir. Özellikle **nesne segmentasyonu** veya **nesne konum tespiti** gibi görevlerde başarı oranı düşer. Bunun nedeni, CNN’lerin nesnelerin konumları hakkında bilgi tutamamasıdır.

---

## 📌 Problemin Sonuçları Nelerdir?

CNN’ler:

- Bir nesnenin **pozisyonu**, **yönü** veya çevresindeki başka nesnelerle olan **ilişkisini** çıkarmakta başarısızdır.
- Yalnızca nesnenin **varlığı** ve onun sınır çizgileriyle ilgilenir.
- Giriş resmi üzerinde bulunan nesnelerin konumları değiştirildiğinde, bu bilgileri ağ boyunca **aktaracak mekanizma** eksiktir.

> Örneğin bir insan yüzünde göz, burun gibi organlar belirli konumlarda yer alır. Ancak bu organların yerlerini değiştirirsek, CNN yine aynı sınıflandırmayı yapabilir çünkü konum bilgisi modele aktarılmaz.

---

## 🧠 Pooling Katmanlarının Sınırlılığı

Pooling işlemleri (örneğin max pooling), ağın **konumsal hassasiyetini** azaltır. CNN'ler, farklı konumlara sahip aynı nesneleri **aynı şekilde** yorumlar. Bu da segmentasyon ve konum duyarlılığı gereken görevlerde zayıf performansa neden olur.

> “Evrişimli sinir ağlarında kullanılan ortaklama (pooling) işlemi büyük bir hataydı, çok iyi bir şekilde çalışması ise bir facia.”  
> — Geoffrey Hinton

---

## 🧩 Kapsül Nedir?

Kapsül, nesnenin yalnızca var olma olasılığını değil, aynı zamanda onunla ilgili:
- **Duruş (pose)**
- **Eğim**
- **Açı**
- **Konum**
- **Yön**
- **Kalınlık**
- **Ölçek (instantiation parameters)**

gibi çeşitli bilgileri içeren bir **nöron grubudur**.

---

## 🔄 Dynamic Routing (Yönlendirme Mekanizması)

Kapsül ağlarının temel fikri, bir sonraki kapsül katmanının çıkışını hesaplamak için **dinamik yönlendirme** (Dynamic Routing) mekanizması kullanmaktır.

- Kapsül çıkışlarının **vektör uzunlukları**, nesnenin var olma olasılığını temsil eder.
- ReLU yerine **squashing function** kullanılır:
  - Kısa vektörler 0’a yakın
  - Uzun vektörler 1’e yakın değer alır

### ✅ Squashing Function

```math
v_j = \frac{{||s_j||^2}}{{1 + ||s_j||^2}} \cdot \frac{{s_j}}{{||s_j||}}
```

## 🧱 CapsNet Mimarisi

CapsNet mimarisinde derinlik, klasik CNN’lerdeki gibi katmanları üst üste eklemekten ziyade, kapsüllerin **iç içe bağlanmasıyla** sağlanır.

### 🔧 Mimarinin Yapısı

1. **İlk Katman (Conv Layer)**  
   - Giriş resmine uygulanır.  
   - ReLU aktivasyon fonksiyonu ile 9x9x256 boyutunda özellik haritası üretilir.

2. **Primary Capsule Layer**  
   - 2 adımlı (stride = 2) ve 32 katmanlı yeni bir evrişim uygulanır.  
   - Çıkışlara diğer ağlardan farklı olarak **squashing** işlemi uygulanır.

3. **Routing by Agreement**  
   - Pooling yerine, hangi bilgilerin bir sonraki kapsüllere gönderileceğine **routing by agreement** ile karar verilir.  
   - Bu mekanizma, **gürültü (noise)** oluşturan bilgileri filtreler.

4. **Çıkış Katmanı**  
   - 10 adet (her bir rakam sınıfı için) 16 boyutlu vektör elde edilir.  
   - Bu vektörler ağın **tahmin çıktısını** temsil eder.

---

## 🎯 Avantajları

- Nesne **konumu**, **yönü**, **ölçeği** gibi bilgileri öğrenme kapasitesi
- Gürültüye karşı dayanıklı bilgi aktarımı
- Segmentasyon ve pozisyon duyarlılığı gereken görevlerde daha yüksek başarı

---

## 📎 İlgili Kaynaklar

- [Capsule Networks Paper (Hinton)](https://arxiv.org/abs/1710.09829)
- [Dynamic Routing Between Capsules - Paper](https://arxiv.org/abs/1710.09829)
- [CapsNet TensorFlow Implementation (GitHub)](https://github.com/naturomics/CapsNet-Tensorflow)

---

> Bu belge, Evrişimli Sinir Ağlarının sınırlılıkları ve bunlara karşı Kapsül Ağlarının getirdiği çözümleri açıklamaktadır. Özellikle **pozisyon-duyarlı görevlerde** kullanılabilecek güçlü bir mimari önerisidir.
