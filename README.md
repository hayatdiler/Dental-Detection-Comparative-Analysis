# Comparative Analysis of RT-DETR, YOLOv8-L, and Faster R-CNN for Tooth Detection in Panoramic Radiographs

This repository contains the official implementation and research findings for automated tooth detection and classification using state-of-the-art Deep Learning architectures.

## 👥 Authors
* **Hayat DİLER** - Marmara University, Computer Engineering
* **Muhammed Enes YILDIRIR** - Marmara University, Computer Engineering
* **Emine Umay KILIÇ** - Marmara University, Computer Engineering

## 📝 Abstract
Panoramic radiography is a fundamental diagnostic tool in dentistry. However, manual tooth identification is time-consuming and prone to human error. This study evaluates the performance of **RT-DETR (Real-Time Detection Transformer)**, **YOLOv8-L**, and **Faster R-CNN** for automated tooth detection and classification across 8 different tooth classes (including both deciduous and permanent teeth).

## 📊 Dataset & Preprocessing
* **Dataset Size:** 968 high-quality panoramic X-ray images.
* **Labeling:** All images were meticulously annotated by dental experts following **FDI (Fédération Dentaire Internationale)** standards.
* **Classes (8 total):** * Permanent: Incisor, Canine, Premolar, Molar
  * Deciduous: Deciduous Incisor, Deciduous Canine, Deciduous Premolar, Deciduous Molar
* **Data Augmentation:** Techniques such as horizontal flipping, random brightness adjustment, and limited rotation were used to improve model generalization and prevent overfitting.

## 🛠️ Technical Specifications
### Hardware & Environment
* **GPU:** NVIDIA GPU optimized with **CUDA 11.8**
* **Framework:** PyTorch & Ultralytics
* **Input Size:** 640x640 pixels

### Training Hyperparameters
| Parameter | YOLOv8-L / RT-DETR | Faster R-CNN |
| :--- | :--- | :--- |
| Epochs | 50 | 50 |
| Batch Size | 4 | 4 |
| Optimizer | AdamW | SGD |

## 🏆 Quantitative Results
The models were evaluated based on the Mean Average Precision (mAP) metric at a 0.5 IoU threshold.

| Architecture | mAP@0.5 | Key Findings |
| :--- | :--- | :--- |
| **YOLOv8-L** | **94.6%** | Best overall performance and detection stability. |
| **RT-DETR** | **92.1%** | Strong global context awareness using Transformers. |
| **Faster R-CNN** | **77.95%** | Excellent for permanent teeth but struggles with deciduous classes. |

### Performance Highlights:
* YOLOv8-L achieved over **98% accuracy** in permanent tooth detection.
* RT-DETR showed highly competitive results in deciduous tooth classification due to its global context modeling.

## 🖼️ Qualitative Analysis
* **YOLOv8-L:** Demonstrated superior localization with bounding boxes tightly fitted to tooth contours.
* **RT-DETR:** Maintained high consistency in tooth numbering across the dental arch.
* **Faster R-CNN:** While accurate for individual teeth, it occasionally showed overlapping boxes in crowded posterior regions.

## 📚 References
1. Temur, K. T., et al. (2020). "Approach of Pediatric Dentists in Turkey to Panoramic Radiography."
2. Jader, G., et al. (2018). "Deep learning for automatic tooth segmentation and identification in panoramic images."
3. Tuzoff, D. V., et al. (2019). "Tooth detection and numbering on panoramic radiographs using convolutional neural networks."
4. Chen, H., et al. (2019). "A deep learning approach to automatic teeth detection and numbering based on object detection in dental periapical films."

# Panoramik Radyografilerde Diş Tespiti: RT-DETR, YOLOv8-L ve Faster R-CNN Mimarilerinin Karşılaştırmalı Analizi

Bu depo, güncel Derin Öğrenme mimarilerini kullanarak panoramik röntgenler üzerinde otomatik diş tespiti ve sınıflandırması yapan akademik çalışmamızın detaylarını ve kodlarını içermektedir.

## 👥 Yazarlar
* **Hayat DİLER** - Marmara Üniversitesi, Bilgisayar Mühendisliği
* **Muhammed Enes YILDIRIR** - Marmara Üniversitesi, Bilgisayar Mühendisliği
* **Emine Umay KILIÇ** - Marmara Üniversitesi, Bilgisayar Mühendisliği

## 📝 Özet
Panoramik radyografi, diş hekimliğinde temel bir tanı aracıdır; ancak dişlerin manuel olarak tespiti zaman alıcıdır ve klinisyen yorgunluğuna bağlı hatalara açıktır. Bu çalışma; diş tespiti ve sınıflandırılması süreçlerinde **RT-DETR (Real-Time Detection Transformer)**, **YOLOv8-L** ve **Faster R-CNN** mimarilerinin performanslarını 8 farklı diş sınıfı (süt ve daimî dişler dahil) üzerinden karşılaştırmalı olarak incelemektedir.

## 📊 Veri Seti ve Ön İşleme
* **Veri Seti Boyutu:** 968 adet yüksek çözünürlüklü panoramik röntgen görüntüsü.
* **Etiketleme:** Veriler, diş hekimliği uzmanları tarafından **FDI (Fédération Dentaire Internationale)** standartlarına uygun şekilde etiketlenmiştir.
* **Sınıflar (Toplam 8):**
  * Daimî: Kesici Diş, Köpek Dişi, Küçük Azı, Büyük Azı
  * Süt: Süt Kesici, Süt Köpek, Süt Küçük Azı, Süt Büyük Azı
* **Veri Artırma:** Yatay çevirme, rastgele parlaklık ayarı ve sınırlı rotasyon teknikleri kullanılarak modelin genelleme kapasitesi artırılmış ve aşırı öğrenme (overfitting) önlenmiştir.

## 🛠️ Teknik Özellikler
### Donanım ve Ortam
* **GPU:** **CUDA 11.8** ile optimize edilmiş NVIDIA GPU
* **Kütüphaneler:** PyTorch & Ultralytics
* **Görüntü Boyutu:** 640x640 piksel

### Eğitim Parametreleri
| Parametre | YOLOv8-L / RT-DETR | Faster R-CNN |
| :--- | :--- | :--- |
| Epoch | 50 | 50 |
| Batch Size | 4 | 4 |
| Optimizer | AdamW | SGD |

## 🏆 Nicel Sonuçlar
Modeller, 0.5 IoU eşiğinde Ortalama Hassasiyet (mAP) metriğine göre değerlendirilmiştir.

| Mimari | mAP@0.5 | Temel Bulgular |
| :--- | :--- | :--- |
| **YOLOv8-L** | **%94.6** | En yüksek genel başarı ve tespit kararlılığı. |
| **RT-DETR** | **%92.1** | Transformer mekanizması ile güçlü küresel bağlam öğrenimi. |
| **Faster R-CNN** | **%77.95** | Daimî dişlerde başarılı ancak süt dişlerinde performans kaybı. |

### Öne Çıkan Başarılar:
* YOLOv8-L, daimî diş sınıflarında **%98'in üzerinde** tespit başarısı elde etmiştir.
* RT-DETR, Transformer yapısı sayesinde süt dişlerinin sınıflandırılmasında oldukça rekabetçi sonuçlar vermiştir.

## 🖼️ Nitel Gözlemler
* **YOLOv8-L:** Sınırlayıcı kutuların (bounding box) diş konturlarına en yakın ve net yerleştiği model olmuştur.
* **RT-DETR:** Dişlerin çene arkı üzerindeki birbirleriyle olan konumsal ilişkilerini daha düzenli yakalamıştır.
* **Faster R-CNN:** Arka bölgelerdeki dişlerin birbirine çok yakın olduğu durumlarda kutuların zaman zaman örtüştüğü gözlemlenmiştir.

## 📚 Kaynakça
1. Temur, K. T., ve ark. (2020). "Türkiye’de Çocuk Diş Hekimlerinin Konik Işınlı Bilgisayarlı Tomografi Kullanımına Yaklaşımı."
2. Jader, G., ve ark. (2018). "Deep learning for automatic tooth segmentation and identification in panoramic images."
3. Tuzoff, D. V., ve ark. (2019). "Tooth detection and numbering on panoramic radiographs using convolutional neural networks."
4. Chen, H., ve ark. (2019). "A deep learning approach to automatic teeth detection and numbering based on object detection in dental periapical films."
