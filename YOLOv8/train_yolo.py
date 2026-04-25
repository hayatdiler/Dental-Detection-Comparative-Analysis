import os
import shutil
import json
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import sys

# --- 1. AYARLAR VE SABİTLER ---

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_IMAGE_DIR = os.path.join(PROJECT_ROOT, 'Radiographs')
RAW_JSON_PATH = os.path.join(PROJECT_ROOT, 'Segmentation', 'teeth_bbox.json')
YOLO_DATASET_DIR = os.path.join(PROJECT_ROOT, 'yolo_dataset')
DATA_YAML_PATH = os.path.join(PROJECT_ROOT, 'data.yaml')

# --- SINIF HARİTALAMASI (52 -> 8) ---
# Senin verdiğin yapıya birebir uygun harita
TOOTH_TYPE_MAP = {
    # --- Kalıcı dişler (1-32) ---
    '1': 0, '2': 0,  # 11-12: Üst sağ kesici -> Kesici
    '3': 1,  # 13: Üst sağ köpek -> Köpek
    '4': 2, '5': 2,  # 14-15: Üst sağ premolar -> Küçük Azı
    '6': 3, '7': 3, '8': 3,  # 16-18: Üst sağ molar -> Büyük Azı

    '9': 0, '10': 0,  # 21-22: Üst sol kesici
    '11': 1,  # 23: Üst sol köpek
    '12': 2, '13': 2,  # 24-25: Üst sol premolar
    '14': 3, '15': 3, '16': 3,  # 26-28: Üst sol molar

    '17': 0, '18': 0,  # 31-32: Alt sol kesici
    '19': 1,  # 33: Alt sol köpek
    '20': 2, '21': 2,  # 34-35: Alt sol premolar
    '22': 3, '23': 3, '24': 3,  # 36-38: Alt sol molar

    '25': 0, '26': 0,  # 41-42: Alt sağ kesici
    '27': 1,  # 43: Alt sağ köpek
    '28': 2, '29': 2,  # 44-45: Alt sağ premolar
    '30': 3, '31': 3, '32': 3,  # 46-48: Alt sağ molar

    # --- Süt dişleri (A-T) ---
    'A': 4, 'B': 4,  # A-B: Üst sağ süt kesici
    'C': 5,  # C: Üst sağ süt köpek
    'D': 6, 'E': 7,  # D: Süt Küçük Azı, E: Süt Büyük Azı (Senin şemana göre)

    'F': 4, 'G': 4,  # F-G: Üst sol süt kesici
    'H': 5,  # H: Üst sol süt köpek
    'I': 6, 'J': 7,  # I: Süt Küçük Azı, J: Süt Büyük Azı

    'K': 4, 'L': 4,  # K-L: Alt sol süt kesici
    'M': 5,  # M: Alt sol süt köpek
    'N': 6, 'O': 7,  # N: Süt Küçük Azı, O: Süt Büyük Azı

    'P': 4, 'Q': 4,  # P-Q: Alt sağ süt kesici
    'R': 5,  # R: Alt sağ süt köpek
    'S': 6, 'T': 7,  # S: Süt Küçük Azı, T: Süt Büyük Azı
}

# YOLO için Sınıf İsimleri (Sırası ID'lerle eşleşmeli: 0, 1, 2... 7)
NEW_CLASS_NAMES = [
    'Kesici_Daimi',  # 0
    'Kopek_Daimi',  # 1
    'Kucuk_Azi_Daimi',  # 2
    'Buyuk_Azi_Daimi',  # 3
    'Sut_Kesici',  # 4
    'Sut_Kopek',  # 5
    'Sut_Kucuk_Azi',  # 6
    'Sut_Buyuk_Azi'  # 7
]


# --- 2. YARDIMCI FONKSİYONLAR ---

def find_image_path_fix(base_dir, json_image_id):
    base_name, _ = os.path.splitext(json_image_id)
    possible_extensions = ['.JPG', '.JPEG', '.PNG', '.TIF', '.TIFF', '.jpg', '.jpeg', '.png', '.tif', '.tiff']
    for ext in possible_extensions:
        full_path = os.path.join(base_dir, base_name + ext)
        if os.path.exists(full_path):
            return base_name + ext
    return None


def get_yolo_class_from_map(tooth_label):
    """
    JSON'dan gelen etiketi (örn: '1', '32', 'A') sözlükten bulup
    0-7 arasındaki sınıf ID'sini döndürür.
    """
    # Gelen veriyi temizle (string yap, boşluk sil, büyük harf yap)
    key = str(tooth_label).strip().upper()

    # Sözlükten bak
    if key in TOOTH_TYPE_MAP:
        return TOOTH_TYPE_MAP[key]
    else:
        return None


def process_data():
    print("--- Veri Hazırlığı Başlıyor (Özel 8 Sınıf Haritası) ---")
    if not os.path.exists(RAW_JSON_PATH):
        print(f"HATA: JSON dosyası bulunamadı: {RAW_JSON_PATH}")
        sys.exit(1)

    # Klasörü temizle
    if os.path.exists(YOLO_DATASET_DIR):
        shutil.rmtree(YOLO_DATASET_DIR)

    for split in ['train', 'val']:
        os.makedirs(os.path.join(YOLO_DATASET_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DATASET_DIR, 'labels', split), exist_ok=True)

    with open(RAW_JSON_PATH, 'r') as f:
        data = json.load(f)

    all_valid_ids = []
    valid_image_map = {}
    json_ids = [item.get("External ID") for item in data if item.get("External ID")]

    print("Dosyalar taranıyor...")
    for json_id in json_ids:
        disk_name = find_image_path_fix(RAW_IMAGE_DIR, json_id)
        if disk_name:
            valid_image_map[json_id] = disk_name
            all_valid_ids.append(json_id)

    if not all_valid_ids:
        print("KRİTİK HATA: Hiçbir görüntü dosyası eşleştirilemedi!")
        sys.exit(1)

    train_ids, val_ids = train_test_split(all_valid_ids, test_size=0.2, random_state=42)
    split_map = {id: 'train' for id in train_ids}
    split_map.update({id: 'val' for id in val_ids})

    processed_count = 0
    for item in data:
        json_id = item.get("External ID")
        if json_id not in split_map: continue

        split = split_map[json_id]
        disk_name = valid_image_map[json_id]

        src = os.path.join(RAW_IMAGE_DIR, disk_name)
        dst = os.path.join(YOLO_DATASET_DIR, 'images', split, disk_name)
        shutil.copy(src, dst)

        try:
            with Image.open(src) as img:
                img_w, img_h = img.size
        except:
            continue

        yolo_lines = []
        objects = item.get("Label", {}).get("objects", [])

        for obj in objects:
            tooth_num = str(obj.get("title"))  # JSON'daki 'title' alanı (1, 32, A, B...)
            bbox = obj.get("bounding box")

            # Haritadan sınıfı bul
            cls_id = get_yolo_class_from_map(tooth_num)

            if (cls_id is not None and isinstance(bbox, list) and len(bbox) == 4):

                # --- DOĞRU KOORDİNAT SİSTEMİ (ymin, xmin, ymax, xmax) ---
                # Önceki başarıyı sağlayan düzeltme burada korunuyor.
                y_min, x_min, y_max, x_max = bbox

                if (x_max - x_min) > 1 and (y_max - y_min) > 1:
                    # YOLO Formatı Hesaplama
                    x_c = ((x_min + x_max) / 2) / img_w
                    y_c = ((y_min + y_max) / 2) / img_h
                    w = (x_max - x_min) / img_w
                    h = (y_max - y_min) / img_h

                    # Clipping
                    x_c = max(0.001, min(0.999, x_c))
                    y_c = max(0.001, min(0.999, y_c))
                    w = max(0.001, min(0.999, w))
                    h = max(0.001, min(0.999, h))

                    yolo_lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

        label_name = os.path.splitext(disk_name)[0] + '.txt'
        label_path = os.path.join(YOLO_DATASET_DIR, 'labels', split, label_name)
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        processed_count += 1

    print(f"Veri Hazırlığı Tamamlandı. Toplam {processed_count} görüntü işlendi.")


def create_yaml():
    train_path = os.path.join(YOLO_DATASET_DIR, 'images', 'train')
    val_path = os.path.join(YOLO_DATASET_DIR, 'images', 'val')

    content = f"""
train: {train_path}
val: {val_path}
nc: {len(NEW_CLASS_NAMES)}
names: {NEW_CLASS_NAMES}
"""
    with open(DATA_YAML_PATH, 'w') as f:
        f.write(content)
    print("data.yaml oluşturuldu.")


def train_model():
    print("\n--- YENİ 8 SINIFLI (52->8) EĞİTİM BAŞLATILIYOR (RTX 5080) ---")

    model_weights = 'yolov8l.pt'

    model = YOLO(model_weights)

    results = model.train(
        data=DATA_YAML_PATH,
        epochs=50,
        imgsz=640,
        batch=4,
        rect=True,
        project='runs/train',
        name='dis_tespit_52to8_final',  # İsim
        workers=0,
        cache=False
    )
    print("Eğitim Tamamlandı!")


if __name__ == '__main__':
    process_data()
    create_yaml()
    train_model()