#pip install opencv-python-headless matplotlib seaborn pandas scipy tqdm
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
import shutil

class FacialDatasetPreprocessor:
    """
    Preprocesamiento completo para dataset de imágenes faciales:
    - Detección y recorte facial
    - Normalización
    - Data Augmentation
    - División en train/val/test
    """

    def __init__(self, input_path, output_path, target_size=(224, 224),
                 detection_method='dnn'):
        """
        Args:
            input_path: Ruta al dataset original
            output_path: Ruta donde guardar el dataset procesado
            target_size: Tamaño objetivo (ancho, alto) en píxeles
            detection_method: 'haar', 'dnn', o 'dlib'
                - 'haar': Rápido pero menos preciso
                - 'dnn': Más preciso, velocidad media (RECOMENDADO)
                - 'dlib': Muy preciso pero más lento
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.target_size = target_size
        self.detection_method = detection_method

        # Configurar detector según el método
        if detection_method == 'haar':
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print("🔧 Usando detector: Haar Cascade (rápido, menos preciso)")

        elif detection_method == 'dnn':
            # Detector DNN de OpenCV (mucho más preciso)
            print("🔧 Usando detector: DNN (preciso, velocidad media) - RECOMENDADO")
            print("📥 Descargando modelos DNN...")
            # Descargar modelo si no existe
            model_file = "res10_300x300_ssd_iter_140000.caffemodel"
            config_file = "deploy.prototxt"

            if not Path(model_file).exists():
                import urllib.request
                print("   Descargando modelo...")
                model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
                urllib.request.urlretrieve(model_url, model_file)

            if not Path(config_file).exists():
                config_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
                urllib.request.urlretrieve(config_url, config_file)

            self.face_net = cv2.dnn.readNetFromCaffe(config_file, model_file)
            print("   ✅ Modelo DNN cargado")

        elif detection_method == 'dlib':
            try:
                import dlib
                self.face_detector = dlib.get_frontal_face_detector()
                print("🔧 Usando detector: Dlib (muy preciso, más lento)")
            except ImportError:
                print("⚠️  Dlib no instalado. Usa: !pip install dlib")
                print("   Cambiando a método DNN...")
                self.detection_method = 'dnn'
                self.__init__(input_path, output_path, target_size, 'dnn')
                return

        # Estadísticas
        self.stats = {
            'total_processed': 0,
            'faces_detected': 0,
            'faces_not_detected': 0,
            'augmented_images': 0,
            'discarded_images': [],
            'low_confidence_detections': []
        }

    def detect_and_crop_face(self, image, margin=0.2, confidence_threshold=0.5):
        """
        Detecta y recorta el rostro de una imagen con múltiples métodos

        Args:
            image: Imagen en formato numpy array (BGR)
            margin: Margen adicional alrededor del rostro (20% por defecto)
            confidence_threshold: Umbral de confianza para DNN (0.5 por defecto)

        Returns:
            Tupla (imagen_rostro, confianza) o (None, 0) si no se detecta
        """
        h, w = image.shape[:2]

        if self.detection_method == 'haar':
            # Método Haar Cascade (original)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            if len(faces) == 0:
                return None, 0

            # Tomar el rostro más grande
            if len(faces) > 1:
                faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

            x, y, w_face, h_face = faces[0]
            confidence = 1.0  # Haar no da confianza, asumimos 1.0

        elif self.detection_method == 'dnn':
            # Método DNN (más preciso)
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)),
                1.0,
                (300, 300),
                (104.0, 177.0, 123.0)
            )

            self.face_net.setInput(blob)
            detections = self.face_net.forward()

            # Buscar la detección con mayor confianza
            best_confidence = 0
            best_box = None

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > confidence_threshold and confidence > best_confidence:
                    best_confidence = confidence
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    best_box = box.astype("int")

            if best_box is None:
                return None, 0

            x, y, x2, y2 = best_box
            w_face = x2 - x
            h_face = y2 - y
            confidence = best_confidence

        elif self.detection_method == 'dlib':
            # Método Dlib
            import dlib
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray, 1)

            if len(faces) == 0:
                return None, 0

            # Tomar el rostro más grande
            if len(faces) > 1:
                faces = sorted(faces, key=lambda f: f.width() * f.height(), reverse=True)

            face = faces[0]
            x, y = face.left(), face.top()
            w_face, h_face = face.width(), face.height()
            confidence = 1.0  # Dlib no da confianza directa

        # Validaciones adicionales para filtrar falsos positivos

        # 1. Verificar proporciones razonables de un rostro (entre 0.7 y 1.5)
        aspect_ratio = w_face / h_face
        if aspect_ratio < 0.7 or aspect_ratio > 1.5:
            return None, 0

        # 2. Verificar que el rostro no sea demasiado pequeño (mínimo 10% del área)
        face_area = w_face * h_face
        image_area = h * w
        if face_area < 0.05 * image_area:
            return None, 0

        # 3. Verificar que el rostro no sea demasiado grande (máximo 80% del área)
        if face_area > 0.8 * image_area:
            return None, 0

        # 4. Verificar posición (debe estar razonablemente centrado)
        center_x = x + w_face // 2
        center_y = y + h_face // 2
        if center_x < w * 0.1 or center_x > w * 0.9 or center_y < h * 0.1 or center_y > h * 0.9:
            return None, 0

        # Agregar margen
        margin_x = int(w_face * margin)
        margin_y = int(h_face * margin)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + w_face + margin_x)
        y2 = min(h, y + h_face + margin_y)

        # Recortar rostro
        face = image[y1:y2, x1:x2]

        return face, confidence

    def resize_and_normalize(self, image):
        """
        Redimensiona la imagen al tamaño objetivo y normaliza píxeles a [0, 1]

        Args:
            image: Imagen en formato numpy array

        Returns:
            Imagen redimensionada y normalizada
        """
        # Redimensionar
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

        # Normalizar a [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        return normalized

    def augment_image(self, image):
        """
        Aplica data augmentation a una imagen

        Args:
            image: Imagen normalizada [0, 1]

        Returns:
            Lista de imágenes aumentadas
        """
        augmented = []
        h, w = image.shape[:2]

        # 1. Flip horizontal
        flipped = cv2.flip(image, 1)
        augmented.append(('flip', flipped))

        # 2. Rotación leve (-15 a +15 grados)
        for angle in [-10, 10]:
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            augmented.append((f'rot{angle}', rotated))

        # 3. Zoom (crop + resize)
        crop_percent = 0.9  # 90% del tamaño original
        crop_w = int(w * crop_percent)
        crop_h = int(h * crop_percent)
        start_x = (w - crop_w) // 2
        start_y = (h - crop_h) // 2
        cropped = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
        zoomed = cv2.resize(cropped, (w, h))
        augmented.append(('zoom', zoomed))

        # 4. Ajuste de brillo
        brightened = np.clip(image * 1.2, 0, 1)
        augmented.append(('bright', brightened))

        darkened = np.clip(image * 0.8, 0, 1)
        augmented.append(('dark', darkened))

        # 5. Shear (deformación leve)
        pts1 = np.float32([[5, 5], [w-5, 5], [5, h-5]])
        pts2 = np.float32([[0, 0], [w, 5], [5, h]])
        M = cv2.getAffineTransform(pts1, pts2)
        sheared = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        augmented.append(('shear', sheared))

        return augmented

    def process_dataset(self, apply_augmentation=True):
        """
        Procesa todo el dataset: detección, recorte, normalización y augmentation

        Args:
            apply_augmentation: Si aplicar data augmentation
        """
        print("=" * 60)
        print("🔧 INICIANDO PREPROCESAMIENTO")
        print("=" * 60)
        print(f"📂 Input: {self.input_path}")
        print(f"📂 Output: {self.output_path}")
        print(f"📏 Tamaño objetivo: {self.target_size}")
        print(f"🔄 Data Augmentation: {'Sí' if apply_augmentation else 'No'}")
        print()

        # Crear directorio temporal para imágenes procesadas
        temp_path = self.output_path / 'temp_processed'
        temp_path.mkdir(parents=True, exist_ok=True)

        # Obtener todas las carpetas de personas usando os para evitar caché
        input_str = str(self.input_path)
        person_folders = []
        for item in os.listdir(input_str):
            item_path = os.path.join(input_str, item)
            if os.path.isdir(item_path):
                person_folders.append(Path(item_path))

        all_processed_data = []

        for person_folder in person_folders:
            person_name = person_folder.name
            print(f"\n👤 Procesando: {person_name}")

            # Crear carpeta para esta persona
            person_output = temp_path / person_name
            person_output.mkdir(exist_ok=True)

            # Obtener todas las imágenes usando os
            folder_str = str(person_folder)
            all_files = os.listdir(folder_str)

            image_files = []
            for file in all_files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(person_folder / file)

            print(f"   📊 Total de imágenes: {len(image_files)}")

            faces_detected = 0
            faces_not_detected = 0

            for idx, img_path in enumerate(tqdm(image_files, desc="   Procesando")):
                # Leer imagen
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Detectar y recortar rostro
                face, confidence = self.detect_and_crop_face(img)

                if face is None:
                    faces_not_detected += 1
                    # Registrar imagen descartada
                    self.stats['discarded_images'].append({
                        'person': person_name,
                        'path': str(img_path),
                        'reason': 'No se detectó rostro o falló validación'
                    })
                    # Saltar esta imagen - no se procesará ni guardará
                    continue

                # Registrar detecciones de baja confianza (pero válidas)
                if confidence < 0.7:
                    self.stats['low_confidence_detections'].append({
                        'person': person_name,
                        'path': str(img_path),
                        'confidence': float(confidence)  # Convertir a float Python
                    })

                faces_detected += 1

                # Redimensionar y normalizar
                processed = self.resize_and_normalize(face)

                # Guardar imagen original procesada
                original_filename = f"{person_name}_{idx:04d}_original.npy"
                np.save(person_output / original_filename, processed)
                all_processed_data.append({
                    'path': str(person_output / original_filename),
                    'person': person_name,
                    'augmented': False
                })

                # Data Augmentation
                if apply_augmentation:
                    augmented_images = self.augment_image(processed)

                    for aug_type, aug_img in augmented_images:
                        aug_filename = f"{person_name}_{idx:04d}_{aug_type}.npy"
                        np.save(person_output / aug_filename, aug_img)
                        all_processed_data.append({
                            'path': str(person_output / aug_filename),
                            'person': person_name,
                            'augmented': True,
                            'aug_type': aug_type
                        })
                        self.stats['augmented_images'] += 1

            print(f"   ✅ Rostros detectados: {faces_detected}")
            print(f"   ⚠️  Rostros NO detectados: {faces_not_detected}")

            self.stats['faces_detected'] += faces_detected
            self.stats['faces_not_detected'] += faces_not_detected
            self.stats['total_processed'] += len(image_files)

        print("\n" + "=" * 60)
        print("📊 ESTADÍSTICAS DE PREPROCESAMIENTO")
        print("=" * 60)
        print(f"Total de imágenes procesadas: {self.stats['total_processed']}")
        print(f"Rostros detectados: {self.stats['faces_detected']}")
        print(f"Rostros NO detectados: {self.stats['faces_not_detected']}")
        print(f"Tasa de detección: {self.stats['faces_detected']/self.stats['total_processed']*100:.1f}%")

        if self.stats['faces_not_detected'] > 0:
            print(f"\n⚠️  Imágenes descartadas (sin rostro detectado):")
            for img_info in self.stats['discarded_images'][:10]:  # Mostrar solo 10
                print(f"   • {img_info['person']}: {Path(img_info['path']).name}")
            if len(self.stats['discarded_images']) > 10:
                print(f"   ... y {len(self.stats['discarded_images']) - 10} más")

            # Guardar lista de descartadas
            discarded_file = self.output_path / 'discarded_images.txt'
            with open(discarded_file, 'w', encoding='utf-8') as f:
                f.write("IMÁGENES DESCARTADAS (No se detectó rostro o falló validación)\n")
                f.write("=" * 60 + "\n\n")
                for img_info in self.stats['discarded_images']:
                    f.write(f"Persona: {img_info['person']}\n")
                    f.write(f"Archivo: {Path(img_info['path']).name}\n")
                    f.write(f"Ruta completa: {img_info['path']}\n")
                    f.write(f"Razón: {img_info['reason']}\n")
                    f.write("-" * 40 + "\n")
            print(f"\n📄 Lista completa guardada en: {discarded_file}")

        if len(self.stats['low_confidence_detections']) > 0:
            print(f"\n⚠️  {len(self.stats['low_confidence_detections'])} detecciones con confianza < 70%")
            print(f"   (Se procesaron pero puede que quieras revisarlas)")
            low_conf_file = self.output_path / 'low_confidence_detections.txt'
            with open(low_conf_file, 'w', encoding='utf-8') as f:
                f.write("DETECCIONES DE BAJA CONFIANZA (procesadas pero verificar)\n")
                f.write("=" * 60 + "\n\n")
                for img_info in self.stats['low_confidence_detections']:
                    f.write(f"Persona: {img_info['person']}\n")
                    f.write(f"Archivo: {Path(img_info['path']).name}\n")
                    f.write(f"Confianza: {img_info['confidence']:.2%}\n")
                    f.write("-" * 40 + "\n")
            print(f"   📄 Lista guardada en: {low_conf_file}")

        if apply_augmentation:
            print(f"\nImágenes aumentadas generadas: {self.stats['augmented_images']}")
            print(f"Total final de imágenes: {self.stats['faces_detected'] + self.stats['augmented_images']}")

        return all_processed_data, temp_path

    def split_dataset(self, data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        """
        Divide el dataset en train, validation y test

        Args:
            data: Lista de diccionarios con info de las imágenes
            train_ratio: Proporción para entrenamiento (0.7 = 70%)
            val_ratio: Proporción para validación (0.15 = 15%)
            test_ratio: Proporción para prueba (0.15 = 15%)
            random_state: Semilla para reproducibilidad
        """
        print("\n" + "=" * 60)
        print("📊 DIVISIÓN DE DATASET")
        print("=" * 60)

        # Verificar que las proporciones sumen 1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
            "Las proporciones deben sumar 1.0"

        # Separar por persona para mantener balance
        splits = {'train': [], 'val': [], 'test': []}

        # Obtener personas únicas
        persons = list(set([d['person'] for d in data]))

        for person in persons:
            # Filtrar datos de esta persona
            person_data = [d for d in data if d['person'] == person]

            # Primera división: train vs (val + test)
            train_data, temp_data = train_test_split(
                person_data,
                test_size=(val_ratio + test_ratio),
                random_state=random_state
            )

            # Segunda división: val vs test
            val_size = val_ratio / (val_ratio + test_ratio)
            val_data, test_data = train_test_split(
                temp_data,
                test_size=(1 - val_size),
                random_state=random_state
            )

            splits['train'].extend(train_data)
            splits['val'].extend(val_data)
            splits['test'].extend(test_data)

        # Crear estructura de carpetas
        for split_name in ['train', 'val', 'test']:
            split_path = self.output_path / split_name
            split_path.mkdir(parents=True, exist_ok=True)

            # Crear carpetas por persona
            for person in persons:
                (split_path / person).mkdir(exist_ok=True)

        # Copiar archivos a las carpetas correspondientes
        print("\n📁 Organizando archivos...")
        for split_name, split_data in splits.items():
            for item in tqdm(split_data, desc=f"   {split_name.upper()}"):
                src_path = Path(item['path'])
                dst_path = self.output_path / split_name / item['person'] / src_path.name
                shutil.copy(src_path, dst_path)

        # Estadísticas de división
        print("\n📊 Distribución del dataset:")
        print(f"   🔹 Train: {len(splits['train'])} imágenes ({len(splits['train'])/len(data)*100:.1f}%)")
        print(f"   🔹 Val:   {len(splits['val'])} imágenes ({len(splits['val'])/len(data)*100:.1f}%)")
        print(f"   🔹 Test:  {len(splits['test'])} imágenes ({len(splits['test'])/len(data)*100:.1f}%)")

        print("\n📊 Distribución por persona:")
        for person in persons:
            train_count = len([d for d in splits['train'] if d['person'] == person])
            val_count = len([d for d in splits['val'] if d['person'] == person])
            test_count = len([d for d in splits['test'] if d['person'] == person])
            total = train_count + val_count + test_count

            print(f"   👤 {person}:")
            print(f"      Train: {train_count} ({train_count/total*100:.1f}%) | "
                  f"Val: {val_count} ({val_count/total*100:.1f}%) | "
                  f"Test: {test_count} ({test_count/total*100:.1f}%)")

        # Guardar metadata
        metadata = {
            'target_size': list(self.target_size),  # Convertir tupla a lista
            'detection_method': self.detection_method,
            'splits': {
                'train': int(len(splits['train'])),  # Convertir a int Python
                'val': int(len(splits['val'])),
                'test': int(len(splits['test']))
            },
            'persons': persons,
            'statistics': {
                'total_processed': int(self.stats['total_processed']),
                'faces_detected': int(self.stats['faces_detected']),
                'faces_not_detected': int(self.stats['faces_not_detected']),
                'augmented_images': int(self.stats['augmented_images'])
            }
        }

        with open(self.output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"\n✅ Metadata guardada en: {self.output_path / 'metadata.json'}")

        return splits

    def visualize_samples(self, num_samples=5):
        """
        Visualiza muestras del dataset procesado
        """
        print("\n📊 Generando visualización de muestras...")

        fig, axes = plt.subplots(3, num_samples, figsize=(15, 9))
        fig.suptitle('Muestras del Dataset Procesado', fontsize=16, fontweight='bold')

        for split_idx, split_name in enumerate(['train', 'val', 'test']):
            split_path = self.output_path / split_name

            # Obtener personas usando os
            split_str = str(split_path)
            if os.path.exists(split_str):
                persons = []
                for item in os.listdir(split_str):
                    item_path = os.path.join(split_str, item)
                    if os.path.isdir(item_path):
                        persons.append(Path(item_path))
            else:
                continue

            for col in range(num_samples):
                if len(persons) == 0:
                    continue

                # Seleccionar persona aleatoria
                person = np.random.choice(persons)

                # Seleccionar imagen aleatoria usando os
                person_str = str(person)
                all_files = os.listdir(person_str)
                images = [person / f for f in all_files if f.endswith('.npy')]

                if len(images) > 0:
                    img_path = np.random.choice(images)
                    img = np.load(img_path)

                    # Convertir de float [0,1] a uint8 [0,255] para visualización
                    img_display = (img * 255).astype(np.uint8)
                    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

                    axes[split_idx, col].imshow(img_display)
                    axes[split_idx, col].axis('off')
                    axes[split_idx, col].set_title(f"{split_name}\n{person.name}",
                                                   fontsize=9)

        plt.tight_layout()
        plt.savefig(self.output_path / 'dataset_samples.png', dpi=150, bbox_inches='tight')
        print(f"✅ Visualización guardada en: {self.output_path / 'dataset_samples.png'}")
        plt.show()

    def run_full_pipeline(self, apply_augmentation=True, train_ratio=0.7,
                         val_ratio=0.15, test_ratio=0.15):
        """
        Ejecuta el pipeline completo de preprocesamiento
        """
        print("\n" + "🚀" * 30)
        print("PIPELINE COMPLETO DE PREPROCESAMIENTO")
        print("🚀" * 30 + "\n")

        # 1. Procesamiento (detección, recorte, normalización, augmentation)
        all_data, temp_path = self.process_dataset(apply_augmentation)

        # 2. División en train/val/test
        splits = self.split_dataset(all_data, train_ratio, val_ratio, test_ratio)

        # 3. Limpiar carpeta temporal
        print(f"\n🧹 Limpiando archivos temporales...")
        shutil.rmtree(temp_path)

        # 4. Visualizar muestras
        self.visualize_samples()

        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETO FINALIZADO")
        print("=" * 60)
        print(f"\n📂 Dataset procesado guardado en: {self.output_path}")
        print(f"\n📁 Estructura de carpetas:")
        print(f"   {self.output_path}/")
        print(f"   ├── train/")
        print(f"   │   ├── {splits['train'][0]['person'] if splits['train'] else 'persona'}/")
        print(f"   │   └── ...")
        print(f"   ├── val/")
        print(f"   ├── test/")
        print(f"   ├── metadata.json")
        print(f"   └── dataset_samples.png")


# EJEMPLO DE USO PARA GOOGLE COLAB
if __name__ == "__main__":
    # Montar Google Drive (si no está montado)
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive montado correctamente")
    except:
        print("ℹ️  No se está ejecutando en Google Colab")

    # Configuración
    INPUT_PATH = "/content/drive/MyDrive/fotitos"
    OUTPUT_PATH = "/content/drive/MyDrive/fotitos_procesadas"
    TARGET_SIZE = (224, 224)  # Puedes cambiar a (128, 128) si prefieres

    # Crear instancia del preprocesador
    preprocessor = FacialDatasetPreprocessor(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        target_size=TARGET_SIZE,
        detection_method='dnn'  # Opciones: 'haar', 'dnn', 'dlib'
    )

    # Ejecutar pipeline completo
    preprocessor.run_full_pipeline(
        apply_augmentation=True,  # Cambiar a False si no quieres augmentation
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15
    )

    print("\n🎉 ¡Listo! Tu dataset está preparado para entrenar modelos.")
