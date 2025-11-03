"""
📱 CONVERTIDOR A TENSORFLOW LITE
=================================
Convierte tu modelo entrenado a formato .tflite optimizado para móviles
Incluye: cuantización, optimización y pruebas de latencia
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import time

print("=" * 80)
print("📱 CONVERTIDOR A TENSORFLOW LITE")
print("=" * 80)
print(f"TensorFlow: {tf.__version__}")
print("=" * 80)


class TFLiteConverter:
    """
    Convierte modelo Keras a TensorFlow Lite optimizado
    """

    def __init__(self, model_path, output_path, class_names):
        """
        Args:
            model_path: Ruta al modelo .h5
            output_path: Carpeta para guardar .tflite
            class_names: Lista de nombres de clases
        """
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names

        print(f"\n📂 Modelo origen: {self.model_path}")
        print(f"📂 Output: {self.output_path}")
        print(f"👥 Clases: {self.class_names}")

    def load_model(self):
        """Carga el modelo Keras"""
        print("\n🔄 Cargando modelo...")
        self.model = tf.keras.models.load_model(self.model_path)
        print("✅ Modelo cargado")
        self.model.summary()
        return self.model

    def convert_to_tflite(self, quantization='float16'):
        """
        Convierte a TensorFlow Lite

        Args:
            quantization: 'none', 'float16', 'int8'
                - 'none': Sin cuantización (más preciso, más pesado)
                - 'float16': Cuantización a 16 bits (RECOMENDADO)
                - 'int8': Cuantización a 8 bits (más ligero, menos preciso)
        """
        print(f"\n🔧 Convirtiendo a TFLite (cuantización: {quantization})...")

        # Crear convertidor
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # Aplicar optimizaciones
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if quantization == 'float16':
            # Cuantización a float16 (RECOMENDADO)
            converter.target_spec.supported_types = [tf.float16]
            output_name = 'model_float16.tflite'

        elif quantization == 'int8':
            # Cuantización completa a int8 (necesita datos representativos)
            print("   ⚠️ int8 requiere dataset representativo")
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            output_name = 'model_int8.tflite'

        else:
            # Sin cuantización
            output_name = 'model.tflite'

        # Convertir
        tflite_model = converter.convert()

        # Guardar
        tflite_path = self.output_path / output_name
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        # Obtener tamaño
        size_mb = os.path.getsize(tflite_path) / (1024 * 1024)

        print(f"✅ Modelo convertido:")
        print(f"   📄 Archivo: {output_name}")
        print(f"   💾 Tamaño: {size_mb:.2f} MB")

        return tflite_path, size_mb

    def test_tflite_inference(self, tflite_path, test_image):
        """
        Prueba inferencia con TFLite y mide latencia

        Args:
            tflite_path: Ruta al modelo .tflite
            test_image: Imagen de prueba (numpy array)
        """
        print(f"\n🧪 Probando inferencia con TFLite...")

        # Cargar intérprete TFLite
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()

        # Obtener detalles de input/output
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"   📥 Input shape: {input_details[0]['shape']}")
        print(f"   📤 Output shape: {output_details[0]['shape']}")

        # Preparar imagen
        input_shape = input_details[0]['shape']

        if test_image.shape[0:3] != tuple(input_shape[1:4]):
            print(f"   ⚠️ Redimensionando imagen a {input_shape[1:3]}")
            test_image = tf.image.resize(test_image, input_shape[1:3]).numpy()

        # Expandir dimensiones si es necesario
        if len(test_image.shape) == 3:
            test_image = np.expand_dims(test_image, axis=0)

        # Asegurar tipo correcto
        input_dtype = input_details[0]['dtype']
        if input_dtype == np.uint8:
            test_image = test_image.astype(np.uint8)
        else:
            test_image = test_image.astype(np.float32)

        # Medir latencia (múltiples iteraciones)
        print(f"\n⏱️  Midiendo latencia (10 inferencias)...")
        latencies = []

        for i in range(10):
            start = time.time()

            interpreter.set_tensor(input_details[0]['index'], test_image)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])

            end = time.time()
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)

        print(f"   ✅ Latencia promedio: {avg_latency:.2f} ms (±{std_latency:.2f} ms)")
        print(f"   🚀 FPS estimado: {1000/avg_latency:.1f}")

        # Obtener predicción
        predicted_class = np.argmax(output[0])
        confidence = output[0][predicted_class]

        print(f"\n🎯 Predicción de prueba:")
        print(f"   Clase: {self.class_names[predicted_class]}")
        print(f"   Confianza: {confidence:.2%}")

        return {
            'avg_latency_ms': float(avg_latency),
            'std_latency_ms': float(std_latency),
            'fps': float(1000/avg_latency),
            'predicted_class': self.class_names[predicted_class],
            'confidence': float(confidence)
        }

    def save_labels_file(self):
        """Guarda archivo de labels para la app móvil"""
        labels_path = self.output_path / 'labels.txt'

        with open(labels_path, 'w', encoding='utf-8') as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")

        print(f"\n📄 Archivo de labels guardado: labels.txt")
        return labels_path

    def create_metadata_json(self, tflite_info):
        """Crea archivo de metadatos para la app"""
        metadata = {
            'model_name': self.model_path.stem,
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'input_shape': self.model.input_shape[1:],
            'tflite_model': tflite_info['filename'],
            'model_size_mb': tflite_info['size_mb'],
            'avg_latency_ms': tflite_info.get('avg_latency_ms', 0),
            'preprocessing': {
                'rescale': '1/255',
                'input_type': 'uint8 or float32'
            }
        }

        metadata_path = self.output_path / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"📄 Metadatos guardados: model_metadata.json")
        return metadata_path

    def convert_all_versions(self):
        """
        Convierte el modelo a múltiples versiones optimizadas
        """
        print("\n" + "=" * 80)
        print("🔄 GENERANDO TODAS LAS VERSIONES")
        print("=" * 80)

        results = []

        # Versión 1: Sin cuantización
        print("\n📦 Versión 1: Sin cuantización (más precisa)")
        path1, size1 = self.convert_to_tflite('none')
        results.append({
            'version': 'sin_cuantizacion',
            'filename': path1.name,
            'size_mb': size1,
            'path': str(path1)
        })

        # Versión 2: Float16 (RECOMENDADA)
        print("\n📦 Versión 2: Float16 (RECOMENDADA)")
        path2, size2 = self.convert_to_tflite('float16')
        results.append({
            'version': 'float16',
            'filename': path2.name,
            'size_mb': size2,
            'path': str(path2)
        })

        # Resumen
        print("\n" + "=" * 80)
        print("📊 RESUMEN DE VERSIONES")
        print("=" * 80)

        for result in results:
            print(f"\n{result['version'].upper()}:")
            print(f"   📄 Archivo: {result['filename']}")
            print(f"   💾 Tamaño: {result['size_mb']:.2f} MB")

        return results

    def run_full_conversion(self, test_image_path=None):
        """
        Pipeline completo de conversión
        """
        print("\n" + "🚀" * 40)
        print("CONVERSIÓN COMPLETA A TENSORFLOW LITE")
        print("🚀" * 40)

        # 1. Cargar modelo
        self.load_model()

        # 2. Convertir todas las versiones
        versions = self.convert_all_versions()

        # 3. Guardar labels
        self.save_labels_file()

        # 4. Probar inferencia (versión float16)
        test_results = None
        if test_image_path and Path(test_image_path).exists():
            print("\n📸 Cargando imagen de prueba...")
            test_img = tf.keras.preprocessing.image.load_img(
                test_image_path,
                target_size=self.model.input_shape[1:3]
            )
            test_img = tf.keras.preprocessing.image.img_to_array(test_img)

            # Probar con versión float16
            float16_path = self.output_path / 'model_float16.tflite'
            test_results = self.test_tflite_inference(float16_path, test_img)

            # Agregar resultados a metadata
            versions[1]['avg_latency_ms'] = test_results['avg_latency_ms']
            versions[1]['fps'] = test_results['fps']

        # 5. Crear metadatos
        self.create_metadata_json(versions[1])  # Usar versión float16

        print("\n" + "🎉" * 40)
        print("✅ CONVERSIÓN COMPLETADA")
        print("🎉" * 40)

        print(f"\n📂 Archivos generados en: {self.output_path}")
        print(f"   • model.tflite (sin cuantización)")
        print(f"   • model_float16.tflite (RECOMENDADO)")
        print(f"   • labels.txt")
        print(f"   • model_metadata.json")

        return versions


# ============================================================================
# SCRIPT DE USO
# ============================================================================

if __name__ == "__main__":
    # Montar Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive montado")
    except:
        print("⚠️ No se está en Colab")

    print("\n" + "📱" * 40)
    print("CONVERTIDOR A TENSORFLOW LITE PARA MÓVILES")
    print("📱" * 40)

    # ⚙️ CONFIGURACIÓN - MODIFICA ESTAS RUTAS
    MODEL_PATH = "/content/drive/MyDrive/resultados_proyecto_cnn/mobilenetv2/model_mobilenetv2.h5"
    OUTPUT_PATH = "/content/drive/MyDrive/modelo_tflite"

    # Cargar nombres de clases desde metadata
    import json
    metadata_path = "/content/drive/MyDrive/fotitos_procesadas/metadata.json"

    if Path(metadata_path).exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            CLASS_NAMES = metadata['persons']
    else:
        # Si no existe metadata, definir manualmente
        CLASS_NAMES = ['Persona1', 'Persona2', 'Persona3']  # ⚠️ MODIFICA ESTO
        print("⚠️ No se encontró metadata.json, usando nombres por defecto")

    print(f"\n👥 Clases detectadas: {CLASS_NAMES}")

    # Verificar que existe el modelo
    if not Path(MODEL_PATH).exists():
        print(f"\n❌ ERROR: No se encuentra el modelo en {MODEL_PATH}")
        print("💡 Asegúrate de haber entrenado el modelo primero")
    else:
        print(f"\n✅ Modelo encontrado: {MODEL_PATH}")

        # Crear convertidor
        converter = TFLiteConverter(
            model_path=MODEL_PATH,
            output_path=OUTPUT_PATH,
            class_names=CLASS_NAMES
        )

        # Ejecutar conversión completa
        # Si tienes una imagen de prueba, pásala aquí:
        # TEST_IMAGE = "/ruta/a/imagen/prueba.jpg"
        TEST_IMAGE = None

        versions = converter.run_full_conversion(test_image_path=TEST_IMAGE)

        print("\n" + "=" * 80)
        print("📱 SIGUIENTE PASO: IMPLEMENTAR APP MÓVIL")
        print("=" * 80)
        print("\n💡 Archivos que necesitas copiar a tu app:")
        print(f"   1. {OUTPUT_PATH}/model_float16.tflite  (modelo optimizado)")
        print(f"   2. {OUTPUT_PATH}/labels.txt             (nombres de clases)")

        print("\n📋 Especificaciones para la app:")
        print(f"   • Número de clases: {len(CLASS_NAMES)}")
        print(f"   • Input shape: {converter.model.input_shape[1:]}")
        print(f"   • Preprocesamiento: Rescale 1/255")

        if versions[1].get('avg_latency_ms'):
            print(f"   • Latencia esperada: {versions[1]['avg_latency_ms']:.2f} ms")
            print(f"   • FPS esperado: {versions[1]['fps']:.1f}")

        print("\n🎓 ¡Listo para implementar en la app móvil!")
