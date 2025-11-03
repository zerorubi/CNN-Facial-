#!pip install opencv-python-headless matplotlib seaborn pandas scipy
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import pandas as pd
from scipy import stats

class FacialDatasetEDA:
    """
    Análisis Exploratorio de Datos para dataset de imágenes faciales
    """

    def __init__(self, dataset_path):
        """
        Args:
            dataset_path: Ruta al directorio raíz del dataset
                         Estructura esperada: dataset_path/persona1/*.jpg
        """
        self.dataset_path = Path(dataset_path)
        self.data = defaultdict(list)
        self.stats = {}

    def load_dataset_info(self):
        """Carga información del dataset sin cargar todas las imágenes en memoria"""
        print("📊 Cargando información del dataset...")

        # Usar os.listdir para evitar problemas de caché con Google Drive
        import os
        dataset_str = str(self.dataset_path)

        # Obtener carpetas de personas usando os
        person_folders = []
        for item in os.listdir(dataset_str):
            item_path = os.path.join(dataset_str, item)
            if os.path.isdir(item_path):
                person_folders.append(Path(item_path))

        for person_folder in person_folders:
            person_name = person_folder.name

            # Buscar imágenes usando os para evitar caché
            folder_str = str(person_folder)
            all_files = os.listdir(folder_str)

            image_files = []
            for file in all_files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(person_folder / file)

            for img_path in image_files:
                # Leer imagen
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"⚠️  No se pudo leer: {img_path}")
                    continue

                # Información básica
                height, width, channels = img.shape
                file_size = img_path.stat().st_size / 1024  # KB
                file_format = img_path.suffix.upper().replace('.', '')  # JPEG, PNG, etc.

                # Análisis de iluminación
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                contrast = np.std(gray)

                # Análisis de nitidez (Laplacian variance)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

                # Análisis de color
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                saturation = np.mean(hsv[:,:,1])

                # Guardar datos
                self.data['person'].append(person_name)
                self.data['width'].append(width)
                self.data['height'].append(height)
                self.data['aspect_ratio'].append(width/height)
                self.data['file_size_kb'].append(file_size)
                self.data['file_format'].append(file_format)
                self.data['brightness'].append(brightness)
                self.data['contrast'].append(contrast)
                self.data['sharpness'].append(laplacian_var)
                self.data['saturation'].append(saturation)
                self.data['path'].append(str(img_path))

        self.df = pd.DataFrame(self.data)
        print(f"✅ Dataset cargado: {len(self.df)} imágenes de {self.df['person'].nunique()} personas")

    def calculate_statistics(self):
        """Calcula estadísticas generales del dataset"""
        print("\n" + "="*60)
        print("📈 ESTADÍSTICAS GENERALES DEL DATASET")
        print("="*60)

        # Estadísticas básicas
        total_images = len(self.df)
        num_persons = self.df['person'].nunique()

        print(f"\n🔢 Cantidad de imágenes:")
        print(f"   • Total: {total_images} imágenes")
        print(f"   • Personas: {num_persons}")
        print(f"   • Promedio por persona: {total_images/num_persons:.1f} imágenes")

        print(f"\n📏 Dimensiones:")
        print(f"   • Ancho promedio: {self.df['width'].mean():.0f} ± {self.df['width'].std():.0f} px")
        print(f"   • Alto promedio: {self.df['height'].mean():.0f} ± {self.df['height'].std():.0f} px")
        print(f"   • Aspect ratio promedio: {self.df['aspect_ratio'].mean():.2f}")
        print(f"   • Tamaño de archivo promedio: {self.df['file_size_kb'].mean():.1f} KB")

        print(f"\n📁 Formatos de archivo:")
        format_counts = self.df['file_format'].value_counts()
        for fmt, count in format_counts.items():
            percentage = (count / total_images) * 100
            print(f"   • {fmt}: {count} imágenes ({percentage:.1f}%)")

        # Balance de clases
        print(f"\n⚖️  Balance de clases:")
        class_counts = self.df['person'].value_counts()
        for person, count in class_counts.items():
            percentage = (count / total_images) * 100
            bar = "█" * int(percentage / 2)
            print(f"   • {person}: {count} imágenes ({percentage:.1f}%) {bar}")

        # Evaluar balance
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count
        if imbalance_ratio > 1.5:
            print(f"   ⚠️  Dataset desbalanceado (ratio: {imbalance_ratio:.2f}:1)")
        else:
            print(f"   ✅ Dataset balanceado (ratio: {imbalance_ratio:.2f}:1)")

        return {
            'total_images': total_images,
            'num_persons': num_persons,
            'class_counts': class_counts.to_dict(),
            'imbalance_ratio': imbalance_ratio
        }

    def analyze_variability(self):
        """Analiza variabilidad en condiciones de captura"""
        print("\n" + "="*60)
        print("🔍 ANÁLISIS DE VARIABILIDAD")
        print("="*60)

        # Iluminación
        print("\n💡 Iluminación (Brightness):")
        print(f"   • Promedio: {self.df['brightness'].mean():.1f}")
        print(f"   • Rango: [{self.df['brightness'].min():.1f}, {self.df['brightness'].max():.1f}]")
        print(f"   • Desviación estándar: {self.df['brightness'].std():.1f}")
        cv_brightness = (self.df['brightness'].std() / self.df['brightness'].mean()) * 100
        print(f"   • Coeficiente de variación: {cv_brightness:.1f}%")

        if cv_brightness > 30:
            print("   ⚠️  Alta variabilidad en iluminación")
        else:
            print("   ✅ Variabilidad moderada en iluminación")

        # Contraste
        print("\n🎨 Contraste:")
        print(f"   • Promedio: {self.df['contrast'].mean():.1f}")
        print(f"   • Rango: [{self.df['contrast'].min():.1f}, {self.df['contrast'].max():.1f}]")
        print(f"   • Desviación estándar: {self.df['contrast'].std():.1f}")

        # Nitidez
        print("\n🔬 Nitidez (Sharpness):")
        print(f"   • Promedio: {self.df['sharpness'].mean():.1f}")
        print(f"   • Rango: [{self.df['sharpness'].min():.1f}, {self.df['sharpness'].max():.1f}]")

        # Identificar imágenes con problemas
        low_sharpness = self.df[self.df['sharpness'] < self.df['sharpness'].quantile(0.1)]
        if len(low_sharpness) > 0:
            print(f"   ⚠️  {len(low_sharpness)} imágenes con baja nitidez detectadas")

        # Saturación (indicador de background variado)
        print("\n🌈 Saturación de color:")
        print(f"   • Promedio: {self.df['saturation'].mean():.1f}")
        print(f"   • Desviación estándar: {self.df['saturation'].std():.1f}")

        # Variabilidad por persona
        print("\n👤 Variabilidad por persona:")
        for person in self.df['person'].unique():
            person_data = self.df[self.df['person'] == person]
            brightness_cv = (person_data['brightness'].std() / person_data['brightness'].mean()) * 100
            print(f"   • {person}: CV iluminación = {brightness_cv:.1f}%")

    def create_visualizations(self):
        """Crea visualizaciones del análisis"""
        print("\n📊 Generando visualizaciones...")

        fig = plt.figure(figsize=(16, 12))

        # 1. Distribución de dimensiones
        ax1 = plt.subplot(3, 3, 1)
        # Convertir personas a códigos numéricos para el color
        person_codes = pd.Categorical(self.df['person']).codes
        scatter = ax1.scatter(self.df['width'], self.df['height'],
                             c=person_codes, cmap='viridis', alpha=0.6)
        ax1.set_title('Distribución de Dimensiones')
        ax1.set_xlabel('Ancho (px)')
        ax1.set_ylabel('Alto (px)')
        # Agregar leyenda
        persons = self.df['person'].unique()
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=plt.cm.viridis(i/len(persons)),
                             markersize=8, label=person)
                  for i, person in enumerate(persons)]
        ax1.legend(handles=handles, loc='best', fontsize=8)

        # 2. Balance de clases
        ax2 = plt.subplot(3, 3, 2)
        self.df['person'].value_counts().plot(kind='bar', ax=ax2, color='steelblue')
        ax2.set_title('Balance de Clases')
        ax2.set_xlabel('Persona')
        ax2.set_ylabel('Número de Imágenes')
        ax2.tick_params(axis='x', rotation=45)

        # 3. Distribución de brillo
        ax3 = plt.subplot(3, 3, 3)
        self.df.boxplot(column='brightness', by='person', ax=ax3)
        ax3.set_title('Distribución de Brillo por Persona')
        ax3.set_xlabel('Persona')
        ax3.set_ylabel('Brillo')
        plt.suptitle('')

        # 4. Histograma de brillo
        ax4 = plt.subplot(3, 3, 4)
        for person in self.df['person'].unique():
            data = self.df[self.df['person'] == person]['brightness']
            ax4.hist(data, alpha=0.5, label=person, bins=20)
        ax4.set_title('Histograma de Iluminación')
        ax4.set_xlabel('Brillo')
        ax4.set_ylabel('Frecuencia')
        ax4.legend()

        # 5. Contraste
        ax5 = plt.subplot(3, 3, 5)
        self.df.boxplot(column='contrast', by='person', ax=ax5)
        ax5.set_title('Distribución de Contraste')
        ax5.set_xlabel('Persona')
        ax5.set_ylabel('Contraste')
        plt.suptitle('')

        # 6. Nitidez
        ax6 = plt.subplot(3, 3, 6)
        self.df.boxplot(column='sharpness', by='person', ax=ax6)
        ax6.set_title('Distribución de Nitidez')
        ax6.set_xlabel('Persona')
        ax6.set_ylabel('Nitidez (Laplacian Var)')
        plt.suptitle('')

        # 7. Saturación
        ax7 = plt.subplot(3, 3, 7)
        self.df.boxplot(column='saturation', by='person', ax=ax7)
        ax7.set_title('Distribución de Saturación')
        ax7.set_xlabel('Persona')
        ax7.set_ylabel('Saturación')
        plt.suptitle('')

        # 8. Tamaño de archivo
        ax8 = plt.subplot(3, 3, 8)
        self.df['file_size_kb'].hist(bins=30, ax=ax8, color='coral', edgecolor='black')
        ax8.set_title('Distribución de Tamaño de Archivo')
        ax8.set_xlabel('Tamaño (KB)')
        ax8.set_ylabel('Frecuencia')

        # 9. Matriz de correlación
        ax9 = plt.subplot(3, 3, 9)
        corr_cols = ['brightness', 'contrast', 'sharpness', 'saturation']
        corr = self.df[corr_cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax9,
                   square=True, cbar_kws={'shrink': 0.8})
        ax9.set_title('Correlación entre Variables')

        plt.tight_layout()
        plt.savefig('eda_facial_dataset.png', dpi=300, bbox_inches='tight')
        print("✅ Visualizaciones guardadas en 'eda_facial_dataset.png'")
        plt.show()

    def generate_report(self):
        """Genera un reporte completo en formato texto"""
        report_path = 'eda_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("REPORTE DE ANÁLISIS EXPLORATORIO DE DATOS (EDA)\n")
            f.write("Dataset de Imágenes Faciales\n")
            f.write("="*70 + "\n\n")

            # Estadísticas generales
            f.write("1. ESTADÍSTICAS GENERALES\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total de imágenes: {len(self.df)}\n")
            f.write(f"Número de personas: {self.df['person'].nunique()}\n")
            f.write(f"Promedio por persona: {len(self.df)/self.df['person'].nunique():.1f}\n\n")

            f.write("Balance de clases:\n")
            for person, count in self.df['person'].value_counts().items():
                f.write(f"  - {person}: {count} imágenes ({count/len(self.df)*100:.1f}%)\n")
            f.write("\n")

            # Dimensiones
            f.write("2. DIMENSIONES DE IMÁGENES\n")
            f.write("-" * 40 + "\n")
            f.write(f"Ancho: {self.df['width'].mean():.0f} ± {self.df['width'].std():.0f} px\n")
            f.write(f"Alto: {self.df['height'].mean():.0f} ± {self.df['height'].std():.0f} px\n")
            f.write(f"Aspect Ratio: {self.df['aspect_ratio'].mean():.2f}\n")
            f.write(f"Tamaño archivo: {self.df['file_size_kb'].mean():.1f} KB\n\n")

            f.write("Formatos de archivo:\n")
            for fmt, count in self.df['file_format'].value_counts().items():
                f.write(f"  - {fmt}: {count} imágenes ({count/len(self.df)*100:.1f}%)\n")
            f.write("\n")

            # Variabilidad
            f.write("3. ANÁLISIS DE VARIABILIDAD\n")
            f.write("-" * 40 + "\n")
            f.write(f"Iluminación (brightness):\n")
            f.write(f"  Promedio: {self.df['brightness'].mean():.1f}\n")
            f.write(f"  Rango: [{self.df['brightness'].min():.1f}, {self.df['brightness'].max():.1f}]\n")
            f.write(f"  Desv. Estándar: {self.df['brightness'].std():.1f}\n")
            f.write(f"  CV: {(self.df['brightness'].std()/self.df['brightness'].mean())*100:.1f}%\n\n")

            f.write(f"Contraste:\n")
            f.write(f"  Promedio: {self.df['contrast'].mean():.1f}\n")
            f.write(f"  Rango: [{self.df['contrast'].min():.1f}, {self.df['contrast'].max():.1f}]\n\n")

            f.write(f"Nitidez (sharpness):\n")
            f.write(f"  Promedio: {self.df['sharpness'].mean():.1f}\n")
            f.write(f"  Rango: [{self.df['sharpness'].min():.1f}, {self.df['sharpness'].max():.1f}]\n\n")

            # Recomendaciones
            f.write("4. RECOMENDACIONES\n")
            f.write("-" * 40 + "\n")

            # Balance
            max_count = self.df['person'].value_counts().max()
            min_count = self.df['person'].value_counts().min()
            if max_count / min_count > 1.5:
                f.write("⚠️  Dataset desbalanceado - considerar data augmentation\n")

            # Iluminación
            cv_brightness = (self.df['brightness'].std() / self.df['brightness'].mean()) * 100
            if cv_brightness > 30:
                f.write("⚠️  Alta variabilidad en iluminación - aplicar normalización\n")

            # Nitidez
            low_sharpness = self.df[self.df['sharpness'] < self.df['sharpness'].quantile(0.1)]
            if len(low_sharpness) > 0:
                f.write(f"⚠️  {len(low_sharpness)} imágenes con baja nitidez - revisar calidad\n")

            f.write("\n" + "="*70 + "\n")

        print(f"✅ Reporte guardado en '{report_path}'")

    def run_full_analysis(self):
        """Ejecuta el análisis completo"""
        self.load_dataset_info()
        self.calculate_statistics()
        self.analyze_variability()
        self.create_visualizations()
        self.generate_report()

        print("\n" + "="*60)
        print("✅ ANÁLISIS COMPLETO FINALIZADO")
        print("="*60)
        print("\nArchivos generados:")
        print("  📊 eda_facial_dataset.png - Visualizaciones")
        print("  📄 eda_report.txt - Reporte detallado")

        return self.df


# EJEMPLO DE USO
if __name__ == "__main__":
    # Para Google Colab - Montar Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive montado correctamente")
    except:
        print("ℹ️  No se está ejecutando en Google Colab")

    # Ruta al dataset en Google Drive
    DATASET_PATH = "/content/drive/MyDrive/fotitos"

    print(f"🔍 Buscando dataset en: {DATASET_PATH}")

    # Verificar que existe la ruta
    if not Path(DATASET_PATH).exists():
        print(f"❌ Error: No se encontró la ruta {DATASET_PATH}")
        print("Verifica que Google Drive esté montado correctamente")
    else:
        # Crear instancia y ejecutar análisis
        eda = FacialDatasetEDA(DATASET_PATH)
        df_results = eda.run_full_analysis()

        # El DataFrame con todos los datos está disponible en df_results
        print("\n📊 DataFrame con resultados disponible en 'df_results'")
        print("\n🔍 Vista previa de los datos:")
        print(df_results.head(10))

        # Mostrar personas encontradas
        print(f"\n👥 Personas encontradas: {df_results['person'].unique().tolist()}")
