#!pip install opencv-python-headless matplotlib seaborn pandas scipy
from pathlib import Path

# Verificar ruta
dataset_path = Path("/content/drive/MyDrive/fotitos")

if dataset_path.exists():
    print("✅ Ruta encontrada!")
    print(f"\n📁 Carpetas encontradas:")
    for folder in dataset_path.iterdir():
        if folder.is_dir():
            num_files = len(list(folder.glob('*.*')))
            print(f"   • {folder.name}: {num_files} archivos")
else:
    print("❌ Ruta no encontrada")
    print("\n🔍 Verifica que la ruta sea correcta")
    print("Intenta listar lo que hay en MyDrive:")
    mydrive = Path("/content/drive/MyDrive")
    if mydrive.exists():
        print("\nCarpetas en MyDrive:")
        for item in list(mydrive.iterdir())[:10]:
            print(f"   • {item.name}")
