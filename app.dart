// ============================================================================
// AndroidManifest.xml
//<!-- Permisos de cámara -->
//    <uses-permission android:name="android.permission.CAMERA"/>
//    <uses-feature android:name="android.hardware.camera" android:required="true"/>
//
// ============================================================================

import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

// ============================================================================
// MAIN
// ============================================================================

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Obtener cámaras disponibles
  try {
    cameras = await availableCameras();
  } catch (e) {
    print('Error obteniendo cámaras: $e');
  }
  
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Reconocimiento Facial',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        brightness: Brightness.dark,
      ),
      home: FacialRecognitionScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}

// ============================================================================
// PANTALLA PRINCIPAL
// ============================================================================

class FacialRecognitionScreen extends StatefulWidget {
  const FacialRecognitionScreen({super.key});

  @override
  _FacialRecognitionScreenState createState() => _FacialRecognitionScreenState();
}

class _FacialRecognitionScreenState extends State<FacialRecognitionScreen> {
  // Cámara
  CameraController? cameraController;
  bool isCameraInitialized = false;
  
  // TensorFlow Lite
  Interpreter? interpreter;
  List<String> labels = [];
  
  // Configuración del modelo
  static const int INPUT_SIZE = 224;  
  static const int NUM_CLASSES = 4; 
  
  // Resultados
  String recognizedName = 'Esperando detección...';
  double confidence = 0.0;
  int latencyMs = 0;
  int fps = 0;
  
  // Control de procesamiento
  bool isProcessing = false;
  int frameCount = 0;
  List<int> fpsQueue = [];
  
  @override
  void initState() {
    super.initState();
    initializeApp();
  }
  
  /// Inicializa cámara y modelo
  Future<void> initializeApp() async {
    await loadModel();
    await loadLabels();
    await initializeCamera();
  }
  
  /// Carga el modelo TensorFlow Lite
  Future<void> loadModel() async {
    try {
      print('🔄 Cargando modelo TFLite...');
      
      // Cargar modelo desde assets
      interpreter = await Interpreter.fromAsset('assets/model_float16.tflite');
      
      print('✅ Modelo cargado exitosamente');
      
      // Mostrar información del modelo
      final inputShape = interpreter!.getInputTensor(0).shape;
      final outputShape = interpreter!.getOutputTensor(0).shape;
      
      print('📊 Input shape: $inputShape');
      print('📊 Output shape: $outputShape');
      
    } catch (e) {
      print('❌ Error cargando modelo: $e');
      showErrorDialog('Error cargando modelo: $e');
    }
  }
  
  /// Carga los nombres de las personas (labels)
  Future<void> loadLabels() async {
    try {
      print('🔄 Cargando labels...');
      
      final labelsData = await rootBundle.loadString('assets/labels.txt');
      labels = labelsData.split('\n').where((label) => label.isNotEmpty).toList();
      
      print('✅ Labels cargados: $labels');
      
    } catch (e) {
      print('❌ Error cargando labels: $e');
      showErrorDialog('Error cargando labels: $e');
    }
  }
  
  /// Inicializa la cámara
  Future<void> initializeCamera() async {
    if (cameras.isEmpty) {
      showErrorDialog('No se encontraron cámaras');
      return;
    }
    
    try {
      // Usar cámara frontal
      final frontCamera = cameras.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );
      
      cameraController = CameraController(
        frontCamera,
        ResolutionPreset.medium,
        enableAudio: false,
      );
      
      await cameraController!.initialize();
      
      setState(() {
        isCameraInitialized = true;
      });
      
      print('✅ Cámara inicializada');
      
      // Iniciar stream de imágenes
      startImageStream();
      
    } catch (e) {
      print('❌ Error inicializando cámara: $e');
      showErrorDialog('Error con la cámara: $e');
    }
  }
  
  /// Inicia el stream de procesamiento de imágenes
  void startImageStream() {
    if (cameraController == null || !cameraController!.value.isInitialized) {
      return;
    }
    
    cameraController!.startImageStream((CameraImage image) {
      // Procesar solo si no está ocupado
      if (!isProcessing) {
        isProcessing = true;
        processFrame(image);
      }
    });
  }
  
  /// Procesa cada frame de la cámara
  Future<void> processFrame(CameraImage cameraImage) async {
    final startTime = DateTime.now().millisecondsSinceEpoch;
    
    try {
      // Convertir CameraImage a imagen procesable
      final imgLib = convertCameraImage(cameraImage);
      
      if (imgLib == null) {
        isProcessing = false;
        return;
      }
      
      // Preprocesar imagen para el modelo
      final input = preprocessImage(imgLib);
      
      // Preparar output
      var output = List.filled(1 * NUM_CLASSES, 0.0).reshape([1, NUM_CLASSES]);
      
      // ⏱️ INFERENCIA (medir latencia)
      final inferenceStart = DateTime.now().millisecondsSinceEpoch;
      interpreter!.run(input, output);
      final inferenceTime = DateTime.now().millisecondsSinceEpoch - inferenceStart;
      
      // Obtener resultados
      final probabilities = output[0] as List<double>;
      final maxIndex = probabilities.indexOf(probabilities.reduce((a, b) => a > b ? a : b));
      final maxConfidence = probabilities[maxIndex];
      
      // Calcular métricas
      final totalTime = DateTime.now().millisecondsSinceEpoch - startTime;
      updateFPS(totalTime);
      
      // Actualizar UI
      setState(() {
        if (maxIndex < labels.length) {
          recognizedName = labels[maxIndex];
          confidence = maxConfidence;
          latencyMs = inferenceTime;
        }
      });
      
    } catch (e) {
      print('❌ Error procesando frame: $e');
    } finally {
      isProcessing = false;
    }
  }
  
  /// Convierte CameraImage a formato img.Image
  img.Image? convertCameraImage(CameraImage cameraImage) {
    try {
      if (cameraImage.format.group == ImageFormatGroup.yuv420) {
        return convertYUV420ToImage(cameraImage);
      } else if (cameraImage.format.group == ImageFormatGroup.bgra8888) {
        return convertBGRA8888ToImage(cameraImage);
      }
      return null;
    } catch (e) {
      print('Error convirtiendo imagen: $e');
      return null;
    }
  }
  
  /// Convierte YUV420 a Image
  img.Image convertYUV420ToImage(CameraImage cameraImage) {
    final int width = cameraImage.width;
    final int height = cameraImage.height;
    
    final int uvRowStride = cameraImage.planes[1].bytesPerRow;
    final int uvPixelStride = cameraImage.planes[1].bytesPerPixel!;
    
    final image = img.Image(width: width, height: height);
    
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int uvIndex = uvPixelStride * (x / 2).floor() + uvRowStride * (y / 2).floor();
        final int index = y * width + x;
        
        final yp = cameraImage.planes[0].bytes[index];
        final up = cameraImage.planes[1].bytes[uvIndex];
        final vp = cameraImage.planes[2].bytes[uvIndex];
        
        int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
        int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91).round().clamp(0, 255);
        int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);
        
        image.setPixelRgb(x, y, r, g, b);
      }
    }
    
    return image;
  }
  
  /// Convierte BGRA8888 a Image
  img.Image convertBGRA8888ToImage(CameraImage cameraImage) {
    return img.Image.fromBytes(
      width: cameraImage.width,
      height: cameraImage.height,
      bytes: cameraImage.planes[0].bytes.buffer,
      order: img.ChannelOrder.bgra,
    );
  }
  
  /// Preprocesa la imagen para el modelo
  List<List<List<List<double>>>> preprocessImage(img.Image image) {
    // Redimensionar a INPUT_SIZE x INPUT_SIZE
    final resized = img.copyResize(image, width: INPUT_SIZE, height: INPUT_SIZE);
    
    // Crear tensor 4D: [1, height, width, 3]
    var input = List.generate(
      1,
      (i) => List.generate(
        INPUT_SIZE,
        (y) => List.generate(
          INPUT_SIZE,
          (x) {
            final pixel = resized.getPixel(x, y);
            return [
              pixel.r / 255.0,  // Normalizar R [0, 1]
              pixel.g / 255.0,  // Normalizar G [0, 1]
              pixel.b / 255.0,  // Normalizar B [0, 1]
            ];
          },
        ),
      ),
    );
    
    return input;
  }
  
  /// Actualiza cálculo de FPS
  void updateFPS(int frameTime) {
    fpsQueue.add(frameTime);
    
    // Mantener últimos 30 frames
    if (fpsQueue.length > 30) {
      fpsQueue.removeAt(0);
    }
    
    if (fpsQueue.isNotEmpty) {
      final avgTime = fpsQueue.reduce((a, b) => a + b) / fpsQueue.length;
      fps = (1000 / avgTime).round();
    }
  }
  
  /// Muestra diálogo de error
  void showErrorDialog(String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Error'),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }
  
  /// Build de la UI
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Vista de la cámara
          if (isCameraInitialized && cameraController != null)
            Positioned.fill(
              child: CameraPreview(cameraController!),
            )
          else
            const Center(
              child: CircularProgressIndicator(),
            ),
          
          // Header
          Positioned(
            top: 0,
            left: 0,
            right: 0,
            child: Container(
              padding: EdgeInsets.only(top: MediaQuery.of(context).padding.top + 10, bottom: 15),
              decoration: const BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [Colors.black87, Colors.transparent],
                ),
              ),
              child: const Text(
                'Reconocimiento Facial',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
            ),
          ),
          
          // Panel de resultados
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: Container(
              padding: const EdgeInsets.all(20),
              decoration: const BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.bottomCenter,
                  end: Alignment.topCenter,
                  colors: [Colors.black87, Colors.transparent],
                ),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  // Nombre de la persona
                  Text(
                    '👤 $recognizedName',
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                      fontSize: 28,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                  
                  const SizedBox(height: 10),
                  
                  // Confianza
                  Text(
                    'Confianza: ${(confidence * 100).toStringAsFixed(0)}%',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 20,
                      color: getConfidenceColor(),
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  
                  const SizedBox(height: 15),
                  
                  // Métricas de rendimiento
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      // Latencia
                      Column(
                        children: [
                          Text(
                            '⏱️ Latencia',
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.grey[400],
                            ),
                          ),
                          Text(
                            '${latencyMs}ms',
                            style: const TextStyle(
                              fontSize: 16,
                              color: Colors.white,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                      
                      // FPS
                      Column(
                        children: [
                          Text(
                            '🎥 FPS',
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.grey[400],
                            ),
                          ),
                          Text(
                            '$fps',
                            style: const TextStyle(
                              fontSize: 16,
                              color: Colors.white,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
  
  /// Obtiene color según confianza
  Color getConfidenceColor() {
    if (confidence >= 0.8) {
      return Colors.greenAccent;
    } else if (confidence >= 0.5) {
      return Colors.orangeAccent;
    } else {
      return Colors.redAccent;
    }
  }
  
  @override
  void dispose() {
    cameraController?.dispose();
    interpreter?.close();
    super.dispose();
  }
}


//
