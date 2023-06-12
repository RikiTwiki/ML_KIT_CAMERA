import android.annotation.SuppressLint
import android.app.*
import android.app.Service.START_NOT_STICKY
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.os.Binder
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.Surface
import android.view.TextureView
import android.widget.ImageView
import android.widget.Toast
import androidx.core.app.NotificationCompat
import com.android.volley.Request
import com.android.volley.toolbox.StringRequest
import com.android.volley.toolbox.Volley
import com.programminghut.realtime_object.R
import com.programminghut.realtime_object.ml.SsdMobilenetV11Metadata1
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.util.*
import kotlin.math.sqrt

class MyForegroundService : Service() {
    lateinit var labels:List<String>
    private val channelId = "ForegroundServiceChannel"
    private val NOTIFICATION_ID = 1

    private val binder = LocalBinder()


    private lateinit var tts: TextToSpeech
    val paint = Paint()
    var previousLocations: MutableMap<Int, RectF> = mutableMapOf()
    var movementThreshold: Float = 100f  // Adjust this value according to your needs
    lateinit var imageView: ImageView




    lateinit var cameraManager: CameraManager

    lateinit var cameraDevice: CameraDevice
    lateinit var surface: Surface
    lateinit var handler: Handler
    lateinit var textureView: TextureView

    var isPersonDetected = false



    private lateinit var imageProcessor: ImageProcessor
    private lateinit var model: SsdMobilenetV11Metadata1
    private lateinit var interpreter: Interpreter
    private lateinit var bitmap: Bitmap

    inner class LocalBinder : Binder() {
        fun getService(): MyForegroundService = this@MyForegroundService
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        Toast.makeText(this@MyForegroundService, "Success openSuccess!", Toast.LENGTH_LONG).show()
        val notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle("Foreground Service")
            .setSmallIcon(R.drawable.ic_notification) // TODO: Замените на ваш значок уведомления
            .build()
        startForeground(NOTIFICATION_ID, notification)


        labels = FileUtil.loadLabels(this, "labels.txt")
        // Здесь должна быть вся ваша логика, связанная с камерой, TextToSpeech, SpeechRecognizer и другими
        imageProcessor = ImageProcessor.Builder().add(ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR)).build()
        val compatList = CompatibilityList()
        val options = Interpreter.Options().apply {
            if (compatList.isDelegateSupportedOnThisDevice) {
                val delegateOptions = compatList.bestOptionsForThisDevice
                val gpuDelegate = GpuDelegate(delegateOptions)
                this.addDelegate(gpuDelegate)
            } else {
                this.setNumThreads(4)
            }
        }
        tts = TextToSpeech(this) { status ->
            if (status != TextToSpeech.ERROR) {
                val russian = Locale("ru")
                if (tts.isLanguageAvailable(russian) >= TextToSpeech.LANG_AVAILABLE) {
                    tts.language = russian
                }
            }
        }
        // Путь к файлу модели может отличаться, в зависимости от того, как вы его добавили в проект
        model = SsdMobilenetV11Metadata1.newInstance(this)


        val handlerThread = HandlerThread("videoThread")
        handlerThread.start()
        handler = Handler(handlerThread.looper)


        textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
                // This method will be called when the SurfaceTexture is available.
                open_camera()
            }

            override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {
                // This method will be called when the SurfaceTexture size changes.
                // Add code here to handle size changes if necessary.
            }

            override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
                // This method will be called just before the SurfaceTexture is destroyed.
                // Return true if you take care of releasing the SurfaceTexture here, or false if you want the system to handle it.
                return false
            }

            override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
                if (isPersonDetected) {
                    return
                }
                // Convert the current frame to a bitmap
                bitmap = textureView.bitmap!!

                // Process the bitmap
                var image = TensorImage.fromBitmap(bitmap)
                image = imageProcessor.process(image)

                // Run the model on the processed bitmap
                val outputs = model.process(image)
                val locations = outputs.locationsAsTensorBuffer.floatArray
                val classes = outputs.classesAsTensorBuffer.floatArray
                val scores = outputs.scoresAsTensorBuffer.floatArray

                // Create a new bitmap to draw bounding boxes and labels on
                var mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                val canvas = Canvas(mutable)

                val h = mutable.height
                val w = mutable.width
                paint.textSize = h / 15f
                paint.strokeWidth = h / 85f

                scores.forEachIndexed { index, fl ->
                    if (fl > 0.60 && classes[index].toInt() == 0) {
                        val x = index * 4
                        val currentRect = RectF(locations[x+1]*w, locations[x]*h, locations[x+3]*w, locations[x+2]*h)

                        val averagePersonHeightInRealWorld = 200  // cm
                        val cameraVerticalFieldOfView = 50.0  // degrees, adjust this to your camera's actual field of view
                        val personHeightInPixels = locations[x+2]*h - locations[x]*h
                        val distanceToPerson = (averagePersonHeightInRealWorld / 2) / Math.tan(Math.toRadians(cameraVerticalFieldOfView / 2)) / personHeightInPixels

                        if (distanceToPerson < 0.3) {
                            val previousRect = previousLocations[index]
                            if (previousRect == null || rectDiff(previousRect, currentRect) > movementThreshold) {

                                tts.speak("Салам, задайте вопрос", TextToSpeech.QUEUE_FLUSH, null, "")

                                // If person detected, close the app and open website

                                initSpeechRecognizer()

                                isPersonDetected = true
                            }

                            // Store the current location for next time
                            previousLocations[index] = currentRect
                        }
                    }
                }


                // Show the bitmap with bounding boxes and labels
                imageView.setImageBitmap(mutable)
            }

            fun rectDiff(rect1: RectF, rect2: RectF): Float {
                return sqrt(((rect1.centerX() - rect2.centerX()) * (rect1.centerX() - rect2.centerX()) + (rect1.centerY() - rect2.centerY()) * (rect1.centerY() - rect2.centerY())).toFloat())
            }

            // Your existing code...
        }


        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }

    fun initSpeechRecognizer() {
        val recognizer = SpeechRecognizer.createSpeechRecognizer(this)
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale("ru").toString())  // Set language to Russian

        val listener = object : RecognitionListener {
            override fun onResults(results: Bundle) {
                val matches = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (matches != null) {
                    for (result in matches) {
                        if (result.contains("weather", ignoreCase = true) || result.contains("привет", ignoreCase = true)) {
                            // The user asked about the weather!
                            fetchWeatherData()
                            break
                        }
                    }
                }
            }

            override fun onReadyForSpeech(params: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onError(error: Int) {}
            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        }

        recognizer.setRecognitionListener(listener)
        recognizer.startListening(intent) // Start listening to the user's speech

        isPersonDetected = false
    }


    fun fetchWeatherData() {
        // For this example, let's assume that the location is hardcoded. You should replace this with
        // real location data in your application.
        val city = "Bishkek"
        val apiKey = "8d323aeb96edb24edffa1656c67efc3b"

        val url = "https://api.openweathermap.org/data/2.5/weather?q=$city&appid=$apiKey&units=metric&lang=ru"

        // Use Android's built-in networking library (Volley) to fetch data.
        val queue = Volley.newRequestQueue(this)
        val stringRequest = StringRequest(
            Request.Method.GET, url,
            { response ->
                // Parse the response to get the current weather.
                val jsonObject = JSONObject(response)
                val weather = jsonObject.getJSONArray("weather").getJSONObject(0).getString("description")
                val temperature = jsonObject.getJSONObject("main").getString("temp")
                val weatherInRussian = "Погода в $city: $weather, температура: $temperature градусов."


                // Speak out the weather.
                tts.speak(weatherInRussian, TextToSpeech.QUEUE_FLUSH, null, "")

                isPersonDetected = false


            },
            { error ->
                // Add this line to handle errors
                Toast.makeText(this@MyForegroundService, "Error fetching weather: ${error.message}", Toast.LENGTH_LONG).show()
            }
        )

        queue.add(stringRequest)


    }



    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        open_camera()
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? {
        return binder
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val serviceChannel = NotificationChannel(
                channelId,
                "Foreground Service Channel",
                NotificationManager.IMPORTANCE_DEFAULT
            )
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(serviceChannel)
        }
    }


    override fun onDestroy() {
        super.onDestroy()
        interpreter.close()
        tts.shutdown()
    }


    @SuppressLint("MissingPermission")
    fun open_camera(){
        cameraManager.openCamera(cameraManager.cameraIdList[0], object:CameraDevice.StateCallback(){
            override fun onOpened(p0: CameraDevice) {
                cameraDevice = p0

                var surfaceTexture = textureView.surfaceTexture
                var surface = Surface(surfaceTexture)

                var captureRequest = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                captureRequest.addTarget(surface)

                cameraDevice.createCaptureSession(listOf(surface), object: CameraCaptureSession.StateCallback(){
                    override fun onConfigured(p0: CameraCaptureSession) {
                        p0.setRepeatingRequest(captureRequest.build(), null, null)
                    }
                    override fun onConfigureFailed(p0: CameraCaptureSession) {
                    }
                }, handler)
            }

            override fun onDisconnected(p0: CameraDevice) {

            }

            override fun onError(p0: CameraDevice, p1: Int) {

            }
        }, handler)
    }


}
