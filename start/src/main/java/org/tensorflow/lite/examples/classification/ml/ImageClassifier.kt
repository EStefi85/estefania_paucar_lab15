package org.tensorflow.lite.examples.classification.ml

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.task.vision.classifier.ImageClassifier as TFLiteImageClassifier
import org.tensorflow.lite.task.core.BaseOptions

class ImageClassifier(context: Context) {
    private var classifier: TFLiteImageClassifier? = null

    init {
        val baseOptions = BaseOptions.builder().setNumThreads(4).build()
        val options = TFLiteImageClassifier.ImageClassifierOptions.builder()
            .setBaseOptions(baseOptions)
            .setMaxResults(3)
            .build()
        try {
            classifier = TFLiteImageClassifier.createFromFileAndOptions(
                context,
                "model.tflite",
                options
            )
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun classify(image: Bitmap): List<Category> {
        val imageProcessor = org.tensorflow.lite.support.image.ImageProcessor.Builder()
            .add(org.tensorflow.lite.support.image.ops.ResizeOp(
                224, 224,
                org.tensorflow.lite.support.image.ops.ResizeOp.ResizeMethod.BILINEAR
            ))
            .build()

        val tensorImage = TensorImage.fromBitmap(image)
        val processedImage = imageProcessor.process(tensorImage)

        val results = classifier?.classify(processedImage)
        return results?.flatMap { it.categories } ?: emptyList()
    }
}