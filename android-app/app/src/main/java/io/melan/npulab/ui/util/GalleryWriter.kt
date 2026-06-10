package io.melan.npulab.ui.util

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Writes a bitmap into the system gallery under Pictures/NpuLab/ using MediaStore.
 * On minSdk 31 we always go through MediaStore — no runtime permission needed.
 */
object GalleryWriter {

    private const val ALBUM = "NpuLab"

    fun save(context: Context, bitmap: Bitmap, prompt: String? = null): Result<Uri> = runCatching {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val displayName = buildString {
            append("npulab_")
            append(timestamp)
            if (!prompt.isNullOrBlank()) {
                append("_")
                append(prompt.take(40)
                    .replace(Regex("[^A-Za-z0-9_\\- ]"), "")
                    .replace(' ', '_')
                    .ifBlank { "image" })
            }
            append(".png")
        }

        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, displayName)
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.RELATIVE_PATH,
                    "${Environment.DIRECTORY_PICTURES}/$ALBUM")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
        }

        val resolver = context.contentResolver
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            ?: error("MediaStore.insert returned null")

        resolver.openOutputStream(uri).use { stream ->
            requireNotNull(stream) { "MediaStore returned null OutputStream for $uri" }
            if (!bitmap.compress(Bitmap.CompressFormat.PNG, /*quality=*/100, stream)) {
                error("Bitmap.compress returned false")
            }
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            values.clear()
            values.put(MediaStore.Images.Media.IS_PENDING, 0)
            resolver.update(uri, values, null, null)
        }
        uri
    }
}
