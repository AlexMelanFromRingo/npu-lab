package io.melan.npulab.account

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.net.HttpURLConnection
import java.net.URL

/**
 * Minimal REST client for Qualcomm AI Hub — we only call the verify endpoint
 * (current user info), so a single function is enough. Avoids pulling in Retrofit
 * / OkHttp.
 *
 * API derived from the qai-hub Python SDK source (api_utils.py):
 *   base URL:    https://workbench.aihub.qualcomm.com
 *   API prefix:  /api/v1/
 *   auth header: "Authorization: token <api_token>"   (lowercase 'token')
 *   verify:      GET /api/v1/users/auth/user/         → returns user JSON
 *
 * On 200 returns the parsed user descriptor; on 401/403 returns
 * VerifyResult.Invalid; on network errors VerifyResult.NetworkError.
 */
object AiHubClient {

    sealed class VerifyResult {
        data class Ok(val displayName: String) : VerifyResult()
        data class Invalid(val message: String) : VerifyResult()
        data class NetworkError(val message: String) : VerifyResult()
    }

    private val json = Json { ignoreUnknownKeys = true }

    fun verifyToken(apiUrl: String, token: String): VerifyResult {
        val urlStr = "${apiUrl.trimEnd('/')}/api/v1/users/auth/user/"
        val conn = (URL(urlStr).openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            setRequestProperty("Authorization", "token $token")
            setRequestProperty("Accept", "application/json")
            connectTimeout = 8_000
            readTimeout = 10_000
        }
        return try {
            val rc = conn.responseCode
            when {
                rc in 200..299 -> {
                    val body = conn.inputStream.bufferedReader().use { it.readText() }
                    VerifyResult.Ok(displayNameFrom(body))
                }
                rc == 401 || rc == 403 -> {
                    val msg = errorBody(conn) ?: "HTTP $rc — invalid token"
                    VerifyResult.Invalid(msg)
                }
                else -> {
                    val msg = errorBody(conn) ?: "HTTP $rc"
                    VerifyResult.NetworkError(msg)
                }
            }
        } catch (t: Throwable) {
            VerifyResult.NetworkError(t.message ?: t::class.java.simpleName)
        } finally {
            conn.disconnect()
        }
    }

    private fun errorBody(conn: HttpURLConnection): String? = try {
        (conn.errorStream ?: conn.inputStream).bufferedReader().use { it.readText() }
    } catch (_: Throwable) { null }

    private fun displayNameFrom(body: String): String {
        val obj = runCatching { json.parseToJsonElement(body).jsonObject }.getOrNull()
            ?: return "verified"
        return obj["email"]?.jsonPrimitive?.content
            ?: obj["username"]?.jsonPrimitive?.content
            ?: obj["name"]?.jsonPrimitive?.content
            ?: "verified"
    }
}
