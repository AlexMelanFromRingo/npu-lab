package io.melan.npulab.account

import kotlinx.serialization.Serializable

@Serializable
data class AiHubAccount(
    val id: String,
    val label: String,
    val token: String,
    val apiUrl: String = DEFAULT_API_URL,
    /** Last known verified user identifier (email/username). null until first verify. */
    val verifiedUser: String? = null,
    /** Epoch millis of last successful verify. 0 if never. */
    val verifiedAtMs: Long = 0L,
) {
    fun tokenMasked(): String =
        if (token.length <= 8) "•".repeat(token.length)
        else "${token.take(4)}${"•".repeat(token.length - 8)}${token.takeLast(4)}"

    companion object {
        const val DEFAULT_API_URL = "https://workbench.aihub.qualcomm.com"
    }
}

@Serializable
data class AccountsState(
    val accounts: List<AiHubAccount> = emptyList(),
    val activeId: String? = null,
) {
    val active: AiHubAccount? get() = accounts.firstOrNull { it.id == activeId }
}
