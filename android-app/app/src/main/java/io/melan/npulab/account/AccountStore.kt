package io.melan.npulab.account

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.serialization.json.Json
import java.util.UUID

/**
 * Encrypted store for AI Hub credentials. Tokens never leave this class in
 * plaintext on disk — they live in EncryptedSharedPreferences (AES-256-GCM
 * with key from AndroidKeyStore).
 *
 * Held as a single JSON blob to make the multi-account update atomic.
 */
class AccountStore(context: Context) {

    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }
    private val prefs: SharedPreferences

    private val _state = MutableStateFlow(AccountsState())
    val state: StateFlow<AccountsState> = _state.asStateFlow()

    init {
        val masterKey = MasterKey.Builder(context.applicationContext)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build()
        prefs = EncryptedSharedPreferences.create(
            context.applicationContext,
            "npulab_accounts_v1",
            masterKey,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
        )
        val raw = prefs.getString(KEY_BLOB, null)
        if (raw != null) {
            runCatching { _state.value = json.decodeFromString<AccountsState>(raw) }
        }
    }

    fun add(label: String, token: String, apiUrl: String = AiHubAccount.DEFAULT_API_URL): AiHubAccount {
        val newAcc = AiHubAccount(
            id = UUID.randomUUID().toString(),
            label = label.ifBlank { "Default" },
            token = token,
            apiUrl = apiUrl,
        )
        val newAccounts = _state.value.accounts + newAcc
        val newActive = _state.value.activeId ?: newAcc.id
        commit(_state.value.copy(accounts = newAccounts, activeId = newActive))
        return newAcc
    }

    fun update(account: AiHubAccount) {
        val list = _state.value.accounts.map { if (it.id == account.id) account else it }
        commit(_state.value.copy(accounts = list))
    }

    fun remove(id: String) {
        val list = _state.value.accounts.filterNot { it.id == id }
        val active = _state.value.activeId.takeIf { id != it } ?: list.firstOrNull()?.id
        commit(AccountsState(list, active))
    }

    fun setActive(id: String) {
        if (_state.value.accounts.any { it.id == id }) {
            commit(_state.value.copy(activeId = id))
        }
    }

    fun markVerified(id: String, user: String?) {
        val list = _state.value.accounts.map {
            if (it.id == id) it.copy(verifiedUser = user, verifiedAtMs = System.currentTimeMillis())
            else it
        }
        commit(_state.value.copy(accounts = list))
    }

    fun activeToken(): String? = _state.value.active?.token

    private fun commit(s: AccountsState) {
        _state.value = s
        prefs.edit().putString(KEY_BLOB, json.encodeToString(AccountsState.serializer(), s)).apply()
    }

    companion object {
        private const val KEY_BLOB = "accounts_json"
    }
}
