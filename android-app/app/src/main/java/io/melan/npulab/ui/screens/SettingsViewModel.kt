package io.melan.npulab.ui.screens

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.melan.npulab.account.AccountStore
import io.melan.npulab.account.AccountsState
import io.melan.npulab.account.AiHubAccount
import io.melan.npulab.account.AiHubClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class SettingsViewModel(app: Application) : AndroidViewModel(app) {

    sealed interface VerifyState {
        data object Idle : VerifyState
        data class Verifying(val accountId: String) : VerifyState
        data class Success(val accountId: String, val displayName: String) : VerifyState
        data class Failure(val accountId: String, val message: String) : VerifyState
    }

    private val store = AccountStore(app.applicationContext)
    private val _verify = MutableStateFlow<VerifyState>(VerifyState.Idle)

    val accountsState: StateFlow<AccountsState> = store.state
    val verifyState: StateFlow<VerifyState> = _verify.asStateFlow()

    fun add(label: String, token: String) {
        if (token.isBlank()) return
        store.add(label.trim(), token.trim())
    }

    fun remove(id: String) = store.remove(id)
    fun setActive(id: String) = store.setActive(id)
    fun updateLabel(account: AiHubAccount, label: String) =
        store.update(account.copy(label = label.trim().ifBlank { "Default" }))

    fun verify(account: AiHubAccount) {
        _verify.value = VerifyState.Verifying(account.id)
        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) {
                AiHubClient.verifyToken(account.apiUrl, account.token)
            }
            _verify.value = when (result) {
                is AiHubClient.VerifyResult.Ok -> {
                    store.markVerified(account.id, result.displayName)
                    VerifyState.Success(account.id, result.displayName)
                }
                is AiHubClient.VerifyResult.Invalid ->
                    VerifyState.Failure(account.id, "Token rejected: ${result.message.take(120)}")
                is AiHubClient.VerifyResult.NetworkError ->
                    VerifyState.Failure(account.id, "Network: ${result.message.take(120)}")
            }
        }
    }

    fun clearVerifyState() {
        _verify.value = VerifyState.Idle
    }
}
