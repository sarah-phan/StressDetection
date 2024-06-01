package com.example.stressrecognitionapp.viewModel

import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.stressrecognitionapp.apiService.RetrofitClient
import com.example.stressrecognitionapp.apiService.UsernameRequest
import com.example.stressrecognitionapp.apiService.UsernameResponse
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class UserSentViewModel: ViewModel() {
//    var state by mutableStateOf(STATE.LOADING)
    private val _state = MutableStateFlow(ApiState.LOADING)
    val state: StateFlow<ApiState> = _state.asStateFlow()
    private var usernameReceivedStatus: UsernameResponse by mutableStateOf(UsernameResponse(""))
    var errorMsg: String by mutableStateOf("")

    fun getUsernameReceivedStatus(usernameRequest: UsernameRequest){
        viewModelScope.launch {
            _state.value = ApiState.LOADING
            val apiResponse = RetrofitClient.instance.getUsernameMessage(usernameRequest)
            try {
                if (apiResponse.isSuccessful && apiResponse.body() != null){
                    usernameReceivedStatus = apiResponse.body()!!
                    _state.value = ApiState.SUCCESS
                    Log.d("ApiSuccess", usernameReceivedStatus.message)
                }
            }
            catch (ex:Exception){
                errorMsg = ex.message!!.toString()
                _state.value = ApiState.FAILED
            }
        }
    }
}