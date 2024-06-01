package com.example.stressrecognitionapp.viewModel

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.stressrecognitionapp.apiService.ModelDataResponse
import com.example.stressrecognitionapp.apiService.RetrofitClient
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class ModelDataViewModel: ViewModel() {
    private val _state = MutableStateFlow(ApiState.LOADING)
    val state: StateFlow<ApiState> = _state.asStateFlow()
    private val _modelDataResponse = MutableStateFlow(ModelDataResponse())
    val modelDataResponse: StateFlow<ModelDataResponse> = _modelDataResponse.asStateFlow()
    var errorMsg: String by mutableStateOf("")

    fun getUserData(){
        viewModelScope.launch {
            _state.value = ApiState.LOADING
            val apiResponse = RetrofitClient.instance.getModelData()
            try {
                if(apiResponse.isSuccessful && apiResponse.body() != null){
                    _state.value = ApiState.SUCCESS
                    _modelDataResponse.value = apiResponse.body()!!
                }
            }
            catch (ex:Exception){
                errorMsg = ex.message!!.toString()
                _state.value = ApiState.FAILED
            }
        }
    }
}