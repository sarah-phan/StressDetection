package com.example.stressrecognitionapp.apiService

import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST

data class UsernameRequest(val username: String)
data class UsernameResponse(val message: String)
data class ModelDataResponse(
    val data: List<List<Float>> = listOf(),
    val label: List<Int> = listOf(),
    val prediction_probability: List<List<Float>> = listOf()
)

interface ApiService {
    @POST("get-username")
    suspend fun getUsernameMessage(@Body usernameData: UsernameRequest): Response<UsernameResponse>

    @GET("data-label")
    suspend fun getModelData(): Response<ModelDataResponse>
}