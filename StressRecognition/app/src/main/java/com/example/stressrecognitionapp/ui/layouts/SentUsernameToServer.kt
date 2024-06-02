package com.example.stressrecognitionapp.ui.layouts

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavController
import com.example.stressrecognitionapp.apiService.UsernameRequest
import com.example.stressrecognitionapp.viewModel.ApiState
import com.example.stressrecognitionapp.viewModel.UserSentViewModel

@Composable
fun SentUsernameToServer(username: String, navController: NavController) {
    val userSentViewModel = viewModel(modelClass = UserSentViewModel::class.java)
    val usernameRequest = UsernameRequest(username = username)

    LaunchedEffect(key1=true){
        userSentViewModel.getUsernameReceivedStatus(usernameRequest = usernameRequest)
    }

    val state by userSentViewModel.state.collectAsState()

    when (state) {
        ApiState.LOADING -> LoadingScreen()
        ApiState.SUCCESS -> {
            navController.navigate("homepage")
        }
        ApiState.FAILED -> ErrorScreen(errorMessage = userSentViewModel.errorMsg)
    }
}