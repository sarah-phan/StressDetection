package com.example.stressrecognitionapp

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.rememberNavController
import com.example.stressrecognitionapp.ui.layouts.PageNavigationController
import com.example.stressrecognitionapp.ui.theme.StressRecognitionAppTheme
import com.example.stressrecognitionapp.viewModel.UserViewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            StressRecognitionAppTheme {
                val navControl = rememberNavController()
                val userViewModel: UserViewModel = viewModel()

                Surface(
                    modifier = Modifier.fillMaxSize(),
                ) {
                    PageNavigationController()
//                    Homepage(navControl, userViewModel)
//                    ReportDetail(3)
//                    Login(userViewModel = userViewModel, navController = navControl)
//                    RespDataChart()
//                    ErrorScreen(errorMessage = "Error")
                }
            }
        }
    }
}