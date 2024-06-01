package com.example.stressrecognitionapp.ui.layouts

import androidx.compose.runtime.Composable
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.stressrecognitionapp.ui.page.Homepage
import com.example.stressrecognitionapp.ui.page.Login
import com.example.stressrecognitionapp.ui.page.ReportDetail
import com.example.stressrecognitionapp.ui.page.RespDataChart
import com.example.stressrecognitionapp.viewModel.UserViewModel

@Composable
fun PageNavigationController(){
    val navController = rememberNavController()
    val userViewModel: UserViewModel = viewModel()

    NavHost(navController = navController, startDestination = "login"){
        composable("login"){
            Login(userViewModel = userViewModel, navController = navController)
        }
        composable("homepage"){
            Homepage(
                navController = navController,
            )
        }
        composable("report-detail/{chosenIndex}"){backStackEntry ->
            val chosenIndexStr = backStackEntry.arguments?.getString("chosenIndex")
            val chosenIndex = chosenIndexStr?.toIntOrNull() ?: 0
            ReportDetail(
                chosenIndex = chosenIndex,
                navController = navController,
            )
        }

        composable("resp-data-chart"){
            RespDataChart()
        }
    }
}
