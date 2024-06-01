package com.example.stressrecognitionapp.ui.page

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavController
import com.example.stressrecognitionapp.R
import com.example.stressrecognitionapp.ui.layouts.ErrorScreen
import com.example.stressrecognitionapp.ui.layouts.FABMenu
import com.example.stressrecognitionapp.ui.layouts.LabelReport
import com.example.stressrecognitionapp.ui.layouts.LoadingScreen
import com.example.stressrecognitionapp.ui.layouts.RespSegmentDataReport
import com.example.stressrecognitionapp.viewModel.ApiState
import com.example.stressrecognitionapp.viewModel.ModelDataViewModel

@Composable
fun ReportDetail(
    chosenIndex: Int,
    navController: NavController,
){
    val modelDataViewModel = viewModel(modelClass = ModelDataViewModel::class.java)
    LaunchedEffect(key1 = true){
        modelDataViewModel.getUserData()
    }
    val userDataResponse by modelDataViewModel.modelDataResponse.collectAsState()
    val state by modelDataViewModel.state.collectAsState()

    val userData = userDataResponse.data
    val label = userDataResponse.label
    val predictionProbability = userDataResponse.prediction_probability

    when(state){
        ApiState.LOADING -> LoadingScreen()
        ApiState.SUCCESS -> {
            val userDataIndex = userData[chosenIndex]
            val labelIndex = label[chosenIndex]
            val predictionProbabilityIndex = predictionProbability[chosenIndex]

            Box(
                modifier = Modifier.background(Color(0xFF51A1C5).copy(alpha = 0.5f))
            ){
                Column {
                    Row(modifier = Modifier.padding(top = 13.dp)) {
                        Icon(
                            painter = painterResource(id = R.drawable.baseline_west_24),
                            contentDescription = "Back Icon",
                            tint = Color.Black,
                            modifier = Modifier
                                .padding(
                                    start = 10.dp,
                                    end = 10.dp,
                                    top = 2.dp
                                )
                                .size(30.dp)
                                .clickable {
                                    navController.popBackStack()
                                }
                            )
                        Text(
                            text = "Segment $chosenIndex",
                            fontSize = 24.sp,
                            color = Color.Black,
                            fontWeight = FontWeight.Bold
                        )
                    }
                    LabelReport(
                        label = labelIndex,
                        predictionProbability = predictionProbabilityIndex,
                    )
                    RespSegmentDataReport(
                        userData = userDataIndex,
                    )
                }
                FABMenu(navController = navController)
            }
        }
        ApiState.FAILED -> ErrorScreen(errorMessage = modelDataViewModel.errorMsg)
    }
}