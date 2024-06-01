package com.example.stressrecognitionapp.ui.page

import android.util.Log
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.stressrecognitionapp.ui.layouts.ErrorScreen
import com.example.stressrecognitionapp.ui.layouts.LoadingScreen
import com.example.stressrecognitionapp.viewModel.ApiState
import com.example.stressrecognitionapp.viewModel.ModelDataViewModel
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet

@Composable
fun RespDataChart(){
    val modelDataViewModel = viewModel(modelClass = ModelDataViewModel::class.java)
    LaunchedEffect(key1 = true){
        modelDataViewModel.getUserData()
    }
    val userDataResponse by modelDataViewModel.modelDataResponse.collectAsState()
    val state by modelDataViewModel.state.collectAsState()

    val userData = userDataResponse.data
    val label = userDataResponse.label

    when(state){
        ApiState.LOADING -> LoadingScreen()
        ApiState.SUCCESS -> {
            val combinedList: List<Float> = userData.flatMap { it }

            Box(
                modifier = Modifier.background(Color(0xFF51A1C5).copy(alpha = 0.5f))
            ){
                ShowAllRespChart(combinedList = combinedList)
            }

        }
        ApiState.FAILED -> ErrorScreen(errorMessage = modelDataViewModel.errorMsg)
    }
}

@Composable
fun ShowAllRespChart(
    combinedList: List<Float>
){
    val scrollState = rememberScrollState()

    Box(
        modifier = Modifier
            .background(
                color = Color.White,
                shape = RoundedCornerShape(10)
            )
    ){
        AndroidView(
            modifier = Modifier
                .horizontalScroll(scrollState)
                .height(1000.dp)
                .width(1800.dp),
            factory = {context ->
                LineChart(context).apply {
                    val entries = combinedList.mapIndexed{index, value ->
                        Entry(index.toFloat(), value)
                    }
                    Log.d("entries", entries.toString())
                    val dataSet = LineDataSet(entries, "Respiration Data").apply {
                        color = 0xFF5156C4.toInt()
                        setDrawValues(true)
                        setCircleColor(0xFF51A1C5.toInt())
                        lineWidth = 3f
                    }
                    data = LineData(dataSet)
                    description.text = ""
                    setTouchEnabled(true)
                    isDragEnabled = true
                    xAxis.setDrawLabels(true)
                } },
            update = { chart ->
                chart.invalidate() // Redraw the chart if data changes
            }
        )
    }

}