package com.example.stressrecognitionapp.ui.layouts

import android.util.Log
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet

@Composable
fun RespSegmentDataReport(
    userData: List<Float>
){
    val scrollState = rememberScrollState()

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(
                top = 20.dp,
                start = 8.dp,
                end = 8.dp
            )
            .background(
                color = Color.White,
                shape = RoundedCornerShape(10)
            )
    ){
        Column(
            modifier = Modifier.padding(
                top = 13.dp,
                start = 13.dp,
                end = 13.dp,
                bottom = 20.dp
            )
        ){
            Text(
                text = "Your respiration data",
                fontSize = 20.sp,
                textAlign = TextAlign.Center,
                modifier = Modifier
                    .padding(bottom = 15.dp)
                    .fillMaxWidth(),
                fontWeight = FontWeight.SemiBold
            )
            AndroidView(
                modifier = Modifier
                    .horizontalScroll(scrollState)
                    .height(200.dp)
                    .width(800.dp),
                factory = {context ->
                    LineChart(context).apply {
                        val entries = userData.mapIndexed{index, value ->
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
}