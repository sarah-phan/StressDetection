package com.example.stressrecognitionapp.ui.layouts

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Fill
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import java.math.BigDecimal
import java.math.RoundingMode

@Composable
fun LabelReport(
    label:Int,
    predictionProbability:List<Float>,
){
    val percentageProbabilityList: MutableList<String> = mutableListOf()

    predictionProbability.forEach{
        val percentageProbability = ChangeToPercentage(value = it.toDouble())
        percentageProbabilityList.add(percentageProbability)
    }

    val labelStringFormFor0 = "Keep up the good work"
    val labelStringFormFor1 = "Action is needed for your stress"

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
    ) {
        Column(
            modifier = Modifier.padding(
                top = 13.dp,
                start = 13.dp,
                end = 13.dp,
                bottom = 20.dp
            )
        ) {
            Text(
                text = "REPORT",
                fontSize = 20.sp,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth(),
                fontWeight = FontWeight.SemiBold
            )
            Text(
                text = if (label == 0) labelStringFormFor0 else labelStringFormFor1,
                modifier = Modifier.padding(top = 13.dp),
                fontSize = 18.sp
            )
            Column() {
                Text(
                    text = "${percentageProbabilityList[0]} for normal state",
                    modifier = Modifier.padding(
                        top = 13.dp,
                        bottom = 4.dp
                    )
                )
                Canvas(
                    modifier = Modifier
                        .height(15.dp)
                        .fillMaxWidth()
                        .background(color = Color(0xFFEAEAEA))
                ) {
                    drawRect(
                        color = Color(0xFF5156C4),
                        size = size.copy(size.width * predictionProbability[0]),
                        style = Fill
                    )
                }
            }

            Box(
                modifier = Modifier.padding(top = 13.dp)
            ) {
                Column() {
                    Text(
                        text = "${percentageProbabilityList[1]} for stress state",
                        modifier = Modifier.padding(bottom = 4.dp)
                    )
                    Canvas(
                        modifier = Modifier
                            .height(15.dp)
                            .background(color = Color(0xFFEAEAEA))
                            .fillMaxWidth()
                    ) {
                        drawRect(
                            color = Color(0xFF5156C4),
                            size = size.copy(size.width * predictionProbability[1]),
                            style = Fill
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun ChangeToPercentage(value: Double):String{
    val percentageProbability = BigDecimal(value * 100).setScale(0, RoundingMode.HALF_UP)
    return "${percentageProbability.toPlainString()}%"
}