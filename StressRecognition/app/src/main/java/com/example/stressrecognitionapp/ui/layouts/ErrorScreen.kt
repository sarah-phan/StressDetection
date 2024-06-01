package com.example.stressrecognitionapp.ui.layouts

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun ErrorScreen(errorMessage: String){
    Box(
        contentAlignment = Alignment.Center,
        modifier = Modifier
            .background(color = Color.Black)

    ){
        Box(
            modifier = Modifier
                .background(color = Color(0xFF51A1C5))
                .padding(20.dp)
        ){
            Column(){
                Text(
                    text="Application Error",
                    color = Color.White,
                    fontWeight = FontWeight.ExtraBold,
                    fontSize = 24.sp,
                    modifier = Modifier.padding(bottom = 15.dp)
                )
                Text(
                    text = errorMessage,
                    color = Color.White,
                    fontSize = 18.sp,
                    modifier = Modifier.padding(bottom = 15.dp)
                )
                Text(
                    text = "OK",
                    fontWeight = FontWeight.Bold,
                    color = Color.White,
                    textAlign = TextAlign.End,
                    modifier = Modifier
                        .align(Alignment.End)
                        .clickable {  }
                )
            }
        }
    }
}
