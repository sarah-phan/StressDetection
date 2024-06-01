package com.example.stressrecognitionapp.ui.layouts

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.width
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

@Composable
fun LoadingScreen(){
    Box(
        contentAlignment = Alignment.Center,
        modifier = Modifier
            .background(color = Color(0xFF51A1C5).copy(alpha = 0.5f))
    ){
        CircularProgressIndicator(
            modifier = Modifier.width(60.dp),
            color = Color(0xFFC45160),
            strokeWidth = 4.dp
        )
    }
}

//@Preview
//@Composable
//fun LoadingScreenPreview(){
//    Box(
//        contentAlignment = Alignment.Center,
//        modifier = Modifier
//            .background(color = Color(0xFF51A1C5).copy(alpha = 0.5f))
//    ){
//        CircularProgressIndicator(
////            modifier = Modifier.width(50.dp),
//            color = Color(0xFFC45160),
//            strokeWidth = 4.dp
//        )
//    }
//}