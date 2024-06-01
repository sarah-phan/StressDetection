package com.example.stressrecognitionapp.ui.layouts

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import com.example.stressrecognitionapp.R

@Composable
fun ShowUserData(
    listState: LazyListState,
    userData: List<List<Float>>,
    label: List<Int>,
    navController: NavController,
) {
    LazyColumn(
        state = listState,
        modifier = Modifier.padding(top = 20.dp),
        userScrollEnabled = true
    ) {
        userData?.let { segmentData ->
            label?.let{
                items(segmentData.size) { index ->
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(8.dp)
                            .background(
                                color = Color.White,
                                shape = RoundedCornerShape(10)
                            )
                            .padding(
                                horizontal = 13.dp,
                                vertical = 18.dp
                            )
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column(){
                                Text(
                                    text = "Segment $index",
                                    fontSize = 20.sp
                                )
                                Text(
                                    text = if(label[index] == 0) "Normal state" else "Stress state"
                                )
                            }
                            Row(
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier
                                    .background(
                                        color = Color(0xFF5156C4),
                                        shape = RoundedCornerShape(15)
                                    )
                                    .padding(8.dp)
                                    .clickable {
                                        navController.navigate("report-detail/$index")
                                    }
                            ) {
                                Text(
                                    text = "Show report",
                                    modifier = Modifier.padding(end = 5.dp),
                                    style = TextStyle(
                                        fontSize = 16.sp,
                                        color = Color.White
                                    ),
                                )
                                Icon(
                                    painter = painterResource(id = R.drawable.baseline_east_24),
                                    contentDescription = "East Icon",
                                    modifier = Modifier.size(20.dp),
                                    tint = Color.White
                                )
                            }
                        }
                    }
                }
            }

        }
    }
    FABMenu(navController = navController)
}



//@Preview
//@Composable
//fun FABPreview(){
//
//
//}