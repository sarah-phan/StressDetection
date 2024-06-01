package com.example.stressrecognitionapp.ui.layouts

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import com.example.stressrecognitionapp.R

@Composable
fun FABMenu(
    navController: NavController
){
    var isExpanded by remember {
        mutableStateOf(false)
    }
    AnimatedVisibility(
        visible = isExpanded,
        enter = fadeIn(animationSpec = tween(300)),
        exit = fadeOut(animationSpec = tween(300))
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black.copy(alpha = 0.7f))
                .clickable { isExpanded = false }
        )
    }

    Box(
        contentAlignment = Alignment.BottomEnd,
        modifier = Modifier
            .fillMaxSize()
            .padding(
                bottom = 50.dp,
                end = 30.dp
            )
    ) {
        AnimatedVisibility(
            visible = isExpanded,
            enter = slideInVertically(initialOffsetY = { it }) + fadeIn(),
            exit = slideOutVertically(targetOffsetY = { it }) + fadeOut()
        ) {
            Column(
                horizontalAlignment = Alignment.End,
                modifier = Modifier.fillMaxWidth()
            ) {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(15.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.padding(bottom = 20.dp)
                ) {
                    Text(
                        text = "Respiration Data",
                        fontSize = 18.sp,
                        color = Color.White,
                        modifier = Modifier
                            .background(
                                color = Color(0xFF51A1C5),
                                shape = RoundedCornerShape(10)
                            )
                            .padding(6.dp)
                    )

                    FloatingActionButton(
                        onClick = {
                            navController.navigate("resp-data-chart")
                                  },
                        shape = CircleShape,
                        modifier = Modifier.size(65.dp),
                        containerColor = Color(0xFFC45160),
                    ) {
                        Icon(
                            painter = painterResource(id = R.drawable.summarize_24px),
                            contentDescription = "Respiration data",
                            modifier = Modifier.size(30.dp),
                            tint = Color.White
                        )
                    }
                }

                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(15.dp),
                    modifier = Modifier.padding(bottom = 20.dp)
                ) {
                    Text(
                        text = "Segment Detail",
                        fontSize = 18.sp,
                        color = Color.White,
                        modifier = Modifier
                            .background(
                                color = Color(0xFF51A1C5),
                                shape = RoundedCornerShape(10)
                            )
                            .padding(6.dp)
                    )
                    FloatingActionButton(
                        onClick = {
                                  navController.navigate("homepage")
                        },
                        shape = CircleShape,
                        modifier = Modifier.size(65.dp),
                        containerColor = Color(0xFFC45160),
                    ) {
                        Icon(
                            painter = painterResource(id = R.drawable.list_24px),
                            contentDescription = "Segment Detail",
                            modifier = Modifier.size(30.dp),
                            tint = Color.White
                        )
                    }
                }

                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(15.dp),
                    modifier = Modifier.padding(bottom = 20.dp)
                ) {
                    Text(
                        text = "Sign Out",
                        fontSize = 18.sp,
                        color = Color.White,
                        modifier = Modifier
                            .background(
                                color = Color(0xFF51A1C5),
                                shape = RoundedCornerShape(10)
                            )
                            .padding(6.dp)
                    )
                    FloatingActionButton(
                        onClick = {
                            navController.navigate("login"){
                                // Remove all entries from the back stack after 'login'
                                popUpTo("login"){
                                    inclusive = true
                                }
                            }

                        },
                        shape = CircleShape,
                        modifier = Modifier.size(65.dp),
                        containerColor = Color(0xFFC45160),
                    ) {
                        Icon(
                            painter = painterResource(id = R.drawable.logout_24px),
                            contentDescription = "Logout",
                            modifier = Modifier.size(30.dp),
                            tint = Color.White
                        )
                    }
                }
                Spacer(modifier = Modifier.height(80.dp))
            }
        }
        FloatingActionButton(
            onClick = { isExpanded = !isExpanded },
            shape = CircleShape,
            modifier = Modifier.size(80.dp),
            containerColor = Color(0xFFC45160),
            contentColor = Color.White
        ) {
            Icon(
                painter = painterResource(id = R.drawable.menu_24px),
                contentDescription = "FAB Menu",
                modifier = Modifier.size(30.dp)
            )
        }
    }
}