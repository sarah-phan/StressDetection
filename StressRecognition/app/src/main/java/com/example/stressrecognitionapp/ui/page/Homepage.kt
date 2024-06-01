package com.example.stressrecognitionapp.ui.page

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavController
import com.example.stressrecognitionapp.ui.layouts.ErrorScreen
import com.example.stressrecognitionapp.ui.layouts.LoadingScreen
import com.example.stressrecognitionapp.ui.layouts.ShowUserData
import com.example.stressrecognitionapp.viewModel.ApiState
import com.example.stressrecognitionapp.viewModel.ModelDataViewModel

@Composable
fun Homepage(
    navController: NavController,
){
    val modelDataViewModel = viewModel(modelClass = ModelDataViewModel::class.java)
    // LaunchedEffect(key1 = true) ensure the network call is made only once
    // (jetpack compose has side effect that composable functions can lead to repeated calls on
    // every recompositions
    LaunchedEffect(key1 = true){
        modelDataViewModel.getUserData()
    }
    val userDataResponse by modelDataViewModel.modelDataResponse.collectAsState()
    val userData = userDataResponse.data
    val label = userDataResponse.label

    val state by modelDataViewModel.state.collectAsState()

//  The LaunchedEffect is used to collect changes to the LazyListState (specifically, changes to
//  the firstVisibleItemIndex and firstVisibleItemScrollOffset) and store them in the state
//  variables scrollIndex and scrollOffset. This way, if the LazyColumn is recomposed, it will
//  retain its scroll position because the listState is preserved due to the rememberLazyListState()
//  outside of the when block.

//    firstVisibleItemIndex refers to the index of the first item that is currently visible
//    firstVisibleItemScrollOffset refers to the scroll offset of the first visible item. This is
//    the pixel offset of the first visible item's start edge from the start edge of the viewport
    val listState = rememberLazyListState()
    val scrollIndex = remember {
        mutableStateOf(listState.firstVisibleItemIndex)
    }
    val scrollOffset = remember {
        mutableStateOf(listState.firstVisibleItemScrollOffset)
    }
    LaunchedEffect(listState) {
        // Collect changes to the list state and save them in state variables
        snapshotFlow { listState.firstVisibleItemScrollOffset to listState.firstVisibleItemIndex }
            .collect { position ->
                scrollIndex.value = position.first
                scrollOffset.value = position.second
            }
    }

    when(state){
        ApiState.LOADING -> LoadingScreen()
        ApiState.SUCCESS -> {
            Box(
                modifier = Modifier.background(Color(0xFF51A1C5).copy(alpha = 0.5f))
            ){
                ShowUserData(
                    listState = listState,
                    userData = userData,
                    label = label,
                    navController = navController,
                )
            }
        }
        ApiState.FAILED -> ErrorScreen(errorMessage = modelDataViewModel.errorMsg)
    }
}

