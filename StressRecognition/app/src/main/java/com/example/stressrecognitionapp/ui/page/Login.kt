package com.example.stressrecognitionapp.ui.page

import android.util.Log
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.LifecycleOwner
import androidx.navigation.NavController
import com.example.stressrecognitionapp.R
import com.example.stressrecognitionapp.ui.layouts.SentUsernameToServer
import com.example.stressrecognitionapp.viewModel.UserViewModel
import org.mindrot.jbcrypt.BCrypt

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun Login(userViewModel: UserViewModel, navController: NavController) {
    var username by remember{
        mutableStateOf("")
    }
    var password by remember{
        mutableStateOf("")
    }
    var showErrorMessage by remember {
        mutableStateOf(false)
    }
    var passwordShow by remember {
        mutableStateOf(false)
    }
    val context = LocalContext.current
    var loginSuccess by remember {
        mutableStateOf(false)
    }

    Log.d("passwordShow", BCrypt.hashpw(password, BCrypt.gensalt()))

    if(loginSuccess){
        SentUsernameToServer(username = username, navController = navController)
    }

    Box(
        modifier = Modifier
        .background(Color(0xFF51A1C5).copy(alpha = 0.5f))
        .fillMaxSize()
    ) {
        Column(
            modifier = Modifier.padding(10.dp),
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = "Login",
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                color = Color.Black,
                modifier = Modifier.padding(
                    bottom = 10.dp
                )
            )
            if (showErrorMessage) {
                Text(
                    text = "Invalid username or password",
                    color = Color(0xFFC45160),
                    modifier = Modifier.padding(
                        bottom = 10.dp
                    )
                )
            }
            TextField(
                modifier = Modifier
                    .padding(bottom = 10.dp)
                    .fillMaxWidth(),
                value = username,
                onValueChange = { username = it },
                label = { Text(text = "Username") }
            )
            TextField(
                modifier = Modifier
                    .padding(bottom = 10.dp)
                    .fillMaxWidth(),
                value = password,
                onValueChange = { password = it },
                label = { Text(text = "Password") },
                visualTransformation =
                if (passwordShow) VisualTransformation.None
                else PasswordVisualTransformation(),
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                trailingIcon = {
                    val image =
                        if (passwordShow) R.drawable.baseline_visibility_24
                        else R.drawable.baseline_visibility_off_24
                    val description = if (passwordShow) "Show password" else "Hide password"
                    IconButton(
                        onClick = { passwordShow = !passwordShow },) {
                        Box(){
                            Icon(
                                        painter = painterResource(id = image),
                                        contentDescription = description,
                                        modifier = Modifier.fillMaxWidth()
                            )
                        }
                    }
                }
            )
            Button(
                onClick = {
                    val userLiveData = userViewModel.getUserByUsername(username)
                    userLiveData.observe(context as LifecycleOwner) { user ->
                        if (user != null) {
                            if (BCrypt.checkpw(password, user.password)) {
                                loginSuccess = true
                            } else {
                                showErrorMessage = true
                                username = ""
                                password = ""
                            }
                        } else {
                            showErrorMessage = true
                            username = ""
                            password = ""
                        }
                    } },
                modifier = Modifier.align(alignment = Alignment.End),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF5156C4))
            ) {
                Text(
                    text = "Login",
                    fontSize = 18.sp
                )
            }
        }
    }
}
