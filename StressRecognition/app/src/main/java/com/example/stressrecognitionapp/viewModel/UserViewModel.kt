package com.example.stressrecognitionapp.viewModel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.example.stressrecognitionapp.data.User
import com.example.stressrecognitionapp.data.UserDatabase
import com.example.stressrecognitionapp.data.UserRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class UserViewModel(application: Application): AndroidViewModel(application) {
    private val repository: UserRepository

    init{
        val userDao = UserDatabase.getDatabase(application).userDao()
        repository = UserRepository(userDao)
    }

    fun getUserByUsername(inputUsername: String): LiveData<User?>{
        val result = MutableLiveData<User?>()
        viewModelScope.launch(Dispatchers.IO) {
            result.postValue(repository.getUserByUsername(inputUsername))
        }
        return result
    }

//    fun logout(){
//
//    }
}