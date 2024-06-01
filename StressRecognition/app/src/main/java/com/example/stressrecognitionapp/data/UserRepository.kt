package com.example.stressrecognitionapp.data

class UserRepository(
    private val userDao: UserDao
) {
    suspend fun getUserByUsername(inputUsername: String): User?{
        return userDao.getUserByUsername(inputUsername)
    }
}