package com.example.stressrecognitionapp.data

import androidx.room.Dao
import androidx.room.Query

@Dao
interface UserDao {
    @Query("Select * from user where username = :inputUsername")
    suspend fun getUserByUsername(inputUsername: String): User?
}