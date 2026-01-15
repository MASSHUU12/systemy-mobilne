package com.example.flightsearch.di

import android.content.Context
import com.example.flightsearch.data.AppDatabase
import com.example.flightsearch.data.FlightDao
import com.example.flightsearch.data.FlightRepository
import com.example.flightsearch.data.FlightRepositoryImpl
import com.example.flightsearch.data.UserPreferencesRepository
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Provides
    @Singleton
    fun provideAppDatabase(@ApplicationContext context: Context): AppDatabase {
        return AppDatabase.getDatabase(context)
    }

    @Provides
    @Singleton
    fun provideFlightDao(appDatabase: AppDatabase): FlightDao {
        return appDatabase.flightDao()
    }

    @Provides
    @Singleton
    fun provideFlightRepository(flightDao: FlightDao): FlightRepository {
        return FlightRepositoryImpl(flightDao)
    }

    @Provides
    @Singleton
    fun provideUserPreferencesRepository(@ApplicationContext context: Context): UserPreferencesRepository {
        return UserPreferencesRepository(context)
    }
}