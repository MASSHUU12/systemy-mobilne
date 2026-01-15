package com.example.flightsearch.data

import kotlinx.coroutines.flow.Flow

interface FlightRepository {
    fun getAirportsByQuery(query: String): Flow<List<Airport>>
    fun getAirportByCode(iataCode: String): Flow<Airport>
    fun getAllOtherAirports(iataCode: String): Flow<List<Airport>>
    fun getAllFavorites(): Flow<List<Favorite>>
    suspend fun addFavorite(favorite: Favorite)
    suspend fun removeFavorite(favorite: Favorite)
}


class FlightRepositoryImpl(private val flightDao: FlightDao) : FlightRepository {
    override fun getAirportsByQuery(query: String): Flow<List<Airport>> = flightDao.getAirportsByQuery(query)
    override fun getAirportByCode(iataCode: String): Flow<Airport> = flightDao.getAirportByCode(iataCode)
    override fun getAllOtherAirports(iataCode: String): Flow<List<Airport>> = flightDao.getAllOtherAirports(iataCode)
    override fun getAllFavorites(): Flow<List<Favorite>> = flightDao.getAllFavorites()
    override suspend fun addFavorite(favorite: Favorite) = flightDao.addFavorite(favorite)
    override suspend fun removeFavorite(favorite: Favorite) = flightDao.removeFavorite(favorite)
}