package com.example.flightsearch.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.flightsearch.data.Airport
import com.example.flightsearch.data.Favorite
import com.example.flightsearch.data.FlightRepository
import com.example.flightsearch.data.UserPreferencesRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import javax.inject.Inject

@OptIn(ExperimentalCoroutinesApi::class)
@HiltViewModel
class FlightViewModel @Inject constructor(
    private val flightRepository: FlightRepository,
    private val userPreferencesRepository: UserPreferencesRepository
) : ViewModel() {
    private val _searchQuery = MutableStateFlow("")
    private val _selectedAirport = MutableStateFlow<Airport?>(null)
    private val _isInitialized = MutableStateFlow(false)

    private val savedSearchQuery: StateFlow<String> = userPreferencesRepository.searchQuery
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), "")

    private val airportSuggestions: Flow<List<Airport>> = _searchQuery
        .filter { it.isNotBlank() }
        .flatMapLatest { query ->
            flightRepository.getAirportsByQuery("%$query%")
        }

    private val favoriteRoutes: Flow<List<FavoriteDetail>> = flightRepository.getAllFavorites()
        .flatMapLatest { favorites ->
            if (favorites.isEmpty()) {
                flowOf(emptyList())
            } else {
                val detailFlows = favorites.map { favorite ->
                    combine(
                        flightRepository.getAirportByCode(favorite.departureCode),
                        flightRepository.getAirportByCode(favorite.destinationCode)
                    ) { departure, destination ->
                        FavoriteDetail(
                            id = favorite.id,
                            departureAirport = departure,
                            destinationAirport = destination,
                            isFavorite = true
                        )
                    }
                }
                combine(detailFlows) { it.toList() }
            }
        }


    val uiState: StateFlow<FlightUiState> = combine(
        _searchQuery,
        savedSearchQuery,
        airportSuggestions.onStart { emit(emptyList()) },
        favoriteRoutes.onStart { emit(emptyList()) },
        _selectedAirport.asStateFlow()
    ) { query, savedQuery, suggestions, favorites, selected ->
        val currentQuery = if (_isInitialized.value) query else savedQuery

        if (currentQuery.isBlank()) {
            FlightUiState(
                searchQuery = "",
                favoriteList = favorites,
                isShowingFavorites = true
            )
        } else if (selected == null) {
            FlightUiState(
                searchQuery = currentQuery,
                airportList = suggestions,
            )
        } else {
            FlightUiState(
                searchQuery = currentQuery,
                selectedAirport = selected,
                flightList = flightRepository.getAllOtherAirports(selected.iataCode).first(),
                favoriteList = favorites
            )
        }
    }.stateIn(
        scope = viewModelScope,
        started = SharingStarted.WhileSubscribed(5000),
        initialValue = FlightUiState()
    )

    init {
        viewModelScope.launch {
            val initialQuery = savedSearchQuery.first()
            _searchQuery.value = initialQuery
            _isInitialized.value = true
        }
    }

    fun onQueryChange(query: String) {
        _searchQuery.value = query
        _selectedAirport.value = null
        viewModelScope.launch {
            userPreferencesRepository.saveSearchQuery(query)
        }
    }

    fun onAirportSelected(airport: Airport) {
        viewModelScope.launch {
            _searchQuery.value = airport.name
            _selectedAirport.value = airport
            userPreferencesRepository.saveSearchQuery(airport.name)
        }
    }

    fun onFavoriteClick(departureCode: String, destinationCode: String) {
        viewModelScope.launch {
            val favorite = uiState.value.favoriteList.find {
                it.departureAirport.iataCode == departureCode && it.destinationAirport.iataCode == destinationCode
            }
            if (favorite != null) {
                flightRepository.removeFavorite(Favorite(favorite.id, departureCode, destinationCode))
            } else {
                flightRepository.addFavorite(Favorite(departureCode = departureCode, destinationCode = destinationCode))
            }
        }
    }
}

data class FlightUiState(
    val searchQuery: String = "",
    val airportList: List<Airport> = emptyList(),
    val flightList: List<Airport> = emptyList(),
    val favoriteList: List<FavoriteDetail> = emptyList(),
    val selectedAirport: Airport? = null,
    val isShowingFavorites: Boolean = false
)

data class FavoriteDetail(
    val id: Int,
    val departureAirport: Airport,
    val destinationAirport: Airport,
    val isFavorite: Boolean
)