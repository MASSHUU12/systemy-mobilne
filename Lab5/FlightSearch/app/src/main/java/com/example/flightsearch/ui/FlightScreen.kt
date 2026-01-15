package com.example.flightsearch.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.FavoriteBorder
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.example.flightsearch.data.Airport

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FlightScreen(
    modifier: Modifier = Modifier,
    viewModel: FlightViewModel = hiltViewModel()
) {
    val uiState by viewModel.uiState.collectAsState()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Flight Search") }
            )
        }
    ) { paddingValues ->
        Column(modifier = modifier.padding(paddingValues)) {
            SearchBar(
                query = uiState.searchQuery,
                onQueryChange = viewModel::onQueryChange,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp)
            )

            when {
                uiState.isShowingFavorites -> {
                    FavoriteList(
                        favorites = uiState.favoriteList,
                        modifier = Modifier.fillMaxSize()
                    )
                }
                uiState.selectedAirport == null -> {
                    AirportSuggestions(
                        airports = uiState.airportList,
                        onAirportClick = viewModel::onAirportSelected
                    )
                }
                else -> {
                    FlightList(
                        departureAirport = uiState.selectedAirport!!,
                        destinationAirports = uiState.flightList,
                        favorites = uiState.favoriteList,
                        onFavoriteClick = viewModel::onFavoriteClick,
                        modifier = Modifier.fillMaxSize()
                    )
                }
            }
        }
    }
}

@Composable
fun SearchBar(
    query: String,
    onQueryChange: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    TextField(
        value = query,
        onValueChange = onQueryChange,
        placeholder = { Text("Enter airport name or code") },
        modifier = modifier,
        singleLine = true
    )
}

@Composable
fun AirportSuggestions(
    airports: List<Airport>,
    onAirportClick: (Airport) -> Unit
) {
    LazyColumn {
        items(airports) { airport ->
            Text(
                text = "${airport.iataCode} - ${airport.name}",
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { onAirportClick(airport) }
                    .padding(16.dp)
            )
        }
    }
}

@Composable
fun FlightList(
    departureAirport: Airport,
    destinationAirports: List<Airport>,
    favorites: List<FavoriteDetail>,
    onFavoriteClick: (String, String) -> Unit,
    modifier: Modifier = Modifier
) {
    LazyColumn(modifier = modifier) {
        item {
            Text(
                text = "Flights from ${departureAirport.name}",
                style = MaterialTheme.typography.headlineSmall,
                modifier = Modifier.padding(16.dp)
            )
        }
        items(destinationAirports) { destination ->
            val isFavorite = favorites.any {
                it.departureAirport.iataCode == departureAirport.iataCode && it.destinationAirport.iataCode == destination.iataCode
            }
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 8.dp)
            ) {
                Row(
                    modifier = Modifier.padding(16.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = androidx.compose.ui.Alignment.CenterVertically
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text("DEPART")
                        Text(departureAirport.iataCode, style = MaterialTheme.typography.bodyLarge)
                        Text(departureAirport.name, style = MaterialTheme.typography.bodySmall)
                    }
                    Column(modifier = Modifier.weight(1f)) {
                        Text("ARRIVE")
                        Text(destination.iataCode, style = MaterialTheme.typography.bodyLarge)
                        Text(destination.name, style = MaterialTheme.typography.bodySmall)
                    }
                    IconButton(onClick = { onFavoriteClick(departureAirport.iataCode, destination.iataCode) }) {
                        Icon(
                            imageVector = if (isFavorite) Icons.Filled.Favorite else Icons.Default.FavoriteBorder,
                            contentDescription = if (isFavorite) "Remove from favorites" else "Add to favorites"
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun FavoriteList(
    favorites: List<FavoriteDetail>,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        if (favorites.isNotEmpty()) {
            Text(
                text = "Favorite Routes",
                style = MaterialTheme.typography.headlineSmall,
                modifier = Modifier.padding(16.dp)
            )
        }
        LazyColumn {
            items(favorites) { favorite ->
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 8.dp)
                ) {
                    Row(
                        modifier = Modifier.padding(16.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = androidx.compose.ui.Alignment.CenterVertically
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text("DEPART")
                            Text(favorite.departureAirport.iataCode, style = MaterialTheme.typography.bodyLarge)
                            Text(favorite.departureAirport.name, style = MaterialTheme.typography.bodySmall)
                        }
                        Column(modifier = Modifier.weight(1f)) {
                            Text("ARRIVE")
                            Text(favorite.destinationAirport.iataCode, style = MaterialTheme.typography.bodyLarge)
                            Text(favorite.destinationAirport.name, style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
            }
        }
    }
}
