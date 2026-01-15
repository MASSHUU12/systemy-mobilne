package com.example.cityrecommendationsapp.ui

import androidx.lifecycle.ViewModel
import com.example.cityrecommendationsapp.data.Category
import com.example.cityrecommendationsapp.data.DataSource
import com.example.cityrecommendationsapp.data.Recommendation
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

class CityViewModel : ViewModel() {
    private val _uiState = MutableStateFlow(CityUiState())
    val uiState: StateFlow<CityUiState> = _uiState.asStateFlow()

    init {
        val categories = DataSource.categories
        _uiState.value = CityUiState(
            categories = categories,
            currentCategory = categories.first(),
            currentRecommendation = categories.first().recommendations.first()
        )
    }

    fun updateCurrentCategory(category: Category) {
        _uiState.update {
            it.copy(
                currentCategory = category,
                currentRecommendation = category.recommendations.first()
            )
        }
    }

    fun updateCurrentRecommendation(recommendation: Recommendation) {
        _uiState.update {
            it.copy(currentRecommendation = recommendation)
        }
    }
}