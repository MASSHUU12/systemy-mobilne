package com.example.cityrecommendationsapp.ui

import com.example.cityrecommendationsapp.data.Category
import com.example.cityrecommendationsapp.data.Recommendation

data class CityUiState(
    val categories: List<Category> = emptyList(),
    val currentCategory: Category? = null,
    val currentRecommendation: Recommendation? = null
)