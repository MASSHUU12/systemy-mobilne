package com.example.cityrecommendationsapp.data

import androidx.annotation.DrawableRes
import androidx.annotation.StringRes

data class Recommendation(
    @StringRes val name: Int,
    @StringRes val description: Int,
    @DrawableRes val image: Int
)

data class Category(
    @StringRes val name: Int,
    val recommendations: List<Recommendation>
)