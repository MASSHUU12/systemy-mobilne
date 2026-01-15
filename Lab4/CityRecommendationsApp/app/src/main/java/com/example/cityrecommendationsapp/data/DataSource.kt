package com.example.cityrecommendationsapp.data

import com.example.cityrecommendationsapp.R

object DataSource {
    val categories = listOf(
        Category(
            name = R.string.category_landmarks,
            recommendations = listOf(
                Recommendation(
                    R.string.landmark_eiffel_tower,
                    R.string.landmark_eiffel_tower_desc,
                    R.drawable.image1
                ),
                Recommendation(
                    R.string.landmark_louvre,
                    R.string.landmark_louvre_desc,
                    R.drawable.image2
                ),
                Recommendation(
                    R.string.landmark_notre_dame,
                    R.string.landmark_notre_dame_desc,
                    R.drawable.image3
                ),
                Recommendation(
                    R.string.landmark_arc_de_triomphe,
                    R.string.landmark_arc_de_triomphe_desc,
                    R.drawable.image4
                )
            )
        ),
        Category(
            name = R.string.category_museums,
            recommendations = listOf(
                Recommendation(
                    R.string.museum_orsay,
                    R.string.museum_orsay_desc,
                    R.drawable.image5
                ),
                Recommendation(
                    R.string.museum_pompidou,
                    R.string.museum_pompidou_desc,
                    R.drawable.image6
                ),
                Recommendation(R.string.museum_rodin, R.string.museum_rodin_desc, R.drawable.image7)
            )
        ),
        Category(
            name = R.string.category_food,
            recommendations = listOf(
                Recommendation(
                    R.string.food_le_relais,
                    R.string.food_le_relais_desc,
                    R.drawable.image8
                ),
                Recommendation(
                    R.string.food_bouillon_chartier,
                    R.string.food_bouillon_chartier_desc,
                    R.drawable.image9
                ),
                Recommendation(
                    R.string.food_l_as_du_fallafel,
                    R.string.food_l_as_du_fallafel_desc,
                    R.drawable.image10
                ),
                Recommendation(
                    R.string.food_pierre_herme,
                    R.string.food_pierre_herme_desc,
                    R.drawable.image11
                )
            )
        )
    )
}