package com.example.cityrecommendationsapp.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material3.windowsizeclass.WindowWidthSizeClass
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.example.cityrecommendationsapp.ui.utils.WindowStateUtils
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.cityrecommendationsapp.data.Category
import com.example.cityrecommendationsapp.data.Recommendation

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CityApp(
    windowSize: WindowWidthSizeClass,
    modifier: Modifier = Modifier,
) {
    val viewModel: CityViewModel = viewModel()
    val uiState by viewModel.uiState.collectAsState()
    val navController = rememberNavController()

    val navigationType = when (windowSize) {
        WindowWidthSizeClass.Compact -> WindowStateUtils.COMPACT
        WindowWidthSizeClass.Medium -> WindowStateUtils.MEDIUM
        else -> WindowStateUtils.EXPANDED
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(id = com.example.cityrecommendationsapp.R.string.app_name)) },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            )
        }
    ) { paddingValues ->
        when (navigationType) {
            WindowStateUtils.EXPANDED -> {
                Row(modifier = modifier.padding(paddingValues)) {
                    CategoryListScreen(
                        categories = uiState.categories,
                        onCategoryClick = {
                            viewModel.updateCurrentCategory(it)
                        },
                        modifier = Modifier.weight(1f)
                    )
                    RecommendationListAndDetailScreen(
                        uiState = uiState,
                        onRecommendationClick = { viewModel.updateCurrentRecommendation(it) },
                        modifier = Modifier.weight(2f)
                    )
                }
            }

            else -> {
                NavHost(
                    navController = navController,
                    startDestination = "categories",
                    modifier = Modifier.padding(paddingValues)
                ) {
                    composable("categories") {
                        CategoryListScreen(
                            categories = uiState.categories,
                            onCategoryClick = { category ->
                                viewModel.updateCurrentCategory(category)
                                navController.navigate("recommendations")
                            }
                        )
                    }
                    composable("recommendations") {
                        RecommendationListAndDetailScreen(
                            uiState = uiState,
                            onRecommendationClick = { recommendation ->
                                viewModel.updateCurrentRecommendation(recommendation)
                            }
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun CategoryListScreen(
    categories: List<Category>,
    onCategoryClick: (Category) -> Unit,
    modifier: Modifier = Modifier
) {
    LazyColumn(modifier = modifier) {
        items(categories) { category ->
            CategoryListItem(
                category = category,
                onCategoryClick = { onCategoryClick(category) }
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CategoryListItem(
    category: Category,
    onCategoryClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier
            .fillMaxWidth()
            .padding(8.dp),
        onClick = onCategoryClick
    ) {
        Text(
            text = stringResource(id = category.name),
            style = MaterialTheme.typography.titleMedium,
            modifier = Modifier.padding(16.dp)
        )
    }
}

@Composable
fun RecommendationListAndDetailScreen(
    uiState: CityUiState,
    onRecommendationClick: (Recommendation) -> Unit,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        uiState.currentCategory?.let { category ->
            Text(
                text = stringResource(id = category.name),
                style = MaterialTheme.typography.headlineMedium,
                modifier = Modifier.padding(16.dp)
            )
            LazyColumn(modifier = Modifier.weight(1f)) {
                items(category.recommendations) { recommendation ->
                    RecommendationListItem(
                        recommendation = recommendation,
                        onRecommendationClick = { onRecommendationClick(recommendation) }
                    )
                }
            }

            uiState.currentRecommendation?.let { recommendation ->
                RecommendationDetail(
                    recommendation = recommendation,
                    modifier = Modifier.padding(16.dp)
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RecommendationListItem(
    recommendation: Recommendation,
    onRecommendationClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 8.dp),
        onClick = onRecommendationClick
    ) {
        Row(
            modifier = Modifier.padding(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Image(
                painter = painterResource(id = recommendation.image),
                contentDescription = null,
                modifier = Modifier
                    .size(64.dp)
                    .padding(end = 16.dp)
            )
            Text(
                text = stringResource(id = recommendation.name),
                style = MaterialTheme.typography.titleSmall
            )
        }
    }
}

@Composable
fun RecommendationDetail(recommendation: Recommendation, modifier: Modifier = Modifier) {
    Column(modifier = modifier) {
        Image(
            painter = painterResource(id = recommendation.image),
            contentDescription = null,
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 16.dp)
        )
        Text(
            text = stringResource(id = recommendation.name),
            style = MaterialTheme.typography.headlineSmall,
            modifier = Modifier.padding(bottom = 8.dp)
        )
        Text(
            text = stringResource(id = recommendation.description),
            style = MaterialTheme.typography.bodyLarge
        )
    }
}