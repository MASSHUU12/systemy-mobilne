package com.example.wifirssi

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.asLiveData
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val wifiScanner = WifiScanner(application)
    private val dao = AppDatabase.getDatabase(application).measurementDao()

    private val _latestMeasurement = MutableLiveData<WifiMeasurement?>()
    val latestMeasurement: LiveData<WifiMeasurement?> = _latestMeasurement

    private val _errorMessage = MutableLiveData<String?>()
    val errorMessage: LiveData<String?> = _errorMessage

    private val _isAutoMeasuring = MutableLiveData(false)
    val isAutoMeasuring: LiveData<Boolean> = _isAutoMeasuring

    private val _minRssi = MutableLiveData<Int>()
    val minRssi: LiveData<Int> = _minRssi

    private val _maxRssi = MutableLiveData<Int>()
    val maxRssi: LiveData<Int> = _maxRssi

    private var autoMeasureJob: Job? = null

    val historyList: LiveData<List<WifiMeasurement>> = dao.getAllMeasurements()
        .map { entities ->
            entities.map { entity -> entity.toDomainModel() }
        }
        .asLiveData()

    private val rssiBuffer = ArrayDeque<Int>()

    companion object {
        const val AUTO_MEASURE_INTERVAL = 3000L
        const val MOVING_AVERAGE_WINDOW = 5
    }

    fun takeSingleMeasurement() {
        val result = wifiScanner.getWifiMeasurement()
        result.fold(
            onSuccess = { measurement ->
                updateMeasurementData(measurement)
            },
            onFailure = { e ->
                _errorMessage.value = e.message
                viewModelScope.launch {
                    delay(2000)
                    _errorMessage.value = null
                }
            }
        )
    }

    fun toggleAutoMeasure(enable: Boolean) {
        if (enable) {
            if (_isAutoMeasuring.value == true) return
            startAutoMeasureLoop()
        } else {
            stopAutoMeasureLoop()
        }
    }

    fun clearHistory() {
        viewModelScope.launch {
            dao.deleteAll()
            _minRssi.value = Int.MAX_VALUE
            _maxRssi.value = Int.MIN_VALUE
        }
    }

    private fun startAutoMeasureLoop() {
        _isAutoMeasuring.value = true
        autoMeasureJob?.cancel()
        autoMeasureJob = viewModelScope.launch {
            while (isActive) {
                takeSingleMeasurement()
                delay(AUTO_MEASURE_INTERVAL)
            }
        }
    }

    private fun stopAutoMeasureLoop() {
        _isAutoMeasuring.value = false
        autoMeasureJob?.cancel()
        autoMeasureJob = null
    }

    private fun updateMeasurementData(measurement: WifiMeasurement) {
        val currentRssi = measurement.rssi

        val currentMin =
            if (_minRssi.value == Int.MAX_VALUE) currentRssi else (_minRssi.value ?: currentRssi)
        val currentMax =
            if (_maxRssi.value == Int.MIN_VALUE) currentRssi else (_maxRssi.value ?: currentRssi)

        if (currentRssi <= currentMin) _minRssi.value = currentRssi
        if (currentRssi >= currentMax) _maxRssi.value = currentRssi

        if (rssiBuffer.size >= MOVING_AVERAGE_WINDOW) {
            rssiBuffer.removeFirst()
        }
        rssiBuffer.addLast(currentRssi)

        val avgRssi = rssiBuffer.average().toInt()
        val smoothedMeasurement = measurement.copy(rssi = avgRssi)

        _latestMeasurement.value = smoothedMeasurement

        viewModelScope.launch {
            dao.insert(smoothedMeasurement.toEntity())
        }
    }

    private fun WifiMeasurement.toEntity(): MeasurementEntity {
        return MeasurementEntity(
            ssid = this.ssid,
            bssid = this.bssid,
            rssi = this.rssi,
            frequency = this.frequency,
            linkSpeed = this.linkSpeed,
            ipAddress = this.ipAddress,
            distanceMeters = this.distanceMeters,
            formattedTimestamp = this.formattedTimestamp,
            timestampMillis = System.currentTimeMillis()
        )
    }

    private fun MeasurementEntity.toDomainModel(): WifiMeasurement {
        return WifiMeasurement(
            ssid = this.ssid,
            bssid = this.bssid,
            rssi = this.rssi,
            frequency = this.frequency,
            linkSpeed = this.linkSpeed,
            ipAddress = this.ipAddress,
        )
    }
}