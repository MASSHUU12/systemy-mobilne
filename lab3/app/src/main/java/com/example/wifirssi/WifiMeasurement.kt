package com.example.wifirssi

import java.text.SimpleDateFormat
import java.util.*

data class WifiMeasurement(
    val ssid: String,
    val bssid: String,
    val rssi: Int,
    val frequency: Int,
    val linkSpeed: Int,
    val ipAddress: String,
    val timestamp: Long = System.currentTimeMillis()
) {
    val distanceMeters: Double
        get() = calculateDistance(rssi, frequency)
    
    val formattedTimestamp: String
        get() = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date(timestamp))
    
    companion object {
        /**
         * Calculate approximate distance using path loss formula
         * Distance (meters) = 10 ^ ((FSPL - 20*log10(f) - 20) / 20)
         * 
         * Simplified formula: d = 10 ^ ((27.55 - (20 * log10(frequency)) + abs(RSSI)) / 20)
         */
        private fun calculateDistance(rssi: Int, frequency: Int): Double {
            val exp = (27.55 - (20 * Math.log10(frequency.toDouble())) + Math.abs(rssi)) / 20.0
            return Math.pow(10.0, exp)
        }
    }
}
