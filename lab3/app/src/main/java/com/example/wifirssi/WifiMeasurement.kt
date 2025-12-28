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
        // Free space path loss constant at 1 meter
        // Derived from: FSPL = 20*log10(d) + 20*log10(f) + 20*log10(4π/c)
        // where c is speed of light
        private const val FSPL_CONSTANT = 27.55
        
        /**
         * Calculate approximate distance using path loss formula
         * Distance (meters) = 10 ^ ((FSPL - 20*log10(f) - 20) / 20)
         * 
         * Simplified formula: d = 10 ^ ((27.55 - (20 * log10(frequency)) + abs(RSSI)) / 20)
         */
        private fun calculateDistance(rssi: Int, frequency: Int): Double {
            val exp = (FSPL_CONSTANT - (20 * Math.log10(frequency.toDouble())) + Math.abs(rssi)) / 20.0
            return Math.pow(10.0, exp)
        }
    }
}
