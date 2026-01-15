package com.example.wifirssi

import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.abs
import kotlin.math.log10
import kotlin.math.pow

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
        get() = calculateMotleyKeenanDistance(rssi, frequency)

    val formattedTimestamp: String
        get() = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date(timestamp))

    companion object {
        // Motley-Keenan / Log-Distance Path Loss Model Constants
        // Formula: PL(d) = PL(d0) + 10 * n * log10(d) + WallLoss

        // Path Loss Exponent (n)
        // 2.0 = Free Space
        // 2.5 - 3.0 = Typical Office/Indoor environment
        private const val PATH_LOSS_EXPONENT = 2.5

        private const val WALL_LOSS = 0.0

        // Reference constant derived from FSPL at 1 meter:
        // FSPL(d=1m) = 20*log10(f_MHz) - 27.55
        private const val FSPL_CONSTANT_TERM = 27.55

        /**
         * Calculate approximate distance using Motley-Keenan Model logic.
         *
         * We derive distance d from the Path Loss formula:
         * PathLoss = |RSSI| (Assuming TxPower is effectively accounted for or 0dBm relative)
         * |RSSI| = (20 * log10(f)) - 27.55 + 10 * n * log10(d)
         *
         * Solving for d:
         * 10 * n * log10(d) = |RSSI| - (20 * log10(f) - 27.55)
         * log10(d) = (|RSSI| - 20*log10(f) + 27.55) / (10 * n)
         * d = 10 ^ exponent
         */
        private fun calculateMotleyKeenanDistance(rssi: Int, frequency: Int): Double {
            val pathLossAt1m = (20 * log10(frequency.toDouble())) - FSPL_CONSTANT_TERM
            val totalPathLoss = abs(rssi).toDouble()
            val exponent = (totalPathLoss - pathLossAt1m - WALL_LOSS) / (10.0 * PATH_LOSS_EXPONENT)

            return 10.0.pow(exponent)
        }
    }
}
