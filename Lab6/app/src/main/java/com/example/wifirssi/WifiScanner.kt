package com.example.wifirssi

import android.content.Context
import android.net.ConnectivityManager
import android.net.wifi.WifiManager
import java.net.Inet4Address
import java.net.Inet6Address

class WifiScanner(context: Context) {

    private val wifiManager =
        context.applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
    private val connectivityManager =
        context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager

    companion object {
        const val UNKNOWN_SSID = "<unknown ssid>"
    }

    fun getWifiMeasurement(): Result<WifiMeasurement> {
        return try {
            val wifiInfo = wifiManager.connectionInfo

            if (wifiInfo == null || wifiInfo.ssid == UNKNOWN_SSID || wifiInfo.ssid.isEmpty()) {
                return Result.failure(Exception("Not connected to Wi-Fi"))
            }

            val measurement = WifiMeasurement(
                ssid = wifiInfo.ssid.replace("\"", ""),
                bssid = wifiInfo.bssid ?: "Unknown",
                rssi = wifiInfo.rssi,
                frequency = wifiInfo.frequency,
                linkSpeed = wifiInfo.linkSpeed,
                ipAddress = getIpAddress()
            )
            Result.success(measurement)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    private fun getIpAddress(): String {
        try {
            val network = connectivityManager.activeNetwork ?: return "N/A"
            val linkProperties = connectivityManager.getLinkProperties(network) ?: return "N/A"

            val ipv4 =
                linkProperties.linkAddresses.firstOrNull { it.address is Inet4Address }?.address?.hostAddress
            if (ipv4 != null) return ipv4

            // Fallback to IPv6
            val ipv6 =
                linkProperties.linkAddresses.firstOrNull { it.address is Inet6Address }?.address?.hostAddress
            return ipv6 ?: "N/A"
        } catch (e: Exception) {
            return "N/A"
        }
    }
}