package com.example.wifirssi

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.wifi.WifiManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Button
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import java.net.Inet4Address
import java.net.NetworkInterface

class MainActivity : AppCompatActivity() {
    
    private lateinit var wifiManager: WifiManager
    private lateinit var connectivityManager: ConnectivityManager
    private lateinit var adapter: MeasurementAdapter
    private lateinit var currentInfoText: TextView
    private lateinit var measureButton: Button
    private lateinit var autoMeasureSwitch: Switch
    private lateinit var recyclerView: RecyclerView
    
    private val handler = Handler(Looper.getMainLooper())
    private var autoMeasureRunnable: Runnable? = null
    private val AUTO_MEASURE_INTERVAL = 5000L // 5 seconds
    
    private val PERMISSION_REQUEST_CODE = 1
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        wifiManager = applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
        connectivityManager = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        
        currentInfoText = findViewById(R.id.currentInfoText)
        measureButton = findViewById(R.id.measureButton)
        autoMeasureSwitch = findViewById(R.id.autoMeasureSwitch)
        recyclerView = findViewById(R.id.recyclerView)
        
        adapter = MeasurementAdapter()
        recyclerView.layoutManager = LinearLayoutManager(this)
        recyclerView.adapter = adapter
        
        measureButton.setOnClickListener {
            if (checkPermissions()) {
                takeMeasurement()
            } else {
                requestPermissions()
            }
        }
        
        autoMeasureSwitch.setOnCheckedChangeListener { _, isChecked ->
            if (isChecked) {
                if (checkPermissions()) {
                    startAutoMeasure()
                } else {
                    autoMeasureSwitch.isChecked = false
                    requestPermissions()
                }
            } else {
                stopAutoMeasure()
            }
        }
        
        if (checkPermissions()) {
            takeMeasurement()
        } else {
            requestPermissions()
        }
    }
    
    private fun checkPermissions(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.ACCESS_FINE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED
    }
    
    private fun requestPermissions() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(
                Manifest.permission.ACCESS_FINE_LOCATION,
                Manifest.permission.ACCESS_COARSE_LOCATION
            ),
            PERMISSION_REQUEST_CODE
        )
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                takeMeasurement()
            } else {
                Toast.makeText(
                    this,
                    "Location permission is required to access WiFi information",
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }
    
    private fun takeMeasurement() {
        try {
            val wifiInfo = wifiManager.connectionInfo
            
            if (wifiInfo.ssid == null || wifiInfo.ssid == "<unknown ssid>" || wifiInfo.ssid.isEmpty()) {
                currentInfoText.text = "Not connected to WiFi"
                Toast.makeText(this, "Not connected to WiFi", Toast.LENGTH_SHORT).show()
                return
            }
            
            val ssid = wifiInfo.ssid.replace("\"", "")
            val bssid = wifiInfo.bssid ?: "Unknown"
            val rssi = wifiInfo.rssi
            val frequency = wifiInfo.frequency
            val linkSpeed = wifiInfo.linkSpeed
            val ipAddress = getIpAddress()
            
            val measurement = WifiMeasurement(
                ssid = ssid,
                bssid = bssid,
                rssi = rssi,
                frequency = frequency,
                linkSpeed = linkSpeed,
                ipAddress = ipAddress
            )
            
            adapter.addMeasurement(measurement)
            updateCurrentInfo(measurement)
            
        } catch (e: Exception) {
            Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }
    
    private fun getIpAddress(): String {
        try {
            val network = connectivityManager.activeNetwork ?: return "N/A"
            val networkCapabilities = connectivityManager.getNetworkCapabilities(network)
            
            if (networkCapabilities?.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) == true) {
                val interfaces = NetworkInterface.getNetworkInterfaces()
                for (intf in interfaces) {
                    if (!intf.isLoopback && intf.isUp) {
                        val addresses = intf.inetAddresses
                        for (addr in addresses) {
                            if (!addr.isLoopbackAddress && addr is Inet4Address) {
                                return addr.hostAddress ?: "N/A"
                            }
                        }
                    }
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return "N/A"
    }
    
    private fun updateCurrentInfo(measurement: WifiMeasurement) {
        currentInfoText.text = """
            Current WiFi Connection
            
            SSID: ${measurement.ssid}
            BSSID: ${measurement.bssid}
            RSSI: ${measurement.rssi} dBm
            Frequency: ${measurement.frequency} MHz
            Link Speed: ${measurement.linkSpeed} Mbps
            IP Address: ${measurement.ipAddress}
            Estimated Distance: ${"%.2f".format(measurement.distanceMeters)} meters
        """.trimIndent()
    }
    
    private fun startAutoMeasure() {
        autoMeasureRunnable = object : Runnable {
            override fun run() {
                takeMeasurement()
                handler.postDelayed(this, AUTO_MEASURE_INTERVAL)
            }
        }
        handler.post(autoMeasureRunnable!!)
        Toast.makeText(this, "Auto-measure started (every 5 seconds)", Toast.LENGTH_SHORT).show()
    }
    
    private fun stopAutoMeasure() {
        autoMeasureRunnable?.let {
            handler.removeCallbacks(it)
            autoMeasureRunnable = null
        }
        Toast.makeText(this, "Auto-measure stopped", Toast.LENGTH_SHORT).show()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        stopAutoMeasure()
    }
}
