package com.example.wifirssi

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.location.LocationManager
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.materialswitch.MaterialSwitch

class MainActivity : AppCompatActivity() {

    private val viewModel: MainViewModel by viewModels()
    private lateinit var adapter: MeasurementAdapter

    private lateinit var currentInfoText: TextView
    private lateinit var measureButton: Button
    private lateinit var clearHistoryButton: Button
    private lateinit var autoMeasureSwitch: MaterialSwitch

    private val locationModeReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (isLocationEnabled() && checkPermissions()) {
                if (autoMeasureSwitch.isChecked) {
                    viewModel.toggleAutoMeasure(true)
                }
            } else {
                if (autoMeasureSwitch.isChecked) {
                    viewModel.toggleAutoMeasure(false)
                    Toast.makeText(
                        this@MainActivity,
                        "Location disabled. Scanning paused.",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }
        }
    }

    companion object {
        const val PERMISSION_REQUEST_CODE = 1
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        setupUI()
        setupObservers()
        checkAndRequestPermissions()
    }

    override fun onStart() {
        super.onStart()
        val filter = IntentFilter(LocationManager.MODE_CHANGED_ACTION)
        registerReceiver(locationModeReceiver, filter)
    }

    override fun onStop() {
        super.onStop()
        unregisterReceiver(locationModeReceiver)
    }

    private fun setupUI() {
        currentInfoText = findViewById(R.id.currentInfoText)
        measureButton = findViewById(R.id.measureButton)
        autoMeasureSwitch = findViewById(R.id.autoMeasureSwitch)
        clearHistoryButton = findViewById(R.id.clearHistoryButton)

        val recyclerView: RecyclerView = findViewById(R.id.recyclerView)
        adapter = MeasurementAdapter()
        recyclerView.layoutManager = LinearLayoutManager(this)
        recyclerView.adapter = adapter

        viewModel.takeSingleMeasurement()

        measureButton.setOnClickListener {
            if (ensurePermissionsAndLocation()) {
                viewModel.takeSingleMeasurement()
            }
        }

        autoMeasureSwitch.setOnCheckedChangeListener { _, isChecked ->
            if (isChecked) {
                if (ensurePermissionsAndLocation()) {
                    viewModel.toggleAutoMeasure(true)
                } else {
                    autoMeasureSwitch.isChecked = false
                }
            } else {
                viewModel.toggleAutoMeasure(false)
            }
        }

        clearHistoryButton.setOnClickListener {
            showClearHistoryDialog()
        }
    }

    private fun showClearHistoryDialog() {
        AlertDialog.Builder(this)
            .setTitle("Clear History")
            .setMessage("Are you sure you want to delete all measurement history?")
            .setPositiveButton("Clear") { _, _ ->
                viewModel.clearHistory()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun setupObservers() {
        viewModel.latestMeasurement.observe(this) { updateCurrentInfoText() }
        viewModel.minRssi.observe(this) { updateCurrentInfoText() }
        viewModel.maxRssi.observe(this) { updateCurrentInfoText() }

        viewModel.historyList.observe(this) { history ->
            adapter.submitList(history)
            if (history.isNotEmpty()) {
                findViewById<RecyclerView>(R.id.recyclerView).smoothScrollToPosition(0)
            }
        }

        viewModel.errorMessage.observe(this) { msg ->
            msg?.let { Toast.makeText(this, it, Toast.LENGTH_SHORT).show() }
        }

        viewModel.isAutoMeasuring.observe(this) { isScanning ->
            if (autoMeasureSwitch.isChecked != isScanning) {
                autoMeasureSwitch.isChecked = isScanning
            }
        }
    }

    private fun updateCurrentInfoText() {
        val measurement = viewModel.latestMeasurement.value ?: return
        val min = viewModel.minRssi.value ?: measurement.rssi
        val max = viewModel.maxRssi.value ?: measurement.rssi

        val displayMin = if (min == Int.MAX_VALUE) "-" else min.toString()
        val displayMax = if (max == Int.MIN_VALUE) "-" else max.toString()

        currentInfoText.text = """
            Current WiFi Connection

            SSID: ${measurement.ssid}
            BSSID: ${measurement.bssid}
            RSSI (Avg): ${measurement.rssi} dBm
            Min: $displayMin dBm | Max: $displayMax dBm

            Frequency: ${measurement.frequency} MHz
            Link Speed: ${measurement.linkSpeed} Mbps
            IP Address: ${measurement.ipAddress}
            Est. Distance: ${"%.2f".format(measurement.distanceMeters)} m
        """.trimIndent()
    }

    private fun ensurePermissionsAndLocation(): Boolean {
        if (!checkPermissions()) {
            requestPermissions()
            return false
        }
        if (!isLocationEnabled()) {
            showEnableLocationDialog()
            return false
        }
        return true
    }

    private fun checkPermissions(): Boolean {
        return ContextCompat.checkSelfPermission(
            this, Manifest.permission.ACCESS_FINE_LOCATION
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

    private fun isLocationEnabled(): Boolean {
        val lm = getSystemService(LOCATION_SERVICE) as LocationManager
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            lm.isLocationEnabled
        } else {
            lm.isProviderEnabled(LocationManager.GPS_PROVIDER) || lm.isProviderEnabled(
                LocationManager.NETWORK_PROVIDER
            )
        }
    }

    private fun showEnableLocationDialog() {
        AlertDialog.Builder(this)
            .setTitle("Turn on Location")
            .setMessage("Location must be enabled to access Wi-Fi details.")
            .setPositiveButton("Settings") { _, _ ->
                startActivity(Intent(Settings.ACTION_LOCATION_SOURCE_SETTINGS))
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun checkAndRequestPermissions() {
        if (!checkPermissions()) {
            requestPermissions()
        } else if (!isLocationEnabled()) {
            showEnableLocationDialog()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                if (isLocationEnabled()) {
                    viewModel.takeSingleMeasurement()
                }
            } else {
                Toast.makeText(this, "Permission required", Toast.LENGTH_SHORT).show()
            }
        }
    }
}