package com.example.wifirssi

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

class MeasurementAdapter : RecyclerView.Adapter<MeasurementAdapter.ViewHolder>() {
    
    private val measurements = mutableListOf<WifiMeasurement>()
    
    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val timeText: TextView = view.findViewById(R.id.timeText)
        val rssiText: TextView = view.findViewById(R.id.rssiText)
        val distanceText: TextView = view.findViewById(R.id.distanceText)
        val detailsText: TextView = view.findViewById(R.id.detailsText)
    }
    
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_measurement, parent, false)
        return ViewHolder(view)
    }
    
    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val measurement = measurements[position]
        holder.timeText.text = measurement.formattedTimestamp
        holder.rssiText.text = "RSSI: ${measurement.rssi} dBm"
        holder.distanceText.text = "Distance: %.2f m".format(measurement.distanceMeters)
        holder.detailsText.text = """
            SSID: ${measurement.ssid}
            BSSID: ${measurement.bssid}
            Frequency: ${measurement.frequency} MHz
            Link Speed: ${measurement.linkSpeed} Mbps
            IP: ${measurement.ipAddress}
        """.trimIndent()
    }
    
    override fun getItemCount() = measurements.size
    
    fun addMeasurement(measurement: WifiMeasurement) {
        measurements.add(0, measurement) // Add at the beginning
        notifyItemInserted(0)
    }
    
    fun clearMeasurements() {
        measurements.clear()
        notifyDataSetChanged()
    }
}
