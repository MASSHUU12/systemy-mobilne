# WiFi RSSI Monitor - Lab 3

Android application that monitors WiFi signal strength and estimates distance to the access point.

## Features

- **RSSI Signal Measurement**: Displays the received signal strength indicator (RSSI) of the connected WiFi access point
- **Distance Estimation**: Calculates approximate distance to the access point using path loss formula
- **Connection Properties**: Shows all readable properties of the active WiFi connection:
  - SSID (Network name)
  - BSSID (MAC address of access point)
  - RSSI (Signal strength in dBm)
  - Frequency (2.4 GHz or 5 GHz)
  - Link Speed (Mbps)
  - IP Address
- **Manual Measurements**: Take a measurement on demand by pressing the button
- **Automatic Measurements**: Enable auto-measure mode to take measurements every 5 seconds
- **Measurement History**: View all measurements in a scrollable list with timestamps

## Requirements

- Android 8.0 (API level 26) or higher
- WiFi connection
- Location permission (required by Android to access WiFi information)

## Permissions

The app requires the following permissions:
- `ACCESS_WIFI_STATE`: To read WiFi connection information
- `CHANGE_WIFI_STATE`: To access detailed WiFi properties
- `ACCESS_FINE_LOCATION`: Required by Android to access WiFi scan results and connection info
- `ACCESS_COARSE_LOCATION`: Alternative location permission

## Building

To build the project:

```bash
cd lab3
./gradlew assembleDebug
```

The APK will be generated in `app/build/outputs/apk/debug/`

## Distance Calculation

The distance is estimated using a simplified path loss formula:

```
d = 10 ^ ((27.55 - (20 * log10(frequency)) + abs(RSSI)) / 20)
```

Where:
- d = distance in meters
- frequency = WiFi frequency in MHz
- RSSI = received signal strength in dBm

Note: This is an approximation and actual distance may vary based on:
- Environmental factors (walls, obstacles)
- WiFi router transmission power
- Antenna characteristics
- Interference from other devices

## Usage

1. Launch the app
2. Grant location permission when prompted
3. Ensure device is connected to a WiFi network
4. Press "Take Measurement" button to capture current WiFi metrics
5. Enable "Auto" switch to automatically take measurements every 5 seconds
6. View measurement history in the list below

## Project Structure

```
lab3/
├── app/
│   ├── src/main/
│   │   ├── java/com/example/wifirssi/
│   │   │   ├── MainActivity.kt          # Main activity with WiFi measurement logic
│   │   │   ├── WifiMeasurement.kt       # Data model for WiFi measurements
│   │   │   └── MeasurementAdapter.kt    # RecyclerView adapter for history list
│   │   ├── res/
│   │   │   ├── layout/
│   │   │   │   ├── activity_main.xml    # Main screen layout
│   │   │   │   └── item_measurement.xml # List item layout
│   │   │   └── values/
│   │   │       ├── strings.xml          # String resources
│   │   │       └── colors.xml           # Color resources
│   │   └── AndroidManifest.xml          # App manifest with permissions
│   └── build.gradle                     # App-level build configuration
├── build.gradle                         # Project-level build configuration
└── settings.gradle                      # Project settings
```
