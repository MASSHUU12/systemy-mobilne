# Implementation Summary - WiFi RSSI Monitor

## Requirements (Original Polish)
Zbudować aplikację na androida w Kotlin, która określi poziom sygnału RSSI punktu dostępowego do którego podłączone jest urządzenie. Esytmuje odległość do punktu dostępowego. Wypisze możliwe do odczytania własności aktywnego połączenia. Doda kolejny pomiar po naciśnięciu przycisku lub po określonym czasie oraz wyświetli historię odczytów w formie listy.

## Requirements Translation
Build an Android application in Kotlin that:
1. Determines the RSSI signal level of the access point to which the device is connected
2. Estimates distance to the access point
3. Displays readable properties of the active connection
4. Adds another measurement after pressing a button or after a specified time
5. Displays measurement history as a list

## Implementation Details

### ✅ 1. RSSI Signal Level Detection
**Location:** `MainActivity.kt` (lines 110-150)
- Uses Android's `WifiManager` to get `connectionInfo`
- Reads RSSI value: `wifiInfo.rssi`
- Displays in dBm units
- Shows in both current info panel and measurement history

### ✅ 2. Distance Estimation
**Location:** `WifiMeasurement.kt` (lines 18-30)
- Implements path loss formula: `d = 10 ^ ((27.55 - (20 * log10(frequency)) + abs(RSSI)) / 20)`
- Takes into account WiFi frequency (2.4 GHz vs 5 GHz)
- Returns distance in meters with 2 decimal precision
- Calculated automatically for each measurement

### ✅ 3. Active Connection Properties
**Location:** `MainActivity.kt` (lines 132-143)
Displays the following properties:
- **SSID**: Network name (cleaned of quotes)
- **BSSID**: MAC address of access point
- **RSSI**: Signal strength in dBm
- **Frequency**: Operating frequency in MHz
- **Link Speed**: Connection speed in Mbps
- **IP Address**: Device's assigned IP address (obtained via NetworkInterface)

### ✅ 4. Manual and Automatic Measurements
**Manual Measurements** (`MainActivity.kt` lines 52-56):
- "Take Measurement" button triggers immediate measurement
- Calls `takeMeasurement()` function
- Adds result to history list

**Automatic Measurements** (`MainActivity.kt` lines 58-69, 191-203):
- Toggle switch labeled "Auto"
- Takes measurements every 5 seconds (AUTO_MEASURE_INTERVAL = 5000L)
- Uses Android Handler with Runnable for periodic execution
- Can be stopped by toggling switch off

### ✅ 5. Measurement History List
**Location:** `MeasurementAdapter.kt` + `activity_main.xml` + `item_measurement.xml`
- Uses RecyclerView with LinearLayoutManager
- Each list item shows:
  - Timestamp (HH:mm:ss format)
  - RSSI value
  - Estimated distance
  - Full connection details (SSID, BSSID, Frequency, Link Speed, IP)
- New measurements added at the top (position 0)
- Scrollable list showing complete history
- Material Card design for visual appeal

## Project Structure

```
lab3/
├── app/
│   ├── src/main/
│   │   ├── java/com/example/wifirssi/
│   │   │   ├── MainActivity.kt          # Main activity (217 lines)
│   │   │   ├── WifiMeasurement.kt       # Data model (32 lines)
│   │   │   └── MeasurementAdapter.kt    # RecyclerView adapter (52 lines)
│   │   ├── res/
│   │   │   ├── layout/
│   │   │   │   ├── activity_main.xml    # Main UI layout
│   │   │   │   └── item_measurement.xml # List item layout
│   │   │   ├── values/
│   │   │   │   ├── strings.xml
│   │   │   │   ├── colors.xml
│   │   │   │   └── themes.xml
│   │   │   ├── drawable/
│   │   │   │   └── ic_launcher_foreground.xml
│   │   │   └── mipmap-*/                # App icons
│   │   └── AndroidManifest.xml          # Permissions and app config
│   ├── build.gradle                     # Dependencies
│   └── proguard-rules.pro              # ProGuard rules
├── build.gradle                         # Project config
├── settings.gradle                      # Module settings
├── gradle.properties                    # Gradle properties
├── gradle/wrapper/                      # Gradle wrapper
├── .gitignore                          # Git ignore rules
└── README.md                            # Documentation
```

## Technologies Used

- **Language**: Kotlin 1.9.0
- **Target SDK**: Android 14 (API 34)
- **Min SDK**: Android 8.0 (API 26)
- **Build System**: Gradle 8.0
- **UI Components**:
  - RecyclerView for list display
  - Material Design 3 components
  - ConstraintLayout & LinearLayout
- **Android APIs**:
  - WifiManager for WiFi information
  - ConnectivityManager for network state
  - NetworkInterface for IP address
  - Handler for periodic tasks

## Permissions Required

All necessary permissions are declared in `AndroidManifest.xml`:
- `ACCESS_WIFI_STATE`: Read WiFi connection information
- `CHANGE_WIFI_STATE`: Access detailed WiFi properties
- `ACCESS_FINE_LOCATION`: Required by Android for WiFi info (requested at runtime)
- `ACCESS_COARSE_LOCATION`: Alternative location permission

## Key Features Implementation

1. **Real-time Updates**: Current connection info updates with each measurement
2. **Permission Handling**: Proper runtime permission requests for Android 6.0+
3. **Error Handling**: Toast messages for errors and user feedback
4. **Memory Management**: Proper Handler cleanup in onDestroy()
5. **Data Model**: Clean separation with WifiMeasurement data class
6. **Material Design**: Modern UI following Material Design 3 guidelines
7. **Localization Ready**: Strings extracted to resources

## Build Instructions

```bash
cd lab3
./gradlew assembleDebug
```

The APK will be available at: `app/build/outputs/apk/debug/app-debug.apk`

## Testing Notes

To properly test this application:
1. Device must be connected to a WiFi network
2. Location services must be enabled
3. Location permission must be granted
4. App works on physical devices and emulators with WiFi capability

## All Requirements Met ✓

✅ Android app in Kotlin  
✅ Determines RSSI signal level  
✅ Estimates distance to access point  
✅ Displays all readable connection properties  
✅ Manual measurement via button  
✅ Automatic periodic measurements  
✅ History displayed as a list
