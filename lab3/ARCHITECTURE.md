# WiFi RSSI Monitor - UI and Architecture Overview

## Application Flow

```
┌─────────────────────────────────────────────────┐
│           MainActivity (Activity)                │
│                                                  │
│  ┌────────────────────────────────────────────┐ │
│  │  Current WiFi Info Panel (TextView)        │ │
│  │  - SSID, BSSID, RSSI, Frequency           │ │
│  │  - Link Speed, IP Address                 │ │
│  │  - Estimated Distance                     │ │
│  └────────────────────────────────────────────┘ │
│                                                  │
│  ┌─────────────────────┐  ┌──────────────────┐ │
│  │ Take Measurement    │  │ [Auto] Switch    │ │
│  │     (Button)        │  │                  │ │
│  └─────────────────────┘  └──────────────────┘ │
│                                                  │
│  ┌────────────────────────────────────────────┐ │
│  │      Measurement History (Title)           │ │
│  └────────────────────────────────────────────┘ │
│                                                  │
│  ┌────────────────────────────────────────────┐ │
│  │         RecyclerView (Scrollable)          │ │
│  │  ┌──────────────────────────────────────┐  │ │
│  │  │ Card: 12:34:56  RSSI: -45 dBm       │  │ │
│  │  │ Distance: 5.23 m                    │  │ │
│  │  │ SSID: MyWiFi | BSSID: AA:BB:CC...   │  │ │
│  │  └──────────────────────────────────────┘  │ │
│  │  ┌──────────────────────────────────────┐  │ │
│  │  │ Card: 12:34:51  RSSI: -47 dBm       │  │ │
│  │  │ Distance: 6.45 m                    │  │ │
│  │  │ SSID: MyWiFi | BSSID: AA:BB:CC...   │  │ │
│  │  └──────────────────────────────────────┘  │ │
│  │             ... more cards ...              │ │
│  └────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

## Component Architecture

```
MainActivity
    │
    ├── WifiManager (System Service)
    │   └── Gets WiFi connection information
    │
    ├── ConnectivityManager (System Service)
    │   └── Checks network connectivity
    │
    ├── Handler + Runnable
    │   └── Manages periodic measurements
    │
    └── MeasurementAdapter (RecyclerView)
        └── Displays list of WifiMeasurement objects

WifiMeasurement (Data Class)
    ├── Properties: SSID, BSSID, RSSI, Frequency, etc.
    └── Calculated: Distance (using path loss formula)
```

## Data Flow

```
User Action (Button/Timer)
        ↓
MainActivity.takeMeasurement()
        ↓
WifiManager.connectionInfo
        ↓
Create WifiMeasurement object
        ↓
Calculate distance (path loss formula)
        ↓
Add to MeasurementAdapter
        ↓
Update RecyclerView (UI)
        ↓
Update Current Info Panel
```

## Key Features Detail

### 1. RSSI Measurement
- **Source**: `WifiInfo.rssi`
- **Unit**: dBm (decibel-milliwatts)
- **Range**: Typically -30 to -90 dBm
  - -30 to -50: Excellent
  - -50 to -60: Good
  - -60 to -70: Fair
  - -70+: Poor

### 2. Distance Calculation
```kotlin
Formula: d = 10 ^ ((27.55 - (20 * log10(frequency)) + abs(RSSI)) / 20)

Example:
  RSSI = -45 dBm
  Frequency = 2437 MHz (Channel 6, 2.4 GHz)
  Distance ≈ 5.23 meters
```

### 3. Automatic Measurements
- **Interval**: 5 seconds (configurable via AUTO_MEASURE_INTERVAL)
- **Implementation**: Android Handler with postDelayed()
- **Lifecycle**: Started/stopped with switch, cleaned up in onDestroy()

### 4. Permissions Flow
```
App Launch
    ↓
Check Permissions
    ├── Granted → Take Measurement
    └── Not Granted → Request Permissions
            ↓
        User Response
            ├── Granted → Take Measurement
            └── Denied → Show Toast Message
```

## File Structure

```
lab3/
├── app/
│   ├── build.gradle              # Dependencies & build config
│   ├── proguard-rules.pro       # Code obfuscation rules
│   └── src/main/
│       ├── AndroidManifest.xml  # App config & permissions
│       ├── java/com/example/wifirssi/
│       │   ├── MainActivity.kt         # 220 lines - Main logic
│       │   ├── WifiMeasurement.kt      # 35 lines - Data model
│       │   └── MeasurementAdapter.kt   # 52 lines - List adapter
│       └── res/
│           ├── layout/
│           │   ├── activity_main.xml    # Main screen layout
│           │   └── item_measurement.xml # List item card
│           ├── values/
│           │   ├── strings.xml          # App strings
│           │   ├── colors.xml           # Color palette
│           │   └── themes.xml           # Material theme
│           ├── drawable/
│           │   └── ic_launcher_foreground.xml
│           └── mipmap-*/               # App icons (various sizes)
├── build.gradle                 # Project-level config
├── settings.gradle             # Module settings
├── gradle.properties           # Gradle settings
├── gradle/wrapper/             # Gradle wrapper
│   └── gradle-wrapper.properties
├── README.md                   # User documentation
├── IMPLEMENTATION.md           # Technical documentation
└── .gitignore                  # Git ignore rules
```

## Technologies Used

| Component | Technology | Version |
|-----------|------------|---------|
| Language | Kotlin | 1.9.0 |
| Build System | Gradle | 8.0 |
| Target SDK | Android 14 | API 34 |
| Min SDK | Android 8.0 | API 26 |
| UI Framework | Material Design 3 | 1.11.0 |
| Layout | LinearLayout, ConstraintLayout | - |
| List View | RecyclerView | 1.3.2 |

## Android APIs Used

### WiFi & Network
- `WifiManager.connectionInfo` - Get WiFi connection details
- `WifiInfo.{ssid, bssid, rssi, frequency, linkSpeed}` - Connection properties
- `ConnectivityManager.activeNetwork` - Check active network
- `NetworkInterface.networkInterfaces` - Get IP address

### UI Components
- `AppCompatActivity` - Base activity
- `RecyclerView` - Scrollable list
- `LinearLayoutManager` - List layout
- `MaterialCardView` - Card design for list items
- `TextView`, `Button`, `Switch` - UI elements

### Permissions
- Runtime permission handling for Android 6.0+
- `ActivityCompat.requestPermissions()`
- `ContextCompat.checkSelfPermission()`

### Threading
- `Handler(Looper.getMainLooper())` - UI thread operations
- `Runnable` - Periodic tasks
- `Handler.postDelayed()` - Scheduled execution

## Color Scheme

| Element | Color | Usage |
|---------|-------|-------|
| Primary | Purple (#6200EE) | App theme, buttons |
| Background | Light Gray (#F5F5F5) | Info panel background |
| RSSI | Orange (#FF6B35) | RSSI value highlight |
| Distance | Green (#4CAF50) | Distance value highlight |
| Text Details | Gray (#666666) | Secondary information |

## Screen Specifications

- **Orientation**: Portrait (default)
- **Min Width**: 320dp
- **Target Width**: 360dp - 480dp
- **Scrollable Content**: Yes (RecyclerView for history)
- **Input Methods**: Touch (button, switch)

## Testing Recommendations

1. **Device Requirements**:
   - Physical Android device or emulator
   - WiFi capability enabled
   - Location services enabled

2. **Test Scenarios**:
   - Connect to 2.4 GHz WiFi network
   - Connect to 5 GHz WiFi network
   - Take manual measurement
   - Enable auto-measurement
   - Walk closer/farther from router
   - Disconnect from WiFi
   - Deny/grant permissions

3. **Expected Behaviors**:
   - RSSI should change with distance
   - Distance estimation should increase when moving away
   - History list should grow with each measurement
   - Auto-measurement should update every 5 seconds
   - App should handle missing WiFi gracefully

## Future Enhancements (Optional)

- Export measurements to CSV
- Graph of RSSI over time
- Multiple AP comparison
- Custom measurement intervals
- WiFi scanner for nearby APs
- Signal strength visualization
- Distance accuracy improvements
- Save measurements to database
