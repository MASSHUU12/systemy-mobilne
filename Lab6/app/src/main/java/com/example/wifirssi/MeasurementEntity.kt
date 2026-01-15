package com.example.wifirssi

import android.content.Context
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Entity
import androidx.room.Insert
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase
import kotlinx.coroutines.flow.Flow

@Entity(tableName = "measurements")
data class MeasurementEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val ssid: String,
    val bssid: String,
    val rssi: Int,
    val frequency: Int,
    val linkSpeed: Int,
    val ipAddress: String,
    val distanceMeters: Double,
    val formattedTimestamp: String,
    val timestampMillis: Long
)

@Dao
interface MeasurementDao {
    @Query("SELECT * FROM measurements ORDER BY timestampMillis DESC")
    fun getAllMeasurements(): Flow<List<MeasurementEntity>>

    @Insert
    suspend fun insert(measurement: MeasurementEntity)

    @Query("DELETE FROM measurements")
    suspend fun deleteAll()
}

@Database(entities = [MeasurementEntity::class], version = 1, exportSchema = false)
abstract class AppDatabase : RoomDatabase() {
    abstract fun measurementDao(): MeasurementDao

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        fun getDatabase(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "wifi_rssi_database"
                ).build()
                INSTANCE = instance
                instance
            }
        }
    }
}