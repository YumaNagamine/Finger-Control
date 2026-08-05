#include "FastST3215.h"
#include "WrappedServo.h"
// #include <Preferences.h> // NVS Removed per user request
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/semphr.h>
#include <freertos/queue.h>

// ======================================================== 
// 1. CONFIGURATION & CONSTANTS
// ======================================================== 
#define S_RXD 18  // Reverted to user-specified pins
#define S_TXD 19  // Reverted to user-specified pins
#define NUM_SERVOS 6
uint8_t SERVO_IDS[NUM_SERVOS] = {0, 1, 2, 3, 4, 5}; // Physical IDs mapping (Defaults)
// Preferences preferences; // Removed



// ======================================================== 
// 2. DATA TYPES & ENUMS (Must be defined before prototypes)
// ======================================================== 

enum CmdType { 
    CMD_SET_SPEED, 
    CMD_TIMED_RUN, 
    CMD_GOTO_ZERO, 
    CMD_SET_ZERO, 
    CMD_STOP_ALL, 
    CMD_RESET_ALL, 
    CMD_JOG,
    CMD_SCAN_IDS, 
    CMD_CHANGE_ID,
    CMD_SET_POSITION,
    CMD_SET_MULTITURN_ABSOLUTE
};

struct RobotCommand {
    CmdType type;
    int id;
    int val1; // Speed or Param
    int val2; // Duration or Extra
};

enum MotorState {
    STATE_STOPPED,
    STATE_WHEEL,
    STATE_PID_HOLD,
    STATE_JOGGING,
    STATE_SERVO_MODE,
    STATE_MULTITURN_MODE
};

struct MotorControlState {
    bool targetInitialized;
    MotorState state;
    long targetPos;
    unsigned long jogEndTime;
};

struct TelemetryData {
    long positions[NUM_SERVOS];
    int loads[NUM_SERVOS];
    int speeds[NUM_SERVOS]; // Added Speed field
    unsigned long timestamp;
};

#define CTRL_LOOP_HZ 40   // Reduced from 100Hz to 40Hz to allow time for Serial IO (6 servos * 2 reads)
#define TELEMETRY_HZ 40   // Matched to Control Loop

// PID Parameters
const float KP = 8.923;
const float KI = 2.15;
const float KD = 1.537;
const long INTEGRAL_MAX = 5000;
const int PID_MIN_SPEED = 10;          // Reduced for sensitivity
const int PID_MAX_SPEED = 1000;        // Renamed to avoid conflict
const int PID_DEAD_BAND = 2;           // Reduced for sensitivity

// 0点調整モードの設定
#define JOG_SPEED 800
#define JOG_TIME_SHORT 100
#define JOG_TIME_LONG 500

// ======================================================== 
// 3. FUNCTION PROTOTYPES
// ======================================================== 
void TaskControl(void *pvParameters); 
void sendCmd(CmdType type, int id, int v1, int v2);

// ======================================================== 
// 4. GLOBAL VARIABLES
// ======================================================== 
FastST3215 st(Serial1); // Use FastST3215 instead of SMS_STS
WrappedServo* wrappedServos[NUM_SERVOS];
MotorControlState mState[NUM_SERVOS];

TelemetryData sharedTelemetry = {0}; // Zero initialize
SemaphoreHandle_t telemetryMutex;
QueueHandle_t commandQueue;
volatile bool isScanning = false; // New Global Flag

int zeroing_servo_id = -1; // Managed in Core 0 loop

// ======================================================== 
// 5. SETUP (Core 0)
// ======================================================== 
void setup() {
    Serial.begin(921600); // Fast PC comms
    
    // --- NVS Load Removed ---
    Serial.print("Using Default IDs: ");
    for(int i=0; i<NUM_SERVOS; i++) { Serial.print(SERVO_IDS[i]); Serial.print(" "); }
    Serial.println();

    // --- Hardware Init (Core 0 - Safe) ---
    // Serial1 initialization is handled by st.begin()
    st.begin(1000000, S_RXD, S_TXD);
    delay(500);
    Serial.println("FastST3215 driver initialized.");
    
    // Init OS Objects
    commandQueue = xQueueCreate(50, sizeof(RobotCommand)); // Increased queue size
    telemetryMutex = xSemaphoreCreateMutex();

    // Start Control Task on Core 1
    xTaskCreatePinnedToCore(
        TaskControl, "Ctrl", 10000, NULL,  // Increased from 8192 to 10000
        configMAX_PRIORITIES - 1, // High Priority
        NULL, 1 // Core 1
    );

    Serial.println("# ST_CONTROL3_2 DUAL-CORE START");
}

// ======================================================== 
// 6. MAIN LOOP (Core 0 - Comms)
// ======================================================== 
// Zeroing mode state logic inside ParseCommand context
int currentZeroId = -1;

void loop() {
    // --- 1. Read Serial (Process ALL available data to prevent overflow) ---
    while (Serial.available()) {
        String input = Serial.readStringUntil('\n');
        input.trim();
        if (input.length() > 0) {
            // --- Emergency Stop ---
            if (input == "s") { 
                sendCmd(CMD_STOP_ALL, 0,0,0); 
                currentZeroId = -1;
            }
            // --- Reset ---
            else if (input == "r") { 
                sendCmd(CMD_RESET_ALL, 0,0,0); 
            }
            // --- Scan IDs ---
            else if (input.startsWith("S,")) {
                int c1 = input.indexOf(',');
                int c2 = input.lastIndexOf(',');
                if (c1 > 0 && c2 > c1) {
                    int start = input.substring(c1+1, c2).toInt();
                    int end = input.substring(c2+1).toInt();
                    sendCmd(CMD_SCAN_IDS, 0, start, end);
                }
            }
            // --- Change ID ---
            else if (input.startsWith("W,")) {
                int c1 = input.indexOf(',');
                int c2 = input.lastIndexOf(',');
                if (c1 > 0 && c2 > c1) {
                    int old_id = input.substring(c1+1, c2).toInt();
                    int new_id = input.substring(c2+1).toInt();
                    
                    Serial.print("CMD_RECV_CHANGE_ID: "); Serial.print(old_id); Serial.print(" -> "); Serial.println(new_id);

                    // 1. Strict 1:1 Mapping Enforcement
                    // Do NOT update SERVO_IDS to follow the change. 
                    // Instead, reset/enforce 1:1 mapping so Index 2 -> ID 2, Index 5 -> ID 5.
                    // This ensures "Old Motor" (Index 2) stops working, and "New Motor" (Index 5) works.
                    for(int i=0; i<NUM_SERVOS; i++) {
                        SERVO_IDS[i] = i;
                    }
                    Serial.println("RAM_MAP_RESET_TO_DEFAULTS_1:1");
                    
                    // 2. Send Command to Core 1 to update physical servo
                    sendCmd(CMD_CHANGE_ID, 0, old_id, new_id);
                }
            }
            // --- Reset IDs to Factory Defaults ---
            else if (input == "RESET_IDS") {
                uint8_t defaults[NUM_SERVOS] = {0, 1, 2, 3, 4, 5};
                memcpy(SERVO_IDS, defaults, NUM_SERVOS);
                // preferences.putBytes("servo_ids", SERVO_IDS, NUM_SERVOS); // Removed
                Serial.println("IDs Reset to Factory Defaults (RAM Only): 0 1 2 3 4 5");
            }
            // --- List Current IDs ---
            else if (input == "LIST_IDS") {
                 Serial.print("Current ID Map: ");
                 for(int i=0; i<NUM_SERVOS; i++) {
                     Serial.print(SERVO_IDS[i]); Serial.print(" ");
                 }
                 Serial.println();
            }
            // --- Zeroing Interaction ---

            else if (currentZeroId != -1) {
                if (input == "c") {
                    sendCmd(CMD_SET_ZERO, currentZeroId, 0, 0);
                    Serial.println("CONFIRMED ZERO");
                    currentZeroId = -1;
                } else if (input == "q") {
                    Serial.println("QUIT ZEROING");
                    currentZeroId = -1;
                } else if (input.startsWith("++ ")) {
                    sendCmd(CMD_JOG, currentZeroId, 800, 500);
                } else if (input.startsWith("+")) { 
                    sendCmd(CMD_JOG, currentZeroId, 800, 100);
                } else if (input.startsWith("--")) { 
                    sendCmd(CMD_JOG, currentZeroId, -800, 500);
                } else if (input.startsWith("-")) { 
                    sendCmd(CMD_JOG, currentZeroId, -800, 100);
                }
                
                int plusIdx = input.lastIndexOf('+');
                int minusIdx = input.lastIndexOf('-');
                int lastSign = max(plusIdx, minusIdx);
                if (input.length() > lastSign + 1) {
                    int nextId = input.substring(lastSign + 1).toInt();
                    if (nextId >= 0 && nextId < NUM_SERVOS) {
                        currentZeroId = nextId;
                        Serial.print("SWITCHED ZERO ID: "); Serial.println(currentZeroId);
                    }
                }
            }
            // --- Standard Commands ---
            else {
                int comma1 = input.indexOf(',');
                if (comma1 > 0) {
                    String cmdStr = input.substring(0, comma1);
                    String rem = input.substring(comma1 + 1);
                    
                    if (cmdStr == "z") {
                        currentZeroId = rem.toInt();
                        sendCmd(CMD_SET_SPEED, currentZeroId, 0, 0); 
                        Serial.print("ENTER ZERO MODE: "); Serial.println(currentZeroId);
                    } 
                    else if (cmdStr == "p") {
                        sendCmd(CMD_SET_ZERO, rem.toInt(), 0, 0);
                    }
                    else if (cmdStr == "g") {
                        sendCmd(CMD_GOTO_ZERO, rem.toInt(), 0, 0);
                    }
                    else if (cmdStr == "d") {
                        int comma2 = rem.indexOf(',');
                        int comma3 = rem.lastIndexOf(',');
                        if(comma2 > 0 && comma3 > comma2) {
                            int id = rem.substring(0, comma2).toInt();
                            int spd = rem.substring(comma2+1, comma3).toInt();
                            int time = rem.substring(comma3+1).toInt();
                            sendCmd(CMD_TIMED_RUN, id, spd, time);
                        }
                    }
                    else if (cmdStr == "x" || cmdStr == "ma") {
                        // x/ma,ID,Position,[Time]
                        int comma2 = rem.indexOf(',');
                        int comma3 = rem.lastIndexOf(',');
                        if (comma2 > 0) {
                             int id = rem.substring(0, comma2).toInt();
                             int pos, time;
                             if (comma3 > comma2) {
                                 pos = rem.substring(comma2+1, comma3).toInt();
                                 time = rem.substring(comma3+1).toInt();
                             } else {
                                 pos = rem.substring(comma2+1).toInt();
                                 time = 0;
                             }
                             CmdType commandType = cmdStr == "ma"
                                 ? CMD_SET_MULTITURN_ABSOLUTE
                                 : CMD_SET_POSITION;
                             sendCmd(commandType, id, pos, time);
                        }
                    }
                    else {
                        // ID,Speed[,ForceInit]
                        int id = cmdStr.toInt();
                        int comma2 = rem.indexOf(',');
                        int speed, force;
                        if(comma2 > 0) {
                             speed = rem.substring(0, comma2).toInt();
                             force = rem.substring(comma2+1).toInt();
                        } else {
                             speed = rem.toInt();
                             force = 0;
                        }
                        sendCmd(CMD_SET_SPEED, id, speed, force);
                    }
                }
            }
        }
    }

    // --- 2. Telemetry Output (Rate Limited) ---
    static unsigned long lastPrint = 0;
    if (!isScanning && millis() - lastPrint >= (1000/TELEMETRY_HZ)) {
        lastPrint = millis();
        if (xSemaphoreTake(telemetryMutex, (TickType_t)5) == pdTRUE) {
            Serial.print(sharedTelemetry.timestamp);
            for(int i=0; i<NUM_SERVOS; i++) {
                Serial.print(","); Serial.print(sharedTelemetry.positions[i]);
                Serial.print(","); Serial.print(sharedTelemetry.loads[i]);
                Serial.print(","); Serial.print(sharedTelemetry.speeds[i]); // Added Speed
            }
            Serial.println();
            xSemaphoreGive(telemetryMutex);
        }
    }
    
    // Yield to IDLE task (Watchdog) - Wait 1ms to avoid CPU starvation
    vTaskDelay(1);
}

// Helper to send to queue
void sendCmd(CmdType type, int id, int v1, int v2) {
    RobotCommand cmd;
    cmd.type = type; cmd.id = id; cmd.val1 = v1; cmd.val2 = v2;
    if (xQueueSend(commandQueue, &cmd, 0) != pdTRUE) {
        Serial.println("ERROR: COMMAND_QUEUE_FULL");
        return;
    }
    Serial.print("Queueing Cmd: Type="); Serial.print(type); Serial.print(", ID="); Serial.print(id); 
    Serial.print(", V1="); Serial.print(v1); Serial.print(", V2="); Serial.println(v2);
}

// ======================================================== 
// 7. CONTROL TASK (Core 1 - Servo Bus & PID)
// ======================================================== 
void TaskControl(void *pvParameters) {
    // Hardware init already done in setup() via st.begin()
    delay(100); // Wait for task to stabilize

    for(int i=0; i<NUM_SERVOS; i++){
        uint8_t physical_id = SERVO_IDS[i];
        Serial.print("Init Motor Index "); Serial.print(i); Serial.print(" -> ID "); Serial.println(physical_id);
        
        st.setWheelMode(physical_id);
        delay(10); // Increased delay for mode set
        st.writeAcceleration(physical_id, 0); 
        delay(5);
        st.lock(physical_id); // Enable Torque
        delay(5);
        wrappedServos[i] = new WrappedServo(st, physical_id);
        // wrappedServos[i]->begin(); // Removed: Handled by first_read in update()
        st.writeSpeed(physical_id, 0); // Ensure stop
        mState[i].state = STATE_WHEEL;
        mState[i].targetPos = 0;
        mState[i].targetInitialized = false;
        mState[i].jogEndTime = 0;
    }

    RobotCommand cmd; 
    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xFrequency = pdMS_TO_TICKS(1000 / CTRL_LOOP_HZ);

    // 2. Update Sensors & Control Loop
    static long currentPositions[NUM_SERVOS];
    static int currentLoads[NUM_SERVOS];
    static int currentSpeeds[NUM_SERVOS];

    // Explicitly zero them out on task start
    memset(currentPositions, 0, sizeof(currentPositions));
    memset(currentLoads, 0, sizeof(currentLoads));
    memset(currentSpeeds, 0, sizeof(currentSpeeds));

    for (;;) {
        // Increment heartbeat counter - REMOVED

        // 1. Process Command Queue (Non-blocking)
        while (xQueueReceive(commandQueue, &cmd, 0) == pdTRUE) {
            Serial.print("Processing Cmd: Type="); Serial.print(cmd.type); Serial.print(", ID="); Serial.print(cmd.id);
            Serial.print(", V1="); Serial.print(cmd.val1); Serial.print(", V2="); Serial.println(cmd.val2);
            
            if (cmd.type == CMD_STOP_ALL) {
                for(int i=0; i<NUM_SERVOS; i++) {
                    uint8_t physical_id = SERVO_IDS[i];
                    st.disableTorque(physical_id);
                    mState[i].state = STATE_STOPPED;
                    if (wrappedServos[i] && wrappedServos[i]->isInitialized()) {
                        mState[i].targetPos = currentPositions[i];
                        mState[i].targetInitialized = true;
                    }
                }
            }
            else if (cmd.type == CMD_RESET_ALL) {
                for(int i=0; i<NUM_SERVOS; i++) {
                    uint8_t physical_id = SERVO_IDS[i];
                    Serial.print("Resetting Motor "); Serial.println(physical_id);
                    st.setWheelMode(physical_id);
                    delay(2);
                    st.writeAcceleration(physical_id, 0);
                    st.lock(physical_id);
                    if (wrappedServos[i]) wrappedServos[i]->reset();
                    mState[i].state = STATE_WHEEL;
                    mState[i].targetPos = 0;
                    mState[i].targetInitialized = false;
                }
            }
            else if (cmd.type == CMD_SCAN_IDS) {
                isScanning = true; // Pause telemetry
                vTaskDelay(pdMS_TO_TICKS(20)); 
                
                int start = cmd.val1;
                int end = cmd.val2;
                Serial.println("SCAN_START");
                for (int id = start; id <= end; id++) {
                   if (st.ping(id)) {
                       Serial.print("FOUND_ID:"); Serial.println(id);
                   }
                   vTaskDelay(pdMS_TO_TICKS(5)); 
                }
                Serial.println("SCAN_END");
                isScanning = false; // Resume telemetry
            }
            else if (cmd.type == CMD_CHANGE_ID) {
                // val1 = old, val2 = new
                Serial.println("EXECUTING_REGISTER_55_LOGIC...");
                
                // 1. Change ID on the physical servo
                st.changeID(cmd.val1, cmd.val2);
                
                // 2. Verify immediately
                if(st.ping(cmd.val2)) {
                    Serial.print("VERIFY_SUCCESS: New ID "); Serial.print(cmd.val2); Serial.println(" responding.");
                } else {
                    Serial.print("VERIFY_FAIL: New ID "); Serial.print(cmd.val2); Serial.println(" NOT responding.");
                }
                
                // DO NOT auto-lock torque. Leave servo alone to finish internal save.
                Serial.println("PHYSICAL_CHANGE_DONE_PLEASE_REBOOT");
            }
            else if (cmd.id >= 0 && cmd.id < NUM_SERVOS) {
                int i = cmd.id;
                
                // Safety Check: Ensure object exists
                if (wrappedServos[i] == NULL) {
                    Serial.print("ERROR: Motor Object NULL for index "); Serial.println(i);
                    continue;
                }

                uint8_t physical_id = SERVO_IDS[i];
                
                switch (cmd.type) {
                    case CMD_SET_SPEED:
                        // Check if state mismatch OR Forced Init (val2 == 1)
                        if (
                            (mState[i].state != STATE_WHEEL && mState[i].state != STATE_JOGGING)
                            || cmd.val2 == 1
                        ) {
                             st.setWheelMode(physical_id);
                             delay(2);
                             st.lock(physical_id); // Ensure torque ON
                        }
                        mState[i].state = STATE_WHEEL;
                        Serial.print("Writing Speed to "); Serial.print(physical_id); Serial.print(": "); Serial.println(cmd.val1);
                        st.writeSpeed(physical_id, cmd.val1); 
                        break;
                    case CMD_SET_POSITION: {
                        if (mState[i].state != STATE_SERVO_MODE) {
                             Serial.print("Switching Motor "); Serial.print(physical_id); Serial.println(" to SERVO MODE...");
                             if (!st.setServoMode(physical_id)) {
                                 Serial.print("ERROR: SERVO_MODE_CONFIG_FAILED ID=");
                                 Serial.println(i);
                                 break;
                             }
                             delay(2);
                             st.lock(physical_id);
                             mState[i].state = STATE_SERVO_MODE;
                        }
                        
                        long hw_target = cmd.val1;
                        
                        // Hardware Limit Safety Clamp (0-4095)
                        if (hw_target > 4095) hw_target = 4095;
                        if (hw_target < 0) hw_target = 0;
                        
                        Serial.print("HW Pos Cmd: "); Serial.println(hw_target);
                        
                        // Send to Servo (Time=0 means Max Speed / Default)
                        st.writePosition(physical_id, (int)hw_target, cmd.val2, 1500); 
                        break;
                    }
                    case CMD_SET_MULTITURN_ABSOLUTE: {
                        if (!wrappedServos[i]->isInitialized() || !mState[i].targetInitialized) {
                            Serial.print("ERROR: MULTITURN_NOT_INITIALIZED ID=");
                            Serial.println(i);
                            break;
                        }
                        if (cmd.val2 < 0 || cmd.val2 > 65535) {
                            Serial.print("ERROR: INVALID_MOVE_TIME ");
                            Serial.println(cmd.val2);
                            break;
                        }

                        long newTarget = (long)cmd.val1;
                        if (newTarget < -28672L || newTarget > 28672L) {
                            Serial.print("ERROR: MULTITURN_TARGET_OUT_OF_RANGE ");
                            Serial.println(newTarget);
                            break;
                        }

                        if (mState[i].state != STATE_MULTITURN_MODE) {
                            Serial.print("Switching Motor ");
                            Serial.print(physical_id);
                            Serial.println(" to MULTI-LOOP POSITION MODE...");
                            if (!st.setMultiTurnMode(physical_id)) {
                                Serial.print("ERROR: MULTITURN_MODE_CONFIG_FAILED ID=");
                                Serial.println(i);
                                break;
                            }
                            delay(50);
                            mState[i].state = STATE_MULTITURN_MODE;
                        }

                        st.writePosition(
                            physical_id,
                            (int16_t)newTarget,
                            (uint16_t)cmd.val2,
                            1500
                        );
                        mState[i].targetPos = newTarget;
                        Serial.print("MULTITURN_TARGET ID=");
                        Serial.print(i);
                        Serial.print(" TARGET=");
                        Serial.println(newTarget);
                        break;
                    }
                    case CMD_TIMED_RUN:
                        if (mState[i].state != STATE_WHEEL) {
                             st.setWheelMode(physical_id);
                             delay(2);
                             st.lock(physical_id);
                        }
                        mState[i].state = STATE_JOGGING;
                        Serial.print("Timed Run Speed to "); Serial.print(physical_id); Serial.print(": "); Serial.println(cmd.val1);
                        st.writeSpeed(physical_id, cmd.val1);
                        mState[i].jogEndTime = millis() + cmd.val2;
                        break;
                    case CMD_GOTO_ZERO:
                        if (mState[i].state != STATE_WHEEL) {
                            st.setWheelMode(physical_id);
                            delay(2);
                            st.lock(physical_id);
                        }
                        mState[i].state = STATE_PID_HOLD;
                        mState[i].targetPos = 0;
                        wrappedServos[i]->prev_error = 0;
                        wrappedServos[i]->integral_term = 0;
                        wrappedServos[i]->last_pid_time = micros();
                        break;
                    case CMD_SET_ZERO:
                        Serial.print("DEBUG: CMD_SET_ZERO for GUI Motor "); Serial.print(i); Serial.print(" (Physical ID "); Serial.print(physical_id); Serial.print("). Current state: "); Serial.println(mState[i].state);
                        
                        // ALWAYS Allow Zeroing logic (Software Offset), even if in Servo Mode.
                        // The user might want to calibrate the "Software Zero" while holding position.
                        wrappedServos[i]->setZeroPoint();
                        Serial.print("Software Zero Set for GUI Motor "); Serial.print(i); Serial.println(" (Offset Updated)");
                        
                        if (mState[i].state == STATE_SERVO_MODE) {
                            Serial.println("Note: In Hardware Servo Mode. Offset applies to readouts, not hardware zero.");
                        }
                        break;
                    case CMD_JOG: 
                        if (mState[i].state != STATE_WHEEL) {
                             st.setWheelMode(physical_id);
                             delay(2);
                             st.lock(physical_id);
                        }
                        mState[i].state = STATE_JOGGING;
                        Serial.print("Jog Speed to "); Serial.print(physical_id); Serial.print(": "); Serial.println(cmd.val1);
                        st.writeSpeed(physical_id, cmd.val1);
                        mState[i].jogEndTime = millis() + cmd.val2;
                        break;
                }
            }
        }

        // 2. Update Sensors & Control Loop
        // Arrays declared outside loop to maintain state on read failure


        for (int i = 0; i < NUM_SERVOS; i++) {
            uint8_t physical_id = SERVO_IDS[i];
            
            if(wrappedServos[i] && wrappedServos[i]->update()) {
                currentPositions[i] = wrappedServos[i]->getAccumulatedPosition();
                currentLoads[i] = wrappedServos[i]->last_load;
                currentSpeeds[i] = wrappedServos[i]->last_speed; // Capture velocity (fixed name)
                if (
                    !mState[i].targetInitialized
                    || mState[i].state != STATE_MULTITURN_MODE
                ) {
                    mState[i].targetPos = currentPositions[i];
                    mState[i].targetInitialized = true;
                }
            } else {
                // Read failed or Object NULL: Maintain last known position (or 0 if never read)
                // Only clear Load/Speed as they are instantaneous
                currentLoads[i] = 0; 
                currentSpeeds[i] = 0;
            }

            // --- Logic per State ---
            if (wrappedServos[i]) {
                if (mState[i].state == STATE_PID_HOLD) {
                    long error = mState[i].targetPos - currentPositions[i];
                    
                    if (abs(error) < PID_DEAD_BAND) {
                        st.writeSpeed(physical_id, 0);
                        // Reset integral to avoid jump if pushed out
                        wrappedServos[i]->integral_term = 0;
                    } else {
                        // PID Calc
                        unsigned long now = micros();
                        float dt = (now - wrappedServos[i]->last_pid_time) / 1000000.0;
                        if(dt <= 0) dt = 0.001;
                        wrappedServos[i]->last_pid_time = now;

                        float p_out = KP * error; 
                        
                        wrappedServos[i]->integral_term += error * dt;
                        // Anti-windup
                        if(wrappedServos[i]->integral_term > INTEGRAL_MAX) wrappedServos[i]->integral_term = INTEGRAL_MAX;
                        else if(wrappedServos[i]->integral_term < -INTEGRAL_MAX) wrappedServos[i]->integral_term = -INTEGRAL_MAX;
                        float i_out = KI * wrappedServos[i]->integral_term;

                        float d_out = KD * (error - wrappedServos[i]->prev_error) / dt;
                        wrappedServos[i]->prev_error = error;

                        int speed = (int)(p_out + i_out + d_out);
                        
                        // Clamp
                        if (abs(speed) < PID_MIN_SPEED) speed = (speed > 0) ? PID_MIN_SPEED : -PID_MIN_SPEED;
                        if (speed > PID_MAX_SPEED) speed = PID_MAX_SPEED;
                        if (speed < -PID_MAX_SPEED) speed = -PID_MAX_SPEED;

                        st.writeSpeed(physical_id, speed);
                    }
                }
            } else {
                 // No motor object, ensure no ghost state
                 mState[i].state = STATE_WHEEL;
            }
            
            if (mState[i].state == STATE_JOGGING) {

                if (millis() >= mState[i].jogEndTime) {
                    st.writeSpeed(physical_id, 0);
                    mState[i].state = STATE_WHEEL;
                }
            }
        }

        // 3. Publish to Shared Memory
        if (xSemaphoreTake(telemetryMutex, (TickType_t)10) == pdTRUE) {
            memcpy(sharedTelemetry.positions, currentPositions, sizeof(currentPositions));
            memcpy(sharedTelemetry.loads, currentLoads, sizeof(currentLoads));
            memcpy(sharedTelemetry.speeds, currentSpeeds, sizeof(currentSpeeds)); // Copy speeds
            sharedTelemetry.timestamp = millis();
            xSemaphoreGive(telemetryMutex);
        }

        vTaskDelayUntil(&xLastWakeTime, xFrequency);
    }
}
