/*
  ST3215サーボ用 角度積算ラッパークラス (0点調整機能付き)
*/
#ifndef WRAPPED_SERVO_H
#define WRAPPED_SERVO_H

#include <Arduino.h>
#include "FastST3215.h" // Use custom driver

class WrappedServo {
private:
    FastST3215& st;              // Custom driver reference
    int servo_id;

    long last_pos;            // Raw 0-4095
    long accumulated_pos;     // Multiturn
    bool first_read;

    long zero_offset;
    long physical_zero_offset; // Track raw hardware position at zero time

public:
    // Telemetry
    int16_t last_load;
    int16_t last_speed;
    int16_t last_volts;

    // PID control state
    long prev_error;
    long integral_term;
    unsigned long last_pid_time;

    WrappedServo(FastST3215& servo_object, int id)
        : st(servo_object), servo_id(id), last_pos(0), 
          accumulated_pos(0), first_read(true), zero_offset(0), physical_zero_offset(0),
          last_load(0), last_speed(0), last_volts(0),
          prev_error(0), integral_term(0), last_pid_time(0) {}

    void begin() {
        int16_t p, v, l;
        if (st.readState(servo_id, p, v, l)) {
            last_pos = p;
            accumulated_pos = p;
            last_speed = v;
            last_load = l;
            first_read = false;
        }
        zero_offset = 0;
    }

    // Returns true if read successful
    bool update() {
        int16_t p, v, l;
        if (!st.readState(servo_id, p, v, l)) {
            return false; 
        }
        
        last_load = l;
        last_speed = v;
        // Voltage read not supported in readState yet, assumed handled elsewhere or dummy
        last_volts = 0; 

        if (first_read) {
            last_pos = p;
            accumulated_pos = p;
            first_read = false;
            return true;
        }

        long delta = (long)p - last_pos;
        if (delta > 2048) delta -= 4096; 
        else if (delta < -2048) delta += 4096;

        accumulated_pos += delta;
        last_pos = p;
        return true;
    }

    long getAccumulatedPosition() const {
        // Return Raw Hardware Position (0-4095) to match Hardware Servo Mode limits
        // and avoid negative accumulated values confusing the GUI tools.
        return last_pos; 
    }

    long getZeroOffset() const {
        return zero_offset;
    }
    
    long getPhysicalZeroOffset() const {
        return physical_zero_offset;
    }

    void setZeroPoint() {
        zero_offset = accumulated_pos;
        physical_zero_offset = last_pos; // Capture absolute hardware pos
    }

    void reset() {
        int16_t p, v, l;
        if (st.readState(servo_id, p, v, l)) {
            last_pos = p;
            accumulated_pos = 0; // Reset software accumulation to 0
            zero_offset = 0;     // Reset the offset as well
            prev_error = 0;
            integral_term = 0;
        }
    }
    
    void hardReset() {
        accumulated_pos = 0;
        zero_offset = 0;
        first_read = true;
    }
};

#endif
