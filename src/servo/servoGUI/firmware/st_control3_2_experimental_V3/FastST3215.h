#ifndef FAST_ST3215_H
#define FAST_ST3215_H

#include <Arduino.h>

class FastST3215 {
public:
    FastST3215(HardwareSerial& serial) : _serial(&serial) {}

    void begin(unsigned long baud, int8_t rxPin, int8_t txPin) {
        _serial->begin(baud, SERIAL_8N1, rxPin, txPin);
    }

    void writeRegister(uint8_t id, uint8_t reg, uint8_t val) {
        // Length = Instr(1) + Reg(1) + Val(1) + Checksum(1) = 4
        uint8_t p[] = {0xFF, 0xFF, id, 0x04, 0x03, reg, val, 0x00};
        uint16_t sum = id + 4 + 3 + reg + val;
        p[7] = ~(sum & 0xFF);
        while (_serial->available()) _serial->read();
        _serial->write(p, 8);
        _serial->flush(); // Ensure transmission complete
        // No delay needed at 1Mbps
    }

    // Set Acceleration (Reg 41 / 0x29)
    // 0 = Max (Instant), 254 = Slowest
    void writeAcceleration(uint8_t id, uint8_t acc) {
        writeRegister(id, 0x29, acc);
    }

    // Torque Enable (Address 40 / 0x28)
    void enableTorque(uint8_t id) { writeRegister(id, 0x28, 1); }
    void disableTorque(uint8_t id) { writeRegister(id, 0x28, 0); }

    // EEPROM Lock (Address 48 / 0x30) - 0=Unlock, 1=Lock
    // Reverted to Reg 48 for standard operations (Mode, etc)
    void unlockEeprom(uint8_t id) { writeRegister(id, 48, 0); }
    void lockEeprom(uint8_t id) { writeRegister(id, 48, 1); }

    // Legacy aliases for compatibility
    void unlock(uint8_t id) { disableTorque(id); } 
    void lock(uint8_t id) { enableTorque(id); }   

    void setMode(uint8_t id, uint8_t mode) {
        disableTorque(id); 
        delay(5);
        unlockEeprom(id); // Uses Reg 48
        delay(5);
        writeRegister(id, 33, mode); // Address 33 (0x21) = Operating Mode
        delay(10); // Reduced delay
        lockEeprom(id); // Uses Reg 48
        delay(5);
    }
    
    void setServoMode(uint8_t id) {
        setMode(id, 0);
    }

    void setWheelMode(uint8_t id) {
        setMode(id, 1);
    }

    // For Servo Mode (Mode 0)
    void writePosition(uint8_t id, uint16_t position, uint16_t time = 0, uint16_t speed = 1500) {
        if (position > 4096) position = 4096;
        
        // Length = Instr(1) + Addr(1) + Data(6) + Checksum(1) = 9
        uint8_t p[13];
        p[0] = 0xFF; p[1] = 0xFF;
        p[2] = id;
        p[3] = 0x09; // Length
        p[4] = 0x03; // WRITE
        p[5] = 0x2A; // Address 42 (Target Position) - Based on st_benchmark and standard maps
        
        p[6] = (uint8_t)(position & 0xFF);
        p[7] = (uint8_t)(position >> 8);
        
        p[8] = (uint8_t)(time & 0xFF);
        p[9] = (uint8_t)(time >> 8);
        
        p[10] = (uint8_t)(speed & 0xFF);
        p[11] = (uint8_t)(speed >> 8);
        
        uint16_t sum = id + 9 + 3 + 0x2A + p[6] + p[7] + p[8] + p[9] + p[10] + p[11];
        p[12] = ~(sum & 0xFF);
        
        _serial->write(p, 13);
        _serial->flush();
    }

    // For Wheel Mode (Mode 1)
    void writeSpeed(uint8_t id, int16_t speed) {
        // Convert to Sign-Magnitude Format for ST3215
        // Bit 15 is direction (0=CW?, 1=CCW?), Bits 0-14 are magnitude
        // Note: Check if Direction bit is 15 or 10? 
        // Based on ReadSpeed having bit 15, we assume WriteSpeed uses bit 15.
        
        uint16_t speed_pkt;
        if (speed < 0) {
            speed_pkt = (uint16_t)(-speed);
            speed_pkt |= 0x8000; // Set Sign Bit
        } else {
            speed_pkt = (uint16_t)speed;
        }
        
        // Address 0x2E (46) 
        uint8_t p[] = {
            0xFF, 0xFF, id, 0x05, 0x03, 0x2E, 
            (uint8_t)(speed_pkt & 0xFF), (uint8_t)(speed_pkt >> 8),
            0x00
        };
        
        uint16_t sum = id + 5 + 3 + 0x2E + p[6] + p[7]; 
        p[8] = ~(sum & 0xFF);
        
        _serial->write(p, 9);
        _serial->flush();
    }

    // Change ID (Register 5)
    void changeID(uint8_t old_id, uint8_t new_id) {
        // ID Change specifically requires Reg 55 (0x37) on ST3215
        
        // 1. Unlock EPROM (Address 55 = 0)
        writeRegister(old_id, 55, 0); 
        delay(20);
        
        // 2. Write New ID (Address 5 = new_id)
        writeRegister(old_id, 5, new_id);
        delay(500); // Generous delay for write
        
        // 3. Lock EPROM (Address 55 = 1) - On NEW ID
        writeRegister(new_id, 55, 1);
        delay(20);
    }

    // Ping: Check if a servo with ID exists by attempting to read its state (known working method)
    bool ping(uint8_t id) {
        int16_t pos, vel, load;
        // Reuse readState as it is verified to work
        return readState(id, pos, vel, load);
    }

    bool readState(uint8_t id, int16_t& pos, int16_t& vel, int16_t& load) {
        while (_serial->available()) _serial->read(); 
        
        // Read Present Pos/Speed/Load starting at 0x38 (per st_load_test)
        // Length = Instr(1) + Addr(1) + DataLen(1) + Checksum(1) = 4
        uint8_t packet[] = {0xFF, 0xFF, id, 0x04, 0x02, 0x38, 0x06, 0x00};
        packet[7] = ~(id + 4 + 2 + 0x38 + 6) & 0xFF; 

        _serial->write(packet, 8);
        
        unsigned long start = micros();
        while (_serial->available() < 12) {
            if (micros() - start > 3000) return false;
        }

        uint8_t rx[12];
        _serial->readBytes(rx, 12);

        if (rx[0] != 0xFF || rx[1] != 0xFF || rx[2] != id) return false;

        pos = (int16_t)(rx[5] | (rx[6] << 8));
        
        // Speed: Sign-Magnitude Format (Bit 15 is sign)
        uint16_t raw_vel = (uint16_t)(rx[7] | (rx[8] << 8));
        if (raw_vel & 0x8000) {
            vel = -(int16_t)(raw_vel & 0x7FFF);
        } else {
            vel = (int16_t)raw_vel;
        }

        load = (int16_t)(rx[9] | (rx[10] << 8));
        
        return true;
    }

    // Add dummy ReadVoltage to satisfy st_control3_2.ino expectation (or remove from ino)
    // But st_control3_2 expects to read voltage. Let's add it.
    int readVoltage(uint8_t id) {
        // Voltage is often at address 62 (0x3E) for STS/SCS?
        // Let's check st_load_test/FastST3215.h... it doesn't have it.
        // SCServo usually reads 0x3E.
        return 0; // Dummy for now to prevent compile error if I call it
    }

private:
    HardwareSerial* _serial;
};

#endif
