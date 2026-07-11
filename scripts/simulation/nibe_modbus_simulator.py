"""NIBE F1155 Modbus TCP simulator for EffektGuard issue #18 testing.

Serves F-series parameter IDs as holding-register addresses on 127.0.0.1:5020,
mirroring a real F1155 exposed to HA via the generic `modbus` integration
(the issue reporter's setup). Values use NIBE scale factors (temp x10).

Registers served (F-series parameter IDs, two's-complement int16):
  40004 BT1 outdoor temp        -32   (-3.2 C)
  40008 BT2 supply temp S1      358   (35.8 C)
  40012 BT3 return temp         312   (31.2 C)
  40013 BT7 HW top              487   (48.7 C)
  40014 BT6 HW charging         442   (44.2 C)
  40033 BT50 room temp          213   (21.3 C)
  43005 Degree minutes (x10)   -1500  (-150.0 DM)  R/W
  43086 Prio                    30    (hot water)   -- NOT a phase current!
  43136 Compressor freq (x10)   620   (62.0 Hz)
  43427 Compressor status       60    (runs)
  47011 Heat offset S1          0     R/W
  48132 Temporary Lux           0     R/W (DHW boost switch)
"""

import asyncio
import logging

from pymodbus.datastore import (
    ModbusDeviceContext,
    ModbusServerContext,
    ModbusSparseDataBlock,
)
from pymodbus.server import StartAsyncTcpServer

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("nibe-sim")


def s16(value: int) -> int:
    """Encode signed int16 as unsigned register value."""
    return value & 0xFFFF


REGISTERS = {
    40004: s16(-32),
    40008: s16(358),
    40012: s16(312),
    40013: s16(487),
    40014: s16(442),
    40033: s16(213),
    43005: s16(-1500),
    43086: s16(30),
    43136: s16(620),
    43427: s16(60),
    47011: s16(0),
    48132: s16(0),
}


class LoggingSparseBlock(ModbusSparseDataBlock):
    """Sparse data block that logs writes (so we can prove e2e writes arrive)."""

    def setValues(self, address, values):
        LOG.info("WRITE register %s = %s", address, values)
        return super().setValues(address, values)


# Second simulated pump: NIBE F750 (exhaust-air ASHP) on modbus unit 2.
# F-series register ids are shared across F750/F1155 (verified: yozik04/nibe
# f750.csv uses the same 40004/40013/43005/47011/48132 ids).
REGISTERS_F750 = {
    40004: s16(-32),   # BT1 outdoor -3.2 C (same site)
    40008: s16(382),   # BT2 supply 38.2 C
    40012: s16(320),   # BT3 return 32.0 C
    40013: s16(512),   # BT7 HW top 51.2 C
    40014: s16(460),   # BT6 HW charging 46.0 C
    40033: s16(218),   # BT50 room 21.8 C
    43005: s16(-850),  # DM -85.0
    43086: s16(30),    # Prio
    43136: s16(450),   # Compressor 45.0 Hz
    43427: s16(60),    # Running
    47011: s16(0),     # Heat offset S1
    48132: s16(0),     # Temporary Lux
}


def main() -> None:
    # pymodbus data blocks are 1-offset relative to protocol address:
    # a request for protocol address N reads block key N+1.
    block = LoggingSparseBlock({addr + 1: val for addr, val in REGISTERS.items()})
    device_f1155 = ModbusDeviceContext(hr=block, ir=block)
    block_f750 = LoggingSparseBlock({addr + 1: val for addr, val in REGISTERS_F750.items()})
    device_f750 = ModbusDeviceContext(hr=block_f750, ir=block_f750)
    context = ModbusServerContext(
        devices={1: device_f1155, 2: device_f750}, single=False
    )
    LOG.info("Starting NIBE F1155 (unit 1) + F750 (unit 2) simulator on 127.0.0.1:5020")
    asyncio.run(StartAsyncTcpServer(context=context, address=("127.0.0.1", 5020)))


if __name__ == "__main__":
    main()
