"""NIBE F1155 Modbus TCP simulator for EffektGuard issue #18 testing.

Serves F-series parameter IDs as holding-register addresses on 127.0.0.1:5020,
mirroring a real F1155 exposed to HA via the generic `modbus` integration
(the issue reporter's setup). Values use NIBE scale factors (temp x10).

Registers served (F-series parameter IDs, int16; every one writable):
  40004 BT1 outdoor temp        -32   (-3.2 C)
  40008 BT2 supply temp S1      358   (35.8 C)
  40012 BT3 return temp         312   (31.2 C)
  40013 BT7 HW top              487   (48.7 C)
  40014 BT6 HW charging         442   (44.2 C)
  40033 BT50 room temp          213   (21.3 C)
  43005 Degree minutes (x10)   -1500  (-150.0 DM)
  43084 Int. el.add. Power      0     (x100: 0.00 kW, the immersion heater off)
  43086 Prio                    30    (heat; 20 is hot water) -- NOT a phase current!
  43136 Compressor freq (x10)   620   (62.0 Hz)
  43427 Compressor status       60    (runs)
  47011 Heat offset S1          0
  48132 Temporary Lux           0     (DHW boost switch)

Built on pymodbus's SimData/SimDevice API (3.13): the datastore classes it replaces fail to start a
multi-device server there. Every write is logged, so an end-to-end write can be proven.
"""

import asyncio
import logging

from pymodbus.server import StartAsyncTcpServer
from pymodbus.simulator import DataType, SimData, SimDevice

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("nibe-sim")

REGISTERS = {
    40004: -32,
    40008: 358,
    40012: 312,
    40013: 487,
    40014: 442,
    40033: 213,
    43005: -1500,
    43084: 0,
    43086: 30,
    43136: 620,
    43427: 60,
    47011: 0,
    48132: 0,
}

# Second simulated pump: NIBE F750 (exhaust-air ASHP) on modbus unit 2.
# F-series register ids are shared across F750/F1155 (verified: yozik04/nibe
# f750.csv uses the same 40004/40013/43005/47011/48132 ids).
REGISTERS_F750 = {
    40004: -32,  # BT1 outdoor -3.2 C (same site)
    40008: 382,  # BT2 supply 38.2 C
    40012: 320,  # BT3 return 32.0 C
    40013: 512,  # BT7 HW top 51.2 C
    40014: 460,  # BT6 HW charging 46.0 C
    40033: 218,  # BT50 room 21.8 C
    43005: -850,  # DM -85.0
    43084: 0,  # Immersion heater off (kW x100)
    43086: 30,  # Prio: heat
    43136: 450,  # Compressor 45.0 Hz
    43427: 60,  # Running
    47011: 0,  # Heat offset S1
    48132: 0,  # Temporary Lux
}


def pump(unit: int, registers: dict[int, int]) -> SimDevice:
    """One pump: its registers, at their own addresses, and a log line for every write."""

    async def log_writes(_func_code, _start, address, _count, _registers, set_values):
        if set_values is not None:
            LOG.info("WRITE unit %s register %s = %s", unit, address, list(set_values))

    return SimDevice(
        unit,
        simdata=[
            SimData(address, values=value, datatype=DataType.INT16)
            for address, value in sorted(registers.items())
        ],
        action=log_writes,
    )


def main() -> None:
    LOG.info("Starting NIBE F1155 (unit 1) + F750 (unit 2) simulator on 127.0.0.1:5020")
    asyncio.run(
        StartAsyncTcpServer(
            context=[pump(1, REGISTERS), pump(2, REGISTERS_F750)], address=("127.0.0.1", 5020)
        )
    )


if __name__ == "__main__":
    main()
