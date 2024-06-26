#!/bin/bash
TARGET=hw
# TARGET=hw_emu
# DEBUG=-g

TOP=Sextans
XO='/home/zbai/Sextans/build/run/run-1/Sextans.xo'
CONSTRAINT='/home/zbai/Sextans/build/run/run-1/Sextans_floorplan.tcl'

# check that the floorplan tcl exists
if [ ! -f "$CONSTRAINT" ]; then
    echo "no constraint file found"
    exit
fi

CONFIG_FILE='/home/zbai/Sextans/build/run/run-1/run/link_config.ini'
TARGET_FREQUENCY=300
PLATFORM=xilinx_u280_xdma_201920_3
OUTPUT_DIR="$(pwd)/vitis_run_${TARGET}"

MAX_SYNTH_JOBS=8
STRATEGY="Explore"
PLACEMENT_STRATEGY="EarlyBlockPlacement"

v++ ${DEBUG} \
  --link \
  --output "${OUTPUT_DIR}/${TOP}_${PLATFORM}.xclbin" \
  --kernel ${TOP} \
  --platform ${PLATFORM} \
  --target ${TARGET} \
  --report_level 2 \
  --temp_dir "${OUTPUT_DIR}/${TOP}_${PLATFORM}.temp" \
  --optimize 3 \
  --connectivity.nk ${TOP}:1:${TOP} \
  --save-temps \
  "${XO}" \
  --vivado.synth.jobs ${MAX_SYNTH_JOBS} \
  --vivado.prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.IS_ENABLED=1 \
  --vivado.prop=run.impl_1.STEPS.OPT_DESIGN.ARGS.DIRECTIVE=$STRATEGY \
  --vivado.prop=run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=$PLACEMENT_STRATEGY \
  --vivado.prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=$STRATEGY \
  --vivado.prop=run.impl_1.STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=$STRATEGY \
  --vivado.prop=run.impl_1.STEPS.OPT_DESIGN.TCL.PRE=$CONSTRAINT \
  --config "${CONFIG_FILE}" \
  --kernel_frequency ${TARGET_FREQUENCY} \
