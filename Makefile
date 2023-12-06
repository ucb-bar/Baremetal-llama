#################################
# Project Settings
#################################

# generated by baremetal-ide version 0.0.1

TARGET ?= firmware

CHIP ?= examplechip

#################################
# RISCV Toolchain
#################################

PREFIX = riscv64-unknown-elf-

CC = $(PREFIX)gcc
CXX = $(PREFIX)g++
CP = $(PREFIX)objcopy
OD = $(PREFIX)objdump
DG = $(PREFIX)gdb
SIZE = $(PREFIX)size


#################################
# Working directories
#################################

BSP_DIR = bsp/
LIB_DIR = lib/
USR_DIR = core/

SRC_DIR = $(USR_DIR)src/
INC_DIR = $(USR_DIR)inc/

BUILD_DIR = build/


#################################
# Source Files
#################################


ifeq ($(USE_HTIF), 1)
LIBRARIES += htif
endif

# USR sources
INCLUDES   = -I$(INC_DIR)
INCLUDES  += -I$(INC_DIR)hal/
A_SOURCES  = $(wildcard $(SRC_DIR)*.S) $(wildcard $(SRC_DIR)*/*.S)
C_SOURCES  = $(wildcard $(SRC_DIR)*.c) $(wildcard $(SRC_DIR)*/*.c)

# BSP sources
INCLUDES  += -I$(BSP_DIR)$(CHIP)/inc
INCLUDES  += -I$(BSP_DIR)common/inc
# A_SOURCES += $(BSP_DIR)$(CHIP)/startup/bootrom.S
A_SOURCES += $(BSP_DIR)$(CHIP)/startup/startup.S

C_SOURCES += $(BSP_DIR)common/src/hal_clint.c
C_SOURCES += $(BSP_DIR)common/src/hal_core.c
C_SOURCES += $(BSP_DIR)common/src/hal_gpio.c
C_SOURCES += $(BSP_DIR)common/src/hal_i2c.c
C_SOURCES += $(BSP_DIR)common/src/hal_plic.c
C_SOURCES += $(BSP_DIR)common/src/hal_spi.c
C_SOURCES += $(BSP_DIR)common/src/hal_uart.c
C_SOURCES += $(BSP_DIR)$(CHIP)/src/hal_rcc.c

# LIB sources
INCLUDES  += $(foreach LIBRARY_NAME,$(LIBRARIES),-I$(LIB_DIR)$(LIBRARY_NAME)/inc)
A_SOURCES += $(foreach LIBRARY_NAME,$(LIBRARIES),$(wildcard $(LIB_DIR)$(LIBRARY_NAME)/src/*.S))
C_SOURCES += $(foreach LIBRARY_NAME,$(LIBRARIES),$(wildcard $(LIB_DIR)$(LIBRARY_NAME)/src/*.c))


#################################
# Object List
#################################

A_OBJECTS = $(addsuffix .o,$(addprefix $(BUILD_DIR),$(basename $(A_SOURCES))))
C_OBJECTS = $(addsuffix .o,$(addprefix $(BUILD_DIR),$(basename $(C_SOURCES))))

OBJECTS = $(A_OBJECTS) $(C_OBJECTS)


#################################
# Target Output Files
#################################

TARGET_ELF = $(BUILD_DIR)$(TARGET).elf
TARGET_BIN = $(BUILD_DIR)$(TARGET).bin
TARGET_HEX = $(BUILD_DIR)$(TARGET).hex
TARGET_VERILOG = $(BUILD_DIR)$(TARGET).out


#################################
# Flags
#################################

# MCU Settings
ARCH = rv64imafdc
ABI = lp64d
CODEMODEL = medany

ifeq ($(USE_HTIF), 1)
LD_SCRIPT = $(USR_DIR)examplechip_htif_large.ld
else
LD_SCRIPT = $(USR_DIR)examplechip_large.ld
endif

# -mcmodel=medany -Wl,--start-group -lc_nano -lgloss_htif -Wl,--end-group -lgcc -static -nostartfiles -dT htif.ld
SPECFLAGS = --specs="nano.specs"
# SPECFLAGS = --specs="htif_nano.specs"

ARCHFLAGS = -march=$(ARCH) -mabi=$(ABI) -mcmodel=$(CODEMODEL) -fno-pie

# compiler Flags
CFLAGS  = -g -std=gnu11 -O0
CFLAGS += -fno-common -fno-builtin-printf
CFLAGS += -Wall -Wextra -Warray-bounds -Wno-unused-parameter -Wcast-qual
# CFLAGS += -Wl,--start-group -lc_nano -lgloss_htif -Wl,--end-group -lgcc
CFLAGS += $(SPECFLAGS)
CFLAGS += $(ARCHFLAGS)
CFLAGS += $(INCLUDES)

# linker Flags
LFLAGS  = -static
LFLAGS += -nostartfiles
# LFLAGS += -nostdlib
# LFLAGS += -u _printf_float
ifdef STACK_SIZE
LFLAGS += -Xlinker --defsym=__stack_size=$(STACK_SIZE)
endif
LFLAGS += -T $(LD_SCRIPT)
LFLAGS_EXTRA = -lm


#################################
# Build
#################################

.DEFAULT_GOAL := build

# default target
build: $(TARGET_ELF)
	@echo "[Build] $(TARGET_ELF) built for target \"$(CHIP)\""

$(TARGET_BIN): $(TARGET_ELF)
	$(CP) -O binary $< $@

$(TARGET_HEX): $(TARGET_ELF)
	$(CP) -O ihex $< $@

$(TARGET_VERILOG): $(TARGET_ELF)
	$(CP) -O verilog $< $@

$(TARGET_ELF): $(OBJECTS)
	@echo "[LD] linking $@"
	@$(CC) $(CFLAGS) $(LFLAGS) $^ -o $@ $(LFLAGS_EXTRA)
	$(SIZE) $(TARGET_ELF)

$(A_OBJECTS): $(BUILD_DIR)%.o: %.S
	@echo "[CC] compiling $@"
	@mkdir -p $(@D)
	@$(CC) $(CFLAGS) -c $< -o $@

$(C_OBJECTS): $(BUILD_DIR)%.o: %.c
	@echo "[CC] compiling $@"
	@mkdir -p $(@D)
	@$(CC) $(CFLAGS) -c $< -o $@


#################################
# OpenOCD Upload Commands
#################################

UPLOAD_COMMANDS_SRAM  = -c "init"
UPLOAD_COMMANDS_SRAM += -c "load_image $(TARGET_ELF) 0x0"
UPLOAD_COMMANDS_SRAM += -c "reset"
# UPLOAD_COMMANDS_SRAM += -c "sleep 100"
# UPLOAD_COMMANDS_SRAM += -c "reg pc 0x08000000"
UPLOAD_COMMANDS_SRAM += -c "exit"

UPLOAD_COMMANDS_FLASH  = -c "exit"
UPLOAD_COMMANDS_FLASH += -c "program $(TARGET_BIN) 0x20000000"
UPLOAD_COMMANDS_FLASH += -c "reset"
UPLOAD_COMMANDS_FLASH += -c "exit"


#################################
# Recipes
#################################

.PHONY: clean
clean:
	@rm -rf $(BUILD_DIR)

bin: $(TARGET_BIN)

hex: $(TARGET_HEX)

verilog: $(TARGET_VERILOG)

.PHONY: dump
dump:
	$(OD) -d $(TARGET_ELF) > $(BUILD_DIR)disassemble.S
	$(OD) -h $(TARGET_ELF) > $(BUILD_DIR)sections.out
	$(OD) -t $(TARGET_ELF) > $(BUILD_DIR)symbol_table.out

# openocd currently only supports 32 bit target, thus we need to use binary loader
.PHONY: upload
upload: $(TARGET_BIN)
	@openocd -f ./debug/$(CHIP).cfg $(UPLOAD_COMMANDS_SRAM)
# @openocd -f ./debug/$(CHIP).cfg -c "init" -c "load_image ./build/firmware.elf 0x0" -c "reset" -c "sleep 100" -c "reg pc 0x08000000" -c "exit"
# @openocd -f ./debug/$(CHIP).cfg -c "program $(TARGET_BIN) 0x20000000 reset exit"

.PHONY: debug
debug: $(TARGET_BIN)
	@openocd -f ./debug/$(CHIP).cfg & $(DG) --eval-command="target extended-remote localhost:3333"

