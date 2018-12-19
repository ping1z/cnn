CC		:= g++
C_FLAGS := -std=c++11 -Wall -Wextra

BIN		:= bin
SRC		:= src
INCLUDE	:= include
LIB		:= lib

LIBRARIES	:=

DEBUG_FLAG	:=

ifeq ($(OS),Windows_NT)
EXECUTABLE	:= cnn.exe
else
EXECUTABLE	:= cnn
endif

all: $(BIN)/$(EXECUTABLE)

debug: DEBUG_FLAG := -g
debug: $(BIN)/$(EXECUTABLE)

clean:
	$(RM) -dR $(BIN)/*

run: all
	./$(BIN)/$(EXECUTABLE)
	
$(BIN)/$(EXECUTABLE): $(SRC)/*
	$(CC) $(DEBUG_FLAG) $(C_FLAGS) -I$(INCLUDE) -L$(LIB) $^ -o $@ $(LIBRARIES)