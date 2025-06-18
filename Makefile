# Compiler
CC = /opt/homebrew/opt/llvm/bin/clang

# Compiler flags (for compiling .c to .o)
CFLAGS = -Wall -Wextra -g -fopenmp -I/opt/homebrew/opt/libomp/include

# Linker flags (for linking .o files to executable)
LDFLAGS = -L/opt/homebrew/opt/libomp/lib -lomp

# Target executable
TARGET = myprogram

# Source files
SRCS = main.c diffusion_func.c

# Object files
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

# Link step
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile step
%.o: %.c diffusion_func.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)


run: $(TARGET)
	@/usr/bin/time -p ./$(TARGET)