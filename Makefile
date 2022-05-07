# Hans A. Winther (2020) (hans.a.winther@gmail.com)

SHELL := /bin/bash

# Set compiler (use =c++17 if you have this availiable)
CC = g++ -std=c++11

# Paths to GSL library
INC  = -I$(HOME)/local/include
LIBS = -L$(HOME)/local/lib -lgsl -lgslcblas

#=======================================================
# Options
#=======================================================
OPTIONS =

# Add bounds checking
OPTIONS += -D_GLIBCXX_DEBUG

# Show warnings if atempting to evaluate a spline out of bounds
OPTIONS += -D_SPLINE_WARNINGS_ON

# Show info about the solution as we integrate
# OPTIONS = -D_FIDUCIAL_VERBOSE_ODE_SOLVER_TRUE

# Add OpenMP parallelization
OPTIONS += -D_USEOPEMP
CC += -fopenmp

#=======================================================

C = -O3 -g $(OPTIONS)

#=======================================================

VPATH=src/
TARGETS := cmb
all: $(TARGETS)

# OBJECT FILES
OBJS = Main.o Utils.o Spline.o ODESolver.o BackgroundCosmology.o RecombinationHistory.o Perturbations.o PowerSpectrum.o

# DEPENDENCIES
Main.o                  : BackgroundCosmology.h RecombinationHistory.h Perturbations.h
Spline.o                : Spline.h
ODESolver.o             : ODESolver.h
Utils.o                 : Utils.h Spline.h ODESolver.h
BackgroundCosmology.o   : BackgroundCosmology.h Utils.h Spline.h ODESolver.h
RecombinationHistory.o  : RecombinationHistory.h BackgroundCosmology.h Utils.h Spline.h ODESolver.h
Perturbations.o					: Perturbations.h RecombinationHistory.h BackgroundCosmology.h Utils.h Spline.h ODESolver.h
PowerSpectrum.o					: PowerSpectrum.h Perturbations.h RecombinationHistory.h BackgroundCosmology.h Utils.h Spline.h ODESolver.h

cmb: $(OBJS)
	${CC} -o $@ $^ $C $(INC) $(LIBS)

%.o: %.cpp
	${CC}  -c -o $@ $< $C $(INC)

clean:
	rm -rf $(TARGETS) *.o
