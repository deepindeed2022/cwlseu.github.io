# -*- Makefile -*-

# target: dependencies
#	action

CC := gcc

all: distance

distance: distance.o gd.o
	$(CC) distance.o gd.o -o distance -l m

distance.o: distance.c
	$(CC) -c distance.c

gd.o: gd.c
	$(CC) -c gd.c

test:
	./distance
clean:
	@rm -f *.o distance
