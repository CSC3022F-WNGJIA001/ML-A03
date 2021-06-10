# makefile for ML Assignment 3
# CSC3022F 2021
#	Author: WNGJIA001

install: venv
	. venv/bin/activate; pip3 install -Ur requirements.txt

venv:
	test -d venv || python3 -m venv venv

clean:
	rm -rf venv
	find . -iname "*.pyc" -delete
