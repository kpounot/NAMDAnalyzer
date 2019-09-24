all: init


init:
	pip install -r requirements.txt


build: setup.py 
ifeq ($(OS), Windows_NT)
	python setup.py build
else
	python3 setup.py build
endif

build_cuda: cuda_setup.py
ifeq ($(OS), Windows_NT)
	python cuda_setup.py build
else
	python3 cuda_setup.py build
endif


.PHONY: install
install:
ifeq ($(OS), Windows_NT)
	python setup.py install
else
	python3 setup.py install
endif



.PHONY: clean
clean:
ifeq ($(OS), Windows_NT)
	rmdir /S /Q build  
	rmdir /S /Q dist  
	rmdir /S /Q NAMDAnalyzer.egg-info  
	cd NAMDAnalyzer/lib/ && del *.pyd
	cd cuda && del *.lib
	cd ../openmp && del *.c *.cpp
else
	rm -rf build  
	rm -rf dist  
	rm -rf NAMDAnalyzer.egg-info  
	rm NAMDAnalyzer/lib/*.pyd 
	rm NAMDAnalyzer/lib/cuda/*.lib
	rm NAMDAnalyzer/lib/openmp/*.c NAMDAnalyzer/lib/openmp/*.cpp
endif

