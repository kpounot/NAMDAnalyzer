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
	cd NAMDAnalyzer/lib/ && del *.c *.cpp
else
	rm -rf build  
	rm NAMDAnalyzer/lib/*.c NAMDAnalyzer/lib/*.cpp
endif

