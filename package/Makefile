all: init


init:
	pip install -r requirements.txt


build: setup.py 
	python setup.py build

build_cuda: cuda_setup.py
	python cuda_setup.py build


.PHONY: install
install:
	python setup.py install

.PHONY: cuda_install
cuda_install:
	python cuda_setup.py install


.PHONY: clean
clean:
ifeq ($(OS), Windows_NT)
	rmdir /S /Q build  
	cd NAMDAnalyzer/lib/cython_pyx/ && del *.c *.cpp
else
	rm -rf build  
	rm NAMDAnalyzer/lib/cython_pyx/*.c NAMDAnalyzer/lib/cython_pyx/*.cpp
endif

