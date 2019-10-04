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


ifneq (,$(wildcard ./build/*))
	rmdir /S /Q build
endif
ifneq (,$(wildcard ./dist/*))
	rmdir /S /Q dist
endif
ifneq (,$(wildcard ./NAMDAnalyzer.egg-info/*))
	rmdir /S /Q NAMDAnalyzer.egg-info
endif
ifneq (,$(wildcard ./NAMDAnalyzer/lib/*.pyd))
	del .\NAMDAnalyzer\lib\*.pyd
endif
ifneq (,$(wildcard ./NAMDAnalyzer/lib/*.cpp))
	del .\NAMDAnalyzer\lib\*.cpp
endif
ifneq (,$(wildcard ./NAMDAnalyzer/lib/cuda/*.lib))
	del .\NAMDAnalyzer\lib\cuda\*.lib
endif


else


ifneq (,$(wildcard ./build/*))
	rm -rf build
endif
ifneq (,$(wildcard ./dist/*))
	rm -rf dist
endif
ifneq (,$(wildcard ./NAMDAnalyzer.egg-info/*))
	rm -rf NAMDAnalyzer.egg-info
endif
ifneq (,$(wildcard ./NAMDAnalyzer/lib/*.pyd))
	rm ./NAMDAnalyzer/lib/*.pyd
endif
ifneq (,$(wildcard ./NAMDAnalyzer/lib/*.cpp))
	rm ./NAMDAnalyzer/lib/*.cpp
endif
ifneq (,$(wildcard ./NAMDAnalyzer/lib/cuda/*.lib))
	rm ./NAMDAnalyzer/lib/cuda/*.lib
endif


endif
