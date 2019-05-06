all: init build


init:
	pip3 install -r requirements.txt


build: setup.py 
ifeq ($(OS), Windows_NT)
	python setup.py build --compiler=mingw32
else
	python3 setup.py build
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
	cd NAMDAnalyzer/lib && del *.c *.cpp
else
	rm -rf build  
	rm NAMDAnalyzer/lib/*.c NAMDAnalyzer/lib/*.cpp
endif

