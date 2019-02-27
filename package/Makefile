all: init NAMDAnalyzer


init:
	pip3 install -r requirements.txt


NAMDAnalyzer: setup.py 
	python setup.py build 


.PHONY: install
install:
	python setup.py install


.PHONY: clean
ifeq ($(OS), Windows_NT)
clean:
	cd NAMDAnalyzer/lib && del *.c && del *.cpp && cd ../.. && rmdir build /s /q && rmdir dist /s /q && del MANIFEST 
else
clean:
	cd NAMDAnalyzer/lib && rm *.c && rm *.cpp && cd ../.. && rm -r build && rm -r dist && rm MANIFEST
endif

