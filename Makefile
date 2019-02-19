CYTHON_PYX 	= package/lib/pycompIntScatFunc.pyx


all: init NAMDAnalyzer clean


init:
	pip3 install -r requirements.txt


NAMDAnalyzer: setup.py $(CYTHON_PYX)
	python setup.py build_ext 


.PHONY: clean
clean:
	cd package/lib && del *.c && del *.cpp && cd ../.. && rmdir build /s /q 

