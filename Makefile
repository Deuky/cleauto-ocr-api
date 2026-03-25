build:
	pip wheel -w wheels -r requirements.txt

install: 
	pip install --no-cache wheels/*
