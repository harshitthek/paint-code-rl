.PHONY: test install clean

install:
	pip install -e .[dev]
	cd renderer && npm ci

test:
	pytest tests/

clean:
	rm -rf artifacts/*
	rm -rf runs/*
	find . -type d -name __pycache__ -exec rm -r {} +
