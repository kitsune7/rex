.PHONY: build run test vet tidy clean

BINARY := bin/rex
PKG    := ./cmd/rex

build:
	@mkdir -p bin
	go build -o $(BINARY) $(PKG)

run:
	go run $(PKG)

test:
	go test ./...

vet:
	go vet ./...

tidy:
	go mod tidy

clean:
	rm -rf bin
