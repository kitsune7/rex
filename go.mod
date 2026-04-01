<<<<<<< HEAD
module rex

go 1.26.1

require (
	github.com/joho/godotenv v1.5.1
	github.com/pelletier/go-toml/v2 v2.2.2
)
||||||| parent of 621d3e3 (Add Go audio I/O package with PortAudio integration)
=======
module rex

go 1.22

require (
	github.com/go-audio/audio v1.0.0
	github.com/go-audio/wav v1.1.0
	github.com/gordonklaus/portaudio v0.0.0-20230709114228-aafa478834f5
	github.com/hajimehoshi/go-mp3 v0.3.4
)

require github.com/go-audio/riff v1.0.0 // indirect
>>>>>>> 621d3e3 (Add Go audio I/O package with PortAudio integration)
