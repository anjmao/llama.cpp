### Compile llama on macOS

One time
```sh
cmake -B build
```

After each code change
```sh
cmake --build build --config Debug -j 8
```

### Testing

Run model

```sh
./build/bin/llama serve -hf allenai/OLMoE-1B-7B-0924-Instruct-GGUF
```

Test api

```sh
curl http://localhost:8080/v1/chat/completions   -H "Content-Type: application/json"   -d '{
"messages": [
{"role": "user", "content": "Write a fast bash one-liner to find all .log files over 100MB."}
],
"stream": true,
"temperature": 0.7
}'
```

Subscribe to routing trace
```sh
curl http://localhost:8080/v1/moe/routed-experts
```

Run UI

```
cd ./ui
go run .
```

#### Dynamic placement test

```
./build/bin/llama serve -hf allenai/OLMoE-1B-7B-0924-Instruct-GGUF -ngl 1 -nr
```

```sh
curl -s http://localhost:8080/v1/model/expert-placement \
  -H 'Content-Type: application/json' \
  -d '{"changes":[{"layer":6,"backend":"gpu"}]}'
```

Or specific expert

```sh
curl -s http://localhost:8080/v1/model/expert-placement \
  -H 'Content-Type: application/json' \
  -d '{"changes":[{"layer":1,"experts_gpu":[0,2]}]}'
```

In UI verify that layer was loaded to GPU