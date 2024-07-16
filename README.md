# Overview

RAG implementation of my personal notes with Bedrock hosted LLM's

## Development

Use docker to run the container locally. Because we do use S3 and Bedrock here we need AWS credentials inside the docker image. The quick and dirty way is to assume the role in your target account and pass in the environment variables

```
docker build -t pdf-reader-admin . && docker run  -e BUCKET_NAME="$BUCKET_NAME" -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" -e AWS_SESSION_TOKEN="$AWS_SESSION_TOKEN" -it -p 8083:8083 pdf-reader-admin
```

or you can just run:

```
./build.sh
```

Note: The `BUCKET_NAME` variable is set to a pre-existing S3 bucket in the target account. Of course you can change it to be whatever

