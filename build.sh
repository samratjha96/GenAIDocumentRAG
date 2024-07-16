#!/bin/bash

docker build -t pdf-reader-admin . && \
docker run  -e BUCKET_NAME="BUCKET_NAME" -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" -e AWS_SESSION_TOKEN="$AWS_SESSION_TOKEN" -it -p 8083:8083 pdf-reader-admin

