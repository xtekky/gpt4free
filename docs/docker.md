### G4F - Docker

If you have Docker installed, you can easily set up and run the project without manually installing dependencies.

1. First, ensure you have both Docker and Docker Compose installed.

   - [Install Docker](https://docs.docker.com/get-docker/)
   - [Install Docker Compose](https://docs.docker.com/compose/install/)

2. Clone the GitHub repo:

```bash
git clone https://github.com/xtekky/gpt4free.git
```

3. Navigate to the project directory:

```bash
cd gpt4free
```

4. Build the Docker image:

```bash
docker pull selenium/node-chrome
docker-compose build
```

5. Start the service using Docker Compose:

```bash
docker-compose up
```

Your server will now be running at `http://localhost:1337`. You can interact with the API or run your tests as you would normally.

To stop the Docker containers, simply run:

```bash
docker-compose down
```

> [!Note]
> When using Docker, any changes you make to your local files will be reflected in the Docker container thanks to the volume mapping in the `docker-compose.yml` file. If you add or remove dependencies, however, you'll need to rebuild the Docker image using `docker-compose build`.

[Return to Home](/)