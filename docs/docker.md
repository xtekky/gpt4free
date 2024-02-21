### G4F - Docker Setup

Easily set up and run the G4F project using Docker without the hassle of manual dependency installation.

1. **Prerequisites:**
   - [Install Docker](https://docs.docker.com/get-docker/)
   - [Install Docker Compose](https://docs.docker.com/compose/install/)

2. **Clone the Repository:**

```bash
git clone https://github.com/xtekky/gpt4free.git
```

3. **Navigate to the Project Directory:**

```bash
cd gpt4free
```

4. **Build the Docker Image:**

```bash
docker pull selenium/node-chrome
docker-compose build
```

5. **Start the Service:**

```bash
docker-compose up
```

Your server will now be accessible at `http://localhost:1337`. Interact with the API or run tests as usual.

To stop the Docker containers, simply run:

```bash
docker-compose down
```

> [!Note]
> Changes made to local files reflect in the Docker container due to volume mapping in `docker-compose.yml`. However, if you add or remove dependencies, rebuild the Docker image using `docker-compose build`.

[Return to Home](/)