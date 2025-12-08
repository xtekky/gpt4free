# Reverse Proxy Configuration for GPT4Free

GPT4Free runs a FastAPI server that can be deployed behind a reverse proxy for production use. This guide covers common reverse proxy setups for deploying GPT4Free in production environments.

## Table of Contents
- [Why Use a Reverse Proxy?](#why-use-a-reverse-proxy)
- [General Requirements](#general-requirements)
- [Nginx Configuration](#nginx-configuration)
- [Caddy Configuration](#caddy-configuration)
- [Apache Configuration](#apache-configuration)
- [Traefik Configuration](#traefik-configuration)
- [Security Considerations](#security-considerations)
- [SSL/TLS Configuration](#ssltls-configuration)
- [Troubleshooting](#troubleshooting)

## Why Use a Reverse Proxy?

A reverse proxy provides several benefits for production deployments:

- **SSL/TLS Termination**: Handle HTTPS encryption at the proxy level
- **Load Balancing**: Distribute traffic across multiple instances
- **Caching**: Cache static assets and API responses
- **Security**: Add rate limiting, IP filtering, and firewall rules
- **Domain Management**: Serve multiple services on different paths/domains
- **WebSocket Support**: Better handling of streaming responses

## General Requirements

GPT4Free typically runs on:
- **Default Port**: 8080 (configurable)
- **Protocol**: HTTP (upgrade to HTTPS via reverse proxy)
- **WebSocket Support**: Required for streaming completions
- **GUI Path**: `/chat/` (web interface)
- **API Path**: `/v1` (OpenAI-compatible API)
- **Docs Path**: `/docs` (Swagger UI)

When deploying behind a reverse proxy:
1. Run GPT4Free on localhost (127.0.0.1) or internal network
2. Configure reverse proxy to forward requests
3. Ensure WebSocket connections are properly forwarded
4. Set appropriate headers for client IP preservation

## Nginx Configuration

### Basic Configuration

Create `/etc/nginx/sites-available/gpt4free`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Optional: Redirect HTTP to HTTPS
    # return 301 https://$server_name$request_uri;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        
        # WebSocket support
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Preserve client information
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts for long-running requests
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

### HTTPS Configuration with Let's Encrypt

```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL certificate configuration
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    # SSL settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        
        # WebSocket support
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Preserve client information
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

### Enable the Configuration

```bash
# Create symbolic link
sudo ln -s /etc/nginx/sites-available/gpt4free /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Reload Nginx
sudo systemctl reload nginx
```

## Caddy Configuration

Caddy provides automatic HTTPS with Let's Encrypt. Create a `Caddyfile`:

### Basic Configuration

```caddy
your-domain.com {
    reverse_proxy localhost:8080
}
```

### Advanced Configuration

```caddy
your-domain.com {
    # Automatic HTTPS
    
    # Custom headers
    header {
        # Security headers
        Strict-Transport-Security "max-age=31536000; includeSubDomains"
        X-Content-Type-Options "nosniff"
        X-Frame-Options "DENY"
        X-XSS-Protection "1; mode=block"
    }

    # Reverse proxy to GPT4Free
    reverse_proxy localhost:8080 {
        # Preserve client information
        header_up X-Real-IP {remote_host}
        header_up X-Forwarded-For {remote_host}
        header_up X-Forwarded-Proto {scheme}
        
        # Timeouts
        transport http {
            read_timeout 60s
            write_timeout 60s
        }
    }

    # Logging
    log {
        output file /var/log/caddy/gpt4free.log
    }
}
```

### Run Caddy

```bash
# Run Caddy with the Caddyfile
caddy run --config Caddyfile

# Or install as a service
caddy start --config Caddyfile
```

## Apache Configuration

### Enable Required Modules

```bash
sudo a2enmod proxy
sudo a2enmod proxy_http
sudo a2enmod proxy_wstunnel
sudo a2enmod ssl
sudo a2enmod headers
```

### Basic Configuration

Create `/etc/apache2/sites-available/gpt4free.conf`:

```apache
<VirtualHost *:80>
    ServerName your-domain.com
    
    # Optional: Redirect to HTTPS
    # Redirect permanent / https://your-domain.com/

    ProxyPreserveHost On
    ProxyPass / http://127.0.0.1:8080/
    ProxyPassReverse / http://127.0.0.1:8080/

    # WebSocket support
    RewriteEngine On
    RewriteCond %{HTTP:Upgrade} =websocket [NC]
    RewriteRule /(.*)           ws://127.0.0.1:8080/$1 [P,L]
    RewriteCond %{HTTP:Upgrade} !=websocket [NC]
    RewriteRule /(.*)           http://127.0.0.1:8080/$1 [P,L]

    # Preserve client information
    RequestHeader set X-Forwarded-Proto "http"
    RequestHeader set X-Forwarded-Port "80"
</VirtualHost>
```

### HTTPS Configuration

```apache
<VirtualHost *:443>
    ServerName your-domain.com

    SSLEngine on
    SSLCertificateFile /etc/letsencrypt/live/your-domain.com/fullchain.pem
    SSLCertificateKeyFile /etc/letsencrypt/live/your-domain.com/privkey.pem

    ProxyPreserveHost On
    ProxyPass / http://127.0.0.1:8080/
    ProxyPassReverse / http://127.0.0.1:8080/

    # WebSocket support
    RewriteEngine On
    RewriteCond %{HTTP:Upgrade} =websocket [NC]
    RewriteRule /(.*)           ws://127.0.0.1:8080/$1 [P,L]
    RewriteCond %{HTTP:Upgrade} !=websocket [NC]
    RewriteRule /(.*)           http://127.0.0.1:8080/$1 [P,L]

    # Preserve client information
    RequestHeader set X-Forwarded-Proto "https"
    RequestHeader set X-Forwarded-Port "443"

    # Security headers
    Header always set Strict-Transport-Security "max-age=31536000"
    Header always set X-Content-Type-Options "nosniff"
    Header always set X-Frame-Options "DENY"
</VirtualHost>
```

### Enable the Configuration

```bash
# Enable the site
sudo a2ensite gpt4free.conf

# Test configuration
sudo apache2ctl configtest

# Reload Apache
sudo systemctl reload apache2
```

## Traefik Configuration

Traefik is a modern reverse proxy with automatic service discovery. It works well with Docker.

### Docker Compose with Traefik

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  traefik:
    image: traefik:v2.10
    command:
      - "--api.insecure=true"
      - "--providers.docker=true"
      - "--providers.docker.exposedbydefault=false"
      - "--entrypoints.web.address=:80"
      - "--entrypoints.websecure.address=:443"
      - "--certificatesresolvers.myresolver.acme.tlschallenge=true"
      - "--certificatesresolvers.myresolver.acme.email=your-email@example.com"
      - "--certificatesresolvers.myresolver.acme.storage=/letsencrypt/acme.json"
    ports:
      - "80:80"
      - "443:443"
      - "8081:8080"  # Traefik dashboard
    volumes:
      - "/var/run/docker.sock:/var/run/docker.sock:ro"
      - "./letsencrypt:/letsencrypt"

  gpt4free:
    image: hlohaus789/g4f:latest
    shm_size: 2gb
    volumes:
      - ./har_and_cookies:/app/har_and_cookies
      - ./generated_media:/app/generated_media
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.gpt4free.rule=Host(`your-domain.com`)"
      - "traefik.http.routers.gpt4free.entrypoints=websecure"
      - "traefik.http.routers.gpt4free.tls.certresolver=myresolver"
      - "traefik.http.services.gpt4free.loadbalancer.server.port=8080"
      # HTTP to HTTPS redirect
      - "traefik.http.routers.gpt4free-http.rule=Host(`your-domain.com`)"
      - "traefik.http.routers.gpt4free-http.entrypoints=web"
      - "traefik.http.routers.gpt4free-http.middlewares=redirect-to-https"
      - "traefik.http.middlewares.redirect-to-https.redirectscheme.scheme=https"
```

### Static File Configuration (traefik.yml)

```yaml
entryPoints:
  web:
    address: ":80"
    http:
      redirections:
        entryPoint:
          to: websecure
          scheme: https
  websecure:
    address: ":443"

certificatesResolvers:
  myresolver:
    acme:
      email: your-email@example.com
      storage: /letsencrypt/acme.json
      tlsChallenge: {}

providers:
  file:
    filename: /etc/traefik/dynamic.yml
```

### Dynamic Configuration (dynamic.yml)

```yaml
http:
  routers:
    gpt4free:
      rule: "Host(`your-domain.com`)"
      service: gpt4free-service
      tls:
        certResolver: myresolver

  services:
    gpt4free-service:
      loadBalancer:
        servers:
          - url: "http://localhost:8080"
```

## Security Considerations

### Rate Limiting (Nginx Example)

```nginx
http {
    # Define rate limit zone
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

    server {
        location /v1 {
            limit_req zone=api_limit burst=20 nodelay;
            proxy_pass http://127.0.0.1:8080;
            # ... other proxy settings
        }
    }
}
```

### IP Whitelisting (Nginx Example)

```nginx
location /v1 {
    # Allow specific IPs
    allow 192.168.1.0/24;
    allow 10.0.0.0/8;
    deny all;

    proxy_pass http://127.0.0.1:8080;
    # ... other proxy settings
}
```

### Authentication (Nginx Example)

```nginx
location /v1 {
    # Basic authentication
    auth_basic "Restricted Access";
    auth_basic_user_file /etc/nginx/.htpasswd;

    proxy_pass http://127.0.0.1:8080;
    # ... other proxy settings
}
```

## SSL/TLS Configuration

### Obtaining SSL Certificates with Certbot (Let's Encrypt)

```bash
# Install Certbot
sudo apt-get update
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate for Nginx
sudo certbot --nginx -d your-domain.com

# Or for Apache
sudo certbot --apache -d your-domain.com

# Auto-renewal (usually set up automatically)
sudo certbot renew --dry-run
```

### Self-Signed Certificate (Development Only)

```bash
# Generate self-signed certificate
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/ssl/private/gpt4free.key \
  -out /etc/ssl/certs/gpt4free.crt
```

## Troubleshooting

### WebSocket Connection Issues

If streaming completions don't work:

1. **Verify WebSocket headers are set**:
   ```nginx
   proxy_set_header Upgrade $http_upgrade;
   proxy_set_header Connection "upgrade";
   ```

2. **Check timeout settings**:
   ```nginx
   proxy_read_timeout 3600s;
   proxy_send_timeout 3600s;
   ```

3. **Verify HTTP version**:
   ```nginx
   proxy_http_version 1.1;
   ```

### 502 Bad Gateway

- Ensure GPT4Free is running: `curl http://localhost:8080`
- Check firewall rules: `sudo ufw status`
- Verify the upstream address is correct (localhost:8080)
- Check logs: `sudo tail -f /var/log/nginx/error.log`

### Connection Timeouts

Increase timeout values in your reverse proxy configuration:

**Nginx**:
```nginx
proxy_connect_timeout 600s;
proxy_send_timeout 600s;
proxy_read_timeout 600s;
```

**Caddy**:
```caddy
reverse_proxy localhost:8080 {
    transport http {
        read_timeout 600s
        write_timeout 600s
    }
}
```

### CORS Issues

If you encounter CORS errors when accessing the API from a web application:

**Nginx**:
```nginx
location /v1 {
    # Handle preflight requests
    if ($request_method = 'OPTIONS') {
        add_header 'Access-Control-Allow-Origin' '*';
        add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS';
        add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization';
        add_header 'Access-Control-Max-Age' 1728000;
        add_header 'Content-Type' 'text/plain; charset=utf-8';
        add_header 'Content-Length' 0;
        return 204;
    }

    proxy_pass http://127.0.0.1:8080;
    # ... other proxy settings
}
```

## Example Production Setup

Here's a complete example of a production-ready setup:

### 1. Run GPT4Free with Docker

```bash
# Create directories
mkdir -p ${PWD}/har_and_cookies ${PWD}/generated_media
chown -R 1000:1000 ${PWD}/har_and_cookies ${PWD}/generated_media

# Run GPT4Free (internal only)
docker run -d \
  --name gpt4free \
  --restart unless-stopped \
  -p 127.0.0.1:8080:8080 \
  -v ${PWD}/har_and_cookies:/app/har_and_cookies \
  -v ${PWD}/generated_media:/app/generated_media \
  hlohaus789/g4f:latest
```

Note: Binding to `127.0.0.1:8080` ensures the service is only accessible locally.

### 2. Configure Nginx with HTTPS

First, add the rate limit zone to your main nginx configuration `/etc/nginx/nginx.conf` in the `http` block:

```nginx
http {
    # ... other http settings ...
    
    # Rate limiting zone
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    
    # ... rest of http block ...
}
```

Then create the site configuration `/etc/nginx/sites-available/gpt4free`:

```nginx
# /etc/nginx/sites-available/gpt4free
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;

    location / {
        # Apply rate limiting
        limit_req zone=api_limit burst=20 nodelay;
        
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    access_log /var/log/nginx/gpt4free-access.log;
    error_log /var/log/nginx/gpt4free-error.log;
}
```

### 3. Enable and Test

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/gpt4free /etc/nginx/sites-enabled/

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Test and reload
sudo nginx -t
sudo systemctl reload nginx
```

Now your GPT4Free instance is accessible at `https://your-domain.com` with:
- Automatic HTTPS
- Rate limiting
- Security headers
- WebSocket support for streaming
- Proper client IP forwarding

## Additional Resources

- [Nginx Documentation](https://nginx.org/en/docs/)
- [Caddy Documentation](https://caddyserver.com/docs/)
- [Apache Documentation](https://httpd.apache.org/docs/)
- [Traefik Documentation](https://doc.traefik.io/traefik/)
- [Let's Encrypt](https://letsencrypt.org/getting-started/)
- [GPT4Free Main Documentation](https://g4f.dev/docs)
