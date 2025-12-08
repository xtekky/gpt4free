# Quick Start Guide: GPT4Free with Reverse Proxy

This guide helps you quickly deploy GPT4Free with a reverse proxy for production use.

## Prerequisites

- Docker and Docker Compose installed
- A domain name (for HTTPS) or use localhost for testing
- Ports 80 and 443 available

## Option 1: Quick Start with Caddy (Recommended)

Caddy provides automatic HTTPS with Let's Encrypt, making it the easiest option.

### Step 1: Clone the repository

```bash
git clone https://github.com/xtekky/gpt4free.git
cd gpt4free
```

### Step 2: Configure Caddyfile

```bash
# Copy the example Caddyfile
cp Caddyfile.example Caddyfile

# Edit the Caddyfile and replace 'your-domain.com' with your actual domain
nano Caddyfile
```

For local testing without a domain, use this simple Caddyfile:
```caddy
localhost {
    reverse_proxy gpt4free:8080
}
```

### Step 3: Start the services

```bash
# Start GPT4Free with Caddy reverse proxy
docker-compose -f docker-compose-proxy.yml up -d
```

### Step 4: Access GPT4Free

- **With domain**: https://your-domain.com
- **Local testing**: http://localhost

That's it! Caddy will automatically obtain and renew SSL certificates.

## Option 2: Manual Setup with Nginx

If you prefer Nginx, follow these steps:

### Step 1: Start GPT4Free (internal only)

```bash
# Create directories
mkdir -p har_and_cookies generated_media

# Run GPT4Free on internal network only
docker run -d \
  --name gpt4free \
  --restart unless-stopped \
  -p 127.0.0.1:8080:8080 \
  -v ${PWD}/har_and_cookies:/app/har_and_cookies \
  -v ${PWD}/generated_media:/app/generated_media \
  hlohaus789/g4f:latest
```

### Step 2: Configure Nginx

Create `/etc/nginx/sites-available/gpt4free`:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        
        # WebSocket support
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_read_timeout 60s;
    }
}
```

### Step 3: Enable site and get SSL

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/gpt4free /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Get SSL certificate (certbot will modify your nginx config)
sudo certbot --nginx -d your-domain.com

# Reload nginx
sudo systemctl reload nginx
```

## Verification

After setup, verify your deployment:

1. **Check service status**:
   ```bash
   # For Docker Compose
   docker-compose -f docker-compose-proxy.yml ps
   
   # For manual Docker
   docker ps
   ```

2. **Test the API**:
   ```bash
   # Local test
   curl http://localhost/v1/models
   
   # With domain
   curl https://your-domain.com/v1/models
   ```

3. **Access the GUI**:
   - Navigate to `https://your-domain.com/chat/` (or `http://localhost/chat/` for local)

4. **Check logs**:
   ```bash
   # Docker Compose
   docker-compose -f docker-compose-proxy.yml logs -f
   
   # Manual Docker
   docker logs -f gpt4free
   ```

## Common Issues

### Port Already in Use

If ports 80 or 443 are already in use:

```bash
# Check what's using the ports
sudo lsof -i :80
sudo lsof -i :443

# Stop the conflicting service or use different ports
```

### SSL Certificate Issues

For Caddy:
- Ensure your domain's DNS points to your server
- Caddy needs to be accessible from the internet on port 80 for Let's Encrypt validation
- Check Caddy logs: `docker-compose -f docker-compose-proxy.yml logs caddy`

For Certbot/Nginx:
- Run `sudo certbot renew --dry-run` to test renewal
- Check nginx error logs: `sudo tail -f /var/log/nginx/error.log`

### WebSocket Connection Failed

Ensure your reverse proxy configuration includes WebSocket support:
- Nginx: `proxy_set_header Upgrade $http_upgrade;` and `proxy_set_header Connection "upgrade";`
- Caddy: This is automatic
- Apache: Enable `mod_proxy_wstunnel`

### Timeout Errors

Increase timeout values in your reverse proxy configuration if you experience timeouts with long-running requests.

## Security Best Practices

1. **Always use HTTPS in production**
2. **Set up rate limiting** to prevent abuse
3. **Use firewall rules** to restrict access if needed
4. **Keep software updated**: Regularly update Docker images and reverse proxy
5. **Monitor logs** for suspicious activity
6. **Use strong API keys** if authentication is enabled

## Next Steps

- **Full documentation**: See [docs/reverse-proxy.md](docs/reverse-proxy.md) for detailed configuration options
- **Security hardening**: Add rate limiting, IP whitelisting, and authentication
- **Monitoring**: Set up monitoring and alerting for your deployment
- **Backups**: Regularly backup the `har_and_cookies` directory

## Getting Help

- **Documentation**: https://g4f.dev/docs
- **Issues**: https://github.com/xtekky/gpt4free/issues
- **Community**: Discord and Telegram (links in main README)
