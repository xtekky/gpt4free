const fetch = globalThis.fetch || require('node-fetch');

module.exports = async (req, res) => {
  const UPSTREAM = process.env.G4F_UPSTREAM || 'http://localhost:8080';
  try {
    const body = req.method === 'POST' ? req.body || await getBody(req) : {};
    const upstreamRes = await fetch(`${UPSTREAM}/chat`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
      // upstream timeouts are managed by platform
    });
    const contentType = upstreamRes.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
      const data = await upstreamRes.json();
      res.status(upstreamRes.status).json(data);
    } else {
      const text = await upstreamRes.text();
      res.status(upstreamRes.status).send(text);
    }
  } catch (err) {
    res.status(200).json({reply: `Proxy error: ${err.message}`});
  }
};

function getBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', chunk => data += chunk);
    req.on('end', () => {
      try { resolve(JSON.parse(data)); } catch (e) { resolve({}); }
    });
    req.on('error', reject);
  });
}
