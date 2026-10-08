// turn.go implements `g4f-go turn`: an embedded STUN/TURN server built on
// pion/turn.
//
// It replaces the external coturn dependency of the remote desktop app: the
// same binary that runs the Python server can relay the WebRTC media between a
// phone on cellular and this machine. Credentials use the TURN REST API scheme
// (draft-uberti-behave-turn-rest-00), which is exactly what
// remote_desktop.config.turn_credentials() mints, so both sides agree on
// authentication without sharing anything but the secret.
package main

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"errors"
	"flag"
	"fmt"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/pion/logging"
	"github.com/pion/stun/v3"
	"github.com/pion/turn/v4"
)

const (
	turnDefaultPort    = 3478
	turnDefaultTLSPort = 5349
	turnDefaultMinPort = 49160
	turnDefaultMaxPort = 49200
	turnDefaultTTL     = time.Hour
	turnDefaultQuota   = 100
	turnSecretBytes    = 32
)

// turnSTUNServers are queried (in order) to learn the public address of a
// NATed host. The first one that answers wins.
var turnSTUNServers = []string{
	"stun.l.google.com:19302",
	"stun.cloudflare.com:3478",
	"stun.nextcloud.com:443",
}

// turnDir holds the shared secret, the TLS material and the runtime state.
func turnDir() string { return filepath.Join(configDir(), "turn") }

// turnSecretPath is the file holding the TURN REST shared secret.
func turnSecretPath() string { return filepath.Join(turnDir(), "secret") }

// turnStatePath describes a running embedded server so that other processes
// (notably the Python remote desktop server) can discover it.
func turnStatePath() string { return filepath.Join(turnDir(), "state.json") }

// turnState is the on-disk description of a running embedded TURN server.
type turnState struct {
	URLs     []string `json:"urls"`
	STUNURL  string   `json:"stun_url"`
	Realm    string   `json:"realm"`
	PublicIP string   `json:"public_ip"`
	Port     int      `json:"port"`
	TLSPort  int      `json:"tls_port"`
	PID      int      `json:"pid"`
	Started  string   `json:"started"`
}

// --- shared secret ---------------------------------------------------------

// newTurnSecret returns a fresh 256-bit secret, URL-safe so it survives being
// pasted into a shell or a query string.
func newTurnSecret() (string, error) {
	buf := make([]byte, turnSecretBytes)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(buf), nil
}

// readTurnSecret returns the persisted secret, or an error when absent.
func readTurnSecret() (string, error) {
	data, err := os.ReadFile(turnSecretPath())
	if err != nil {
		return "", err
	}
	secret := strings.TrimSpace(string(data))
	if secret == "" {
		return "", fmt.Errorf("empty secret in %s", turnSecretPath())
	}
	return secret, nil
}

// writeTurnSecret persists the secret with owner-only permissions.
func writeTurnSecret(secret string) error {
	if err := os.MkdirAll(turnDir(), 0o700); err != nil {
		return err
	}
	return os.WriteFile(turnSecretPath(), []byte(secret), 0o600)
}

// ensureTurnSecret returns the persisted secret, generating one on first use.
func ensureTurnSecret() (string, error) {
	if secret, err := readTurnSecret(); err == nil {
		return secret, nil
	}
	secret, err := newTurnSecret()
	if err != nil {
		return "", err
	}
	if err := writeTurnSecret(secret); err != nil {
		return "", err
	}
	return secret, nil
}

// --- runtime state ---------------------------------------------------------

func writeTurnState(st turnState) error {
	if err := os.MkdirAll(turnDir(), 0o700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(turnStatePath(), append(data, '\n'), 0o600)
}

func readTurnState() (turnState, error) {
	var st turnState
	data, err := os.ReadFile(turnStatePath())
	if err != nil {
		return st, err
	}
	if err := json.Unmarshal(data, &st); err != nil {
		return st, err
	}
	return st, nil
}

// processAlive reports whether pid is still running. On Windows a signal 0
// probe would terminate the process, so the state file is trusted there.
func processAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	if runtime.GOOS == "windows" {
		return true
	}
	p, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	return p.Signal(syscall.Signal(0)) == nil
}

// --- address detection -----------------------------------------------------

// localIPv4 returns the first global unicast IPv4 address of this host.
func localIPv4() net.IP {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return net.IPv4(127, 0, 0, 1)
	}
	for _, a := range addrs {
		ipnet, ok := a.(*net.IPNet)
		if !ok || ipnet.IP.IsLoopback() {
			continue
		}
		ip4 := ipnet.IP.To4()
		if ip4 == nil || !ip4.IsGlobalUnicast() {
			continue
		}
		return ip4
	}
	return net.IPv4(127, 0, 0, 1)
}

// stunQuery asks one STUN server for our reflexive address.
func stunQuery(ctx context.Context, addr string) (net.IP, error) {
	dialer := net.Dialer{Timeout: 3 * time.Second}
	conn, err := dialer.DialContext(ctx, "udp4", addr)
	if err != nil {
		return nil, err
	}
	defer conn.Close() //nolint:errcheck

	deadline := time.Now().Add(3 * time.Second)
	if dl, ok := ctx.Deadline(); ok && dl.Before(deadline) {
		deadline = dl
	}
	if err := conn.SetDeadline(deadline); err != nil {
		return nil, err
	}

	client, err := stun.NewClient(conn)
	if err != nil {
		return nil, err
	}
	defer client.Close() //nolint:errcheck

	msg, err := stun.Build(stun.TransactionID, stun.BindingRequest)
	if err != nil {
		return nil, err
	}

	var mapped net.IP
	if err := client.Do(msg, func(ev stun.Event) {
		if ev.Error != nil {
			return
		}
		var xor stun.XORMappedAddress
		if err := xor.GetFrom(ev.Message); err != nil {
			return
		}
		mapped = xor.IP
	}); err != nil {
		return nil, err
	}
	if mapped == nil {
		return nil, fmt.Errorf("no XOR-MAPPED-ADDRESS from %s", addr)
	}
	return mapped, nil
}

// stunPublicIP asks the well-known STUN servers for our public address.
func stunPublicIP(ctx context.Context) (net.IP, error) {
	var lastErr error
	for _, srv := range turnSTUNServers {
		ip, err := stunQuery(ctx, srv)
		if err == nil && ip != nil {
			return ip, nil
		}
		lastErr = err
	}
	if lastErr == nil {
		lastErr = errors.New("no STUN server answered")
	}
	return nil, lastErr
}

// resolvePublicIP determines the address peers outside the LAN should be told
// to send media to: the explicit flag, the environment, a STUN probe, or
// finally the local address (which is correct on a host without NAT).
func resolvePublicIP(ctx context.Context, explicit string) (net.IP, error) {
	for _, candidate := range []string{explicit, os.Getenv("G4F_TURN_PUBLIC_IP")} {
		candidate = strings.TrimSpace(candidate)
		if candidate == "" {
			continue
		}
		ip := net.ParseIP(candidate)
		if ip == nil {
			return nil, fmt.Errorf("invalid public IP %q", candidate)
		}
		return ip, nil
	}
	if ip, err := stunPublicIP(ctx); err == nil {
		return ip, nil
	}
	return localIPv4(), nil
}

// --- options ---------------------------------------------------------------

// turnOptions is the resolved configuration of an embedded TURN server.
type turnOptions struct {
	PublicIP  net.IP
	BindIP    string
	Port      int
	TLSPort   int
	Realm     string
	Secret    string
	MinPort   int
	MaxPort   int
	EnableTCP bool
	EnableTLS bool
	CertFile  string
	KeyFile   string
	Quota     int
	LogLevel  logging.LogLevel
}

// turnEnvInt reads an integer override from the environment.
func turnEnvInt(name string, fallback int) int {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return fallback
	}
	value, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	return value
}

// turnEnvDuration reads a duration from the environment, accepting both Go
// duration strings ("1h", "30m") and a bare number of seconds.
func turnEnvDuration(name string, fallback time.Duration) time.Duration {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return fallback
	}
	if value, err := parseTurnDuration(raw); err == nil {
		return value
	}
	return fallback
}

// parseTurnDuration accepts a Go duration ("1h", "30m") or a bare number of
// seconds ("600"), matching how the remote desktop server reads RD_TURN_TTL.
func parseTurnDuration(raw string) (time.Duration, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return 0, errors.New("empty duration")
	}
	if value, err := time.ParseDuration(raw); err == nil {
		return value, nil
	}
	seconds, err := strconv.Atoi(raw)
	if err != nil {
		return 0, fmt.Errorf("invalid duration %q", raw)
	}
	return time.Duration(seconds) * time.Second, nil
}

// turnDurationFlag is a flag.Value that accepts both duration syntaxes.
type turnDurationFlag struct{ d *time.Duration }

func (f turnDurationFlag) String() string {
	if f.d == nil {
		return ""
	}
	return f.d.String()
}

func (f turnDurationFlag) Set(raw string) error {
	value, err := parseTurnDuration(raw)
	if err != nil {
		return err
	}
	*f.d = value
	return nil
}

// defaultTurnOptions builds the options used when the server is started
// implicitly (by `-m remote_desktop`) rather than through `turn serve`.
func defaultTurnOptions(ctx context.Context) (*turnOptions, error) {
	publicIP, err := resolvePublicIP(ctx, "")
	if err != nil {
		return nil, err
	}
	secret, err := ensureTurnSecret()
	if err != nil {
		return nil, err
	}
	realm := strings.TrimSpace(os.Getenv("G4F_TURN_REALM"))
	if realm == "" {
		realm = publicIP.String()
	}
	return &turnOptions{
		PublicIP:  publicIP,
		BindIP:    strings.TrimSpace(os.Getenv("G4F_TURN_BIND")),
		Port:      turnEnvInt("G4F_TURN_PORT", turnDefaultPort),
		TLSPort:   turnEnvInt("G4F_TURN_TLS_PORT", turnDefaultTLSPort),
		Realm:     realm,
		Secret:    secret,
		MinPort:   turnEnvInt("G4F_TURN_MIN_PORT", turnDefaultMinPort),
		MaxPort:   turnEnvInt("G4F_TURN_MAX_PORT", turnDefaultMaxPort),
		EnableTCP: true,
		Quota:     turnEnvInt("G4F_TURN_QUOTA", turnDefaultQuota),
		LogLevel:  logging.LogLevelError,
	}, nil
}

// --- quota -----------------------------------------------------------------

// turnQuota enforces a per-source-IP allocation limit, the equivalent of
// coturn's total-quota. Without it a single client could exhaust the relay
// port range.
type turnQuota struct {
	mu    sync.Mutex
	limit int
	byIP  map[string]int
}

func newTurnQuota(limit int) *turnQuota {
	return &turnQuota{limit: limit, byIP: make(map[string]int)}
}

func (q *turnQuota) allow(srcAddr net.Addr) bool {
	if q == nil || q.limit <= 0 {
		return true
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.byIP[turnAddrIP(srcAddr)] < q.limit
}

func (q *turnQuota) add(srcAddr net.Addr) {
	if q == nil || q.limit <= 0 {
		return
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	q.byIP[turnAddrIP(srcAddr)]++
}

func (q *turnQuota) remove(srcAddr net.Addr) {
	if q == nil || q.limit <= 0 {
		return
	}
	key := turnAddrIP(srcAddr)
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.byIP[key] <= 1 {
		delete(q.byIP, key)
		return
	}
	q.byIP[key]--
}

func turnAddrIP(addr net.Addr) string {
	if addr == nil {
		return ""
	}
	if host, _, err := net.SplitHostPort(addr.String()); err == nil {
		return host
	}
	return addr.String()
}

// --- server ----------------------------------------------------------------

// embeddedTurn is a running pion/turn server plus the details needed to
// advertise it.
type embeddedTurn struct {
	server  *turn.Server
	urls    []string
	stunURL string
	secret  string
	realm   string
	state   turnState

	closeOnce sync.Once
}

// Close shuts the server down and clears the discovery state file.
func (t *embeddedTurn) Close() error {
	if t == nil {
		return nil
	}
	var err error
	t.closeOnce.Do(func() {
		err = t.server.Close()
		if st, serr := readTurnState(); serr == nil && st.PID == os.Getpid() {
			os.Remove(turnStatePath()) //nolint:errcheck
		}
	})
	return err
}

// turnLoggerFactory builds a pion logger factory for the requested level.
func turnLoggerFactory(level logging.LogLevel) *logging.DefaultLoggerFactory {
	return &logging.DefaultLoggerFactory{
		Writer:          os.Stderr,
		DefaultLogLevel: level,
		ScopeLevels: map[string]logging.LogLevel{
			"turn":       level,
			"turnc":      level,
			"allocation": level,
		},
	}
}

// turnLogLevel maps a CLI string onto a pion log level.
func turnLogLevel(name string) logging.LogLevel {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "disable", "disabled", "off", "none", "silent":
		return logging.LogLevelDisabled
	case "warn", "warning":
		return logging.LogLevelWarn
	case "info":
		return logging.LogLevelInfo
	case "debug":
		return logging.LogLevelDebug
	case "trace":
		return logging.LogLevelTrace
	default:
		return logging.LogLevelError
	}
}

// turnNetwork returns the socket family matching the public address.
func turnNetwork(ip net.IP) (string, string) {
	if ip != nil && ip.To4() == nil {
		return "udp6", "tcp6"
	}
	return "udp4", "tcp4"
}

// startEmbeddedTurn brings up the STUN/TURN listeners described by opts.
func startEmbeddedTurn(ctx context.Context, opts *turnOptions) (*embeddedTurn, error) {
	if opts.MinPort <= 0 || opts.MaxPort < opts.MinPort {
		return nil, fmt.Errorf("invalid relay port range %d-%d", opts.MinPort, opts.MaxPort)
	}
	if opts.Secret == "" {
		return nil, errors.New("empty TURN shared secret")
	}

	udpNet, tcpNet := turnNetwork(opts.PublicIP)
	bindIP := opts.BindIP
	if bindIP == "" {
		if udpNet == "udp6" {
			bindIP = "::"
		} else {
			bindIP = "0.0.0.0"
		}
	}

	relay := &turn.RelayAddressGeneratorPortRange{
		RelayAddress: opts.PublicIP,
		Address:      bindIP,
		MinPort:      uint16(opts.MinPort), //nolint:gosec // validated above
		MaxPort:      uint16(opts.MaxPort), //nolint:gosec // validated above
	}

	loggerFactory := turnLoggerFactory(opts.LogLevel)
	quota := newTurnQuota(opts.Quota)

	udpConn, err := net.ListenPacket(udpNet, net.JoinHostPort(bindIP, strconv.Itoa(opts.Port)))
	if err != nil {
		return nil, fmt.Errorf("listening on %s: %w", net.JoinHostPort(bindIP, strconv.Itoa(opts.Port)), err)
	}

	config := turn.ServerConfig{
		PacketConnConfigs: []turn.PacketConnConfig{{
			PacketConn:            udpConn,
			RelayAddressGenerator: relay,
		}},
		Realm:         opts.Realm,
		AuthHandler:   turn.LongTermTURNRESTAuthHandler(opts.Secret, loggerFactory.NewLogger("turn")),
		LoggerFactory: loggerFactory,
		QuotaHandler: func(username, realm string, srcAddr net.Addr) bool {
			return quota.allow(srcAddr)
		},
		EventHandler: turn.EventHandler{
			OnAllocationCreated: func(srcAddr, dstAddr net.Addr, protocol, username, realm string,
				relayAddr net.Addr, requestedPort int) {
				quota.add(srcAddr)
				loggerFactory.NewLogger("turn").Infof(
					"allocation created src=%s relay=%s protocol=%s", srcAddr, relayAddr, protocol)
			},
			OnAllocationDeleted: func(srcAddr, dstAddr net.Addr, protocol, username, realm string) {
				quota.remove(srcAddr)
				loggerFactory.NewLogger("turn").Infof("allocation deleted src=%s protocol=%s", srcAddr, protocol)
			},
		},
	}

	urls := []string{fmt.Sprintf("turn:%s:%d?transport=udp", opts.PublicIP, opts.Port)}

	// TCP and TLS listeners share the same UDP relay generator: pion/turn only
	// ever allocates UDP relays (RFC 6062 TCP relay is not implemented), the
	// listener merely carries the control channel.
	if opts.EnableTCP {
		tcpListener, lerr := net.Listen(tcpNet, net.JoinHostPort(bindIP, strconv.Itoa(opts.Port)))
		if lerr != nil {
			udpConn.Close() //nolint:errcheck
			return nil, fmt.Errorf("listening on tcp %s: %w", net.JoinHostPort(bindIP, strconv.Itoa(opts.Port)), lerr)
		}
		config.ListenerConfigs = append(config.ListenerConfigs, turn.ListenerConfig{
			Listener:              tcpListener,
			RelayAddressGenerator: relay,
		})
		urls = append(urls, fmt.Sprintf("turn:%s:%d?transport=tcp", opts.PublicIP, opts.Port))
	}

	if opts.EnableTLS {
		cert, cerr := tls.LoadX509KeyPair(opts.CertFile, opts.KeyFile)
		if cerr != nil {
			closeTurnListeners(config)
			udpConn.Close() //nolint:errcheck
			return nil, fmt.Errorf("loading TLS key pair: %w", cerr)
		}
		tlsListener, lerr := net.Listen(tcpNet, net.JoinHostPort(bindIP, strconv.Itoa(opts.TLSPort)))
		if lerr != nil {
			closeTurnListeners(config)
			udpConn.Close() //nolint:errcheck
			return nil, fmt.Errorf("listening on tls %s: %w", net.JoinHostPort(bindIP, strconv.Itoa(opts.TLSPort)), lerr)
		}
		config.ListenerConfigs = append(config.ListenerConfigs, turn.ListenerConfig{
			Listener:              tls.NewListener(tlsListener, &tls.Config{Certificates: []tls.Certificate{cert}}),
			RelayAddressGenerator: relay,
		})
		urls = append(urls, fmt.Sprintf("turns:%s:%d?transport=tcp", opts.PublicIP, opts.TLSPort))
	}

	server, err := turn.NewServer(config)
	if err != nil {
		closeTurnListeners(config)
		udpConn.Close() //nolint:errcheck
		return nil, err
	}

	instance := &embeddedTurn{
		server:  server,
		urls:    urls,
		stunURL: fmt.Sprintf("stun:%s:%d", opts.PublicIP, opts.Port),
		secret:  opts.Secret,
		realm:   opts.Realm,
		state: turnState{
			URLs:     urls,
			STUNURL:  fmt.Sprintf("stun:%s:%d", opts.PublicIP, opts.Port),
			Realm:    opts.Realm,
			PublicIP: opts.PublicIP.String(),
			Port:     opts.Port,
			TLSPort:  opts.TLSPort,
			PID:      os.Getpid(),
			Started:  time.Now().UTC().Format(time.RFC3339),
		},
	}

	if err := writeTurnState(instance.state); err != nil {
		server.Close() //nolint:errcheck
		return nil, err
	}

	// Shut down when the caller's context is cancelled.
	go func() {
		<-ctx.Done()
		instance.Close() //nolint:errcheck
	}()

	return instance, nil
}

// closeTurnListeners releases the listeners of a partially built config.
func closeTurnListeners(config turn.ServerConfig) {
	for _, lc := range config.ListenerConfigs {
		lc.Listener.Close() //nolint:errcheck
	}
}

// --- self-signed TLS -------------------------------------------------------

// ensureSelfSignedCert generates (once) a self-signed certificate for the TLS
// listener. Browsers reject untrusted certificates for `turns:` URLs, so this
// is only useful for testing or when the certificate is trusted manually.
func ensureSelfSignedCert() (string, string, error) {
	certPath := filepath.Join(turnDir(), "tls.crt")
	keyPath := filepath.Join(turnDir(), "tls.key")
	if fileExists(certPath) && fileExists(keyPath) {
		return certPath, keyPath, nil
	}
	if err := os.MkdirAll(turnDir(), 0o700); err != nil {
		return "", "", err
	}

	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return "", "", err
	}
	serial, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		return "", "", err
	}
	template := x509.Certificate{
		SerialNumber:          serial,
		Subject:               pkix.Name{CommonName: "g4f-go turn"},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().AddDate(10, 0, 0),
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IsCA:                  true,
	}
	der, err := x509.CreateCertificate(rand.Reader, &template, &template, &key.PublicKey, key)
	if err != nil {
		return "", "", err
	}
	keyDER, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		return "", "", err
	}

	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER})
	if err := os.WriteFile(certPath, certPEM, 0o600); err != nil {
		return "", "", err
	}
	if err := os.WriteFile(keyPath, keyPEM, 0o600); err != nil {
		return "", "", err
	}
	return certPath, keyPath, nil
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// --- CLI -------------------------------------------------------------------

// runTurnCommand handles `g4f-go turn [subcommand]`.
func runTurnCommand(ctx context.Context, args []string) int {
	if len(args) == 0 {
		printTurnHelp()
		return 0
	}
	switch args[0] {
	case "help", "--help", "-h":
		printTurnHelp()
		return 0
	case "serve", "run", "start":
		return turnServe(ctx, args[1:])
	case "status":
		return turnStatus()
	case "secret":
		return turnSecretCommand(args[1:])
	case "env":
		return turnEnvCommand(args[1:])
	case "credentials", "cred":
		return turnCredentialsCommand(args[1:])
	case "path":
		fmt.Println(turnDir())
		return 0
	}
	fmt.Fprintf(os.Stderr, "g4f-go: unknown turn subcommand %q\n", args[0])
	printTurnHelp()
	return 2
}

// printTurnHelp shows the `g4f-go turn` usage.
func printTurnHelp() {
	fmt.Printf(`g4f-go turn - embedded STUN/TURN server (pion/turn)

Usage:
  g4f-go turn serve [flags]     run the STUN/TURN server in the foreground
  g4f-go turn status            show the configured/running server
  g4f-go turn env [--json]      print RD_TURN_* variables for the server
  g4f-go turn secret [--new]    print (or rotate) the shared secret
  g4f-go turn credentials       mint a time-limited username/password pair
  g4f-go turn path              print the configuration directory

Serve flags:
  --public-ip <IP>     address peers should send media to (default: STUN probe)
  --bind <IP>          local address to bind (default: all interfaces)
  --port <PORT>        UDP/TCP port (default %d)
  --tls-port <PORT>    TLS port (default %d)
  --realm <REALM>      authentication realm (default: the public IP)
  --secret <SECRET>    shared secret (default: the persisted one)
  --min-port <PORT>    first relay port (default %d)
  --max-port <PORT>    last relay port (default %d)
  --no-tcp             disable the TCP listener
  --cert <FILE>        TLS certificate; enables the TLS listener
  --key <FILE>         TLS private key
  --tls-self-signed    generate a self-signed certificate for TLS
  --max-allocations N  per-IP allocation quota, 0 disables (default %d)
  --log-level <LEVEL>  disable, error, warn, info, debug, trace (default error)

Credentials flags:
  --ttl <DURATION>     credential lifetime, e.g. 1h, 600s or 600 (default 1h)
  --user <ID>          user id embedded in the username (default g4f)
  --secret <SECRET>    shared secret (default: the persisted one)
  --json               print a JSON object instead of plain lines

Every flag also has an environment variable (G4F_TURN_PUBLIC_IP, G4F_TURN_BIND,
G4F_TURN_PORT, G4F_TURN_TLS_PORT, G4F_TURN_REALM, G4F_TURN_SECRET,
G4F_TURN_MIN_PORT, G4F_TURN_MAX_PORT, G4F_TURN_QUOTA, G4F_TURN_CERT,
G4F_TURN_KEY); flags win over the environment.

The server speaks STUN and TURN on the same port and authenticates with the
TURN REST scheme, so the credentials minted by the remote desktop server
(remote_desktop.config.turn_credentials) are accepted as-is.

`+"`g4f-go -m remote_desktop`"+` starts this server automatically unless
G4F_TURN_AUTOSTART=0 is set or a TURN server is already configured.

State: %s
`, turnDefaultPort, turnDefaultTLSPort,
		turnDefaultMinPort, turnDefaultMaxPort, turnDefaultQuota, turnDir())
}

// turnServeFlags holds the `turn serve` options. Every flag defaults to its
// G4F_TURN_* environment variable so the server can be configured without
// arguments; an explicit flag always wins.
type turnServeFlags struct {
	publicIP *string
	bindIP   *string
	port     *int
	tlsPort  *int
	realm    *string
	secret   *string
	minPort  *int
	maxPort  *int
	noTCP    *bool
	certFile *string
	keyFile  *string
	selfSign *bool
	quota    *int
	logLevel *string
}

func newTurnServeFlags() (*flag.FlagSet, *turnServeFlags) {
	fs := flag.NewFlagSet("turn serve", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	f := &turnServeFlags{
		publicIP: fs.String("public-ip", strings.TrimSpace(os.Getenv("G4F_TURN_PUBLIC_IP")), "public IP peers should send media to"),
		bindIP:   fs.String("bind", strings.TrimSpace(os.Getenv("G4F_TURN_BIND")), "local address to bind"),
		port:     fs.Int("port", turnEnvInt("G4F_TURN_PORT", turnDefaultPort), "UDP/TCP listening port"),
		tlsPort:  fs.Int("tls-port", turnEnvInt("G4F_TURN_TLS_PORT", turnDefaultTLSPort), "TLS listening port"),
		realm:    fs.String("realm", strings.TrimSpace(os.Getenv("G4F_TURN_REALM")), "authentication realm"),
		secret:   fs.String("secret", strings.TrimSpace(os.Getenv("G4F_TURN_SECRET")), "shared secret"),
		minPort:  fs.Int("min-port", turnEnvInt("G4F_TURN_MIN_PORT", turnDefaultMinPort), "first relay port"),
		maxPort:  fs.Int("max-port", turnEnvInt("G4F_TURN_MAX_PORT", turnDefaultMaxPort), "last relay port"),
		noTCP:    fs.Bool("no-tcp", false, "disable the TCP listener"),
		certFile: fs.String("cert", strings.TrimSpace(os.Getenv("G4F_TURN_CERT")), "TLS certificate (PEM)"),
		keyFile:  fs.String("key", strings.TrimSpace(os.Getenv("G4F_TURN_KEY")), "TLS private key (PEM)"),
		selfSign: fs.Bool("tls-self-signed", false, "generate a self-signed TLS certificate"),
		quota:    fs.Int("max-allocations", turnEnvInt("G4F_TURN_QUOTA", turnDefaultQuota), "per-IP allocation quota"),
		logLevel: fs.String("log-level", "error", "pion log level"),
	}
	return fs, f
}

// turnServe runs the embedded server until the context is cancelled.
func turnServe(ctx context.Context, args []string) int {
	fs, f := newTurnServeFlags()
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(os.Stderr, "g4f-go: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	resolvedIP, err := resolvePublicIP(ctx, *f.publicIP)
	if err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
		return 1
	}

	resolvedSecret := strings.TrimSpace(*f.secret)
	if resolvedSecret == "" {
		resolvedSecret, err = ensureTurnSecret()
		if err != nil {
			fmt.Fprintln(os.Stderr, "g4f-go:", err)
			return 1
		}
	}

	resolvedRealm := strings.TrimSpace(*f.realm)
	if resolvedRealm == "" {
		resolvedRealm = resolvedIP.String()
	}

	opts := &turnOptions{
		PublicIP:  resolvedIP,
		BindIP:    strings.TrimSpace(*f.bindIP),
		Port:      *f.port,
		TLSPort:   *f.tlsPort,
		Realm:     resolvedRealm,
		Secret:    resolvedSecret,
		MinPort:   *f.minPort,
		MaxPort:   *f.maxPort,
		EnableTCP: !*f.noTCP,
		Quota:     *f.quota,
		LogLevel:  turnLogLevel(*f.logLevel),
	}

	switch {
	case *f.certFile != "" || *f.keyFile != "":
		if *f.certFile == "" || *f.keyFile == "" {
			fmt.Fprintln(os.Stderr, "g4f-go: --cert and --key must be given together")
			return 2
		}
		opts.EnableTLS = true
		opts.CertFile, opts.KeyFile = *f.certFile, *f.keyFile
	case *f.selfSign:
		cert, key, cerr := ensureSelfSignedCert()
		if cerr != nil {
			fmt.Fprintln(os.Stderr, "g4f-go:", cerr)
			return 1
		}
		opts.EnableTLS = true
		opts.CertFile, opts.KeyFile = cert, key
		fmt.Fprintln(os.Stderr,
			"g4f-go: using a self-signed TLS certificate; browsers reject it unless it is trusted")
	}

	instance, err := startEmbeddedTurn(ctx, opts)
	if err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
		return 1
	}
	defer instance.Close() //nolint:errcheck

	printTurnBanner(opts, instance)

	<-ctx.Done()
	fmt.Println("g4f-go: shutting down the TURN server")
	return 0
}

// printTurnBanner reports the listeners and the variables to hand to the
// remote desktop server.
func printTurnBanner(opts *turnOptions, instance *embeddedTurn) {
	listeners := []string{fmt.Sprintf("udp %s:%d", displayBind(opts.BindIP), opts.Port)}
	if opts.EnableTCP {
		listeners = append(listeners, fmt.Sprintf("tcp %s:%d", displayBind(opts.BindIP), opts.Port))
	}
	if opts.EnableTLS {
		listeners = append(listeners, fmt.Sprintf("tls %s:%d", displayBind(opts.BindIP), opts.TLSPort))
	}

	fmt.Println()
	fmt.Println("  Embedded STUN/TURN server (pion/turn)")
	fmt.Println("  " + strings.Repeat("-", 46))
	fmt.Printf("  public IP   : %s\n", opts.PublicIP)
	fmt.Printf("  realm       : %s\n", opts.Realm)
	fmt.Printf("  listeners   : %s\n", strings.Join(listeners, ", "))
	fmt.Printf("  relay ports : %d-%d\n", opts.MinPort, opts.MaxPort)
	fmt.Printf("  secret file : %s\n", turnSecretPath())
	fmt.Printf("  state file  : %s\n", turnStatePath())
	fmt.Println("  " + strings.Repeat("-", 46))
	fmt.Println("  Start the remote desktop server with:")
	fmt.Println()
	fmt.Printf("    RD_TURN_URL=%q \\\n", strings.Join(instance.urls, ","))
	fmt.Printf("    RD_TURN_SECRET=%q \\\n", instance.secret)
	fmt.Printf("    RD_STUN_URL=%q \\\n", instance.stunURL)
	fmt.Println("    g4f-go -m remote_desktop")
	fmt.Println()
	fmt.Println("  Or simply run `g4f-go -m remote_desktop`: it discovers this server")
	fmt.Println("  through the state file above.")
	fmt.Println()
	fmt.Printf("  Forward on your router: UDP %d, UDP %d-%d", opts.Port, opts.MinPort, opts.MaxPort)
	if opts.EnableTCP {
		fmt.Printf(", TCP %d", opts.Port)
	}
	if opts.EnableTLS {
		fmt.Printf(", TCP %d", opts.TLSPort)
	}
	fmt.Println()
	fmt.Println()
}

func displayBind(bind string) string {
	if bind == "" {
		return "0.0.0.0"
	}
	return bind
}

// turnStatus reports the persisted configuration and whether a server runs.
func turnStatus() int {
	fmt.Printf("config dir:  %s\n", turnDir())
	if _, err := readTurnSecret(); err == nil {
		fmt.Printf("secret:      %s (present)\n", turnSecretPath())
	} else {
		fmt.Printf("secret:      %s (not generated yet)\n", turnSecretPath())
	}

	st, err := readTurnState()
	if err != nil {
		fmt.Println("server:      not running (no state file)")
		return 0
	}
	if processAlive(st.PID) {
		fmt.Printf("server:      running (pid %d, started %s)\n", st.PID, st.Started)
	} else {
		fmt.Printf("server:      not running (stale state file, pid %d)\n", st.PID)
	}
	fmt.Printf("public IP:   %s\n", st.PublicIP)
	fmt.Printf("realm:       %s\n", st.Realm)
	fmt.Printf("urls:        %s\n", strings.Join(st.URLs, ", "))
	if st.STUNURL != "" {
		fmt.Printf("stun:        %s\n", st.STUNURL)
	}
	return 0
}

// turnSecretCommand prints or rotates the shared secret.
func turnSecretCommand(args []string) int {
	rotate := false
	for _, a := range args {
		switch a {
		case "--new", "-n", "--rotate":
			rotate = true
		case "help", "--help", "-h":
			fmt.Println("usage: g4f-go turn secret [--new]")
			return 0
		default:
			fmt.Fprintf(os.Stderr, "g4f-go: unknown argument %q\n", a)
			return 2
		}
	}
	if rotate {
		secret, err := newTurnSecret()
		if err != nil {
			fmt.Fprintln(os.Stderr, "g4f-go:", err)
			return 1
		}
		if err := writeTurnSecret(secret); err != nil {
			fmt.Fprintln(os.Stderr, "g4f-go:", err)
			return 1
		}
		fmt.Fprintln(os.Stderr, "g4f-go: rotated the TURN secret; restart the server to apply it")
		fmt.Println(secret)
		return 0
	}
	secret, err := ensureTurnSecret()
	if err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
		return 1
	}
	fmt.Println(secret)
	return 0
}

// turnEnvCommand prints the environment for the configured server.
func turnEnvCommand(args []string) int {
	asJSON := false
	for _, a := range args {
		switch a {
		case "--json":
			asJSON = true
		case "help", "--help", "-h":
			fmt.Println("usage: g4f-go turn env [--json]")
			return 0
		default:
			fmt.Fprintf(os.Stderr, "g4f-go: unknown argument %q\n", a)
			return 2
		}
	}

	st, err := readTurnState()
	if err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go: no TURN server configured; run `g4f-go turn serve` first")
		return 1
	}
	secret, err := readTurnSecret()
	if err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
		return 1
	}

	if asJSON {
		payload := map[string]any{
			"RD_TURN_URL":    strings.Join(st.URLs, ","),
			"RD_TURN_SECRET": secret,
			"RD_STUN_URL":    st.STUNURL,
			"realm":          st.Realm,
			"public_ip":      st.PublicIP,
			"pid":            st.PID,
		}
		data, merr := json.MarshalIndent(payload, "", "  ")
		if merr != nil {
			fmt.Fprintln(os.Stderr, "g4f-go:", merr)
			return 1
		}
		fmt.Println(string(data))
		return 0
	}

	fmt.Printf("export RD_TURN_URL=%s\n", shellQuote(strings.Join(st.URLs, ",")))
	fmt.Printf("export RD_TURN_SECRET=%s\n", shellQuote(secret))
	if st.STUNURL != "" {
		fmt.Printf("export RD_STUN_URL=%s\n", shellQuote(st.STUNURL))
	}
	return 0
}

// shellQuote wraps a value in single quotes for POSIX shells.
func shellQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", `'\''`) + "'"
}

// turnCredentialsCommand mints a TURN REST credential pair, matching what the
// Python server hands to the browsers.
func turnCredentialsCommand(args []string) int {
	fs := flag.NewFlagSet("turn credentials", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	ttl := turnEnvDuration("G4F_TURN_TTL", turnDefaultTTL)
	fs.Var(turnDurationFlag{&ttl}, "ttl", "credential `duration` (e.g. 1h, 600s or 600)")
	user := fs.String("user", "g4f", "user id embedded in the username")
	secret := fs.String("secret", "", "shared secret (default: the persisted one)")
	asJSON := fs.Bool("json", false, "print a JSON object instead of plain lines")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}

	resolved := strings.TrimSpace(*secret)
	if resolved == "" {
		var err error
		resolved, err = ensureTurnSecret()
		if err != nil {
			fmt.Fprintln(os.Stderr, "g4f-go:", err)
			return 1
		}
	}

	username, password, err := turn.GenerateLongTermTURNRESTCredentials(resolved, *user, ttl)
	if err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
		return 1
	}

	if *asJSON {
		payload := map[string]any{
			"username": username,
			"password": password,
			"ttl":      int(ttl.Seconds()),
		}
		data, merr := json.MarshalIndent(payload, "", "  ")
		if merr != nil {
			fmt.Fprintln(os.Stderr, "g4f-go:", merr)
			return 1
		}
		fmt.Println(string(data))
		return 0
	}

	fmt.Printf("username: %s\npassword: %s\n", username, password)
	return 0
}

// --- remote desktop integration --------------------------------------------

// turnAutostartEnabled reports whether `-m remote_desktop` should bring up the
// embedded TURN server. It is on by default and disabled with
// G4F_TURN_AUTOSTART=0.
func turnAutostartEnabled() bool {
	raw := strings.TrimSpace(os.Getenv("G4F_TURN_AUTOSTART"))
	if raw == "" {
		return true
	}
	return isTruthy(raw)
}

// remoteDesktopTurnEnv starts the embedded TURN server for the remote desktop
// module and returns the environment that points the Python server at it.
// It is a no-op when autostart is disabled or a relay is already configured.
func remoteDesktopTurnEnv(ctx context.Context) ([]string, func()) {
	if !turnAutostartEnabled() {
		return nil, nil
	}
	for _, name := range []string{"RD_TURN_URL", "RD_ICE_SERVERS"} {
		if strings.TrimSpace(os.Getenv(name)) != "" {
			return nil, nil
		}
	}

	opts, err := defaultTurnOptions(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "g4f-go: warning: embedded TURN server not started: %v\n", err)
		return nil, nil
	}
	instance, err := startEmbeddedTurn(ctx, opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "g4f-go: warning: embedded TURN server not started: %v\n", err)
		return nil, nil
	}

	fmt.Printf("Embedded TURN relay: %s (realm %s)\n", strings.Join(instance.urls, ", "), instance.realm)
	return []string{
		"RD_TURN_URL=" + strings.Join(instance.urls, ","),
		"RD_TURN_SECRET=" + instance.secret,
		"RD_STUN_URL=" + instance.stunURL,
	}, func() { instance.Close() } //nolint:errcheck
}
