package main

import (
	"context"
	"crypto/hmac"
	"crypto/sha1" //nolint:gosec // the TURN REST scheme mandates HMAC-SHA1
	"encoding/base64"
	"encoding/json"
	"io"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/pion/logging"
	"github.com/pion/stun/v3"
	"github.com/pion/turn/v4"
)

// isolateTurnConfig points the config directory at a throwaway HOME so tests
// never touch the real ~/.g4f/turn.
func isolateTurnConfig(t *testing.T) string {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv("APPDATA", filepath.Join(home, "AppData", "Roaming"))
	return filepath.Join(home, ".g4f", "turn")
}

// freeUDPPort asks the OS for an unused UDP port.
func freeUDPPort(t *testing.T) int {
	t.Helper()
	conn, err := net.ListenPacket("udp4", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("reserving a UDP port: %v", err)
	}
	defer conn.Close() //nolint:errcheck
	return conn.LocalAddr().(*net.UDPAddr).Port
}

// captureStdout runs fn with os.Stdout redirected to a pipe and returns what it
// printed.
func captureStdout(t *testing.T, fn func()) string {
	t.Helper()
	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe: %v", err)
	}
	saved := os.Stdout
	os.Stdout = writer
	defer func() { os.Stdout = saved }()

	fn()

	if err := writer.Close(); err != nil {
		t.Fatalf("closing the pipe writer: %v", err)
	}
	data, err := io.ReadAll(reader)
	if err != nil {
		t.Fatalf("reading the captured output: %v", err)
	}
	if err := reader.Close(); err != nil {
		t.Fatalf("closing the pipe reader: %v", err)
	}
	return string(data)
}

func TestTurnSecretRoundTrip(t *testing.T) {
	isolateTurnConfig(t)

	if _, err := readTurnSecret(); err == nil {
		t.Fatal("expected no secret before one is generated")
	}

	secret, err := ensureTurnSecret()
	if err != nil {
		t.Fatalf("ensureTurnSecret: %v", err)
	}
	if len(secret) < 32 {
		t.Fatalf("secret is too short: %q", secret)
	}
	if strings.ContainsAny(secret, "+/=") {
		t.Fatalf("secret is not URL-safe: %q", secret)
	}

	again, err := ensureTurnSecret()
	if err != nil {
		t.Fatalf("ensureTurnSecret (second call): %v", err)
	}
	if again != secret {
		t.Fatalf("secret is not stable: %q != %q", again, secret)
	}

	info, err := os.Stat(turnSecretPath())
	if err != nil {
		t.Fatalf("stat secret: %v", err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Fatalf("secret permissions = %o, want 600", perm)
	}

	other, err := newTurnSecret()
	if err != nil {
		t.Fatalf("newTurnSecret: %v", err)
	}
	if other == secret {
		t.Fatal("two generated secrets are identical")
	}
}

func TestTurnStateRoundTrip(t *testing.T) {
	isolateTurnConfig(t)

	if _, err := readTurnState(); err == nil {
		t.Fatal("expected no state before one is written")
	}

	want := turnState{
		URLs:     []string{"turn:203.0.113.7:3478?transport=udp"},
		STUNURL:  "stun:203.0.113.7:3478",
		Realm:    "203.0.113.7",
		PublicIP: "203.0.113.7",
		Port:     3478,
		TLSPort:  5349,
		PID:      os.Getpid(),
		Started:  time.Now().UTC().Format(time.RFC3339),
	}
	if err := writeTurnState(want); err != nil {
		t.Fatalf("writeTurnState: %v", err)
	}

	got, err := readTurnState()
	if err != nil {
		t.Fatalf("readTurnState: %v", err)
	}
	if got.PublicIP != want.PublicIP || got.Port != want.Port || got.Realm != want.Realm {
		t.Fatalf("state mismatch: %+v != %+v", got, want)
	}
	if len(got.URLs) != 1 || got.URLs[0] != want.URLs[0] {
		t.Fatalf("urls mismatch: %v", got.URLs)
	}
}

func TestProcessAlive(t *testing.T) {
	if !processAlive(os.Getpid()) {
		t.Fatal("the current process should be reported as alive")
	}
	if processAlive(0) || processAlive(-1) {
		t.Fatal("non-positive pids must not be reported as alive")
	}
}

// TestTurnCredentialsMatchPython pins the wire format to the one produced by
// remote_desktop.config.turn_credentials(), so credentials minted by either
// side authenticate against the other.
func TestTurnCredentialsMatchPython(t *testing.T) {
	const (
		secret   = "test-shared-secret"
		username = "4102444800:abcd1234" // 2100-01-01, so the handler accepts it
		// Computed with Python:
		//   base64.b64encode(hmac.new(secret, username, hashlib.sha1).digest())
		wantPassword = "23VaYvQVBZztrPARsJZ2MGFAjuA="
	)

	mac := hmac.New(sha1.New, []byte(secret))
	if _, err := mac.Write([]byte(username)); err != nil {
		t.Fatalf("hmac write: %v", err)
	}
	if got := base64.StdEncoding.EncodeToString(mac.Sum(nil)); got != wantPassword {
		t.Fatalf("password = %q, want %q", got, wantPassword)
	}

	// The server-side handler must derive exactly the same key.
	handler := turn.LongTermTURNRESTAuthHandler(secret, nil)
	key, ok := handler(username, "realm.example", &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 1234})
	if !ok {
		t.Fatal("handler rejected a valid, unexpired credential")
	}
	want := turn.GenerateAuthKey(username, "realm.example", wantPassword)
	if !hmac.Equal(key, want) {
		t.Fatal("handler derived a different auth key than the Python scheme")
	}
}

func TestTurnAuthHandlerRejectsExpiredAndMalformed(t *testing.T) {
	handler := turn.LongTermTURNRESTAuthHandler("secret", nil)
	addr := &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 1234}

	expired := strconv.FormatInt(time.Now().Add(-time.Minute).Unix(), 10) + ":abcd1234"
	if _, ok := handler(expired, "realm", addr); ok {
		t.Fatal("handler accepted an expired credential")
	}
	if _, ok := handler("not-a-timestamp:abcd1234", "realm", addr); ok {
		t.Fatal("handler accepted a malformed credential")
	}
}

func TestGenerateLongTermTURNRESTCredentials(t *testing.T) {
	secret := "another-secret"
	username, password, err := turn.GenerateLongTermTURNRESTCredentials(secret, "g4f", time.Hour)
	if err != nil {
		t.Fatalf("GenerateLongTermTURNRESTCredentials: %v", err)
	}

	parts := strings.Split(username, ":")
	if len(parts) != 2 || parts[1] != "g4f" {
		t.Fatalf("unexpected username %q", username)
	}
	expiry, err := strconv.Atoi(parts[0])
	if err != nil {
		t.Fatalf("username expiry is not an integer: %q", parts[0])
	}
	if int64(expiry) < time.Now().Unix() {
		t.Fatalf("credential is already expired: %d", expiry)
	}

	mac := hmac.New(sha1.New, []byte(secret))
	if _, err := mac.Write([]byte(username)); err != nil {
		t.Fatalf("hmac write: %v", err)
	}
	if want := base64.StdEncoding.EncodeToString(mac.Sum(nil)); password != want {
		t.Fatalf("password = %q, want %q", password, want)
	}
}

func TestTurnQuota(t *testing.T) {
	quota := newTurnQuota(2)
	a := &net.UDPAddr{IP: net.IPv4(10, 0, 0, 1), Port: 1000}
	b := &net.UDPAddr{IP: net.IPv4(10, 0, 0, 2), Port: 1000}

	if !quota.allow(a) {
		t.Fatal("first allocation should be allowed")
	}
	quota.add(a)
	quota.add(a)
	if quota.allow(a) {
		t.Fatal("allocation over the quota should be rejected")
	}
	if !quota.allow(b) {
		t.Fatal("the quota must be tracked per source IP")
	}

	quota.remove(a)
	if !quota.allow(a) {
		t.Fatal("allocation should be allowed again after a release")
	}
	quota.remove(a)
	quota.remove(a)
	if !quota.allow(a) {
		t.Fatal("releasing below zero must not corrupt the counter")
	}

	unlimited := newTurnQuota(0)
	for i := 0; i < 100; i++ {
		unlimited.add(a)
	}
	if !unlimited.allow(a) {
		t.Fatal("a zero quota must disable the limit")
	}
}

func TestTurnAddrIP(t *testing.T) {
	if got := turnAddrIP(&net.UDPAddr{IP: net.IPv4(192, 0, 2, 5), Port: 9}); got != "192.0.2.5" {
		t.Fatalf("turnAddrIP = %q", got)
	}
	if got := turnAddrIP(nil); got != "" {
		t.Fatalf("turnAddrIP(nil) = %q", got)
	}
}

func TestTurnLogLevel(t *testing.T) {
	cases := map[string]logging.LogLevel{
		"disable": logging.LogLevelDisabled,
		"off":     logging.LogLevelDisabled,
		"warn":    logging.LogLevelWarn,
		"info":    logging.LogLevelInfo,
		"debug":   logging.LogLevelDebug,
		"trace":   logging.LogLevelTrace,
		"":        logging.LogLevelError,
		"bogus":   logging.LogLevelError,
	}
	for input, want := range cases {
		if got := turnLogLevel(input); got != want {
			t.Fatalf("turnLogLevel(%q) = %v, want %v", input, got, want)
		}
	}
}

func TestTurnNetwork(t *testing.T) {
	if udp, tcp := turnNetwork(net.IPv4(192, 0, 2, 1)); udp != "udp4" || tcp != "tcp4" {
		t.Fatalf("IPv4 network = %q/%q", udp, tcp)
	}
	if udp, tcp := turnNetwork(net.ParseIP("2001:db8::1")); udp != "udp6" || tcp != "tcp6" {
		t.Fatalf("IPv6 network = %q/%q", udp, tcp)
	}
	if udp, _ := turnNetwork(nil); udp != "udp4" {
		t.Fatalf("nil network = %q", udp)
	}
}

func TestTurnEnvParsing(t *testing.T) {
	t.Setenv("G4F_TURN_TEST_INT", "1234")
	if got := turnEnvInt("G4F_TURN_TEST_INT", 7); got != 1234 {
		t.Fatalf("turnEnvInt = %d", got)
	}
	t.Setenv("G4F_TURN_TEST_INT", "not-a-number")
	if got := turnEnvInt("G4F_TURN_TEST_INT", 7); got != 7 {
		t.Fatalf("turnEnvInt fallback = %d", got)
	}
	os.Unsetenv("G4F_TURN_TEST_INT")
	if got := turnEnvInt("G4F_TURN_TEST_INT", 7); got != 7 {
		t.Fatalf("turnEnvInt default = %d", got)
	}

	t.Setenv("G4F_TURN_TEST_DUR", "90m")
	if got := turnEnvDuration("G4F_TURN_TEST_DUR", time.Hour); got != 90*time.Minute {
		t.Fatalf("turnEnvDuration = %v", got)
	}
	t.Setenv("G4F_TURN_TEST_DUR", "600")
	if got := turnEnvDuration("G4F_TURN_TEST_DUR", time.Hour); got != 10*time.Minute {
		t.Fatalf("turnEnvDuration seconds = %v", got)
	}
	t.Setenv("G4F_TURN_TEST_DUR", "nonsense")
	if got := turnEnvDuration("G4F_TURN_TEST_DUR", time.Hour); got != time.Hour {
		t.Fatalf("turnEnvDuration fallback = %v", got)
	}
}

func TestResolvePublicIPExplicit(t *testing.T) {
	ip, err := resolvePublicIP(context.Background(), "203.0.113.9")
	if err != nil {
		t.Fatalf("resolvePublicIP: %v", err)
	}
	if ip.String() != "203.0.113.9" {
		t.Fatalf("resolvePublicIP = %s", ip)
	}
	if _, err := resolvePublicIP(context.Background(), "not-an-ip"); err == nil {
		t.Fatal("expected an error for an invalid address")
	}
}

func TestTurnAutostartEnabled(t *testing.T) {
	t.Setenv("G4F_TURN_AUTOSTART", "")
	os.Unsetenv("G4F_TURN_AUTOSTART")
	if !turnAutostartEnabled() {
		t.Fatal("autostart should default to enabled")
	}
	for _, off := range []string{"0", "false", "no", "off"} {
		t.Setenv("G4F_TURN_AUTOSTART", off)
		if turnAutostartEnabled() {
			t.Fatalf("autostart should be disabled by %q", off)
		}
	}
	t.Setenv("G4F_TURN_AUTOSTART", "1")
	if !turnAutostartEnabled() {
		t.Fatal("autostart should be enabled by \"1\"")
	}
}

func TestRemoteDesktopTurnEnvSkipsWhenConfigured(t *testing.T) {
	isolateTurnConfig(t)

	t.Setenv("G4F_TURN_AUTOSTART", "0")
	if env, stop := remoteDesktopTurnEnv(context.Background()); env != nil || stop != nil {
		t.Fatal("autostart disabled should be a no-op")
	}

	t.Setenv("G4F_TURN_AUTOSTART", "1")
	t.Setenv("RD_TURN_URL", "turn:turn.example.com:3478")
	if env, stop := remoteDesktopTurnEnv(context.Background()); env != nil || stop != nil {
		t.Fatal("an existing RD_TURN_URL should be respected")
	}

	os.Unsetenv("RD_TURN_URL")
	t.Setenv("RD_ICE_SERVERS", `[{"urls":["stun:stun.example.com"]}]`)
	if env, stop := remoteDesktopTurnEnv(context.Background()); env != nil || stop != nil {
		t.Fatal("an existing RD_ICE_SERVERS should be respected")
	}
}

func TestShellQuote(t *testing.T) {
	cases := map[string]string{
		"plain":     "'plain'",
		"with sp":   "'with sp'",
		"it's":      `'it'\''s'`,
		"turn:a:b":  "'turn:a:b'",
		"":          "''",
		"a\nb":      "'a\nb'",
		"$HOME`id`": "'$HOME`id`'",
	}
	for input, want := range cases {
		if got := shellQuote(input); got != want {
			t.Fatalf("shellQuote(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestEnsureSelfSignedCert(t *testing.T) {
	isolateTurnConfig(t)

	cert, key, err := ensureSelfSignedCert()
	if err != nil {
		t.Fatalf("ensureSelfSignedCert: %v", err)
	}
	if !fileExists(cert) || !fileExists(key) {
		t.Fatalf("certificate files were not created: %s %s", cert, key)
	}

	// A second call must reuse the existing material.
	cert2, key2, err := ensureSelfSignedCert()
	if err != nil {
		t.Fatalf("ensureSelfSignedCert (second call): %v", err)
	}
	if cert2 != cert || key2 != key {
		t.Fatal("certificate paths changed between calls")
	}
}

// startTestTurn brings up an embedded server on the loopback interface.
func startTestTurn(t *testing.T, quota int) (*embeddedTurn, *turnOptions) {
	t.Helper()
	isolateTurnConfig(t)

	secret, err := ensureTurnSecret()
	if err != nil {
		t.Fatalf("ensureTurnSecret: %v", err)
	}
	relayPort := freeUDPPort(t)
	opts := &turnOptions{
		PublicIP:  net.IPv4(127, 0, 0, 1),
		BindIP:    "127.0.0.1",
		Port:      freeUDPPort(t),
		Realm:     "g4f.test",
		Secret:    secret,
		MinPort:   relayPort,
		MaxPort:   relayPort,
		EnableTCP: true,
		Quota:     quota,
		LogLevel:  logging.LogLevelError,
	}

	ctx, cancel := context.WithCancel(context.Background())
	instance, err := startEmbeddedTurn(ctx, opts)
	if err != nil {
		cancel()
		t.Fatalf("startEmbeddedTurn: %v", err)
	}
	t.Cleanup(func() {
		cancel()
		instance.Close() //nolint:errcheck
	})
	return instance, opts
}

// newTestTurnClient dials the embedded server with freshly minted credentials.
func newTestTurnClient(t *testing.T, opts *turnOptions) *turn.Client {
	t.Helper()

	username, password, err := turn.GenerateLongTermTURNRESTCredentials(opts.Secret, "g4f", time.Hour)
	if err != nil {
		t.Fatalf("GenerateLongTermTURNRESTCredentials: %v", err)
	}

	conn, err := net.ListenPacket("udp4", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("client socket: %v", err)
	}
	t.Cleanup(func() { conn.Close() }) //nolint:errcheck

	addr := net.JoinHostPort("127.0.0.1", strconv.Itoa(opts.Port))
	client, err := turn.NewClient(&turn.ClientConfig{
		STUNServerAddr: addr,
		TURNServerAddr: addr,
		Conn:           conn,
		Username:       username,
		Password:       password,
		Realm:          opts.Realm,
		LoggerFactory:  turnLoggerFactory(logging.LogLevelError),
	})
	if err != nil {
		t.Fatalf("turn.NewClient: %v", err)
	}
	t.Cleanup(client.Close)
	if err := client.Listen(); err != nil {
		t.Fatalf("client.Listen: %v", err)
	}
	return client
}

func TestEmbeddedTurnSTUNBinding(t *testing.T) {
	_, opts := startTestTurn(t, 0)
	client := newTestTurnClient(t, opts)

	mapped, err := client.SendBindingRequest()
	if err != nil {
		t.Fatalf("SendBindingRequest: %v", err)
	}
	if mapped == nil {
		t.Fatal("no mapped address returned")
	}
	udpAddr, ok := mapped.(*net.UDPAddr)
	if !ok {
		t.Fatalf("mapped address is %T", mapped)
	}
	if !udpAddr.IP.IsLoopback() {
		t.Fatalf("mapped address %s is not loopback", mapped)
	}
}

// TestEmbeddedTurnRawSTUNBinding speaks STUN directly, the way a browser's ICE
// agent does, to prove the server is a usable STUN server and not only a TURN
// endpoint.
func TestEmbeddedTurnRawSTUNBinding(t *testing.T) {
	_, opts := startTestTurn(t, 0)

	conn, err := net.ListenPacket("udp4", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("client socket: %v", err)
	}
	defer conn.Close() //nolint:errcheck

	server := &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1), Port: opts.Port}
	request := stun.MustBuild(stun.TransactionID, stun.BindingRequest)
	if _, err := conn.WriteTo(request.Raw, server); err != nil {
		t.Fatalf("writing the binding request: %v", err)
	}

	if err := conn.SetReadDeadline(time.Now().Add(5 * time.Second)); err != nil {
		t.Fatalf("SetReadDeadline: %v", err)
	}
	buf := make([]byte, 1500)
	n, _, err := conn.ReadFrom(buf)
	if err != nil {
		t.Fatalf("reading the binding response: %v", err)
	}

	response := &stun.Message{Raw: buf[:n]}
	if err := response.Decode(); err != nil {
		t.Fatalf("decoding the response: %v", err)
	}
	if response.Type != stun.BindingSuccess {
		t.Fatalf("response type = %s, want %s", response.Type, stun.BindingSuccess)
	}
	if response.TransactionID != request.TransactionID {
		t.Fatal("the response does not echo the transaction id")
	}

	var xorAddr stun.XORMappedAddress
	if err := xorAddr.GetFrom(response); err != nil {
		t.Fatalf("XOR-MAPPED-ADDRESS: %v", err)
	}
	if !xorAddr.IP.IsLoopback() {
		t.Fatalf("mapped address %s is not loopback", xorAddr)
	}
	if xorAddr.Port != conn.LocalAddr().(*net.UDPAddr).Port {
		t.Fatalf("mapped port = %d, want %d", xorAddr.Port, conn.LocalAddr().(*net.UDPAddr).Port)
	}
}

func TestEmbeddedTurnAllocationAndRelay(t *testing.T) {
	_, opts := startTestTurn(t, 0)
	client := newTestTurnClient(t, opts)

	relayConn, err := client.Allocate()
	if err != nil {
		t.Fatalf("Allocate: %v", err)
	}
	defer relayConn.Close() //nolint:errcheck

	relayAddr, ok := relayConn.LocalAddr().(*net.UDPAddr)
	if !ok {
		t.Fatalf("relay address is %T", relayConn.LocalAddr())
	}
	if !relayAddr.IP.IsLoopback() {
		t.Fatalf("relay address %s is not loopback", relayAddr)
	}
	if relayAddr.Port < opts.MinPort || relayAddr.Port > opts.MaxPort {
		t.Fatalf("relay port %d is outside %d-%d", relayAddr.Port, opts.MinPort, opts.MaxPort)
	}

	// A peer socket that the relay will be told about.
	peer, err := net.ListenPacket("udp4", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("peer socket: %v", err)
	}
	defer peer.Close() //nolint:errcheck

	deadline := time.Now().Add(5 * time.Second)
	if err := peer.SetDeadline(deadline); err != nil {
		t.Fatalf("peer deadline: %v", err)
	}
	if err := relayConn.SetDeadline(deadline); err != nil {
		t.Fatalf("relay deadline: %v", err)
	}

	// Sending through the relay installs the permission for the peer's IP.
	if _, err := relayConn.WriteTo([]byte("ping"), peer.LocalAddr()); err != nil {
		t.Fatalf("relay write: %v", err)
	}

	buf := make([]byte, 1500)
	n, from, err := peer.ReadFrom(buf)
	if err != nil {
		t.Fatalf("peer read: %v", err)
	}
	if string(buf[:n]) != "ping" {
		t.Fatalf("peer received %q", buf[:n])
	}
	if from.String() != relayAddr.String() {
		t.Fatalf("peer saw source %s, want %s", from, relayAddr)
	}

	// And the reverse direction.
	if _, err := peer.WriteTo([]byte("pong"), relayAddr); err != nil {
		t.Fatalf("peer write: %v", err)
	}
	n, _, err = relayConn.ReadFrom(buf)
	if err != nil {
		t.Fatalf("relay read: %v", err)
	}
	if string(buf[:n]) != "pong" {
		t.Fatalf("relay received %q", buf[:n])
	}
}

func TestEmbeddedTurnRejectsBadCredentials(t *testing.T) {
	_, opts := startTestTurn(t, 0)

	conn, err := net.ListenPacket("udp4", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("client socket: %v", err)
	}
	defer conn.Close() //nolint:errcheck

	addr := net.JoinHostPort("127.0.0.1", strconv.Itoa(opts.Port))
	client, err := turn.NewClient(&turn.ClientConfig{
		STUNServerAddr: addr,
		TURNServerAddr: addr,
		Conn:           conn,
		Username:       strconv.FormatInt(time.Now().Add(time.Hour).Unix(), 10) + ":intruder",
		Password:       "wrong-password",
		Realm:          opts.Realm,
		LoggerFactory:  turnLoggerFactory(logging.LogLevelDisabled),
	})
	if err != nil {
		t.Fatalf("turn.NewClient: %v", err)
	}
	defer client.Close()
	if err := client.Listen(); err != nil {
		t.Fatalf("client.Listen: %v", err)
	}

	if _, err := client.Allocate(); err == nil {
		t.Fatal("allocation with a wrong password should fail")
	}
}

func TestEmbeddedTurnQuotaRejectsSecondAllocation(t *testing.T) {
	_, opts := startTestTurn(t, 1)

	first := newTestTurnClient(t, opts)
	relayConn, err := first.Allocate()
	if err != nil {
		t.Fatalf("first Allocate: %v", err)
	}
	defer relayConn.Close() //nolint:errcheck

	second := newTestTurnClient(t, opts)
	if _, err := second.Allocate(); err == nil {
		t.Fatal("the second allocation from the same IP should hit the quota")
	}
}

func TestEmbeddedTurnWritesAndClearsState(t *testing.T) {
	instance, opts := startTestTurn(t, 0)

	st, err := readTurnState()
	if err != nil {
		t.Fatalf("readTurnState: %v", err)
	}
	if st.PID != os.Getpid() {
		t.Fatalf("state pid = %d, want %d", st.PID, os.Getpid())
	}
	if st.Port != opts.Port || st.Realm != opts.Realm {
		t.Fatalf("state mismatch: %+v", st)
	}
	if st.STUNURL != instance.stunURL {
		t.Fatalf("state stun url = %q, want %q", st.STUNURL, instance.stunURL)
	}
	if len(st.URLs) != len(instance.urls) {
		t.Fatalf("state urls = %v, want %v", st.URLs, instance.urls)
	}

	if err := instance.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := readTurnState(); err == nil {
		t.Fatal("the state file should be removed on shutdown")
	}
}

func TestStartEmbeddedTurnValidatesOptions(t *testing.T) {
	isolateTurnConfig(t)
	ctx := context.Background()

	if _, err := startEmbeddedTurn(ctx, &turnOptions{
		PublicIP: net.IPv4(127, 0, 0, 1), Secret: "s", MinPort: 100, MaxPort: 50,
	}); err == nil {
		t.Fatal("an inverted port range should be rejected")
	}
	if _, err := startEmbeddedTurn(ctx, &turnOptions{
		PublicIP: net.IPv4(127, 0, 0, 1), MinPort: 100, MaxPort: 200,
	}); err == nil {
		t.Fatal("an empty secret should be rejected")
	}
}

func TestTurnStatusAndEnvCommands(t *testing.T) {
	isolateTurnConfig(t)

	if code := turnStatus(); code != 0 {
		t.Fatalf("turnStatus = %d", code)
	}
	if code := turnEnvCommand(nil); code == 0 {
		t.Fatal("turnEnvCommand should fail without a configured server")
	}

	instance, _ := startTestTurn(t, 0)
	_ = instance

	if code := turnStatus(); code != 0 {
		t.Fatalf("turnStatus (running) = %d", code)
	}
	if code := turnEnvCommand([]string{"--json"}); code != 0 {
		t.Fatalf("turnEnvCommand --json = %d", code)
	}
	if code := turnEnvCommand([]string{"--bogus"}); code != 2 {
		t.Fatalf("turnEnvCommand with a bad flag = %d", code)
	}
}

func TestTurnSecretCommand(t *testing.T) {
	isolateTurnConfig(t)

	if code := turnSecretCommand(nil); code != 0 {
		t.Fatalf("turnSecretCommand = %d", code)
	}
	before, err := readTurnSecret()
	if err != nil {
		t.Fatalf("readTurnSecret: %v", err)
	}
	if code := turnSecretCommand([]string{"--new"}); code != 0 {
		t.Fatalf("turnSecretCommand --new = %d", code)
	}
	after, err := readTurnSecret()
	if err != nil {
		t.Fatalf("readTurnSecret: %v", err)
	}
	if before == after {
		t.Fatal("--new did not rotate the secret")
	}
	if code := turnSecretCommand([]string{"--bogus"}); code != 2 {
		t.Fatalf("turnSecretCommand with a bad flag = %d", code)
	}
}

func TestTurnCredentialsCommand(t *testing.T) {
	isolateTurnConfig(t)

	out := captureStdout(t, func() {
		if code := turnCredentialsCommand([]string{"--ttl", "10m", "--user", "phone"}); code != 0 {
			t.Fatalf("turnCredentialsCommand = %d", code)
		}
	})
	if !strings.Contains(out, "username: ") || !strings.Contains(out, "password: ") {
		t.Fatalf("unexpected plain output: %q", out)
	}
	if !strings.Contains(out, ":phone") {
		t.Fatalf("the user id is missing from the username: %q", out)
	}

	// A bare number of seconds is accepted too, matching RD_TURN_TTL.
	if code := turnCredentialsCommand([]string{"--ttl", "600"}); code != 0 {
		t.Fatalf("turnCredentialsCommand with seconds = %d", code)
	}
	if code := turnCredentialsCommand([]string{"--ttl", "nonsense"}); code != 2 {
		t.Fatalf("turnCredentialsCommand with a bad duration = %d", code)
	}

	jsonOut := captureStdout(t, func() {
		if code := turnCredentialsCommand([]string{"--ttl", "600", "--json"}); code != 0 {
			t.Fatalf("turnCredentialsCommand --json = %d", code)
		}
	})
	var payload struct {
		Username string `json:"username"`
		Password string `json:"password"`
		TTL      int    `json:"ttl"`
	}
	if err := json.Unmarshal([]byte(jsonOut), &payload); err != nil {
		t.Fatalf("decoding the JSON output %q: %v", jsonOut, err)
	}
	if payload.TTL != 600 {
		t.Fatalf("ttl = %d, want 600", payload.TTL)
	}
	secret, err := readTurnSecret()
	if err != nil {
		t.Fatalf("readTurnSecret: %v", err)
	}
	wantUser, wantPass, err := turn.GenerateLongTermTURNRESTCredentials(secret, "g4f", 600*time.Second)
	if err != nil {
		t.Fatalf("GenerateLongTermTURNRESTCredentials: %v", err)
	}
	if payload.Username != wantUser || payload.Password != wantPass {
		t.Fatalf("JSON credentials = %q/%q, want %q/%q",
			payload.Username, payload.Password, wantUser, wantPass)
	}
}

func TestTurnServeFlagsReadEnvironment(t *testing.T) {
	isolateTurnConfig(t)
	t.Setenv("G4F_TURN_PUBLIC_IP", "203.0.113.7")
	t.Setenv("G4F_TURN_BIND", "127.0.0.1")
	t.Setenv("G4F_TURN_PORT", "3479")
	t.Setenv("G4F_TURN_TLS_PORT", "5350")
	t.Setenv("G4F_TURN_REALM", "env.realm")
	t.Setenv("G4F_TURN_SECRET", "env-secret")
	t.Setenv("G4F_TURN_MIN_PORT", "50000")
	t.Setenv("G4F_TURN_MAX_PORT", "50010")
	t.Setenv("G4F_TURN_QUOTA", "7")
	t.Setenv("G4F_TURN_CERT", "/tmp/env.crt")
	t.Setenv("G4F_TURN_KEY", "/tmp/env.key")

	fs, f := newTurnServeFlags()
	if err := fs.Parse(nil); err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if *f.publicIP != "203.0.113.7" || *f.bindIP != "127.0.0.1" {
		t.Fatalf("address env not applied: %q %q", *f.publicIP, *f.bindIP)
	}
	if *f.port != 3479 || *f.tlsPort != 5350 {
		t.Fatalf("port env not applied: %d %d", *f.port, *f.tlsPort)
	}
	if *f.realm != "env.realm" || *f.secret != "env-secret" {
		t.Fatalf("realm/secret env not applied: %q %q", *f.realm, *f.secret)
	}
	if *f.minPort != 50000 || *f.maxPort != 50010 || *f.quota != 7 {
		t.Fatalf("relay env not applied: %d %d %d", *f.minPort, *f.maxPort, *f.quota)
	}
	if *f.certFile != "/tmp/env.crt" || *f.keyFile != "/tmp/env.key" {
		t.Fatalf("TLS env not applied: %q %q", *f.certFile, *f.keyFile)
	}

	// Flags must win over the environment.
	fs, f = newTurnServeFlags()
	if err := fs.Parse([]string{"--port", "9999", "--realm", "flag.realm"}); err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if *f.port != 9999 || *f.realm != "flag.realm" {
		t.Fatalf("flags did not override the environment: %d %q", *f.port, *f.realm)
	}
}

func TestTurnCredentialsTTLFromEnvironment(t *testing.T) {
	isolateTurnConfig(t)
	t.Setenv("G4F_TURN_TTL", "600")

	ttl := turnEnvDuration("G4F_TURN_TTL", turnDefaultTTL)
	if ttl != 10*time.Minute {
		t.Fatalf("G4F_TURN_TTL = %v, want 10m", ttl)
	}
}

func TestParseTurnDuration(t *testing.T) {
	cases := []struct {
		in   string
		want time.Duration
		ok   bool
	}{
		{"1h", time.Hour, true},
		{"30m", 30 * time.Minute, true},
		{"600s", 10 * time.Minute, true},
		{"600", 10 * time.Minute, true},
		{" 90 ", 90 * time.Second, true},
		{"", 0, false},
		{"nonsense", 0, false},
	}
	for _, tc := range cases {
		got, err := parseTurnDuration(tc.in)
		if tc.ok && (err != nil || got != tc.want) {
			t.Fatalf("parseTurnDuration(%q) = %v, %v; want %v", tc.in, got, err, tc.want)
		}
		if !tc.ok && err == nil {
			t.Fatalf("parseTurnDuration(%q) should fail", tc.in)
		}
	}
}

func TestRunTurnCommandDispatch(t *testing.T) {
	isolateTurnConfig(t)
	ctx := context.Background()

	if code := runTurnCommand(ctx, []string{"help"}); code != 0 {
		t.Fatalf("turn help = %d", code)
	}
	if code := runTurnCommand(ctx, []string{"path"}); code != 0 {
		t.Fatalf("turn path = %d", code)
	}
	if code := runTurnCommand(ctx, []string{"status"}); code != 0 {
		t.Fatalf("turn status = %d", code)
	}
	if code := runTurnCommand(ctx, []string{"nonsense"}); code != 2 {
		t.Fatalf("unknown turn subcommand = %d", code)
	}
}

func TestHasSubcommandRecognizesTurn(t *testing.T) {
	if !hasSubcommand([]string{"turn", "status"}) {
		t.Fatal("`turn` should be recognized as a subcommand")
	}
}
