# This is a public namespace, so we don't want to expose any non-underscored
# attributes that aren't actually part of our public API. But it's very
# annoying to carefully always use underscored names for module-level
# temporaries, imports, etc. when implementing the module. So we put the
# implementation in an underscored module, and then re-export the public parts
# here.
# We still have some underscore names though but only a few.

from . import _socket
import sys
import typing as _t

# The socket module exports a bunch of platform-specific constants. We want to
# re-export them. Since the exact set of constants varies depending on Python
# version, platform, the libc installed on the system where Python was built,
# etc., we figure out which constants to re-export dynamically at runtime (see
# below). But that confuses static analysis tools like jedi and mypy. So this
# import statement statically lists every constant that *could* be
# exported. It always fails at runtime, since no single Python build exports
# all these constants, but it lets static analysis tools understand what's
# going on. There's a test in test_exports.py to make sure that the list is
# kept up to date.
try:
    # fmt: off
    from socket import (  # type: ignore
        CMSG_LEN, CMSG_SPACE, CAPI, AF_UNSPEC, AF_INET, AF_UNIX, AF_IPX,
        AF_APPLETALK, AF_INET6, AF_ROUTE, AF_LINK, AF_SNA, PF_SYSTEM,
        AF_SYSTEM, SOCK_STREAM, SOCK_DGRAM, SOCK_RAW, SOCK_SEQPACKET, SOCK_RDM,
        SO_DEBUG, SO_ACCEPTCONN, SO_REUSEADDR, SO_KEEPALIVE, SO_DONTROUTE,
        SO_BROADCAST, SO_USELOOPBACK, SO_LINGER, SO_OOBINLINE, SO_REUSEPORT,
        SO_SNDBUF, SO_RCVBUF, SO_SNDLOWAT, SO_RCVLOWAT, SO_SNDTIMEO,
        SO_RCVTIMEO, SO_ERROR, SO_TYPE, LOCAL_PEERCRED, SOMAXCONN, SCM_RIGHTS,
        SCM_CREDS, MSG_OOB, MSG_PEEK, MSG_DONTROUTE, MSG_DONTWAIT, MSG_EOR,
        MSG_TRUNC, MSG_CTRUNC, MSG_WAITALL, MSG_EOF, SOL_SOCKET, SOL_IP,
        SOL_TCP, SOL_UDP, IPPROTO_IP, IPPROTO_HOPOPTS, IPPROTO_ICMP,
        IPPROTO_IGMP, IPPROTO_GGP, IPPROTO_IPV4, IPPROTO_IPIP, IPPROTO_TCP,
        IPPROTO_EGP, IPPROTO_PUP, IPPROTO_UDP, IPPROTO_IDP, IPPROTO_HELLO,
        IPPROTO_ND, IPPROTO_TP, IPPROTO_ROUTING, IPPROTO_FRAGMENT,
        IPPROTO_RSVP, IPPROTO_GRE, IPPROTO_ESP, IPPROTO_AH, IPPROTO_ICMPV6,
        IPPROTO_NONE, IPPROTO_DSTOPTS, IPPROTO_XTP, IPPROTO_EON, IPPROTO_PIM,
        IPPROTO_IPCOMP, IPPROTO_SCTP, IPPROTO_RAW, IPPROTO_MAX, IPPROTO_MPTCP,
        SYSPROTO_CONTROL, IPPORT_RESERVED, IPPORT_USERRESERVED, INADDR_ANY,
        INADDR_BROADCAST, INADDR_LOOPBACK, INADDR_UNSPEC_GROUP,
        INADDR_ALLHOSTS_GROUP, INADDR_MAX_LOCAL_GROUP, INADDR_NONE, IP_OPTIONS,
        IP_HDRINCL, IP_TOS, IP_TTL, IP_RECVOPTS, IP_RECVRETOPTS,
        IP_RECVDSTADDR, IP_RETOPTS, IP_MULTICAST_IF, IP_MULTICAST_TTL,
        IP_MULTICAST_LOOP, IP_ADD_MEMBERSHIP, IP_DROP_MEMBERSHIP,
        IP_DEFAULT_MULTICAST_TTL, IP_DEFAULT_MULTICAST_LOOP,
        IP_MAX_MEMBERSHIPS, IPV6_JOIN_GROUP, IPV6_LEAVE_GROUP,
        IPV6_MULTICAST_HOPS, IPV6_MULTICAST_IF, IPV6_MULTICAST_LOOP,
        IPV6_UNICAST_HOPS, IPV6_V6ONLY, IPV6_CHECKSUM, IPV6_RECVTCLASS,
        IPV6_RTHDR_TYPE_0, IPV6_TCLASS, TCP_NODELAY, TCP_MAXSEG, TCP_KEEPINTVL,
        TCP_KEEPCNT, TCP_FASTOPEN, TCP_NOTSENT_LOWAT, EAI_ADDRFAMILY,
        EAI_AGAIN, EAI_BADFLAGS, EAI_FAIL, EAI_FAMILY, EAI_MEMORY, EAI_NODATA,
        EAI_NONAME, EAI_OVERFLOW, EAI_SERVICE, EAI_SOCKTYPE, EAI_SYSTEM,
        EAI_BADHINTS, EAI_PROTOCOL, EAI_MAX, AI_PASSIVE, AI_CANONNAME,
        AI_NUMERICHOST, AI_NUMERICSERV, AI_MASK, AI_ALL, AI_V4MAPPED_CFG,
        AI_ADDRCONFIG, AI_V4MAPPED, AI_DEFAULT, NI_MAXHOST, NI_MAXSERV,
        NI_NOFQDN, NI_NUMERICHOST, NI_NAMEREQD, NI_NUMERICSERV, NI_DGRAM,
        SHUT_RD, SHUT_WR, SHUT_RDWR, EBADF, EAGAIN, EWOULDBLOCK, AF_ASH,
        AF_ATMPVC, AF_ATMSVC, AF_AX25, AF_BLUETOOTH, AF_BRIDGE, AF_ECONET,
        AF_IRDA, AF_KEY, AF_LLC, AF_NETBEUI, AF_NETLINK, AF_NETROM, AF_PACKET,
        AF_PPPOX, AF_ROSE, AF_SECURITY, AF_WANPIPE, AF_X25, BDADDR_ANY,
        BDADDR_LOCAL, FD_SETSIZE, IPV6_DSTOPTS, IPV6_HOPLIMIT, IPV6_HOPOPTS,
        IPV6_NEXTHOP, IPV6_PKTINFO, IPV6_RECVDSTOPTS, IPV6_RECVHOPLIMIT,
        IPV6_RECVHOPOPTS, IPV6_RECVPKTINFO, IPV6_RECVRTHDR, IPV6_RTHDR,
        IPV6_RTHDRDSTOPTS, MSG_ERRQUEUE, NETLINK_DNRTMSG, NETLINK_FIREWALL,
        NETLINK_IP6_FW, NETLINK_NFLOG, NETLINK_ROUTE, NETLINK_USERSOCK,
        NETLINK_XFRM, PACKET_BROADCAST, PACKET_FASTROUTE, PACKET_HOST,
        PACKET_LOOPBACK, PACKET_MULTICAST, PACKET_OTHERHOST, PACKET_OUTGOING,
        POLLERR, POLLHUP, POLLIN, POLLMSG, POLLNVAL, POLLOUT, POLLPRI,
        POLLRDBAND, POLLRDNORM, POLLWRNORM, SIOCGIFINDEX, SIOCGIFNAME,
        SOCK_CLOEXEC, TCP_CORK, TCP_DEFER_ACCEPT, TCP_INFO, TCP_KEEPIDLE,
        TCP_LINGER2, TCP_QUICKACK, TCP_SYNCNT, TCP_WINDOW_CLAMP, AF_ALG,
        AF_CAN, AF_RDS, AF_TIPC, AF_VSOCK, ALG_OP_DECRYPT, ALG_OP_ENCRYPT,
        ALG_OP_SIGN, ALG_OP_VERIFY, ALG_SET_AEAD_ASSOCLEN,
        ALG_SET_AEAD_AUTHSIZE, ALG_SET_IV, ALG_SET_KEY, ALG_SET_OP,
        ALG_SET_PUBKEY, CAN_BCM, CAN_BCM_RX_CHANGED, CAN_BCM_RX_DELETE,
        CAN_BCM_RX_READ, CAN_BCM_RX_SETUP, CAN_BCM_RX_STATUS,
        CAN_BCM_RX_TIMEOUT, CAN_BCM_TX_DELETE, CAN_BCM_TX_EXPIRED,
        CAN_BCM_TX_READ, CAN_BCM_TX_SEND, CAN_BCM_TX_SETUP, CAN_BCM_TX_STATUS,
        CAN_EFF_FLAG, CAN_EFF_MASK, CAN_ERR_FLAG, CAN_ERR_MASK, CAN_ISOTP,
        CAN_RAW, CAN_RAW_ERR_FILTER, CAN_RAW_FD_FRAMES, CAN_RAW_FILTER,
        CAN_RAW_LOOPBACK, CAN_RAW_RECV_OWN_MSGS, CAN_RTR_FLAG, CAN_SFF_MASK,
        IOCTL_VM_SOCKETS_GET_LOCAL_CID, IPV6_DONTFRAG, IPV6_PATHMTU,
        IPV6_RECVPATHMTU, IP_TRANSPARENT, MSG_CMSG_CLOEXEC, MSG_CONFIRM,
        MSG_FASTOPEN, MSG_MORE, MSG_NOSIGNAL, NETLINK_CRYPTO, PF_CAN,
        PF_PACKET, PF_RDS, SCM_CREDENTIALS, SOCK_NONBLOCK, SOL_ALG,
        SOL_CAN_BASE, SOL_CAN_RAW, SOL_TIPC, SO_BINDTODEVICE, SO_DOMAIN,
        SO_MARK, SO_PASSCRED, SO_PASSSEC, SO_PEERCRED, SO_PEERSEC, SO_PRIORITY,
        SO_PROTOCOL, SO_VM_SOCKETS_BUFFER_MAX_SIZE,
        SO_VM_SOCKETS_BUFFER_MIN_SIZE, SO_VM_SOCKETS_BUFFER_SIZE,
        TCP_CONGESTION, TCP_USER_TIMEOUT, TIPC_ADDR_ID, TIPC_ADDR_NAME,
        TIPC_ADDR_NAMESEQ, TIPC_CFG_SRV, TIPC_CLUSTER_SCOPE, TIPC_CONN_TIMEOUT,
        TIPC_CRITICAL_IMPORTANCE, TIPC_DEST_DROPPABLE, TIPC_HIGH_IMPORTANCE,
        TIPC_IMPORTANCE, TIPC_LOW_IMPORTANCE, TIPC_MEDIUM_IMPORTANCE,
        TIPC_NODE_SCOPE, TIPC_PUBLISHED, TIPC_SRC_DROPPABLE,
        TIPC_SUBSCR_TIMEOUT, TIPC_SUB_CANCEL, TIPC_SUB_PORTS, TIPC_SUB_SERVICE,
        TIPC_TOP_SRV, TIPC_WAIT_FOREVER, TIPC_WITHDRAWN, TIPC_ZONE_SCOPE,
        VMADDR_CID_ANY, VMADDR_CID_HOST, VMADDR_PORT_ANY,
        VM_SOCKETS_INVALID_VERSION, MSG_BCAST, MSG_MCAST, RCVALL_MAX,
        RCVALL_OFF, RCVALL_ON, RCVALL_SOCKETLEVELONLY, SIO_KEEPALIVE_VALS,
        SIO_LOOPBACK_FAST_PATH, SIO_RCVALL, SO_EXCLUSIVEADDRUSE, HCI_FILTER,
        BTPROTO_SCO, BTPROTO_HCI, HCI_TIME_STAMP, SOL_RDS, BTPROTO_L2CAP,
        BTPROTO_RFCOMM, HCI_DATA_DIR, SOL_HCI, CAN_BCM_RX_ANNOUNCE_RESUME,
        CAN_BCM_RX_CHECK_DLC, CAN_BCM_RX_FILTER_ID, CAN_BCM_RX_NO_AUTOTIMER,
        CAN_BCM_RX_RTR_FRAME, CAN_BCM_SETTIMER, CAN_BCM_STARTTIMER,
        CAN_BCM_TX_ANNOUNCE, CAN_BCM_TX_COUNTEVT, CAN_BCM_TX_CP_CAN_ID,
        CAN_BCM_TX_RESET_MULTI_IDX, IPPROTO_CBT, IPPROTO_ICLFXBM, IPPROTO_IGP,
        IPPROTO_L2TP, IPPROTO_PGM, IPPROTO_RDP, IPPROTO_ST, AF_QIPCRTR,
        CAN_BCM_CAN_FD_FRAME, IPPROTO_MOBILE, IPV6_USE_MIN_MTU,
        MSG_NOTIFICATION, SO_SETFIB, CAN_J1939, CAN_RAW_JOIN_FILTERS,
        IPPROTO_UDPLITE, J1939_EE_INFO_NONE, J1939_EE_INFO_TX_ABORT,
        J1939_FILTER_MAX, J1939_IDLE_ADDR, J1939_MAX_UNICAST_ADDR,
        J1939_NLA_BYTES_ACKED, J1939_NLA_PAD, J1939_NO_ADDR, J1939_NO_NAME,
        J1939_NO_PGN, J1939_PGN_ADDRESS_CLAIMED, J1939_PGN_ADDRESS_COMMANDED,
        J1939_PGN_MAX, J1939_PGN_PDU1_MAX, J1939_PGN_REQUEST,
        SCM_J1939_DEST_ADDR, SCM_J1939_DEST_NAME, SCM_J1939_ERRQUEUE,
        SCM_J1939_PRIO, SO_J1939_ERRQUEUE, SO_J1939_FILTER, SO_J1939_PROMISC,
        SO_J1939_SEND_PRIO, UDPLITE_RECV_CSCOV, UDPLITE_SEND_CSCOV, IP_RECVTOS,
        TCP_KEEPALIVE, SO_INCOMING_CPU
    )
    # fmt: on
except ImportError:
    pass

# Dynamically re-export whatever constants this particular Python happens to
# have:
import socket as _stdlib_socket

_bad_symbols: _t.Set[str] = set()
if sys.platform == "win32":
    # See https://github.com/python-trio/trio/issues/39
    # Do not import for windows platform
    # (you can still get it from stdlib socket, of course, if you want it)
    _bad_symbols.add("SO_REUSEADDR")

globals().update(
    {
        _name: getattr(_stdlib_socket, _name)
        for _name in _stdlib_socket.__all__  # type: ignore
        if _name.isupper() and _name not in _bad_symbols
    }
)

# import the overwrites
from ._socket import (
    fromfd,
    from_stdlib_socket,
    getprotobyname,
    socketpair,
    getnameinfo,
    socket,
    getaddrinfo,
    set_custom_hostname_resolver,
    set_custom_socket_factory,
    SocketType,
)

# not always available so expose only if
if sys.platform == "win32" or not _t.TYPE_CHECKING:
    try:
        from ._socket import fromshare
    except ImportError:
        pass

# expose these functions to trio.socket
from socket import (
    gaierror,
    herror,
    gethostname,
    ntohs,
    htonl,
    htons,
    inet_aton,
    inet_ntoa,
    inet_pton,
    inet_ntop,
)

# not always available so expose only if
if sys.platform != "win32" or not _t.TYPE_CHECKING:
    try:
        from socket import sethostname, if_nameindex, if_nametoindex, if_indextoname
    except ImportError:
        pass

# get names used by Trio that we define on our own
from ._socket import IPPROTO_IPV6

if _t.TYPE_CHECKING:
    IP_BIND_ADDRESS_NO_PORT: int
else:
    try:
        IP_BIND_ADDRESS_NO_PORT
    except NameError:
        if sys.platform == "linux":
            IP_BIND_ADDRESS_NO_PORT = 24

del sys
