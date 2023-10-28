// MurmurHash3 related functions
// https://github.com/markogresak/fingerprintjs2/blob/master/src/x64hash128.js

// Given two 64bit ints (as an array of two 32bit ints) returns the two
// added together as a 64bit int (as an array of two 32bit ints).
var x64Add = function (t, r) {
        (t = [t[0] >>> 16, 65535 & t[0], t[1] >>> 16, 65535 & t[1]]),
            (r = [r[0] >>> 16, 65535 & r[0], r[1] >>> 16, 65535 & r[1]]);
        var e = [0, 0, 0, 0];
        return (
            (e[3] += t[3] + r[3]),
            (e[2] += e[3] >>> 16),
            (e[3] &= 65535),
            (e[2] += t[2] + r[2]),
            (e[1] += e[2] >>> 16),
            (e[2] &= 65535),
            (e[1] += t[1] + r[1]),
            (e[0] += e[1] >>> 16),
            (e[1] &= 65535),
            (e[0] += t[0] + r[0]),
            (e[0] &= 65535),
            [(e[0] << 16) | e[1], (e[2] << 16) | e[3]]
        );
    },
    // Given two 64bit ints (as an array of two 32bit ints) returns the two
    // multiplied together as a 64bit int (as an array of two 32bit ints).
    x64Multiply = function (t, r) {
        (t = [t[0] >>> 16, 65535 & t[0], t[1] >>> 16, 65535 & t[1]]),
            (r = [r[0] >>> 16, 65535 & r[0], r[1] >>> 16, 65535 & r[1]]);
        var e = [0, 0, 0, 0];
        return (
            (e[3] += t[3] * r[3]),
            (e[2] += e[3] >>> 16),
            (e[3] &= 65535),
            (e[2] += t[2] * r[3]),
            (e[1] += e[2] >>> 16),
            (e[2] &= 65535),
            (e[2] += t[3] * r[2]),
            (e[1] += e[2] >>> 16),
            (e[2] &= 65535),
            (e[1] += t[1] * r[3]),
            (e[0] += e[1] >>> 16),
            (e[1] &= 65535),
            (e[1] += t[2] * r[2]),
            (e[0] += e[1] >>> 16),
            (e[1] &= 65535),
            (e[1] += t[3] * r[1]),
            (e[0] += e[1] >>> 16),
            (e[1] &= 65535),
            (e[0] += t[0] * r[3] + t[1] * r[2] + t[2] * r[1] + t[3] * r[0]),
            (e[0] &= 65535),
            [(e[0] << 16) | e[1], (e[2] << 16) | e[3]]
        );
    },
    // Given a 64bit int (as an array of two 32bit ints) and an int
    // representing a number of bit positions, returns the 64bit int (as an
    // array of two 32bit ints) rotated left by that number of positions.
    x64Rotl = function (t, r) {
        return 32 === (r %= 64)
            ? [t[1], t[0]]
            : r < 32
            ? [
                  (t[0] << r) | (t[1] >>> (32 - r)),
                  (t[1] << r) | (t[0] >>> (32 - r)),
              ]
            : ((r -= 32),
              [
                  (t[1] << r) | (t[0] >>> (32 - r)),
                  (t[0] << r) | (t[1] >>> (32 - r)),
              ]);
    },
    // Given a 64bit int (as an array of two 32bit ints) and an int
    // representing a number of bit positions, returns the 64bit int (as an
    // array of two 32bit ints) shifted left by that number of positions.
    x64LeftShift = function (t, r) {
        return 0 === (r %= 64)
            ? t
            : r < 32
            ? [(t[0] << r) | (t[1] >>> (32 - r)), t[1] << r]
            : [t[1] << (r - 32), 0];
    },
    // Given two 64bit ints (as an array of two 32bit ints) returns the two
    // xored together as a 64bit int (as an array of two 32bit ints).
    x64Xor = function (t, r) {
        return [t[0] ^ r[0], t[1] ^ r[1]];
    },
    // Given a block, returns murmurHash3's final x64 mix of that block.
    // (`[0, h[0] >>> 1]` is a 33 bit unsigned right shift. This is the
    // only place where we need to right shift 64bit ints.)
    x64Fmix = function (t) {
        return (
            (t = x64Xor(t, [0, t[0] >>> 1])),
            (t = x64Multiply(t, [4283543511, 3981806797])),
            (t = x64Xor(t, [0, t[0] >>> 1])),
            (t = x64Multiply(t, [3301882366, 444984403])),
            (t = x64Xor(t, [0, t[0] >>> 1]))
        );
    },
    // Given a string and an optional seed as an int, returns a 128 bit
    // hash using the x64 flavor of MurmurHash3, as an unsigned hex.
    x64hash128 = function (t, r) {
        r = r || 0;
        for (
            var e = (t = t || "").length % 16,
                o = t.length - e,
                x = [0, r],
                c = [0, r],
                h = [0, 0],
                a = [0, 0],
                d = [2277735313, 289559509],
                i = [1291169091, 658871167],
                l = 0;
            l < o;
            l += 16
        )
            (h = [
                (255 & t.charCodeAt(l + 4)) |
                    ((255 & t.charCodeAt(l + 5)) << 8) |
                    ((255 & t.charCodeAt(l + 6)) << 16) |
                    ((255 & t.charCodeAt(l + 7)) << 24),
                (255 & t.charCodeAt(l)) |
                    ((255 & t.charCodeAt(l + 1)) << 8) |
                    ((255 & t.charCodeAt(l + 2)) << 16) |
                    ((255 & t.charCodeAt(l + 3)) << 24),
            ]),
                (a = [
                    (255 & t.charCodeAt(l + 12)) |
                        ((255 & t.charCodeAt(l + 13)) << 8) |
                        ((255 & t.charCodeAt(l + 14)) << 16) |
                        ((255 & t.charCodeAt(l + 15)) << 24),
                    (255 & t.charCodeAt(l + 8)) |
                        ((255 & t.charCodeAt(l + 9)) << 8) |
                        ((255 & t.charCodeAt(l + 10)) << 16) |
                        ((255 & t.charCodeAt(l + 11)) << 24),
                ]),
                (h = x64Multiply(h, d)),
                (h = x64Rotl(h, 31)),
                (h = x64Multiply(h, i)),
                (x = x64Xor(x, h)),
                (x = x64Rotl(x, 27)),
                (x = x64Add(x, c)),
                (x = x64Add(x64Multiply(x, [0, 5]), [0, 1390208809])),
                (a = x64Multiply(a, i)),
                (a = x64Rotl(a, 33)),
                (a = x64Multiply(a, d)),
                (c = x64Xor(c, a)),
                (c = x64Rotl(c, 31)),
                (c = x64Add(c, x)),
                (c = x64Add(x64Multiply(c, [0, 5]), [0, 944331445]));
        switch (((h = [0, 0]), (a = [0, 0]), e)) {
            case 15:
                a = x64Xor(a, x64LeftShift([0, t.charCodeAt(l + 14)], 48));
            case 14:
                a = x64Xor(a, x64LeftShift([0, t.charCodeAt(l + 13)], 40));
            case 13:
                a = x64Xor(a, x64LeftShift([0, t.charCodeAt(l + 12)], 32));
            case 12:
                a = x64Xor(a, x64LeftShift([0, t.charCodeAt(l + 11)], 24));
            case 11:
                a = x64Xor(a, x64LeftShift([0, t.charCodeAt(l + 10)], 16));
            case 10:
                a = x64Xor(a, x64LeftShift([0, t.charCodeAt(l + 9)], 8));
            case 9:
                (a = x64Xor(a, [0, t.charCodeAt(l + 8)])),
                    (a = x64Multiply(a, i)),
                    (a = x64Rotl(a, 33)),
                    (a = x64Multiply(a, d)),
                    (c = x64Xor(c, a));
            case 8:
                h = x64Xor(h, x64LeftShift([0, t.charCodeAt(l + 7)], 56));
            case 7:
                h = x64Xor(h, x64LeftShift([0, t.charCodeAt(l + 6)], 48));
            case 6:
                h = x64Xor(h, x64LeftShift([0, t.charCodeAt(l + 5)], 40));
            case 5:
                h = x64Xor(h, x64LeftShift([0, t.charCodeAt(l + 4)], 32));
            case 4:
                h = x64Xor(h, x64LeftShift([0, t.charCodeAt(l + 3)], 24));
            case 3:
                h = x64Xor(h, x64LeftShift([0, t.charCodeAt(l + 2)], 16));
            case 2:
                h = x64Xor(h, x64LeftShift([0, t.charCodeAt(l + 1)], 8));
            case 1:
                (h = x64Xor(h, [0, t.charCodeAt(l)])),
                    (h = x64Multiply(h, d)),
                    (h = x64Rotl(h, 31)),
                    (h = x64Multiply(h, i)),
                    (x = x64Xor(x, h));
        }
        return (
            (x = x64Xor(x, [0, t.length])),
            (c = x64Xor(c, [0, t.length])),
            (x = x64Add(x, c)),
            (c = x64Add(c, x)),
            (x = x64Fmix(x)),
            (c = x64Fmix(c)),
            (x = x64Add(x, c)),
            (c = x64Add(c, x)),
            ("00000000" + (x[0] >>> 0).toString(16)).slice(-8) +
                ("00000000" + (x[1] >>> 0).toString(16)).slice(-8) +
                ("00000000" + (c[0] >>> 0).toString(16)).slice(-8) +
                ("00000000" + (c[1] >>> 0).toString(16)).slice(-8)
        );
    };
export default x64hash128;
