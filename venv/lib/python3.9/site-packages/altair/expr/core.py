from ..utils import SchemaBase


class DatumType(object):
    """An object to assist in building Vega-Lite Expressions"""

    def __repr__(self):
        return "datum"

    def __getattr__(self, attr):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return GetAttrExpression("datum", attr)

    def __getitem__(self, attr):
        return GetItemExpression("datum", attr)

    def __call__(self, datum, **kwargs):
        """Specify a datum for use in an encoding"""
        return dict(datum=datum, **kwargs)


datum = DatumType()


def _js_repr(val):
    """Return a javascript-safe string representation of val"""
    if val is True:
        return "true"
    elif val is False:
        return "false"
    elif val is None:
        return "null"
    else:
        return repr(val)


class Expression(SchemaBase):
    """Expression

    Base object for enabling build-up of Javascript expressions using
    a Python syntax. Calling ``repr(obj)`` will return a Javascript
    representation of the object and the operations it encodes.
    """

    _schema = {"type": "string"}

    def to_dict(self, *args, **kwargs):
        return repr(self)

    def __setattr__(self, attr, val):
        # We don't need the setattr magic defined in SchemaBase
        return object.__setattr__(self, attr, val)

    def __add__(self, other):
        return BinaryExpression("+", self, other)

    def __radd__(self, other):
        return BinaryExpression("+", other, self)

    def __sub__(self, other):
        return BinaryExpression("-", self, other)

    def __rsub__(self, other):
        return BinaryExpression("-", other, self)

    def __mul__(self, other):
        return BinaryExpression("*", self, other)

    def __rmul__(self, other):
        return BinaryExpression("*", other, self)

    def __truediv__(self, other):
        return BinaryExpression("/", self, other)

    def __rtruediv__(self, other):
        return BinaryExpression("/", other, self)

    __div__ = __truediv__

    __rdiv__ = __rtruediv__

    def __mod__(self, other):
        return BinaryExpression("%", self, other)

    def __rmod__(self, other):
        return BinaryExpression("%", other, self)

    def __pow__(self, other):
        # "**" Javascript operator is not supported in all browsers
        return FunctionExpression("pow", (self, other))

    def __rpow__(self, other):
        # "**" Javascript operator is not supported in all browsers
        return FunctionExpression("pow", (other, self))

    def __neg__(self):
        return UnaryExpression("-", self)

    def __pos__(self):
        return UnaryExpression("+", self)

    # comparison operators

    def __eq__(self, other):
        return BinaryExpression("===", self, other)

    def __ne__(self, other):
        return BinaryExpression("!==", self, other)

    def __gt__(self, other):
        return BinaryExpression(">", self, other)

    def __lt__(self, other):
        return BinaryExpression("<", self, other)

    def __ge__(self, other):
        return BinaryExpression(">=", self, other)

    def __le__(self, other):
        return BinaryExpression("<=", self, other)

    def __abs__(self):
        return FunctionExpression("abs", (self,))

    # logical operators

    def __and__(self, other):
        return BinaryExpression("&&", self, other)

    def __rand__(self, other):
        return BinaryExpression("&&", other, self)

    def __or__(self, other):
        return BinaryExpression("||", self, other)

    def __ror__(self, other):
        return BinaryExpression("||", other, self)

    def __invert__(self):
        return UnaryExpression("!", self)

    # item access
    def __getitem__(self, val):
        return GetItemExpression(self, val)


class UnaryExpression(Expression):
    def __init__(self, op, val):
        super(UnaryExpression, self).__init__(op=op, val=val)

    def __repr__(self):
        return "({op}{val})".format(op=self.op, val=_js_repr(self.val))


class BinaryExpression(Expression):
    def __init__(self, op, lhs, rhs):
        super(BinaryExpression, self).__init__(op=op, lhs=lhs, rhs=rhs)

    def __repr__(self):
        return "({lhs} {op} {rhs})".format(
            op=self.op, lhs=_js_repr(self.lhs), rhs=_js_repr(self.rhs)
        )


class FunctionExpression(Expression):
    def __init__(self, name, args):
        super(FunctionExpression, self).__init__(name=name, args=args)

    def __repr__(self):
        args = ",".join(_js_repr(arg) for arg in self.args)
        return "{name}({args})".format(name=self.name, args=args)


class ConstExpression(Expression):
    def __init__(self, name, doc):
        self.__doc__ = """{}: {}""".format(name, doc)
        super(ConstExpression, self).__init__(name=name, doc=doc)

    def __repr__(self):
        return str(self.name)


class GetAttrExpression(Expression):
    def __init__(self, group, name):
        super(GetAttrExpression, self).__init__(group=group, name=name)

    def __repr__(self):
        return "{}.{}".format(self.group, self.name)


class GetItemExpression(Expression):
    def __init__(self, group, name):
        super(GetItemExpression, self).__init__(group=group, name=name)

    def __repr__(self):
        return "{}[{!r}]".format(self.group, self.name)
