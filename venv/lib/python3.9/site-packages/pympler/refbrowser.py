"""Tree-like exploration of object referrers.

This module provides a base implementation for tree-like referrers browsing.
The two non-interactive classes ConsoleBrowser and FileBrowser output a tree
to the console or a file. One graphical user interface for referrers browsing
is provided as well. Further types can be subclassed.

All types share a similar initialisation. That is, you provide a root object
and may specify further settings such as the initial depth of the tree or an
output function.
Afterwards you can print the tree which will be arranged based on your previous
settings.

The interactive browser is based on a TreeWidget implemented in IDLE. It is
available only if you have Tcl/Tk installed. If you try to instantiate the
interactive browser without having Tkinter installed, an ImportError will be
raised.

"""
import gc
import inspect
import sys

from pympler import muppy
from pympler import summary

from pympler.util.compat import tkinter


class _Node(object):
    """A node as it is used in the tree structure.

    Each node contains the object it represents and a list of children.
    Children can be other nodes or arbitrary other objects. Any object
    in a tree which is not of the type _Node is considered a leaf.

    """
    def __init__(self, o, str_func=None):
        """You have to define the object this node represents. Also you can
        define an output function which will be used to represent this node.
        If no function is defined, the default str representation is used.

        keyword arguments
        str_func -- output function

        """
        self.o = o
        self.children = []
        self.str_func = str_func

    def __str__(self):
        """Override str(self.o) if str_func is defined."""
        if self.str_func is not None:
            return self.str_func(self.o)
        else:
            return str(self.o)


class RefBrowser(object):
    """Base class to other RefBrowser implementations.

    This base class provides means to extract a tree from a given root object
    and holds information on already known objects (to avoid repetition
    if requested).

    """

    def __init__(self, rootobject, maxdepth=3, str_func=summary._repr,
                 repeat=True, stream=None):
        """You have to provide the root object used in the refbrowser.

        keyword arguments
        maxdepth -- maximum depth of the initial tree
        str_func -- function used when calling str(node)
        repeat -- should nodes appear repeatedly in the tree, or should be
                  referred to existing nodes
        stream -- output stream (used in derived classes)

        """
        self.root = rootobject
        self.maxdepth = maxdepth
        self.str_func = str_func
        self.repeat = repeat
        self.stream = stream
        # objects which should be ignored while building the tree
        # e.g. the current frame
        self.ignore = []
        # set of object ids which are already included
        self.already_included = set()
        self.ignore.append(self.already_included)

    def get_tree(self):
        """Get a tree of referrers of the root object."""
        self.ignore.append(inspect.currentframe())
        return self._get_tree(self.root, self.maxdepth)

    def _get_tree(self, root, maxdepth):
        """Workhorse of the get_tree implementation.

        This is a recursive method which is why we have a wrapper method.
        root is the current root object of the tree which should be returned.
        Note that root is not of the type _Node.
        maxdepth defines how much further down the from the root the tree
        should be build.

        """
        objects = gc.get_referrers(root)
        res = _Node(root, self.str_func)
        self.already_included.add(id(root))
        if maxdepth == 0:
            return res
        self.ignore.append(inspect.currentframe())
        self.ignore.append(objects)
        for o in objects:
            # Ignore dict of _Node and RefBrowser objects
            if isinstance(o, dict):
                if any(isinstance(ref, (_Node, RefBrowser))
                       for ref in gc.get_referrers(o)):
                    continue
            _id = id(o)
            if not self.repeat and (_id in self.already_included):
                s = self.str_func(o)
                res.children.append("%s (already included, id %s)" %
                                    (s, _id))
                continue
            if (not isinstance(o, _Node)) and (o not in self.ignore):
                res.children.append(self._get_tree(o, maxdepth - 1))
        return res


class StreamBrowser(RefBrowser):
    """RefBrowser implementation which prints the tree to the console.

    If you don't like the looks, you can change it a little bit.
    The class attributes 'hline', 'vline', 'cross', and 'space' can be
    modified to your needs.

    """
    hline = '-'
    vline = '|'
    cross = '+'
    space = ' '

    def print_tree(self, tree=None):
        """ Print referrers tree to console.

        keyword arguments
        tree -- if not None, the passed tree will be printed. Otherwise it is
        based on the rootobject.

        """
        if tree is None:
            tree = self.get_tree()
        self._print(tree, '', '')

    def _print(self, tree, prefix, carryon):
        """Compute and print a new line of the tree.

        This is a recursive function.

        arguments
        tree -- tree to print
        prefix -- prefix to the current line to print
        carryon -- prefix which is used to carry on the vertical lines

        """
        level = prefix.count(self.cross) + prefix.count(self.vline)
        len_children = 0
        if isinstance(tree, _Node):
            len_children = len(tree.children)

        # add vertex
        prefix += str(tree)
        # and as many spaces as the vertex is long
        carryon += self.space * len(str(tree))
        if (level == self.maxdepth) or (not isinstance(tree, _Node)) or\
           (len_children == 0):
            self.stream.write(prefix + '\n')
            return
        else:
            # add in between connections
            prefix += self.hline
            carryon += self.space
            # if there is more than one branch, add a cross
            if len(tree.children) > 1:
                prefix += self.cross
                carryon += self.vline
            prefix += self.hline
            carryon += self.space

            if len_children > 0:
                # print the first branch (on the same line)
                self._print(tree.children[0], prefix, carryon)
                for b in range(1, len_children):
                    # the carryon becomes the prefix for all following children
                    prefix = carryon[:-2] + self.cross + self.hline
                    # remove the vlines for any children of last branch
                    if b == (len_children - 1):
                        carryon = carryon[:-2] + 2 * self.space
                    self._print(tree.children[b], prefix, carryon)
                    # leave a free line before the next branch
                    if b == (len_children - 1):
                        if len(carryon.strip(' ')) == 0:
                            return
                        self.stream.write(carryon[:-2].rstrip() + '\n')


class ConsoleBrowser(StreamBrowser):
    """RefBrowser that prints to the console (stdout)."""

    def __init__(self, *args, **kwargs):
        super(ConsoleBrowser, self).__init__(*args, **kwargs)
        if not self.stream:
            self.stream = sys.stdout


class FileBrowser(StreamBrowser):
    """RefBrowser implementation which prints the tree to a file."""

    def print_tree(self, filename, tree=None):
        """ Print referrers tree to file (in text format).

        keyword arguments
        tree -- if not None, the passed tree will be printed.

        """
        old_stream = self.stream
        self.stream = open(filename, 'w')
        try:
            super(FileBrowser, self).print_tree(tree=tree)
        finally:
            self.stream.close()
            self.stream = old_stream


# Code for interactive browser (GUI)
# ==================================

# The interactive browser requires Tkinter which is not always available. To
# avoid an import error when loading the module, we encapsulate most of the
# code in the following try-except-block. The InteractiveBrowser itself
# remains outside this block. If you try to instantiate it without having
# Tkinter installed, the import error will be raised.
try:
    if sys.version_info < (3, 5, 2):
        from idlelib import TreeWidget as _TreeWidget
    else:
        from idlelib import tree as _TreeWidget

    class _TreeNode(_TreeWidget.TreeNode):
        """TreeNode used by the InteractiveBrowser.

        Not to be confused with _Node. This one is used in the GUI
        context.

        """
        def reload_referrers(self):
            """Reload all referrers for this _TreeNode."""
            self.item.node = self.item.reftree._get_tree(self.item.node.o, 1)
            self.item._clear_children()
            self.expand()
            self.update()

        def print_object(self):
            """Print object which this _TreeNode represents to console."""
            print(self.item.node.o)

        def drawtext(self):
            """Override drawtext from _TreeWidget.TreeNode.

            This seems to be a good place to add the popup menu.

            """
            _TreeWidget.TreeNode.drawtext(self)
            # create a menu
            menu = tkinter.Menu(self.canvas, tearoff=0)
            menu.add_command(label="reload referrers",
                             command=self.reload_referrers)
            menu.add_command(label="print", command=self.print_object)
            menu.add_separator()
            menu.add_command(label="expand", command=self.expand)
            menu.add_separator()
            # the popup only disappears when to click on it
            menu.add_command(label="Close Popup Menu")

            def do_popup(event):
                menu.post(event.x_root, event.y_root)

            self.label.bind("<Button-3>", do_popup)
            # override, i.e. disable the editing of items

            # disable editing of TreeNodes
            def edit(self, event=None):
                pass  # see comment above

            def edit_finish(self, event=None):
                pass  # see comment above

            def edit_cancel(self, event=None):
                pass  # see comment above

    class _ReferrerTreeItem(_TreeWidget.TreeItem, tkinter.Label):
        """Tree item wrapper around _Node object."""

        def __init__(self, parentwindow, node, reftree):  # constr calls
            """You need to provide the parent window, the node this TreeItem
            represents, as well as the tree (_Node) which the node
            belongs to.

            """
            _TreeWidget.TreeItem.__init__(self)
            tkinter.Label.__init__(self, parentwindow)
            self.node = node
            self.parentwindow = parentwindow
            self.reftree = reftree

        def _clear_children(self):
            """Clear children list from any TreeNode instances.

            Normally these objects are not required for memory profiling, as
            they are part of the profiler.

            """
            new_children = []
            for child in self.node.children:
                if not isinstance(child, _TreeNode):
                    new_children.append(child)
            self.node.children = new_children

        def GetText(self):
            return str(self.node)

        def GetIconName(self):
            """Different icon when object cannot be expanded, i.e. has no
            referrers.

            """
            if not self.IsExpandable():
                return "python"

        def IsExpandable(self):
            """An object is expandable when it is a node which has children and
            is a container object.

            """
            if not isinstance(self.node, _Node):
                return False
            else:
                if len(self.node.children) > 0:
                    return True
                else:
                    return muppy._is_containerobject(self.node.o)

        def GetSubList(self):
            """This method is the point where further referrers are computed.

            Thus, the computation is done on-demand and only when needed.

            """
            sublist = []

            children = self.node.children
            if (len(children) == 0) and\
                    (muppy._is_containerobject(self.node.o)):
                self.node = self.reftree._get_tree(self.node.o, 1)
                self._clear_children()
                children = self.node.children

            for child in children:
                item = _ReferrerTreeItem(self.parentwindow, child,
                                         self.reftree)
                sublist.append(item)
            return sublist

except ImportError:
    _TreeWidget = None


def gui_default_str_function(o):
    """Default str function for InteractiveBrowser."""
    return summary._repr(o) + '(id=%s)' % id(o)


class InteractiveBrowser(RefBrowser):
    """Interactive referrers browser.

    The interactive browser is based on a TreeWidget implemented in IDLE. It is
    available only if you have Tcl/Tk installed. If you try to instantiate the
    interactive browser without having Tkinter installed, an ImportError will
    be raised.

    """
    def __init__(self, rootobject, maxdepth=3,
                 str_func=gui_default_str_function, repeat=True):
        """You have to provide the root object used in the refbrowser.

        keyword arguments
        maxdepth -- maximum depth of the initial tree
        str_func -- function used when calling str(node)
        repeat -- should nodes appear repeatedly in the tree, or should be
                  referred to existing nodes

        """
        if tkinter is None:
            raise ImportError(
                "InteractiveBrowser requires Tkinter to be installed.")
        RefBrowser.__init__(self, rootobject, maxdepth, str_func, repeat)

    def main(self, standalone=False):
        """Create interactive browser window.

        keyword arguments
        standalone -- Set to true, if the browser is not attached to other
        windows

        """
        window = tkinter.Tk()
        sc = _TreeWidget.ScrolledCanvas(window, bg="white",
                                        highlightthickness=0, takefocus=1)
        sc.frame.pack(expand=1, fill="both")
        item = _ReferrerTreeItem(window, self.get_tree(), self)
        node = _TreeNode(sc.canvas, None, item)
        node.expand()
        if standalone:
            window.mainloop()


# list to hold to referrers
superlist = []
root = "root"
for i in range(3):
    tmp = [root]
    superlist.append(tmp)


def foo(o):
    return str(type(o))


def print_sample():
    cb = ConsoleBrowser(root, str_func=foo)
    cb.print_tree()


def write_sample():
    fb = FileBrowser(root, str_func=foo)
    fb.print_tree('sample.txt')


if __name__ == "__main__":
    write_sample()
