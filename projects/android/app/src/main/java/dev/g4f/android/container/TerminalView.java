package dev.g4f.android.container;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.graphics.drawable.ColorDrawable;
import android.text.InputType;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.inputmethod.BaseInputConnection;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputConnection;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.PopupWindow;

/**
 * Native terminal renderer for the container PTY.
 *
 * <p>The previous implementation rendered PTY bytes with xterm.js inside a
 * browser view. This view keeps the same job in the app process: it decodes
 * UTF-8, interprets the ANSI/VT100 subset produced by bash, apt, g4f-go and the
 * bundled tools, and draws a grid of cells with Canvas. Input comes from the
 * soft keyboard, hardware keys and the on-screen key row, and is written back
 * to the PTY through the listener.</p>
 */
public final class TerminalView extends View {

    /** Receives keystrokes and resize events for the attached PTY session. */
    public interface Listener {
        void onInput(int sessionId, byte[] data);

        void onResize(int sessionId, int cols, int rows);

        void onCopy(String text);
    }

    private static final int ST_GROUND = 0;
    private static final int ST_ESC = 1;
    private static final int ST_CSI = 2;
    private static final int ST_OSC = 3;
    private static final int ST_OSC_ESC = 4;
    private static final int ST_CHARSET = 5;

    private static final byte BOLD = 1;
    private static final byte DIM = 2;
    private static final byte ITALIC = 4;
    private static final byte UNDERLINE = 8;
    private static final byte REVERSE = 16;
    private static final byte STRIKE = 32;

    private static final int DEFAULT_FG = 0xffd6dad3;
    private static final int DEFAULT_BG = 0xff101411;
    private static final int SB_LIMIT = 5000;
    private static final long DOUBLE_TAP_TIMEOUT_MS = 300;

    private static final int[] ANSI = {
            0xff000000, 0xffcd0000, 0xff00cd00, 0xffcdcd00,
            0xff0000ee, 0xffcd00cd, 0xff00cdcd, 0xffe5e5e5,
            0xff7f7f7f, 0xffff0000, 0xff00ff00, 0xffffff00,
            0xff5c5cff, 0xffff00ff, 0xff00ffff, 0xffffffff,
    };

    private static final class Line {
        final int[] cps;
        final int[] fgs;
        final int[] bgs;
        final byte[] flags;

        Line(int cols) {
            cps = new int[cols];
            fgs = new int[cols];
            bgs = new int[cols];
            flags = new byte[cols];
            java.util.Arrays.fill(fgs, DEFAULT_FG);
            java.util.Arrays.fill(bgs, DEFAULT_BG);
        }
    }

    private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint fillPaint = new Paint();
    private final Paint handlePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint handleBorderPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Typeface mono = Typeface.MONOSPACE;
    private final Typeface monoItalic = Typeface.create(Typeface.MONOSPACE, Typeface.ITALIC);
    private final float density;
    private final float handleRadius;
    private final float handleTouchRadius;

    private float cellWidth;
    private float cellHeight;
    private float baseline;
    private float padding;

    private int cols = 80;
    private int rows = 24;
    private int[] cps = new int[cols * rows];
    private int[] fgs = new int[cols * rows];
    private int[] bgs = new int[cols * rows];
    private byte[] flags = new byte[cols * rows];

    private final Line[] scrollback = new Line[SB_LIMIT];
    private int sbStart;
    private int sbCount;
    private int scrollOffset;

    private int cursorRow;
    private int cursorCol;
    private boolean cursorVisible = true;
    private boolean pendingWrap;
    private int savedRow;
    private int savedCol;
    private int scrollTop;
    private int scrollBottom;

    private int fg = DEFAULT_FG;
    private int bg = DEFAULT_BG;
    private byte attr;

    private int state = ST_GROUND;
    private final StringBuilder seq = new StringBuilder();
    private int utf8Needed;
    private int utf8Code;
    private int utf8Min;

    private boolean altScreen;
    private int[] savedCps;
    private int[] savedFgs;
    private int[] savedBgs;
    private byte[] savedFlags;
    private int savedAltRow;
    private int savedAltCol;
    private int savedAltTop;
    private int savedAltBottom;

    private int selAnchorLine = -1;
    private int selAnchorCol;
    private int selFocusLine = -1;
    private int selFocusCol;
    private boolean selectionMode;
    private PopupWindow selectionMenu;

    private boolean ctrlPressed;
    private boolean altPressed;

    private int sessionId = -1;
    private Listener listener;

    private float touchStartX;
    private float touchStartY;
    private float lastTouchY;
    private float scrollAccum;
    private boolean longPressFired;
    private boolean draggingSelection;
    private boolean draggingScroll;
    private int draggingHandle = -1; // -1 none, 0 start pip, 1 end pip
    private boolean handleDragsAnchor;
    private long lastTapTime;
    private float lastTapX;
    private float lastTapY;
    private int downLine;
    private int downCol;
    private final Runnable longPress = this::startLongPressSelection;
    private final int touchSlop;

    public TerminalView(Context context) {
        super(context);
        density = getResources().getDisplayMetrics().density;
        padding = 4f * density;
        touchSlop = (int) (6f * density);
        handleRadius = 6f * density;
        handleTouchRadius = 14f * density;
        handlePaint.setColor(0xffc4f06c);
        handlePaint.setStyle(Paint.Style.FILL);
        handleBorderPaint.setColor(DEFAULT_BG);
        handleBorderPaint.setStyle(Paint.Style.STROKE);
        handleBorderPaint.setStrokeWidth(2f * density);
        textPaint.setTypeface(mono);
        textPaint.setTextSize(TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_SP, 12f, getResources().getDisplayMetrics()));
        textPaint.setColor(DEFAULT_FG);
        cellWidth = textPaint.measureText("M");
        if (cellWidth <= 0) cellWidth = 7f;
        Paint.FontMetrics metrics = textPaint.getFontMetrics();
        cellHeight = (metrics.descent - metrics.ascent) * 1.18f;
        baseline = -metrics.ascent + (cellHeight - (metrics.descent - metrics.ascent)) / 2f;
        setBackgroundColor(DEFAULT_BG);
        setFocusable(true);
        setFocusableInTouchMode(true);
        setClickable(true);
        java.util.Arrays.fill(fgs, DEFAULT_FG);
        java.util.Arrays.fill(bgs, DEFAULT_BG);
        resetAttributes();
        scrollTop = 0;
        scrollBottom = rows - 1;
    }

    // ── Session wiring ─────────────────────────────────────────────────

    public void setSession(int sessionId, Listener listener) {
        this.sessionId = sessionId;
        this.listener = listener;
        if (listener != null) listener.onResize(sessionId, cols, rows);
    }

    public int sessionId() {
        return sessionId;
    }

    /** Feed raw PTY bytes. Safe to call from the service callback thread. */
    public synchronized void write(byte[] data) {
        if (data == null || data.length == 0) return;
        for (byte value : data) {
            int b = value & 0xff;
            decodeByte(b);
        }
        trimScrollOffset();
        postInvalidateOnAnimation();
    }

    public void write(String text) {
        if (text == null) return;
        write(text.getBytes(java.nio.charset.StandardCharsets.UTF_8));
    }

    public void showProcessExit(int exitCode, boolean stopped) {
        String message = stopped ? "[stopped]" : "[process exited " + exitCode + "]";
        write("\r\n\u001b[38;2;150;160;150m" + message + "\u001b[0m\r\n");
    }

    public void showBanner() {
        write("\u001b[38;2;150;160;150mg4f Shell\u001b[0m\r\n"
                + "Tools: \u001b[38;2;196;240;108msandbox-cli list\u001b[0m\r\n"
                + "Run:   \u001b[38;2;196;240;108msandbox-cli <tool>\u001b[0m\r\n"
                + "Also:  \u001b[38;2;196;240;108mg4f-go\u001b[0m\r\n\r\n");
    }

    // ── Input ──────────────────────────────────────────────────────────

    public void setCtrl(boolean value) {
        ctrlPressed = value;
    }

    public void setAlt(boolean value) {
        altPressed = value;
    }

    public boolean isCtrl() {
        return ctrlPressed;
    }

    public boolean isAlt() {
        return altPressed;
    }

    public void clearModifiers() {
        ctrlPressed = false;
        altPressed = false;
    }

    /** Send text typed on the keyboard or by the IME, applying Ctrl/Alt. */
    public void sendText(String text) {
        if (text == null || text.isEmpty()) return;
        if (ctrlPressed) {
            StringBuilder out = new StringBuilder(text.length());
            for (int i = 0; i < text.length(); i++) {
                char c = text.charAt(i);
                if (c >= '@' && c <= '_') out.append((char) (c & 0x1f));
                else if (c >= 'a' && c <= 'z') out.append((char) (c & 0x1f));
                else out.append(c);
            }
            text = out.toString();
        }
        if (altPressed && !text.startsWith("\u001b")) text = "\u001b" + text;
        ctrlPressed = false;
        altPressed = false;
        dispatch(text.getBytes(java.nio.charset.StandardCharsets.UTF_8));
    }

    /** Send one of the Termux-style key names from the on-screen key row. */
    public void sendKey(String name) {
        String value;
        switch (name) {
            case "esc":
                value = "\u001b";
                break;
            case "tab":
                value = "\t";
                break;
            case "left":
            case "up":
            case "down":
            case "right":
            case "home":
            case "end":
            case "pgup":
            case "pgdn":
                value = cursorSequence(name);
                break;
            case "ctrlc":
                value = "\u0003";
                break;
            case "ctrld":
                value = "\u0004";
                break;
            default:
                value = "";
        }
        if (value.isEmpty()) return;
        if (altPressed && !value.startsWith("\u001b")) value = "\u001b" + value;
        ctrlPressed = false;
        altPressed = false;
        dispatch(value.getBytes(java.nio.charset.StandardCharsets.UTF_8));
    }

    private String cursorSequence(String name) {
        char finalChar;
        switch (name) {
            case "up":
                finalChar = 'A';
                break;
            case "down":
                finalChar = 'B';
                break;
            case "right":
                finalChar = 'C';
                break;
            case "left":
                finalChar = 'D';
                break;
            case "home":
                finalChar = 'H';
                break;
            case "end":
                finalChar = 'F';
                break;
            case "pgup":
                return "\u001b[5~";
            case "pgdn":
                return "\u001b[6~";
            default:
                return "";
        }
        if (ctrlPressed) return "\u001b[1;5" + finalChar;
        if (altPressed) return "\u001b[1;3" + finalChar;
        return "\u001b[" + finalChar;
    }

    private void dispatch(byte[] bytes) {
        if (listener != null && sessionId >= 0) listener.onInput(sessionId, bytes);
    }

    @Override
    public boolean onCheckIsTextEditor() {
        return true;
    }

    @Override
    public InputConnection onCreateInputConnection(EditorInfo outAttrs) {
        outAttrs.inputType = InputType.TYPE_CLASS_TEXT
                | InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS
                | InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD;
        outAttrs.imeOptions = EditorInfo.IME_ACTION_NONE
                | EditorInfo.IME_FLAG_NO_FULLSCREEN
                | EditorInfo.IME_FLAG_NO_EXTRACT_UI;
        return new BaseInputConnection(this, false) {
            @Override
            public boolean commitText(CharSequence text, int newCursorPosition) {
                sendText(text == null ? "" : text.toString());
                return true;
            }

            @Override
            public boolean setComposingText(CharSequence text, int newCursorPosition) {
                return true;
            }

            @Override
            public boolean finishComposingText() {
                return true;
            }

            @Override
            public boolean deleteSurroundingText(int beforeLength, int afterLength) {
                for (int i = 0; i < beforeLength; i++) sendText("\u007f");
                return true;
            }

            @Override
            public boolean sendKeyEvent(KeyEvent event) {
                if (event.getAction() == KeyEvent.ACTION_DOWN && handleKey(event)) return true;
                return super.sendKeyEvent(event);
            }
        };
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        if (handleKey(event)) return true;
        return super.onKeyDown(keyCode, event);
    }

    private boolean handleKey(KeyEvent event) {
        int keyCode = event.getKeyCode();
        switch (keyCode) {
            case KeyEvent.KEYCODE_ENTER:
            case KeyEvent.KEYCODE_NUMPAD_ENTER:
                sendText("\r");
                return true;
            case KeyEvent.KEYCODE_DEL:
                sendText("\u007f");
                return true;
            case KeyEvent.KEYCODE_FORWARD_DEL:
                sendText("\u001b[3~");
                return true;
            case KeyEvent.KEYCODE_TAB:
                sendText("\t");
                return true;
            case KeyEvent.KEYCODE_ESCAPE:
                sendText("\u001b");
                return true;
            case KeyEvent.KEYCODE_DPAD_UP:
                sendKey("up");
                return true;
            case KeyEvent.KEYCODE_DPAD_DOWN:
                sendKey("down");
                return true;
            case KeyEvent.KEYCODE_DPAD_LEFT:
                sendKey("left");
                return true;
            case KeyEvent.KEYCODE_DPAD_RIGHT:
                sendKey("right");
                return true;
            case KeyEvent.KEYCODE_MOVE_HOME:
                sendKey("home");
                return true;
            case KeyEvent.KEYCODE_MOVE_END:
                sendKey("end");
                return true;
            case KeyEvent.KEYCODE_PAGE_UP:
                sendKey("pgup");
                return true;
            case KeyEvent.KEYCODE_PAGE_DOWN:
                sendKey("pgdn");
                return true;
            default:
                break;
        }
        int unicode = event.getUnicodeChar(event.getMetaState());
        if (unicode == 0) return false;
        sendText(new String(Character.toChars(unicode)));
        return true;
    }

    /** Copy the current selection (returns null when nothing is selected). */
    public String getSelectedText() {
        int[] bounds = selectionBounds();
        if (bounds == null) return null;
        int startLine = bounds[0];
        int startCol = bounds[1];
        int endLine = bounds[2];
        int endCol = bounds[3];
        StringBuilder out = new StringBuilder();
        for (int line = startLine; line <= endLine; line++) {
            int length = lineLength(line);
            int from = line == startLine ? Math.min(startCol, length) : 0;
            int to = line == endLine ? Math.min(endCol + 1, length) : length;
            if (line > startLine) out.append('\n');
            for (int col = from; col < to; col++) {
                int cp = cpAt(line, col);
                if (cp > 0) out.appendCodePoint(cp);
            }
        }
        return out.length() == 0 ? null : out.toString();
    }

    public boolean hasSelection() {
        return selAnchorLine >= 0 && selFocusLine >= 0;
    }

    /** Ordered selection endpoints: start line/col, then end line/col. */
    private int[] selectionBounds() {
        if (!hasSelection()) return null;
        if (selAnchorLine < selFocusLine
                || (selAnchorLine == selFocusLine && selAnchorCol <= selFocusCol)) {
            return new int[] {selAnchorLine, selAnchorCol, selFocusLine, selFocusCol};
        }
        return new int[] {selFocusLine, selFocusCol, selAnchorLine, selAnchorCol};
    }

    public void clearSelection() {
        dismissSelectionMenu();
        selAnchorLine = -1;
        selFocusLine = -1;
        draggingHandle = -1;
        invalidate();
    }

    public void selectAll() {
        selAnchorLine = 0;
        selAnchorCol = 0;
        selFocusLine = Math.max(0, sbCount + rows - 1);
        selFocusCol = Math.max(0, cols - 1);
        invalidate();
    }

    public void toggleSelectionMode() {
        selectionMode = !selectionMode;
        if (!selectionMode) clearSelection();
    }

    public boolean isSelectionMode() {
        return selectionMode;
    }

    /** Copy the current selection and notify the host (for the Copy button). */
    public boolean copySelection() {
        String text = getSelectedText();
        if (text == null) return false;
        if (listener != null) listener.onCopy(text);
        return true;
    }

    /** Offer selection actions instead of copying to the clipboard immediately. */
    private void showSelectionMenu() {
        if (!hasSelection() || getSelectedText() == null) return;
        dismissSelectionMenu();

        LinearLayout content = new LinearLayout(getContext());
        content.setOrientation(LinearLayout.VERTICAL);
        content.setBackgroundColor(Color.parseColor("#181d19"));
        content.addView(menuButton("Copy", () -> {
            copySelection();
            dismissSelectionMenu();
        }));
        content.addView(menuButton("Select all", () -> {
            selectAll();
            post(this::showSelectionMenu);
        }));

        PopupWindow menu = new PopupWindow(content,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT, true);
        menu.setOutsideTouchable(true);
        menu.setBackgroundDrawable(new ColorDrawable(Color.parseColor("#181d19")));
        menu.setOnDismissListener(() -> {
            if (selectionMenu == menu) selectionMenu = null;
        });
        selectionMenu = menu;

        int[] bounds = selectionBounds();
        float x = handleX(bounds[3], false);
        float y = handleY(bounds[2]);
        if (y < 0f) {
            x = handleX(bounds[1], true);
            y = handleY(bounds[0]);
        }
        if (y < 0f) y = cellHeight;
        int menuX = (int) Math.max(0f, Math.min(x, getWidth() - 150f * density));
        int menuY = (int) Math.max(0f,
                Math.min(y + handleRadius, getHeight() - 3f * cellHeight));
        menu.showAtLocation(this, Gravity.TOP | Gravity.START, menuX, menuY);
    }

    private Button menuButton(String label, Runnable action) {
        Button button = new Button(getContext());
        button.setText(label);
        button.setTextSize(12f);
        button.setAllCaps(false);
        button.setMinWidth(0);
        button.setMinimumWidth(0);
        button.setPadding((int) (12f * density), 0, (int) (12f * density), 0);
        button.setTextColor(Color.parseColor("#c4f06c"));
        button.setBackgroundColor(Color.TRANSPARENT);
        button.setOnClickListener(v -> action.run());
        return button;
    }

    private void dismissSelectionMenu() {
        if (selectionMenu != null) {
            selectionMenu.dismiss();
            selectionMenu = null;
        }
    }

    // ── Touch: scroll, tap/long-press selection, draggable pips ────────

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();
        int action = event.getActionMasked();
        switch (action) {
            case MotionEvent.ACTION_DOWN:
                dismissSelectionMenu();
                touchStartX = x;
                touchStartY = y;
                lastTouchY = y;
                scrollAccum = 0;
                longPressFired = false;
                draggingSelection = false;
                draggingScroll = false;
                draggingHandle = handleAt(x, y);
                if (draggingHandle >= 0) {
                    handleDragsAnchor = endpointIsAnchor(draggingHandle == 0);
                    removeCallbacks(longPress);
                    return true;
                }
                downLine = lineForY(y);
                downCol = colForX(x);
                postDelayed(longPress, 350);
                return true;
            case MotionEvent.ACTION_MOVE:
                handleTouchMove(event);
                return true;
            case MotionEvent.ACTION_UP:
            case MotionEvent.ACTION_CANCEL:
                removeCallbacks(longPress);
                boolean up = action == MotionEvent.ACTION_UP;
                if (draggingHandle >= 0 || draggingSelection) {
                    if (up && getSelectedText() != null) showSelectionMenu();
                } else if (!longPressFired && !draggingScroll) {
                    handleTap(x, y, up);
                }
                draggingHandle = -1;
                draggingSelection = false;
                draggingScroll = false;
                performClick();
                return true;
            default:
                return super.onTouchEvent(event);
        }
    }

    /** A tap shows the keyboard; a double tap selects the word under it. */
    private void handleTap(float x, float y, boolean up) {
        if (!up) {
            lastTapTime = 0;
            return;
        }
        boolean tap = Math.abs(x - touchStartX) + Math.abs(y - touchStartY) < touchSlop;
        if (!tap) {
            lastTapTime = 0;
            requestFocus();
            showKeyboard();
            return;
        }
        long now = System.currentTimeMillis();
        boolean doubleTap = now - lastTapTime <= DOUBLE_TAP_TIMEOUT_MS
                && Math.abs(x - lastTapX) <= touchSlop * 2f
                && Math.abs(y - lastTapY) <= touchSlop * 2f;
        lastTapTime = now;
        lastTapX = x;
        lastTapY = y;
        if (doubleTap) {
            beginSelection(downLine, downCol, true);
            lastTapTime = 0;
            showSelectionMenu();
            invalidate();
            return;
        }
        clearSelection();
        requestFocus();
        showKeyboard();
    }

    private void handleTouchMove(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();
        if (draggingHandle >= 0) {
            autoScrollSelection(y);
            moveHandle(x, y);
            invalidate();
            return;
        }
        float dx = x - touchStartX;
        float dy = y - touchStartY;
        if (!longPressFired && !draggingSelection && !draggingScroll) {
            if (Math.abs(dx) + Math.abs(dy) < touchSlop) return;
            removeCallbacks(longPress);
            if (selectionMode || Math.abs(dx) > Math.abs(dy) * 1.5f) {
                beginSelection(downLine, downCol, false);
                draggingSelection = true;
            } else if (Math.abs(dy) > touchSlop) {
                dismissSelectionMenu();
                draggingScroll = true;
                lastTouchY = touchStartY;
            }
        }
        if (draggingSelection) {
            autoScrollSelection(y);
            selFocusLine = lineForY(y);
            selFocusCol = colForX(x);
            invalidate();
        } else if (draggingScroll) {
            // Natural scrolling: moving the finger down reveals older lines.
            scrollAccum += y - lastTouchY;
            lastTouchY = y;
            while (scrollAccum >= cellHeight) {
                scrollOffset = Math.min(sbCount, scrollOffset + 1);
                scrollAccum -= cellHeight;
            }
            while (scrollAccum <= -cellHeight) {
                scrollOffset = Math.max(0, scrollOffset - 1);
                scrollAccum += cellHeight;
            }
            invalidate();
        }
    }

    private void startLongPressSelection() {
        longPressFired = true;
        draggingSelection = true;
        beginSelection(downLine, downCol, true);
        invalidate();
    }

    private void beginSelection(int line, int col, boolean word) {
        selAnchorLine = clampLine(line);
        selAnchorCol = Math.max(0, col);
        selFocusLine = selAnchorLine;
        selFocusCol = selAnchorCol;
        if (word) {
            int start = selAnchorCol;
            int end = selAnchorCol;
            int length = lineLength(selAnchorLine);
            while (start > 0 && isWordChar(cpAt(selAnchorLine, start - 1))) start--;
            while (end < length - 1 && isWordChar(cpAt(selAnchorLine, end + 1))) end++;
            selAnchorCol = start;
            selFocusCol = end;
        }
    }

    /** Move the grabbed pip while keeping the opposite selection end fixed. */
    private void moveHandle(float x, float y) {
        int line = clampLine(lineForY(y));
        int col = Math.max(0, Math.min(cols - 1, colForX(x)));
        if (handleDragsAnchor) {
            selAnchorLine = line;
            selAnchorCol = col;
        } else {
            selFocusLine = line;
            selFocusCol = col;
        }
    }

    private boolean endpointIsAnchor(boolean startEndpoint) {
        boolean anchorIsStart = selAnchorLine < selFocusLine
                || (selAnchorLine == selFocusLine && selAnchorCol <= selFocusCol);
        return startEndpoint == anchorIsStart;
    }

    /** Keep a dragged pip reachable at the top/bottom edge of the view. */
    private void autoScrollSelection(float y) {
        float edge = cellHeight * 1.5f;
        if (y < padding + edge && scrollOffset < sbCount) {
            scrollOffset++;
        } else if (y > getHeight() - padding - edge && scrollOffset > 0) {
            scrollOffset--;
        }
    }

    /** Return 0 for the start pip, 1 for the end pip, or -1 for no pip. */
    private int handleAt(float x, float y) {
        int[] bounds = selectionBounds();
        if (bounds == null) return -1;
        float startX = handleX(bounds[1], true);
        float startY = handleY(bounds[0]);
        float endX = handleX(bounds[3], false);
        float endY = handleY(bounds[2]);
        boolean nearStart = nearHandle(x, y, startX, startY);
        boolean nearEnd = nearHandle(x, y, endX, endY);
        if (nearStart && nearEnd) return x < (startX + endX) / 2f ? 0 : 1;
        if (nearStart) return 0;
        return nearEnd ? 1 : -1;
    }

    private boolean nearHandle(float x, float y, float hx, float hy) {
        if (hy < 0f) return false;
        float dx = x - hx;
        float dy = y - hy;
        return dx * dx + dy * dy <= handleTouchRadius * handleTouchRadius;
    }

    private float handleX(int col, boolean startHandle) {
        float x = padding + (startHandle ? col : col + 1) * cellWidth;
        return Math.max(handleRadius, Math.min(getWidth() - handleRadius, x));
    }

    private float handleY(int line) {
        float top = lineTop(line);
        if (top < 0f) return -1f;
        float y = top + cellHeight + handleRadius * 0.5f;
        return Math.max(handleRadius, Math.min(getHeight() - handleRadius, y));
    }

    private float lineTop(int line) {
        int start = visibleStartLine();
        if (line < start || line >= start + visibleRowCount()) return -1f;
        return padding + (line - start) * cellHeight;
    }

    private static boolean isWordChar(int cp) {
        if (cp <= 0) return false;
        return Character.isLetterOrDigit(cp) || cp == '_' || cp == '-' || cp == '.'
                || cp == '/' || cp == ':' || cp == '@';
    }

    @Override
    public boolean performClick() {
        super.performClick();
        return true;
    }

    private void showKeyboard() {
        InputMethodManager imm = (InputMethodManager) getContext()
                .getSystemService(Context.INPUT_METHOD_SERVICE);
        if (imm != null) imm.showSoftInput(this, InputMethodManager.SHOW_IMPLICIT);
    }

    @Override
    protected void onDetachedFromWindow() {
        removeCallbacks(longPress);
        dismissSelectionMenu();
        super.onDetachedFromWindow();
    }

    // ── Layout ─────────────────────────────────────────────────────────

    @Override
    protected void onSizeChanged(int width, int height, int oldWidth, int oldHeight) {
        super.onSizeChanged(width, height, oldWidth, oldHeight);
        int newCols = Math.max(20,
                (int) Math.floor((width - 2 * padding) / Math.max(1f, cellWidth)));
        int newRows = Math.max(4,
                (int) Math.floor((height - 2 * padding) / Math.max(1f, cellHeight)));
        resizeGrid(newCols, newRows);
    }

    private void resizeGrid(int newCols, int newRows) {
        if (newCols == cols && newRows == rows) return;
        if (altScreen) {
            // The saved primary buffer keeps its old width; dropping the
            // alternate screen is safer than restoring mismatched arrays.
            savedCps = null;
            altScreen = false;
        }
        int[] nextCps = new int[newCols * newRows];
        int[] nextFgs = new int[newCols * newRows];
        int[] nextBgs = new int[newCols * newRows];
        byte[] nextFlags = new byte[newCols * newRows];
        java.util.Arrays.fill(nextFgs, DEFAULT_FG);
        java.util.Arrays.fill(nextBgs, DEFAULT_BG);
        int copyRows = Math.min(rows, newRows);
        int copyCols = Math.min(cols, newCols);
        for (int row = 0; row < copyRows; row++) {
            System.arraycopy(cps, row * cols, nextCps, row * newCols, copyCols);
            System.arraycopy(fgs, row * cols, nextFgs, row * newCols, copyCols);
            System.arraycopy(bgs, row * cols, nextBgs, row * newCols, copyCols);
            System.arraycopy(flags, row * cols, nextFlags, row * newCols, copyCols);
        }
        cols = newCols;
        rows = newRows;
        cps = nextCps;
        fgs = nextFgs;
        bgs = nextBgs;
        flags = nextFlags;
        cursorRow = Math.max(0, Math.min(cursorRow, rows - 1));
        cursorCol = Math.max(0, Math.min(cursorCol, cols - 1));
        scrollTop = 0;
        scrollBottom = rows - 1;
        pendingWrap = false;
        if (listener != null && sessionId >= 0) listener.onResize(sessionId, cols, rows);
    }

    private int visibleRowCount() {
        int count = (int) Math.floor((getHeight() - 2 * padding) / Math.max(1f, cellHeight));
        return Math.max(1, count);
    }

    private int visibleStartLine() {
        int total = sbCount + rows;
        int end = total - Math.min(scrollOffset, sbCount);
        return Math.max(0, end - visibleRowCount());
    }

    private int lineForY(float y) {
        int row = (int) Math.floor((y - padding) / Math.max(1f, cellHeight));
        return clampLine(visibleStartLine() + row);
    }

    private int colForX(float x) {
        int col = (int) Math.floor((x - padding) / Math.max(1f, cellWidth));
        return Math.max(0, Math.min(cols - 1, col));
    }

    // ── Rendering ──────────────────────────────────────────────────────

    @Override
    protected synchronized void onDraw(Canvas canvas) {
        canvas.drawColor(DEFAULT_BG);
        int count = visibleRowCount();
        int total = sbCount + rows;
        int end = total - Math.min(scrollOffset, sbCount);
        int start = Math.max(0, end - count);
        for (int line = start; line < end; line++) {
            float top = padding + (line - start) * cellHeight;
            drawLine(canvas, line, top);
        }
        if (scrollOffset == 0 && cursorVisible && hasFocus() && !altScreen) {
            drawCursor(canvas, start);
        }
        drawSelectionHandles(canvas);
    }

    private void drawLine(Canvas canvas, int line, float top) {
        int length = lineLength(line);
        for (int col = 0; col < length; col++) {
            int cellBg = bgAt(line, col);
            byte cellFlags = flagsAt(line, col);
            if ((cellFlags & REVERSE) != 0) cellBg = fgAt(line, col);
            if (isSelected(line, col)) cellBg = blend(cellBg, 0xff31402f);
            if (cellBg != DEFAULT_BG) {
                fillPaint.setColor(cellBg);
                canvas.drawRect(padding + col * cellWidth, top,
                        padding + (col + 1) * cellWidth, top + cellHeight, fillPaint);
            }
        }
        // Group equally styled cells into text runs to keep draw calls low.
        int col = 0;
        while (col < length) {
            int cp = cpAt(line, col);
            if (cp == 0) {
                col++;
                continue;
            }
            int cellFg = fgAt(line, col);
            byte cellFlags = flagsAt(line, col);
            int cellBg = bgAt(line, col);
            if ((cellFlags & REVERSE) != 0) {
                int swap = cellFg;
                cellFg = cellBg;
                cellBg = swap;
            }
            StringBuilder run = new StringBuilder();
            int runStart = col;
            while (col < length) {
                int value = cpAt(line, col);
                if (value == 0) break;
                byte valueFlags = flagsAt(line, col);
                int valueFg = fgAt(line, col);
                int valueBg = bgAt(line, col);
                if ((valueFlags & REVERSE) != 0) {
                    int swap = valueFg;
                    valueFg = valueBg;
                    valueBg = swap;
                }
                if (valueFg != cellFg || (valueFlags & ~REVERSE) != (cellFlags & ~REVERSE)) break;
                run.appendCodePoint(value);
                col++;
            }
            if (run.length() == 0) {
                col = runStart + 1;
                continue;
            }
            textPaint.setTextSize(TypedValue.applyDimension(
                    TypedValue.COMPLEX_UNIT_SP, 12f, getResources().getDisplayMetrics()));
            textPaint.setTypeface((cellFlags & ITALIC) != 0 ? monoItalic : mono);
            textPaint.setFakeBoldText((cellFlags & BOLD) != 0);
            textPaint.setColor((cellFlags & DIM) != 0 ? dim(cellFg) : cellFg);
            canvas.drawText(run.toString(), padding + runStart * cellWidth,
                    top + baseline, textPaint);
            if ((cellFlags & UNDERLINE) != 0) {
                fillPaint.setColor(cellFg);
                canvas.drawRect(padding + runStart * cellWidth,
                        top + cellHeight - 2 * density,
                        padding + col * cellWidth,
                        top + cellHeight - density, fillPaint);
            }
            if ((cellFlags & STRIKE) != 0) {
                fillPaint.setColor(cellFg);
                float middle = top + cellHeight / 2f;
                canvas.drawRect(padding + runStart * cellWidth, middle - density / 2f,
                        padding + col * cellWidth, middle + density / 2f, fillPaint);
            }
        }
    }

    private void drawCursor(Canvas canvas, int startLine) {
        int line = sbCount + cursorRow;
        if (line < startLine) return;
        float left = padding + cursorCol * cellWidth;
        float top = padding + (line - startLine) * cellHeight;
        fillPaint.setColor(DEFAULT_FG);
        canvas.drawRect(left, top, left + cellWidth, top + cellHeight, fillPaint);
        int cp = cpAt(line, cursorCol);
        if (cp > 0) {
            textPaint.setTypeface(mono);
            textPaint.setFakeBoldText(false);
            textPaint.setColor(DEFAULT_BG);
            canvas.drawText(new String(Character.toChars(cp)), left, top + baseline, textPaint);
        }
    }

    /** Draw the draggable start/end pips for the active selection. */
    private void drawSelectionHandles(Canvas canvas) {
        int[] bounds = selectionBounds();
        if (bounds == null) return;
        drawHandle(canvas, handleX(bounds[1], true), handleY(bounds[0]));
        drawHandle(canvas, handleX(bounds[3], false), handleY(bounds[2]));
    }

    private void drawHandle(Canvas canvas, float x, float y) {
        if (y < 0f) return;
        canvas.drawCircle(x, y, handleRadius, handlePaint);
        canvas.drawCircle(x, y, handleRadius, handleBorderPaint);
    }

    private boolean isSelected(int line, int col) {
        if (selAnchorLine < 0 || selFocusLine < 0) return false;
        int startLine;
        int startCol;
        int endLine;
        int endCol;
        if (selAnchorLine < selFocusLine
                || (selAnchorLine == selFocusLine && selAnchorCol <= selFocusCol)) {
            startLine = selAnchorLine;
            startCol = selAnchorCol;
            endLine = selFocusLine;
            endCol = selFocusCol;
        } else {
            startLine = selFocusLine;
            startCol = selFocusCol;
            endLine = selAnchorLine;
            endCol = selAnchorCol;
        }
        if (line < startLine || line > endLine) return false;
        if (line == startLine && col < startCol) return false;
        if (line == endLine && col > endCol) return false;
        return true;
    }

    private static int blend(int from, int to) {
        return Color.rgb((Color.red(from) + Color.red(to)) / 2,
                (Color.green(from) + Color.green(to)) / 2,
                (Color.blue(from) + Color.blue(to)) / 2);
    }

    private static int dim(int color) {
        return Color.rgb(Color.red(color) * 3 / 4, Color.green(color) * 3 / 4,
                Color.blue(color) * 3 / 4);
    }

    private int lineLength(int line) {
        int sb = sbCount;
        if (line < 0) return 0;
        if (line < sb) return scrollbackAt(line).cps.length;
        int row = line - sb;
        return row < rows ? cols : 0;
    }

    private int cpAt(int line, int col) {
        int sb = sbCount;
        if (line < 0 || col < 0) return 0;
        if (line < sb) {
            Line value = scrollbackAt(line);
            return col < value.cps.length ? value.cps[col] : 0;
        }
        int row = line - sb;
        if (row >= rows || col >= cols) return 0;
        return cps[row * cols + col];
    }

    private int fgAt(int line, int col) {
        int sb = sbCount;
        if (line < sb) {
            Line value = scrollbackAt(line);
            return col < value.fgs.length ? value.fgs[col] : DEFAULT_FG;
        }
        int row = line - sb;
        if (row >= rows || col >= cols) return DEFAULT_FG;
        return fgs[row * cols + col];
    }

    private int bgAt(int line, int col) {
        int sb = sbCount;
        if (line < sb) {
            Line value = scrollbackAt(line);
            return col < value.bgs.length ? value.bgs[col] : DEFAULT_BG;
        }
        int row = line - sb;
        if (row >= rows || col >= cols) return DEFAULT_BG;
        return bgs[row * cols + col];
    }

    private byte flagsAt(int line, int col) {
        int sb = sbCount;
        if (line < sb) {
            Line value = scrollbackAt(line);
            return col < value.flags.length ? value.flags[col] : 0;
        }
        int row = line - sb;
        if (row >= rows || col >= cols) return 0;
        return flags[row * cols + col];
    }

    private int clampLine(int line) {
        return Math.max(0, Math.min(sbCount + rows - 1, line));
    }

    private Line scrollbackAt(int index) {
        return scrollback[(sbStart + index) % SB_LIMIT];
    }

    private void pushScrollback(Line line) {
        if (sbCount < SB_LIMIT) {
            scrollback[(sbStart + sbCount) % SB_LIMIT] = line;
            sbCount++;
        } else {
            scrollback[sbStart] = line;
            sbStart = (sbStart + 1) % SB_LIMIT;
        }
        if (scrollOffset > 0) {
            // Keep the visible region anchored while the user reads history.
            scrollOffset = Math.min(sbCount, scrollOffset + 1);
        }
    }

    private void trimScrollOffset() {
        if (scrollOffset > sbCount) scrollOffset = sbCount;
    }

    // ── ANSI / UTF-8 parser ────────────────────────────────────────────

    private void decodeByte(int b) {
        if (utf8Needed == 0) {
            if (b < 0x80) {
                process((char) b);
            } else if ((b & 0xe0) == 0xc0) {
                utf8Needed = 1;
                utf8Code = b & 0x1f;
                utf8Min = 0x80;
            } else if ((b & 0xf0) == 0xe0) {
                utf8Needed = 2;
                utf8Code = b & 0x0f;
                utf8Min = 0x800;
            } else if ((b & 0xf8) == 0xf0) {
                utf8Needed = 3;
                utf8Code = b & 0x07;
                utf8Min = 0x10000;
            } else {
                process('\ufffd');
            }
        } else if ((b & 0xc0) == 0x80) {
            utf8Code = (utf8Code << 6) | (b & 0x3f);
            if (--utf8Needed == 0) {
                int cp = utf8Code;
                if (cp < utf8Min || cp > 0x10ffff
                        || (cp >= 0xd800 && cp <= 0xdfff)) {
                    process('\ufffd');
                } else {
                    processCodePoint(cp);
                }
            }
        } else {
            utf8Needed = 0;
            process('\ufffd');
            decodeByte(b);
        }
    }

    private void processCodePoint(int cp) {
        if (cp < 0x10000) {
            process((char) cp);
        } else if (state == ST_GROUND) {
            putCodePoint(cp);
        }
    }

    private void process(char c) {
        switch (state) {
            case ST_GROUND:
                if (c == 0x1b) {
                    state = ST_ESC;
                } else {
                    handleGround(c);
                }
                break;
            case ST_ESC:
                handleEsc(c);
                break;
            case ST_CSI:
                if (c >= 0x40 && c <= 0x7e) {
                    handleCsi(seq.toString(), c);
                    state = ST_GROUND;
                } else if (c == 0x1b) {
                    state = ST_ESC;
                } else {
                    seq.append(c);
                }
                break;
            case ST_OSC:
                if (c == 0x07) state = ST_GROUND;
                else if (c == 0x1b) state = ST_OSC_ESC;
                break;
            case ST_OSC_ESC:
                if (c == '\\') state = ST_GROUND;
                else {
                    state = ST_OSC;
                    if (c == 0x1b) state = ST_OSC_ESC;
                }
                break;
            case ST_CHARSET:
                state = ST_GROUND;
                break;
            default:
                state = ST_GROUND;
        }
    }

    private void handleEsc(char c) {
        state = ST_GROUND;
        switch (c) {
            case '[':
                seq.setLength(0);
                state = ST_CSI;
                break;
            case ']':
                state = ST_OSC;
                break;
            case 'P':
            case '^':
            case '_':
                state = ST_OSC;
                break;
            case '(':
            case ')':
            case '*':
            case '+':
            case '#':
                state = ST_CHARSET;
                break;
            case '7':
                savedRow = cursorRow;
                savedCol = cursorCol;
                break;
            case '8':
                cursorRow = savedRow;
                cursorCol = savedCol;
                break;
            case 'D':
                lineFeed();
                break;
            case 'E':
                cursorCol = 0;
                lineFeed();
                break;
            case 'M':
                reverseIndex();
                break;
            case 'c':
                resetTerminal();
                break;
            default:
                break;
        }
    }

    private void handleGround(char c) {
        if (c == '\r') {
            cursorCol = 0;
            pendingWrap = false;
        } else if (c == '\n') {
            lineFeed();
        } else if (c == '\b') {
            if (cursorCol > 0) cursorCol--;
            pendingWrap = false;
        } else if (c == '\t') {
            cursorCol = Math.min(cols - 1, (cursorCol / 8 + 1) * 8);
            pendingWrap = false;
        } else if (c >= 0x20) {
            putCodePoint(c);
        }
    }

    private void putCodePoint(int cp) {
        int width = charWidth(cp);
        if (width == 0) return;
        if (pendingWrap) {
            cursorCol = 0;
            lineFeed();
            pendingWrap = false;
        }
        if (width == 2 && cursorCol == cols - 1) {
            cursorCol = 0;
            lineFeed();
        }
        int index = cursorRow * cols + cursorCol;
        cps[index] = cp;
        fgs[index] = fg;
        bgs[index] = bg;
        flags[index] = attr;
        if (width == 2 && cursorCol + 1 < cols) {
            cps[index + 1] = 0;
            fgs[index + 1] = fg;
            bgs[index + 1] = bg;
            flags[index + 1] = attr;
        }
        cursorCol += width;
        if (cursorCol >= cols) {
            cursorCol = cols - 1;
            pendingWrap = true;
        }
    }

    private static int charWidth(int cp) {
        if (cp == 0 || cp < 32) return 0;
        if (cp == 0x200b || cp == 0x200c || cp == 0x200d) return 0;
        if (cp >= 0x0300 && cp <= 0x036f) return 0;
        if (cp >= 0x1100 && (cp <= 0x115f
                || cp == 0x2329 || cp == 0x232a
                || (cp >= 0x2e80 && cp <= 0xa4cf && cp != 0x303f)
                || (cp >= 0xac00 && cp <= 0xd7a3)
                || (cp >= 0xf900 && cp <= 0xfaff)
                || (cp >= 0xfe10 && cp <= 0xfe19)
                || (cp >= 0xfe30 && cp <= 0xfe6f)
                || (cp >= 0xff00 && cp <= 0xff60)
                || (cp >= 0xffe0 && cp <= 0xffe6)
                || (cp >= 0x1f300 && cp <= 0x1faff)
                || (cp >= 0x20000 && cp <= 0x3fffd))) {
            return 2;
        }
        return 1;
    }

    private void lineFeed() {
        pendingWrap = false;
        if (cursorRow == scrollBottom) {
            scrollUp(1);
        } else if (cursorRow < rows - 1) {
            cursorRow++;
        }
    }

    private void reverseIndex() {
        pendingWrap = false;
        if (cursorRow == scrollTop) {
            scrollDown(1);
        } else if (cursorRow > 0) {
            cursorRow--;
        }
    }

    private void scrollUp(int count) {
        for (int i = 0; i < count; i++) {
            if (scrollTop == 0 && scrollBottom == rows - 1) {
                Line line = new Line(cols);
                System.arraycopy(cps, 0, line.cps, 0, cols);
                System.arraycopy(fgs, 0, line.fgs, 0, cols);
                System.arraycopy(bgs, 0, line.bgs, 0, cols);
                System.arraycopy(flags, 0, line.flags, 0, cols);
                pushScrollback(line);
            }
            for (int row = scrollTop; row < scrollBottom; row++) {
                System.arraycopy(cps, (row + 1) * cols, cps, row * cols, cols);
                System.arraycopy(fgs, (row + 1) * cols, fgs, row * cols, cols);
                System.arraycopy(bgs, (row + 1) * cols, bgs, row * cols, cols);
                System.arraycopy(flags, (row + 1) * cols, flags, row * cols, cols);
            }
            clearRow(scrollBottom);
        }
    }

    private void scrollDown(int count) {
        for (int i = 0; i < count; i++) {
            for (int row = scrollBottom; row > scrollTop; row--) {
                System.arraycopy(cps, (row - 1) * cols, cps, row * cols, cols);
                System.arraycopy(fgs, (row - 1) * cols, fgs, row * cols, cols);
                System.arraycopy(bgs, (row - 1) * cols, bgs, row * cols, cols);
                System.arraycopy(flags, (row - 1) * cols, flags, row * cols, cols);
            }
            clearRow(scrollTop);
        }
    }

    private void clearRow(int row) {
        int offset = row * cols;
        java.util.Arrays.fill(cps, offset, offset + cols, 0);
        java.util.Arrays.fill(fgs, offset, offset + cols, DEFAULT_FG);
        java.util.Arrays.fill(bgs, offset, offset + cols, DEFAULT_BG);
        java.util.Arrays.fill(flags, offset, offset + cols, (byte) 0);
    }

    private void clearScreen(int fromRow, int fromCol, int toRow, int toCol) {
        for (int row = fromRow; row <= toRow && row < rows; row++) {
            int start = row == fromRow ? fromCol : 0;
            int end = row == toRow ? toCol : cols - 1;
            for (int col = start; col <= end && col < cols; col++) {
                int index = row * cols + col;
                cps[index] = 0;
                fgs[index] = DEFAULT_FG;
                bgs[index] = DEFAULT_BG;
                flags[index] = 0;
            }
        }
    }

    private void resetTerminal() {
        java.util.Arrays.fill(cps, 0);
        java.util.Arrays.fill(fgs, DEFAULT_FG);
        java.util.Arrays.fill(bgs, DEFAULT_BG);
        java.util.Arrays.fill(flags, (byte) 0);
        cursorRow = 0;
        cursorCol = 0;
        savedRow = 0;
        savedCol = 0;
        scrollTop = 0;
        scrollBottom = rows - 1;
        pendingWrap = false;
        cursorVisible = true;
        scrollOffset = 0;
        resetAttributes();
        altScreen = false;
    }

    private void resetAttributes() {
        fg = DEFAULT_FG;
        bg = DEFAULT_BG;
        attr = 0;
    }

    private void handleCsi(String value, char finalChar) {
        String body = value;
        boolean privateMode = false;
        if (body.startsWith("?")) {
            privateMode = true;
            body = body.substring(1);
        } else if (body.startsWith(">") || body.startsWith("<") || body.startsWith("=")) {
            return;
        }
        int[] params = parseParams(body);
        switch (finalChar) {
            case 'A':
                cursorRow = Math.max(scrollTop, cursorRow - param(params, 0, 1));
                break;
            case 'B':
                cursorRow = Math.min(scrollBottom, cursorRow + param(params, 0, 1));
                break;
            case 'C':
                cursorCol = Math.min(cols - 1, cursorCol + param(params, 0, 1));
                break;
            case 'D':
                cursorCol = Math.max(0, cursorCol - param(params, 0, 1));
                break;
            case 'E':
                cursorRow = Math.min(scrollBottom, cursorRow + param(params, 0, 1));
                cursorCol = 0;
                break;
            case 'F':
                cursorRow = Math.max(scrollTop, cursorRow - param(params, 0, 1));
                cursorCol = 0;
                break;
            case 'G':
                cursorCol = Math.max(0, Math.min(cols - 1, param(params, 0, 1) - 1));
                break;
            case 'H':
            case 'f':
                cursorRow = Math.max(0, Math.min(rows - 1, param(params, 0, 1) - 1));
                cursorCol = Math.max(0, Math.min(cols - 1, param(params, 1, 1) - 1));
                break;
            case 'd':
                cursorRow = Math.max(0, Math.min(rows - 1, param(params, 0, 1) - 1));
                break;
            case 'J':
                eraseDisplay(param(params, 0, 0));
                break;
            case 'K':
                eraseLine(param(params, 0, 0));
                break;
            case 'L':
                insertLines(param(params, 0, 1));
                break;
            case 'M':
                deleteLines(param(params, 0, 1));
                break;
            case 'P':
                deleteChars(param(params, 0, 1));
                break;
            case '@':
                insertChars(param(params, 0, 1));
                break;
            case 'X':
                eraseChars(param(params, 0, 1));
                break;
            case 'S':
                scrollUp(param(params, 0, 1));
                break;
            case 'T':
                scrollDown(param(params, 0, 1));
                break;
            case 'm':
                applySgr(params);
                break;
            case 'r':
                scrollTop = Math.max(0, param(params, 0, 1) - 1);
                scrollBottom = Math.min(rows - 1, param(params, 1, rows) - 1);
                if (scrollBottom <= scrollTop) {
                    scrollTop = 0;
                    scrollBottom = rows - 1;
                }
                cursorRow = scrollTop;
                cursorCol = 0;
                break;
            case 's':
                savedRow = cursorRow;
                savedCol = cursorCol;
                break;
            case 'u':
                cursorRow = savedRow;
                cursorCol = savedCol;
                break;
            case 'h':
            case 'l':
                if (privateMode) handlePrivateMode(params, finalChar == 'h');
                break;
            default:
                break;
        }
        pendingWrap = false;
    }

    private void handlePrivateMode(int[] params, boolean enabled) {
        for (int value : params) {
            int mode = value < 0 ? 0 : value;
            if (mode == 25) {
                cursorVisible = enabled;
            } else if (mode == 1049 || mode == 47 || mode == 1047) {
                setAltScreen(enabled);
            }
        }
    }

    private void setAltScreen(boolean enabled) {
        if (enabled == altScreen) return;
        if (enabled) {
            savedCps = cps.clone();
            savedFgs = fgs.clone();
            savedBgs = bgs.clone();
            savedFlags = flags.clone();
            savedAltRow = cursorRow;
            savedAltCol = cursorCol;
            savedAltTop = scrollTop;
            savedAltBottom = scrollBottom;
            java.util.Arrays.fill(cps, 0);
            java.util.Arrays.fill(fgs, DEFAULT_FG);
            java.util.Arrays.fill(bgs, DEFAULT_BG);
            java.util.Arrays.fill(flags, (byte) 0);
            cursorRow = 0;
            cursorCol = 0;
            scrollTop = 0;
            scrollBottom = rows - 1;
            altScreen = true;
        } else if (savedCps != null) {
            cps = savedCps;
            fgs = savedFgs;
            bgs = savedBgs;
            flags = savedFlags;
            cursorRow = savedAltRow;
            cursorCol = savedAltCol;
            scrollTop = savedAltTop;
            scrollBottom = savedAltBottom;
            altScreen = false;
        }
        pendingWrap = false;
    }

    private void eraseDisplay(int mode) {
        if (mode == 0) {
            clearScreen(cursorRow, cursorCol, rows - 1, cols - 1);
        } else if (mode == 1) {
            clearScreen(0, 0, cursorRow, cursorCol);
        } else if (mode == 2) {
            clearScreen(0, 0, rows - 1, cols - 1);
        } else if (mode == 3) {
            sbStart = 0;
            sbCount = 0;
            scrollOffset = 0;
        }
    }

    private void eraseLine(int mode) {
        if (mode == 0) {
            clearScreen(cursorRow, cursorCol, cursorRow, cols - 1);
        } else if (mode == 1) {
            clearScreen(cursorRow, 0, cursorRow, cursorCol);
        } else if (mode == 2) {
            clearScreen(cursorRow, 0, cursorRow, cols - 1);
        }
    }

    private void insertLines(int count) {
        if (cursorRow < scrollTop || cursorRow > scrollBottom) return;
        int limit = Math.min(count, scrollBottom - cursorRow + 1);
        for (int i = 0; i < limit; i++) {
            for (int row = scrollBottom; row > cursorRow; row--) {
                System.arraycopy(cps, (row - 1) * cols, cps, row * cols, cols);
                System.arraycopy(fgs, (row - 1) * cols, fgs, row * cols, cols);
                System.arraycopy(bgs, (row - 1) * cols, bgs, row * cols, cols);
                System.arraycopy(flags, (row - 1) * cols, flags, row * cols, cols);
            }
            clearRow(cursorRow);
        }
    }

    private void deleteLines(int count) {
        if (cursorRow < scrollTop || cursorRow > scrollBottom) return;
        int limit = Math.min(count, scrollBottom - cursorRow + 1);
        for (int i = 0; i < limit; i++) {
            for (int row = cursorRow; row < scrollBottom; row++) {
                System.arraycopy(cps, (row + 1) * cols, cps, row * cols, cols);
                System.arraycopy(fgs, (row + 1) * cols, fgs, row * cols, cols);
                System.arraycopy(bgs, (row + 1) * cols, bgs, row * cols, cols);
                System.arraycopy(flags, (row + 1) * cols, flags, row * cols, cols);
            }
            clearRow(scrollBottom);
        }
    }

    private void deleteChars(int count) {
        int limit = Math.min(count, cols - cursorCol);
        int offset = cursorRow * cols + cursorCol;
        System.arraycopy(cps, offset + limit, cps, offset, cols - cursorCol - limit);
        System.arraycopy(fgs, offset + limit, fgs, offset, cols - cursorCol - limit);
        System.arraycopy(bgs, offset + limit, bgs, offset, cols - cursorCol - limit);
        System.arraycopy(flags, offset + limit, flags, offset, cols - cursorCol - limit);
        for (int col = cols - limit; col < cols; col++) {
            int index = cursorRow * cols + col;
            cps[index] = 0;
            fgs[index] = DEFAULT_FG;
            bgs[index] = DEFAULT_BG;
            flags[index] = 0;
        }
    }

    private void insertChars(int count) {
        int limit = Math.min(count, cols - cursorCol);
        int offset = cursorRow * cols + cursorCol;
        System.arraycopy(cps, offset, cps, offset + limit, cols - cursorCol - limit);
        System.arraycopy(fgs, offset, fgs, offset + limit, cols - cursorCol - limit);
        System.arraycopy(bgs, offset, bgs, offset + limit, cols - cursorCol - limit);
        System.arraycopy(flags, offset, flags, offset + limit, cols - cursorCol - limit);
        for (int col = cursorCol; col < cursorCol + limit; col++) {
            int index = cursorRow * cols + col;
            cps[index] = ' ';
            fgs[index] = fg;
            bgs[index] = bg;
            flags[index] = attr;
        }
    }

    private void eraseChars(int count) {
        int limit = Math.min(count, cols - cursorCol);
        for (int col = cursorCol; col < cursorCol + limit; col++) {
            int index = cursorRow * cols + col;
            cps[index] = 0;
            fgs[index] = DEFAULT_FG;
            bgs[index] = DEFAULT_BG;
            flags[index] = 0;
        }
    }

    private void applySgr(int[] params) {
        if (params.length == 0) {
            resetAttributes();
            return;
        }
        for (int i = 0; i < params.length; i++) {
            int code = params[i] < 0 ? 0 : params[i];
            switch (code) {
                case 0:
                    resetAttributes();
                    break;
                case 1:
                    attr |= BOLD;
                    break;
                case 2:
                    attr |= DIM;
                    break;
                case 3:
                    attr |= ITALIC;
                    break;
                case 4:
                    attr |= UNDERLINE;
                    break;
                case 7:
                    attr |= REVERSE;
                    break;
                case 9:
                    attr |= STRIKE;
                    break;
                case 21:
                case 22:
                    attr &= ~(BOLD | DIM);
                    break;
                case 23:
                    attr &= ~ITALIC;
                    break;
                case 24:
                    attr &= ~UNDERLINE;
                    break;
                case 27:
                    attr &= ~REVERSE;
                    break;
                case 29:
                    attr &= ~STRIKE;
                    break;
                case 39:
                    fg = DEFAULT_FG;
                    break;
                case 49:
                    bg = DEFAULT_BG;
                    break;
                case 38:
                    i = readExtendedColor(params, i, true);
                    break;
                case 48:
                    i = readExtendedColor(params, i, false);
                    break;
                default:
                    if (code >= 30 && code <= 37) {
                        fg = ANSI[code - 30];
                    } else if (code >= 40 && code <= 47) {
                        bg = ANSI[code - 40];
                    } else if (code >= 90 && code <= 97) {
                        fg = ANSI[code - 90 + 8];
                    } else if (code >= 100 && code <= 107) {
                        bg = ANSI[code - 100 + 8];
                    }
                    break;
            }
        }
    }

    private int readExtendedColor(int[] params, int index, boolean foreground) {
        if (index + 1 >= params.length) return index;
        int mode = params[index + 1];
        if (mode == 5 && index + 2 < params.length) {
            int color = ansi256(params[index + 2]);
            if (foreground) fg = color;
            else bg = color;
            return index + 2;
        }
        if (mode == 2 && index + 4 < params.length) {
            int color = Color.rgb(clampColor(params[index + 2]),
                    clampColor(params[index + 3]), clampColor(params[index + 4]));
            if (foreground) fg = color;
            else bg = color;
            return index + 4;
        }
        return index + 1;
    }

    private static int clampColor(int value) {
        return Math.max(0, Math.min(255, value));
    }

    private static int ansi256(int value) {
        int index = Math.max(0, Math.min(255, value));
        if (index < 16) return ANSI[index];
        if (index < 232) {
            int i = index - 16;
            return Color.rgb(level(i / 36), level((i % 36) / 6), level(i % 6));
        }
        int gray = 8 + (index - 232) * 10;
        return Color.rgb(gray, gray, gray);
    }

    private static int level(int value) {
        return value == 0 ? 0 : 55 + value * 40;
    }

    private static int param(int[] params, int index, int fallback) {
        if (index >= params.length || params[index] < 0) return fallback;
        return params[index];
    }

    private static int[] parseParams(String body) {
        if (body.isEmpty()) return new int[0];
        String[] parts = body.split(";", -1);
        java.util.ArrayList<Integer> values = new java.util.ArrayList<>(parts.length);
        for (String part : parts) {
            if (part.isEmpty()) {
                values.add(-1);
                continue;
            }
            try {
                values.add(Integer.parseInt(part));
            } catch (NumberFormatException error) {
                values.add(-1);
            }
        }
        int[] result = new int[values.size()];
        for (int i = 0; i < result.length; i++) result[i] = values.get(i);
        return result;
    }
}
