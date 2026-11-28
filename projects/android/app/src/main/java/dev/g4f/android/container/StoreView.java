package dev.g4f.android.container;

import android.content.Context;
import android.graphics.Color;
import android.graphics.Typeface;
import android.text.TextWatcher;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

/** Native catalog published by `sandbox-cli list`.
 *
 *  The container owns the registry: sandbox-cli tools can be installed, while
 *  the `g4f-bin commands` group lists every bundled command and opens one in a
 *  terminal. New entries need no APK update. */
public final class StoreView extends ScrollView {

    public interface Listener {
        void onInstall(String tool);

        /** Open a shell with a bundled /g4f-bin command ready to run. */
        void onOpenCommand(String command);
    }

    private static final Pattern ANSI = Pattern.compile("\\u001b\\[[0-9;?]*[ -/]*[@-~]");
    // sandbox-cli appends this group to `list` from the container's /g4f-bin.
    private static final String BIN_GROUP = "g4f-bin commands";
    // sandbox-cli dispatches these before run_tool(), so they have no install
    // path. Show them in the catalog but do not offer an Install button.
    private static final List<String> BUILT_IN = Arrays.asList(
            "sandbox-cli", "package", "update", "upgrade", "uninstall",
            "settings", "reset", "urls", "hosts", "harness", "polli-sandbox");

    private final Listener listener;
    private final LinearLayout groups;
    private final EditText search;
    private final TextView message;
    private List<Tool> tools = new ArrayList<>();
    private String filter = "";

    public StoreView(Context context, Listener listener) {
        super(context);
        this.listener = listener;
        setBackgroundColor(Color.parseColor("#101411"));
        setFillViewport(true);

        LinearLayout root = new LinearLayout(context);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(16, 12, 16, 24);
        addView(root, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        TextView title = new TextView(context);
        title.setText("Sandbox App Store");
        title.setTextColor(Color.parseColor("#c4f06c"));
        title.setTextSize(18f);
        title.setTypeface(null, Typeface.BOLD);
        root.addView(title);

        message = new TextView(context);
        message.setTextColor(Color.parseColor("#8b9a8c"));
        message.setTextSize(12f);
        message.setPadding(0, 4, 0, 8);
        message.setText("Loading tools from sandbox-cli…");
        root.addView(message);

        search = new EditText(context);
        search.setHint("Search tools");
        search.setTextColor(Color.parseColor("#d6dad3"));
        search.setHintTextColor(Color.parseColor("#6f806f"));
        search.setSingleLine(true);
        search.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) { }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                filter = s == null ? "" : s.toString();
                render();
            }

            @Override
            public void afterTextChanged(android.text.Editable s) { }
        });
        root.addView(search, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        groups = new LinearLayout(context);
        groups.setOrientation(LinearLayout.VERTICAL);
        root.addView(groups, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
    }

    public int getToolsCount() {
        return tools.size();
    }

    public void showLoading() {
        tools = new ArrayList<>();
        filter = "";
        message.setText("Loading tools from sandbox-cli…");
        groups.removeAllViews();
    }

    public void showError(String text) {
        tools = new ArrayList<>();
        message.setText(text);
        groups.removeAllViews();
    }

    public void showTools(String output) {
        List<Tool> parsed = parse(output);
        if (parsed.isEmpty()) {
            showError("No tools found. Is sandbox-cli installed in the container?");
            return;
        }
        tools = parsed;
        message.setText(tools.size() + " entries — sandbox-cli apps to install, "
                + "g4f-bin commands to open in a terminal");
        render();
    }

    private void render() {
        groups.removeAllViews();
        String query = filter.trim().toLowerCase(Locale.ROOT);
        String currentGroup = null;
        for (Tool tool : tools) {
            if (!query.isEmpty() && !tool.matches(query)) continue;
            if (!tool.group.equals(currentGroup)) {
                currentGroup = tool.group;
                TextView header = new TextView(getContext());
                header.setText(currentGroup);
                header.setTextColor(Color.parseColor("#c4f06c"));
                header.setTextSize(14f);
                header.setTypeface(null, Typeface.BOLD);
                header.setPadding(0, 12, 0, 4);
                groups.addView(header);
            }
            LinearLayout row = new LinearLayout(getContext());
            row.setOrientation(LinearLayout.HORIZONTAL);
            row.setGravity(Gravity.CENTER_VERTICAL);
            row.setPadding(0, 2, 0, 2);

            TextView label = new TextView(getContext());
            label.setText(tool.id + "  —  " + tool.description);
            label.setTextColor(Color.parseColor("#d6dad3"));
            label.setTextSize(12f);
            row.addView(label, new LinearLayout.LayoutParams(
                    0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

            if (BIN_GROUP.equals(tool.group)) {
                // These commands ship in /g4f-bin and need no install. Open a
                // terminal with the name at the prompt so arguments can be
                // added before it runs.
                Button terminal = new Button(getContext());
                terminal.setText("Terminal");
                terminal.setTextSize(11f);
                terminal.setAllCaps(false);
                terminal.setMinWidth(0);
                terminal.setMinimumWidth(0);
                terminal.setPadding(12, 2, 12, 2);
                final String command = tool.id;
                terminal.setOnClickListener(v -> listener.onOpenCommand(command));
                row.addView(terminal);
            } else if (BUILT_IN.contains(tool.id)) {
                TextView badge = new TextView(getContext());
                badge.setText("built in");
                badge.setTextColor(Color.parseColor("#8b9a8c"));
                badge.setTextSize(11f);
                badge.setPadding(12, 2, 12, 2);
                row.addView(badge);
            } else {
                Button install = new Button(getContext());
                install.setText("Install");
                install.setTextSize(11f);
                install.setAllCaps(false);
                install.setMinWidth(0);
                install.setMinimumWidth(0);
                install.setPadding(12, 2, 12, 2);
                final String id = tool.id;
                install.setOnClickListener(v -> listener.onInstall(id));
                row.addView(install);
            }
            groups.addView(row);
        }
    }

    /** `list_tools()` prints `<Group>` followed by `  <id padded> <description>`. */
    static List<Tool> parse(String output) {
        List<Tool> result = new ArrayList<>();
        String group = "";
        if (output == null) return result;
        for (String raw : output.replace("\r\n", "\n").replace('\r', '\n').split("\n")) {
            String line = ANSI.matcher(raw).replaceAll("").trim();
            if (line.isEmpty()) continue;
            if (!raw.startsWith(" ") && !raw.startsWith("\t")) {
                group = line;
                continue;
            }
            String[] parts = line.split("\\s+", 2);
            if (group.isEmpty() || parts.length < 2) continue;
            result.add(new Tool(group, parts[0], parts[1]));
        }
        return result;
    }

    static final class Tool {
        final String group;
        final String id;
        final String description;

        Tool(String group, String id, String description) {
            this.group = group;
            this.id = id;
            this.description = description;
        }

        boolean matches(String query) {
            return id.toLowerCase(Locale.ROOT).contains(query)
                    || description.toLowerCase(Locale.ROOT).contains(query)
                    || group.toLowerCase(Locale.ROOT).contains(query);
        }
    }
}
