package dev.g4f.android.mcp;

import android.Manifest;
import android.content.ActivityNotFoundException;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.net.Uri;
import android.provider.CalendarContract;
import android.provider.ContactsContract;
import android.provider.Settings;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.net.URLEncoder;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Collections;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * On-device actions exposed as MCP tools.
 *
 * <p>The tool names, descriptions and parameters intentionally match the
 * function calling tools from Google AI Edge Gallery's MobileActions task:
 * flashlight, contact, email, map, WiFi settings and calendar. Actions that
 * need an app (contact/email/map/calendar/WiFi) are started as Android
 * intents, so the user still confirms or edits the result in the native app.
 */
public final class MobileActions {
    private static final String TAG = "g4f-mobile-actions";
    private static final int MAX_ARG_LENGTH = 4096;

    /** A single MCP tool: JSON schema plus its on-device implementation. */
    public interface Tool {
        String name();

        String description();

        JSONObject inputSchema();

        /** Execute the action and return a JSON payload (or an "error" key). */
        JSONObject execute(Context context, JSONObject arguments) throws Exception;
    }

    private interface Action {
        JSONObject run(Context context, JSONObject arguments) throws Exception;
    }

    private static final Map<String, Tool> TOOLS = new LinkedHashMap<>();

    static {
      try {
        add("turnOnFlashlight", "Turns the flashlight on", emptySchema(),
                (context, args) -> setFlashlight(context, true));
        add("turnOffFlashlight", "Turns the flashlight off", emptySchema(),
                (context, args) -> setFlashlight(context, false));
        add("createContact", "Creates a contact in the phone's contact list.",
                stringSchema(
                        "firstName", "The first name of the contact.",
                        "lastName", "The last name of the contact.",
                        "phoneNumber", "The phone number of the contact.",
                        "email", "The email address of the contact."),
                (context, args) -> {
                    String firstName = required(args, "firstName");
                    String lastName = required(args, "lastName");
                    String phoneNumber = required(args, "phoneNumber");
                    String email = required(args, "email");
                    Intent intent = new Intent(ContactsContract.Intents.Insert.ACTION)
                            .setType(ContactsContract.RawContacts.CONTENT_TYPE)
                            .putExtra(ContactsContract.Intents.Insert.NAME, firstName + " " + lastName)
                            .putExtra(ContactsContract.Intents.Insert.EMAIL, email)
                            .putExtra(ContactsContract.Intents.Insert.EMAIL_TYPE,
                                    ContactsContract.CommonDataKinds.Email.TYPE_WORK)
                            .putExtra(ContactsContract.Intents.Insert.PHONE, phoneNumber)
                            .putExtra(ContactsContract.Intents.Insert.PHONE_TYPE,
                                    ContactsContract.CommonDataKinds.Phone.TYPE_WORK);
                    JSONObject result = startActivity(context, intent, "create a contact");
                    if (!result.has("error")) {
                        result.put("first_name", firstName);
                        result.put("last_name", lastName);
                        result.put("phone_number", phoneNumber);
                        result.put("email", email);
                    }
                    return result;
                });
        add("sendEmail", "Sends an email.",
                stringSchema(
                        "to", "The email address of the recipient.",
                        "subject", "The subject of the email.",
                        "body", "The body of the email."),
                (context, args) -> {
                    String to = required(args, "to");
                    String subject = required(args, "subject");
                    String body = required(args, "body");
                    Intent intent = new Intent(Intent.ACTION_SEND)
                            .setType("text/plain")
                            .putExtra(Intent.EXTRA_EMAIL, new String[] { to })
                            .putExtra(Intent.EXTRA_SUBJECT, subject)
                            .putExtra(Intent.EXTRA_TEXT, body);
                    JSONObject result = startActivity(context, intent, "send an email");
                    if (!result.has("error")) {
                        result.put("to", to);
                        result.put("subject", subject);
                        result.put("body", body);
                    }
                    return result;
                });
        add("showLocationOnMap", "Shows a location on the map.",
                stringSchema("location",
                        "The location to search for. May be the name of a place, "
                                + "a business, or an address."),
                (context, args) -> {
                    String location = required(args, "location");
                    String encoded = URLEncoder.encode(location, "UTF-8");
                    Intent intent = new Intent(Intent.ACTION_VIEW)
                            .setData(Uri.parse("geo:0,0?q=" + encoded));
                    JSONObject result = startActivity(context, intent, "show the location on a map");
                    if (!result.has("error")) result.put("location", location);
                    return result;
                });
        add("openWifiSettings", "Opens the WiFi settings.", emptySchema(),
                (context, args) -> startActivity(
                        context, new Intent(Settings.ACTION_WIFI_SETTINGS), "open WiFi settings"));
        add("createCalendarEvent", "Creates a new calendar event.",
                stringSchema(
                        "datetime", "The date and time of the event in the format YYYY-MM-DDTHH:MM:SS.",
                        "title", "The title of the event."),
                (context, args) -> {
                    String datetime = required(args, "datetime");
                    String title = required(args, "title");
                    long startTime = parseDateTime(datetime);
                    Intent intent = new Intent(Intent.ACTION_INSERT)
                            .setData(CalendarContract.Events.CONTENT_URI)
                            .putExtra(CalendarContract.Events.TITLE, title)
                            .putExtra(CalendarContract.EXTRA_EVENT_BEGIN_TIME, startTime)
                            .putExtra(CalendarContract.EXTRA_EVENT_END_TIME, startTime + 3600000L);
                    JSONObject result = startActivity(context, intent, "create a calendar event");
                    if (!result.has("error")) {
                        result.put("datetime", datetime);
                        result.put("title", title);
                    }
                    return result;
                });
      } catch (JSONException e) {
          throw new ExceptionInInitializerError(e);
      }
    }

    private MobileActions() {
    }

    private static void add(String name, String description, JSONObject schema, Action action) {
        TOOLS.put(name, new Tool() {
            @Override
            public String name() {
                return name;
            }

            @Override
            public String description() {
                return description;
            }

            @Override
            public JSONObject inputSchema() {
                return schema;
            }

            @Override
            public JSONObject execute(Context context, JSONObject arguments) throws Exception {
                return action.run(context, arguments == null ? new JSONObject() : arguments);
            }
        });
    }

    public static Map<String, Tool> tools() {
        return Collections.unmodifiableMap(TOOLS);
    }

    private static JSONObject emptySchema() throws JSONException {
        return new JSONObject()
                .put("type", "object")
                .put("properties", new JSONObject())
                .put("required", new JSONArray());
    }

    /** Build a JSON schema whose properties are all required strings. */
    private static JSONObject stringSchema(String... namesAndDescriptions) throws JSONException {
        JSONObject properties = new JSONObject();
        JSONArray required = new JSONArray();
        for (int i = 0; i < namesAndDescriptions.length; i += 2) {
            String name = namesAndDescriptions[i];
            properties.put(name, new JSONObject()
                    .put("type", "string")
                    .put("description", namesAndDescriptions[i + 1]));
            required.put(name);
        }
        return new JSONObject()
                .put("type", "object")
                .put("properties", properties)
                .put("required", required);
    }

    private static String required(JSONObject args, String key) {
        Object value = args.opt(key);
        if (!(value instanceof String) || ((String) value).trim().isEmpty()) {
            throw new IllegalArgumentException("Argument '" + key + "' must be a non-empty string");
        }
        String text = (String) value;
        if (text.length() > MAX_ARG_LENGTH) {
            throw new IllegalArgumentException(
                    "Argument '" + key + "' is too long (max " + MAX_ARG_LENGTH + " characters)");
        }
        return text;
    }

    private static long parseDateTime(String datetime) {
        SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US);
        format.setLenient(false);
        try {
            Date parsed = format.parse(datetime);
            if (parsed == null) throw new IllegalArgumentException("Invalid datetime");
            return parsed.getTime();
        } catch (ParseException e) {
            throw new IllegalArgumentException(
                    "datetime must use the format YYYY-MM-DDTHH:MM:SS");
        }
    }

    private static JSONObject success() throws JSONException {
        return new JSONObject().put("result", "success");
    }

    private static JSONObject error(String message) throws JSONException {
        return new JSONObject().put("error", message);
    }

    private static JSONObject setFlashlight(Context context, boolean enabled) throws JSONException {
        if (context.checkSelfPermission(Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            return error("Camera permission is required to control the flashlight. "
                    + "Allow Camera for G4F in Android Settings, then retry.");
        }
        CameraManager manager = (CameraManager) context.getSystemService(Context.CAMERA_SERVICE);
        if (manager == null) return error("The camera service is unavailable");
        try {
            String cameraId = null;
            for (String id : manager.getCameraIdList()) {
                Boolean available = manager.getCameraCharacteristics(id)
                        .get(CameraCharacteristics.FLASH_INFO_AVAILABLE);
                if (Boolean.TRUE.equals(available)) {
                    cameraId = id;
                    break;
                }
            }
            if (cameraId == null) return error("This device has no camera with a flash unit");
            manager.setTorchMode(cameraId, enabled);
            return success().put("enabled", enabled);
        } catch (Exception e) {
            Log.w(TAG, "Flashlight failed", e);
            return error("Could not set the flashlight: " + e.getClass().getSimpleName());
        }
    }

    /** Start an intent from the application context (no activity in the MCP service). */
    private static JSONObject startActivity(Context context, Intent intent, String actionName)
            throws JSONException {
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        try {
            context.startActivity(intent);
            return success();
        } catch (ActivityNotFoundException e) {
            Log.w(TAG, "No app found to " + actionName);
            return error("No app is installed to " + actionName + ".");
        } catch (Exception e) {
            Log.w(TAG, "Failed to " + actionName, e);
            return error("Failed to " + actionName + ": " + e.getClass().getSimpleName());
        }
    }
}
