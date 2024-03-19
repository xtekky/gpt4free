from kivy.logger import Logger
from kivy.clock import Clock

from jnius import autoclass
from jnius import cast
from android import activity

PythonActivity = autoclass('org.kivy.android.PythonActivity')
Intent = autoclass('android.content.Intent')
Uri = autoclass('android.net.Uri')

MEDIA_DATA = "_data"
RESULT_LOAD_IMAGE = 1

Activity = autoclass('android.app.Activity')

def user_select_image(on_selection):
    """Open Gallery Activity and call callback with absolute image filepath of image user selected.
    None if user canceled.
    """

    currentActivity = cast('android.app.Activity', PythonActivity.mActivity)

    # Forum discussion: https://groups.google.com/forum/#!msg/kivy-users/bjsG2j9bptI/-Oe_aGo0newJ
    def on_activity_result(request_code, result_code, intent):
        if request_code != RESULT_LOAD_IMAGE:
            Logger.warning('user_select_image: ignoring activity result that was not RESULT_LOAD_IMAGE')
            return

        if result_code == Activity.RESULT_CANCELED:
            Clock.schedule_once(lambda dt: on_selection(None), 0)
            return

        if result_code != Activity.RESULT_OK:
            # This may just go into the void...
            raise NotImplementedError('Unknown result_code "{}"'.format(result_code))

        selectedImage = intent.getData();  # Uri
        filePathColumn = [MEDIA_DATA]; # String[]
        # Cursor
        cursor = currentActivity.getContentResolver().query(selectedImage,
                filePathColumn, None, None, None);
        cursor.moveToFirst();

        # int
        columnIndex = cursor.getColumnIndex(filePathColumn[0]);
        # String
        picturePath = cursor.getString(columnIndex);
        cursor.close();
        Logger.info('android_ui: user_select_image() selected %s', picturePath)

        # This is possibly in a different thread?
        Clock.schedule_once(lambda dt: on_selection(picturePath), 0)

    # See: http://pyjnius.readthedocs.org/en/latest/android.html
    activity.bind(on_activity_result=on_activity_result)

    intent = Intent()

    # http://programmerguru.com/android-tutorial/how-to-pick-image-from-gallery/
    # http://stackoverflow.com/questions/18416122/open-gallery-app-in-android
    intent.setAction(Intent.ACTION_PICK)
    # TODO internal vs external?
    intent.setData(Uri.parse('content://media/internal/images/media'))
    # TODO setType(Image)?

    currentActivity.startActivityForResult(intent, RESULT_LOAD_IMAGE)