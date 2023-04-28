# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from textwrap import dedent
from typing import TYPE_CHECKING, Mapping, Optional, Union, cast
from urllib.parse import urlparse

from typing_extensions import Final, Literal, TypeAlias

from streamlit.elements import image
from streamlit.errors import StreamlitAPIException
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg as ForwardProto
from streamlit.proto.PageConfig_pb2 import PageConfig as PageConfigProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.string_util import is_emoji
from streamlit.util import lower_clean_dict_keys

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

GET_HELP_KEY: Final = "get help"
REPORT_A_BUG_KEY: Final = "report a bug"
ABOUT_KEY: Final = "about"

PageIcon: TypeAlias = Union[image.AtomicImage, str]
Layout: TypeAlias = Literal["centered", "wide"]
InitialSideBarState: TypeAlias = Literal["auto", "expanded", "collapsed"]
_GetHelp: TypeAlias = Literal["Get help", "Get Help", "get help"]
_ReportABug: TypeAlias = Literal["Report a bug", "report a bug"]
_About: TypeAlias = Literal["About", "about"]
MenuKey: TypeAlias = Literal[_GetHelp, _ReportABug, _About]
MenuItems: TypeAlias = Mapping[MenuKey, Optional[str]]

# Emojis recommended by https://share.streamlit.io/rensdimmendaal/emoji-recommender/main/app/streamlit.py
# for the term "streamlit". Watch out for zero-width joiners,
# as they won't parse correctly in the list() call!
RANDOM_EMOJIS: Final = list(
    "🔥™🎉🚀🌌💣✨🌙🎆🎇💥🤩🤙🌛🤘⬆💡🤪🥂⚡💨🌠🎊🍿😛🔮🤟🌃🍃🍾💫▪🌴🎈🎬🌀🎄😝☔⛽🍂💃😎🍸🎨🥳☀😍🅱🌞😻🌟😜💦💅🦄😋😉👻🍁🤤👯🌻‼🌈👌🎃💛😚🔫🙌👽🍬🌅☁🍷👭☕🌚💁👅🥰🍜😌🎥🕺❕🧡☄💕🍻✅🌸🚬🤓🍹®☺💪😙☘🤠✊🤗🍵🤞😂💯😏📻🎂💗💜🌊❣🌝😘💆🤑🌿🦋😈⛄🚿😊🌹🥴😽💋😭🖤🙆👐⚪💟☃🙈🍭💻🥀🚗🤧🍝💎💓🤝💄💖🔞⁉⏰🕊🎧☠♥🌳🏾🙉⭐💊🍳🌎🙊💸❤🔪😆🌾✈📚💀🏠✌🏃🌵🚨💂🤫🤭😗😄🍒👏🙃🖖💞😅🎅🍄🆓👉💩🔊🤷⌚👸😇🚮💏👳🏽💘💿💉👠🎼🎶🎤👗❄🔐🎵🤒🍰👓🏄🌲🎮🙂📈🚙📍😵🗣❗🌺🙄👄🚘🥺🌍🏡♦💍🌱👑👙☑👾🍩🥶📣🏼🤣☯👵🍫➡🎀😃✋🍞🙇😹🙏👼🐝⚫🎁🍪🔨🌼👆👀😳🌏📖👃🎸👧💇🔒💙😞⛅🏻🍴😼🗿🍗♠🦁✔🤖☮🐢🐎💤😀🍺😁😴📺☹😲👍🎭💚🍆🍋🔵🏁🔴🔔🧐👰☎🏆🤡🐠📲🙋📌🐬✍🔑📱💰🐱💧🎓🍕👟🐣👫🍑😸🍦👁🆗🎯📢🚶🦅🐧💢🏀🚫💑🐟🌽🏊🍟💝💲🐍🍥🐸☝♣👊⚓❌🐯🏈📰🌧👿🐳💷🐺📞🆒🍀🤐🚲🍔👹🙍🌷🙎🐥💵🔝📸⚠❓🎩✂🍼😑⬇⚾🍎💔🐔⚽💭🏌🐷🍍✖🍇📝🍊🐙👋🤔🥊🗽🐑🐘🐰💐🐴♀🐦🍓✏👂🏴👇🆘😡🏉👩💌😺✝🐼🐒🐶👺🖕👬🍉🐻🐾⬅⏬▶👮🍌♂🔸👶🐮👪⛳🐐🎾🐕👴🐨🐊🔹©🎣👦👣👨👈💬⭕📹📷"
)

# Also pick out some vanity emojis.
ENG_EMOJIS: Final = [
    "🎈",  # st.balloons 🎈🎈
    "🤓",  # Abhi
    "🏈",  # Amey
    "🚲",  # Thiago
    "🐧",  # Matteo
    "🦒",  # Ken
    "🐳",  # Karrie
    "🕹️",  # Jonathan
    "🇦🇲",  # Henrikh
    "🎸",  # Guido
    "🦈",  # Austin
    "💎",  # Emiliano
    "👩‍🎤",  # Naomi
    "🧙‍♂️",  # Jon
    "🐻",  # Brandon
    "🎎",  # James
    # TODO: Solicit emojis from the rest of Streamlit
]


def _get_favicon_string(page_icon: PageIcon) -> str:
    """Return the string to pass to the frontend to have it show
    the given PageIcon.

    If page_icon is a string that looks like an emoji (or an emoji shortcode),
    we return it as-is. Otherwise we use `image_to_url` to return a URL.

    (If `image_to_url` raises an error and page_icon is a string, return
    the unmodified page_icon string instead of re-raising the error.)
    """

    # Choose a random emoji.
    if page_icon == "random":
        return get_random_emoji()

    # If page_icon is an emoji, return it as is.
    if isinstance(page_icon, str) and is_emoji(page_icon):
        return page_icon

    # Fall back to image_to_url.
    try:
        return image.image_to_url(
            page_icon,
            width=-1,  # Always use full width for favicons
            clamp=False,
            channels="RGB",
            output_format="auto",
            image_id="favicon",
        )
    except Exception:
        if isinstance(page_icon, str):
            # This fall-thru handles emoji shortcode strings (e.g. ":shark:"),
            # which aren't valid filenames and so will cause an Exception from
            # `image_to_url`.
            return page_icon
        raise


@gather_metrics("set_page_config")
def set_page_config(
    page_title: Optional[str] = None,
    page_icon: Optional[PageIcon] = None,
    layout: Layout = "centered",
    initial_sidebar_state: InitialSideBarState = "auto",
    menu_items: Optional[MenuItems] = None,
) -> None:
    """
    Configures the default settings of the page.

    .. note::
        This must be the first Streamlit command used in your app, and must only
        be set once.

    Parameters
    ----------
    page_title: str or None
        The page title, shown in the browser tab. If None, defaults to the
        filename of the script ("app.py" would show "app • Streamlit").
    page_icon : Anything supported by st.image or str or None
        The page favicon.
        Besides the types supported by `st.image` (like URLs or numpy arrays),
        you can pass in an emoji as a string ("🦈") or a shortcode (":shark:").
        If you're feeling lucky, try "random" for a random emoji!
        Emoji icons are courtesy of Twemoji and loaded from MaxCDN.
    layout: "centered" or "wide"
        How the page content should be laid out. Defaults to "centered",
        which constrains the elements into a centered column of fixed width;
        "wide" uses the entire screen.
    initial_sidebar_state: "auto" or "expanded" or "collapsed"
        How the sidebar should start out. Defaults to "auto",
        which hides the sidebar on mobile-sized devices, and shows it otherwise.
        "expanded" shows the sidebar initially; "collapsed" hides it.
    menu_items: dict
        Configure the menu that appears on the top-right side of this app.
        The keys in this dict denote the menu item you'd like to configure:

        - "Get help": str or None
            The URL this menu item should point to.
            If None, hides this menu item.
        - "Report a Bug": str or None
            The URL this menu item should point to.
            If None, hides this menu item.
        - "About": str or None
            A markdown string to show in the About dialog.
            If None, only shows Streamlit's default About text.

        The URL may also refer to an email address e.g. ``mailto:john@example.com``.

    Example
    -------
    >>> import streamlit as st
    >>>
    >>> st.set_page_config(
    ...     page_title="Ex-stream-ly Cool App",
    ...     page_icon="🧊",
    ...     layout="wide",
    ...     initial_sidebar_state="expanded",
    ...     menu_items={
    ...         'Get Help': 'https://www.extremelycoolapp.com/help',
    ...         'Report a bug': "https://www.extremelycoolapp.com/bug",
    ...         'About': "# This is a header. This is an *extremely* cool app!"
    ...     }
    ... )
    """

    msg = ForwardProto()

    if page_title is not None:
        msg.page_config_changed.title = page_title

    if page_icon is not None:
        msg.page_config_changed.favicon = _get_favicon_string(page_icon)

    pb_layout: "PageConfigProto.Layout.ValueType"
    if layout == "centered":
        pb_layout = PageConfigProto.CENTERED
    elif layout == "wide":
        pb_layout = PageConfigProto.WIDE
    else:
        raise StreamlitAPIException(
            f'`layout` must be "centered" or "wide" (got "{layout}")'
        )
    msg.page_config_changed.layout = pb_layout

    pb_sidebar_state: "PageConfigProto.SidebarState.ValueType"
    if initial_sidebar_state == "auto":
        pb_sidebar_state = PageConfigProto.AUTO
    elif initial_sidebar_state == "expanded":
        pb_sidebar_state = PageConfigProto.EXPANDED
    elif initial_sidebar_state == "collapsed":
        pb_sidebar_state = PageConfigProto.COLLAPSED
    else:
        raise StreamlitAPIException(
            "`initial_sidebar_state` must be "
            '"auto" or "expanded" or "collapsed" '
            f'(got "{initial_sidebar_state}")'
        )

    msg.page_config_changed.initial_sidebar_state = pb_sidebar_state

    if menu_items is not None:
        lowercase_menu_items = cast(MenuItems, lower_clean_dict_keys(menu_items))
        validate_menu_items(lowercase_menu_items)
        menu_items_proto = msg.page_config_changed.menu_items
        set_menu_items_proto(lowercase_menu_items, menu_items_proto)

    ctx = get_script_run_ctx()
    if ctx is None:
        return
    ctx.enqueue(msg)


def get_random_emoji() -> str:

    # Weigh our emojis 10x, cuz we're awesome!
    # TODO: fix the random seed with a hash of the user's app code, for stability?
    return random.choice(RANDOM_EMOJIS + 10 * ENG_EMOJIS)


def set_menu_items_proto(lowercase_menu_items, menu_items_proto) -> None:
    if GET_HELP_KEY in lowercase_menu_items:
        if lowercase_menu_items[GET_HELP_KEY] is not None:
            menu_items_proto.get_help_url = lowercase_menu_items[GET_HELP_KEY]
        else:
            menu_items_proto.hide_get_help = True

    if REPORT_A_BUG_KEY in lowercase_menu_items:
        if lowercase_menu_items[REPORT_A_BUG_KEY] is not None:
            menu_items_proto.report_a_bug_url = lowercase_menu_items[REPORT_A_BUG_KEY]
        else:
            menu_items_proto.hide_report_a_bug = True

    if ABOUT_KEY in lowercase_menu_items:
        if lowercase_menu_items[ABOUT_KEY] is not None:
            menu_items_proto.about_section_md = dedent(lowercase_menu_items[ABOUT_KEY])


def validate_menu_items(menu_items: MenuItems) -> None:
    for k, v in menu_items.items():
        if not valid_menu_item_key(k):
            raise StreamlitAPIException(
                "We only accept the keys: "
                '"Get help", "Report a bug", and "About" '
                f'("{k}" is not a valid key.)'
            )
        if v is not None:
            if not valid_url(v) and k != ABOUT_KEY:
                raise StreamlitAPIException(f'"{v}" is a not a valid URL!')


def valid_menu_item_key(key: str) -> "TypeGuard[MenuKey]":
    return key in {GET_HELP_KEY, REPORT_A_BUG_KEY, ABOUT_KEY}


def valid_url(url: str) -> bool:
    # Function taken from https://stackoverflow.com/questions/7160737/how-to-validate-a-url-in-python-malformed-or-not
    try:
        result = urlparse(url)
        if result.scheme == "mailto":
            return all([result.scheme, result.path])
        return all([result.scheme, result.netloc])
    except Exception:
        return False
