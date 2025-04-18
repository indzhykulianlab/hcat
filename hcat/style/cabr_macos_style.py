"""
#E5DDC8
Teal
#01949A
Navy Blue
#004369
Red
#DB1F48



TEST
#FAFAFA - whiteish
#E5F7F9 - light blue
#2B70AE - blue
#C25200 - orange
#27282E - slate


"""
slate = "#27282E"
whiteish = "#FAFAFA"
lightblue = "#D1DFFF"
blue = "#2B70AE"
orange = "#C25200"

darkgrey = "#121212"
smoke = "#848884"
lightgrey = "#A9A9A9"
white = "#FFFFFF"
black = "#000000"


# Harvard Pallette
brick = "#A67563"
skyblue = "#77ABD9"
cloudblue = "#A0BED9"
deepgreen = "#172601"
leafgreen = "#465902"
cloud = "#DEE7F1"
stone = "#DDD1B5"
default_colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

super_light_grey = "#ededed"

class Colors:
    BACKGROUND = white
    LIGHT_COLOR = white
    DARK_COLOR = slate
    ACCENT = orange
    DARK = black
    TEXT = black


MACOS_STYLE = f"""



QPushButton {{
    border: 1px solid {Colors.DARK} ;
    border-style: outset; 
    margin: 0.5px;
    height: 14px;
    padding: 1px 0px 1px 0px;
    background-color: {Colors.LIGHT_COLOR};
    color: {Colors.TEXT};
    font: {Colors.TEXT} Office Code Pro;
    font-weight: Bold;
    font-size: 11px;
}}
QPushButton:Hover{{
    background-color: {smoke};
}}
QPushButton:Pressed {{
    background-color: {Colors.DARK_COLOR};
}}

QPushButton#next_button {{
    border-right: 3px solid {default_colors[0]};

}}
QPushButton:Pressed#next_button {{
    background-color: {default_colors[0]};
    border-right: 3px solid {default_colors[0]};

}}

QPushButton#previous_button {{
    border-left: 3px solid {default_colors[3]};

}}

QPushButton:Pressed#previous_button {{
    background-color: {default_colors[3]};
    border-left: 3px solid {default_colors[3]};
}}

QPushButton#all_above_button{{
    color: {default_colors[2]};
    background-color: {Colors.LIGHT_COLOR};

}}
QPushButton:Pressed#all_above_button{{
    color: {default_colors[2]};
    background-color: {Colors.DARK_COLOR};

}}
QPushButton:Hover#all_above_button{{
    color: {default_colors[2]};
    background-color: {smoke};

}}

QPushButton#all_below_button{{
    color: {default_colors[3]};
    background-color: {Colors.LIGHT_COLOR};

}}
QPushButton:Pressed#all_below_button{{
    color: {default_colors[3]};
    background-color: {Colors.DARK_COLOR};

}}
QPushButton:Hover#all_below_button{{
    color: {default_colors[3]};
    background-color: {smoke};

}}

QPushButton#field_delete_button {{
    height: 12px;
    width: 12px;

}}


QMainWindow::separator {{
    background: {Colors.DARK};
    width: 1.5px;
    height: 1.5px;
    margin-left: 0px;
    padding: 0px;
    }}

QGroupBox {{
    background-color: rgba(0,0,0,0);
    border: 1px solid rgba(0,0,0,0);
    border-radius: 0px;
    margin-top: 3ex; /* leave space at the top for the title */
    font: {Colors.TEXT} Office Code Pro;
    font-weight: 300;
    font-size: 10px;
    margin-bottom: 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left; /* position at the top center */
    padding: 0px 3px;

}}


ABRViewerWidget {{
    background-color: white;
}}

ABRControls {{
    color: {Colors.LIGHT_COLOR}
}}
ABRFooter {{
    color: {Colors.TEXT};
}}

QFrame#frame {{
    border: 1.5px solid rgba(0,0,0,0);
    border-top: 1.5px solid rgba(0,0,0,255);
}}

QSpinBox{{
    width: 60px;
    border: 1px solid black;
}}

QSpinBox::up-button {{
    border-image: url(":/resources/up_selector.svg") 0;
    margin: 3px;
    padding: 1px;
}}
QSpinBox::down-button {{
    border-image: url(:/resources/down_selector.svg) 0;
    margin: 3px;
    padding: 1px;
}}

QComboBox {{
    border: 1px solid black;
    padding: 2px;
}}
QComboBox::down-arrow {{
    border-image: url(:/resources/down_selector.svg) 0;
    margin: 1px;
    margin-top: 2.5px;
    margin-bottom: 2.5px;
}}

QComboBox::drop-down {{
    background-color: white;
}}


QSlider::groove:horizontal {{
    background: {smoke};
    height: 3px; /* the groove expands to the size of the slider by default. by giving it a height, it has a fixed size */
}}

QSlider::handle:horizontal {{
    image: url(:/resources/drag_handle.svg);
    margin: -4px -4px;
}}



QCheckBox::indicator:checked {{
    background-color: white;
    image: url(:/resources/check.svg);
    width: 10px;
    height: 10px;
    padding: 1px;
    border: 1px solid black;
    margin: 2px;
    
}}

QCheckBox::indicator:unchecked {{
    background-color: white;
    width: 10px;
    height: 10px;
    padding: 1px;
    border: 1px solid black;
    margin: 2px;
    
}}

QTabWidget {{
    margin: 0px;
    padding: 20px;
    background: red;
}}
QTabBar {{
background-color: black;

}}
QTabBar::tab {{
    padding: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
    background: lightgrey;
    border: 1px solid darkgrey;
    border-bottom: 0;
}}
QTabBar::tab:selected {{
    background-color: lightgrey;
    color: black;
    font: bold;
}}
QTabBar::tab:hover{{
    background-color: {smoke};
}}

AttrRegexListWidget{{
    background: lightgrey;

}}

QScrollArea {{
    border: none;
    background: lightgrey;
}}

"""