import torch
import torch.nn as nn


def weight_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight.data, 0, 0.01)
            m.bias.data.zero_()


def color_table(color_name):
    if color_name == "black":
        return "0"
    elif color_name == "red":
        return "1"
    elif color_name == "green":
        return "2"
    elif color_name == "yellow":
        return "3"
    elif color_name == "blue":
        return "4"
    elif color_name == "purple":
        return "5"
    elif color_name == "cyan":
        return "6"
    elif color_name == "white":
        return "7"
    else:
        raise ValueError("Invalid color name!")


def console_highlight(input_string, font_color='', background_color='', flash=False, bold=False):
    prefix = "\033["

    if len(font_color) != 0:
        if prefix[-1].isdigit():
            prefix = prefix + ";" + "3" + color_table(font_color)
        else:
            prefix = prefix + "3" + color_table(font_color)

    if len(background_color) != 0:
        if prefix[-1].isdigit():
            prefix = prefix + ";" + "4" + color_table(background_color)
        else:
            prefix = prefix + "4" + color_table(background_color)

    if flash:
        if prefix[-1].isdigit():
            prefix = prefix + ";" + "5"
        else:
            prefix = prefix + "5"

    if bold:
        if prefix[-1].isdigit():
            prefix = prefix + ";" + "1"
        else:
            prefix = prefix + "1"

    prefix = prefix + "m"

    return prefix + input_string + "\033[0m"


def print_red_info(msg, thread=None):
    if thread is not None:
        print(console_highlight('PROCESS[%d] > [INFO]: %s' % (thread, msg), 'red'))
    else:
        print(console_highlight('[INFO]: %s' % (msg), 'red'))


def print_blue_info(msg, thread=None):
    if thread is not None:
        print(console_highlight('PROCESS[%d] > [INFO]: %s' % (thread, msg), 'blue'))
    else:
        print(console_highlight('[INFO]: %s' % (msg), 'blue'))


def print_green_info(msg, thread=None):
    if thread is not None:
        print(console_highlight('PROCESS[%d] > [INFO]: %s' % (thread, msg), 'green'))
    else:
        print(console_highlight('[INFO]: %s' % (msg), 'green'))


def print_purple_info(msg, thread=None):
    if thread is not None:
        print(console_highlight('PROCESS[%d] > [INFO]: %s' % (thread, msg), 'purple'))
    else:
        print(console_highlight('[INFO]: %s' % (msg), 'purple'))
