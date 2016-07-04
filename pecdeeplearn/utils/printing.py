from __future__ import division

import time


def print_progress(elapsed_time, units_complete, total_units):

    frac_complete = units_complete / total_units
    remaining_time = ((1 - frac_complete) / frac_complete) * elapsed_time

    def hours_minutes_seconds(total_secs):

        hours, secs = divmod(total_secs, 60 ** 2)
        mins, secs = divmod(secs, 60)

        return hours, mins, secs

    elapsed_message = "{:02.0f}:{:02.0f}:{:02.0f} elapsed." \
        .format(*hours_minutes_seconds(elapsed_time))
    print('\n' + elapsed_message)

    remaining_message = "{:02.0f}:{:02.0f}:{:02.0f} maximum remaining." \
        .format(*hours_minutes_seconds(remaining_time))
    print(remaining_message + '\n')
